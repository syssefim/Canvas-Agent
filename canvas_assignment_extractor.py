"""Export assignments for every Canvas course listed in ``courses.json``.

The script reuses a Playwright browser profile for interactive authentication,
then reads assignments through Canvas's documented REST API. Successful exports
are written atomically. Incomplete exports are saved separately so a transient
failure cannot replace the last known-good dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlencode, urljoin, urlparse

from playwright.sync_api import APIRequestContext, BrowserContext
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CANVAS_BASE_URL = os.environ.get(
    "CANVAS_BASE_URL", "https://sfsu.instructure.com"
)
DEFAULT_PROFILE_DIR = Path(
    os.environ.get("CANVAS_PROFILE_DIR", str(PROJECT_DIR / ".canvas-profile"))
)
DEFAULT_HISTORY_DIR = Path(
    os.environ.get(
        "CANVAS_ASSIGNMENT_HISTORY_DIR",
        str(PROJECT_DIR / "assignment_history"),
    )
)
DEFAULT_COURSES_FILE = PROJECT_DIR / "courses.json"
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "assignments.json"

PER_PAGE = 100
DEFAULT_REQUEST_TIMEOUT_MS = 60_000
DEFAULT_MAX_ATTEMPTS = 4
MAX_RETRY_DELAY_SECONDS = 60.0


class CanvasExtractorError(RuntimeError):
    """Base class for expected, user-actionable extraction failures."""


class CanvasHTTPError(CanvasExtractorError):
    """A non-success response from Canvas."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class CanvasResponseError(CanvasExtractorError):
    """A successful HTTP response with an unexpected body or header."""


class CanvasAuthenticationError(CanvasExtractorError):
    """The browser session is not authenticated to the Canvas API."""


class AssignmentValidationError(CanvasExtractorError):
    """A Canvas assignment does not satisfy the expected minimal schema."""


def normalize_base_url(value: str) -> str:
    """Validate and normalize the origin used for Canvas API requests."""
    parsed = urlparse(value.strip())
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "Canvas base URL must be an HTTPS origin, for example "
            "https://school.instructure.com"
        )

    try:
        parsed.port
    except ValueError as error:
        raise ValueError(f"Canvas base URL has an invalid port: {value!r}") from error

    return f"https://{parsed.netloc.rstrip('/')}"


def _origin(url: str) -> tuple[str, str, int] | None:
    parsed = urlparse(url)
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return None
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError:
        return None
    return (parsed.scheme.lower(), parsed.hostname.lower(), port)


def _is_safe_canvas_url(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme.lower() == "https"
        and not parsed.fragment
        and _origin(url) == _origin(base_url)
    )


def _safe_embedded_url(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme.lower() in {"http", "https", "mailto"}:
        return value
    if not parsed.scheme and value.startswith("/"):
        return value
    return None


class _HTMLTextExtractor(HTMLParser):
    """Turn Canvas rich HTML into compact text without discarding useful links."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "main",
        "nav", "ol", "p", "pre", "section", "table", "tr", "ul",
    }
    IGNORED_TAGS = {"script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0
        self.anchors: list[tuple[str | None, int]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return

        if tag == "li":
            self.parts.append("\n- ")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

        if tag == "a":
            self.anchors.append(
                (_safe_embedded_url(attributes.get("href")), len(self.parts))
            )
        elif tag == "img":
            alt = " ".join((attributes.get("alt") or "").split())
            src = _safe_embedded_url(attributes.get("src"))
            if alt and src:
                self.parts.append(f"{alt} ({src})")
            elif alt:
                self.parts.append(alt)
            elif src:
                self.parts.append(src)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.IGNORED_TAGS:
            self.ignored_depth = max(0, self.ignored_depth - 1)
            return
        if self.ignored_depth:
            return

        if tag == "a" and self.anchors:
            href, start_index = self.anchors.pop()
            visible_text = "".join(self.parts[start_index:])
            if href and href not in visible_text:
                self.parts.append(f" ({href})")

        if tag == "li" or tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts).replace("\xa0", " ")
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        value = re.sub(r"(?: \| ){2,}", " | ", value)
        return value.strip(" \n|")


def html_to_text(value: str | None) -> str | None:
    if not value:
        return None
    if not isinstance(value, str):
        raise AssignmentValidationError("description must be a string or null")

    parser = _HTMLTextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text() or None


def load_courses(
    path: Path, base_url: str = DEFAULT_CANVAS_BASE_URL
) -> list[dict[str, str]]:
    """Read and validate the small course records produced by the first script."""
    try:
        with path.open(encoding="utf-8") as file:
            raw_courses = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(
            f"Course file not found: {path}. Run canvas_course_extractor.py first."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Course file is not valid JSON: {path}: {error}") from error

    if not isinstance(raw_courses, list):
        raise ValueError(f"Expected a JSON list in {path}")

    courses: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, course in enumerate(raw_courses):
        if not isinstance(course, dict):
            raise ValueError(f"Course #{index + 1} is not a JSON object")

        raw_id = course.get("id")
        course_id = "" if isinstance(raw_id, bool) else str(raw_id or "").strip()
        raw_name = course.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not course_id.isdigit():
            raise ValueError(f"Course #{index + 1} has an invalid id: {course_id!r}")
        if raw_name is not None and not isinstance(raw_name, str):
            raise ValueError(f"Course #{index + 1} has a non-string name")
        if course_id in seen_ids:
            continue

        seen_ids.add(course_id)
        courses.append(
            {
                "id": course_id,
                "name": name or f"Course {course_id}",
                "url": f"{base_url}/courses/{course_id}",
            }
        )
    return courses


def is_logged_in_to_canvas(
    url: str, base_url: str = DEFAULT_CANVAS_BASE_URL
) -> bool:
    parsed = urlparse(url)
    return _origin(url) == _origin(base_url) and not parsed.path.startswith("/login")


def _normalized_headers(headers: dict[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _retry_after_seconds(headers: dict[str, str], fallback: float) -> float:
    value = headers.get("retry-after")
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = fallback
        return min(MAX_RETRY_DELAY_SECONDS, max(0.0, delay))
    return min(MAX_RETRY_DELAY_SECONDS, fallback)


def request_json(
    request: APIRequestContext,
    url: str,
    description: str,
    *,
    timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Any, dict[str, str]]:
    """GET JSON with bounded retries and prompt response-body disposal."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            response = request.get(
                url, timeout=timeout_ms, fail_on_status_code=False
            )
        except PlaywrightError as error:
            if attempt == max_attempts:
                raise CanvasExtractorError(
                    f"Canvas request failed for {description} after "
                    f"{max_attempts} attempt(s): {error}"
                ) from error
            sleeper(min(MAX_RETRY_DELAY_SECONDS, 2 ** (attempt - 1)))
            continue

        retry_delay: float | None = None
        try:
            headers = _normalized_headers(response.headers)
            if response.ok:
                try:
                    return response.json(), headers
                except Exception as error:
                    raise CanvasResponseError(
                        f"Canvas returned non-JSON data for {description}; "
                        "the login session may have expired"
                    ) from error

            status = response.status
            if (status == 429 or 500 <= status < 600) and attempt < max_attempts:
                retry_delay = _retry_after_seconds(
                    headers, fallback=float(2 ** (attempt - 1))
                )
            else:
                detail = " ".join(response.text()[:500].split())
                suffix = f": {detail}" if detail and status not in {401, 403} else ""
                raise CanvasHTTPError(
                    status,
                    f"Canvas returned HTTP {status} for {description}{suffix}",
                )
        finally:
            response.dispose()

        if retry_delay is not None:
            sleeper(retry_delay)

    raise AssertionError("request retry loop terminated unexpectedly")


def verify_canvas_api(
    context: BrowserContext,
    base_url: str,
    *,
    timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> None:
    """Verify that browser cookies authenticate an actual Canvas API call."""
    try:
        profile, _ = request_json(
            context.request,
            f"{base_url}/api/v1/users/self/profile",
            "the current-user profile",
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
        )
    except CanvasHTTPError as error:
        if error.status in {401, 403}:
            raise CanvasAuthenticationError(
                "The saved Canvas login is missing or expired."
            ) from error
        raise
    except CanvasResponseError as error:
        raise CanvasAuthenticationError(
            "Canvas did not return an authenticated API response."
        ) from error

    if not isinstance(profile, dict) or profile.get("id") is None:
        raise CanvasAuthenticationError(
            "Canvas returned an unexpected current-user profile."
        )


def ensure_canvas_login(
    context: BrowserContext,
    base_url: str,
    login_url: str,
    *,
    headless: bool,
    timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    courses_url = f"{base_url}/courses"
    page.goto(courses_url, wait_until="domcontentloaded")

    if is_logged_in_to_canvas(page.url, base_url):
        try:
            verify_canvas_api(
                context,
                base_url,
                timeout_ms=timeout_ms,
                max_attempts=max_attempts,
            )
            print("Using the existing Canvas login.")
            return
        except CanvasAuthenticationError:
            pass

    if headless:
        raise CanvasAuthenticationError(
            "The saved Canvas login is missing or expired; rerun without --headless."
        )

    page.goto(login_url, wait_until="domcontentloaded")
    print("\nLog into Canvas in the browser window.")
    print("Wait until you can see your Canvas dashboard.")
    input("\nPress Enter after you're logged in... ")

    page.goto(courses_url, wait_until="domcontentloaded")
    if not is_logged_in_to_canvas(page.url, base_url):
        raise CanvasAuthenticationError("Canvas login is not complete.")
    verify_canvas_api(
        context,
        base_url,
        timeout_ms=timeout_ms,
        max_attempts=max_attempts,
    )


def _split_link_header(value: str) -> list[str]:
    """Split a Link header without splitting commas inside URLs or quotes."""
    parts: list[str] = []
    start = 0
    in_angle = False
    in_quote = False
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_quote:
            escaped = True
        elif character == '"':
            in_quote = not in_quote
        elif character == "<" and not in_quote:
            in_angle = True
        elif character == ">" and not in_quote:
            in_angle = False
        elif character == "," and not in_angle and not in_quote:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def next_link_from_header(value: str | None) -> str | None:
    if not value:
        return None

    next_urls: list[str] = []
    parameter_pattern = re.compile(
        r";\s*([^=;\s]+)\s*=\s*(?:\"([^\"]*)\"|([^;,\s]+))"
    )
    for part in _split_link_header(value):
        match = re.match(r"^\s*<([^>]+)>(.*)$", part)
        if not match:
            raise CanvasResponseError("Canvas returned a malformed Link header")
        link_url, parameters = match.groups()
        relations: list[str] = []
        for parameter in parameter_pattern.finditer(parameters):
            name = parameter.group(1).lower()
            parameter_value = parameter.group(2) or parameter.group(3) or ""
            if name == "rel":
                relations.extend(item.lower() for item in parameter_value.split())
        if "next" in relations:
            next_urls.append(link_url)

    if len(next_urls) > 1:
        raise CanvasResponseError("Canvas returned multiple next-page links")
    return next_urls[0] if next_urls else None


def _assignments_url(base_url: str, course_id: str, page_number: int) -> str:
    query = urlencode(
        [
            ("per_page", str(PER_PAGE)),
            ("page", str(page_number)),
            ("order_by", "due_at"),
            ("include[]", "submission"),
            ("include[]", "all_dates"),
        ]
    )
    return f"{base_url}/api/v1/courses/{quote(course_id, safe='')}/assignments?{query}"


def fetch_course_assignments(
    context: BrowserContext,
    course_id: str,
    base_url: str = DEFAULT_CANVAS_BASE_URL,
    *,
    timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[Any]:
    """Fetch every page of assignments visible to the logged-in Canvas user."""
    assignments: list[Any] = []
    page_number = 1
    url = _assignments_url(base_url, course_id, page_number)
    seen_urls: set[str] = set()

    while url:
        if url in seen_urls:
            raise CanvasResponseError(
                f"Canvas repeated a pagination URL while fetching course {course_id}"
            )
        if not _is_safe_canvas_url(url, base_url):
            raise CanvasResponseError(
                f"Canvas returned an unsafe pagination URL for course {course_id}"
            )
        seen_urls.add(url)

        page_items, headers = request_json(
            context.request,
            url,
            f"assignments for course {course_id}",
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
            sleeper=sleeper,
        )
        if not isinstance(page_items, list):
            raise CanvasResponseError(
                f"Canvas returned an unexpected assignments response for course {course_id}"
            )
        assignments.extend(page_items)

        link_header = headers.get("link")
        next_url = next_link_from_header(link_header)
        if next_url:
            url = urljoin(url, next_url)
            continue
        if link_header is not None or len(page_items) < PER_PAGE:
            break

        # Defensive fallback for intermediaries that strip Link headers.
        page_number += 1
        url = _assignments_url(base_url, course_id, page_number)

    return assignments


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AssignmentValidationError(f"{field} must be a string or null")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AssignmentValidationError(f"{field} must be a list of strings")
    return value


def submission_summary(submission: Any) -> dict[str, Any] | None:
    if submission is None:
        return None
    if not isinstance(submission, dict):
        raise AssignmentValidationError("submission must be an object or null")

    result = {
        "workflow_state": _optional_string(
            submission.get("workflow_state"), "submission.workflow_state"
        ),
        "submitted_at": _optional_string(
            submission.get("submitted_at"), "submission.submitted_at"
        ),
        "graded_at": _optional_string(
            submission.get("graded_at"), "submission.graded_at"
        ),
        "score": submission.get("score"),
        "grade": submission.get("grade"),
        "attempt": submission.get("attempt"),
    }
    for field in ("late", "missing", "excused"):
        value = submission.get(field, False)
        if not isinstance(value, bool):
            raise AssignmentValidationError(f"submission.{field} must be a boolean")
        result[field] = value
    return result


def compact_assignment(assignment: Any) -> dict[str, Any]:
    """Validate and retain the assignment fields used by downstream consumers."""
    if not isinstance(assignment, dict):
        raise AssignmentValidationError("assignment must be an object")

    raw_id = assignment.get("id")
    assignment_id = "" if isinstance(raw_id, bool) else str(raw_id or "").strip()
    if not assignment_id.isdigit():
        raise AssignmentValidationError("assignment has a missing or invalid id")

    name = assignment.get("name")
    if not isinstance(name, str) or not name.strip():
        raise AssignmentValidationError(
            f"assignment {assignment_id} has a missing or invalid name"
        )

    published = assignment.get("published", False)
    if not isinstance(published, bool):
        raise AssignmentValidationError(
            f"assignment {assignment_id} published must be a boolean"
        )

    points_possible = assignment.get("points_possible")
    if points_possible is not None and (
        isinstance(points_possible, bool)
        or not isinstance(points_possible, (int, float))
    ):
        raise AssignmentValidationError(
            f"assignment {assignment_id} points_possible must be numeric or null"
        )

    result: dict[str, Any] = {
        "id": assignment_id,
        "name": name,
        "due_at": _optional_string(assignment.get("due_at"), "due_at"),
        "unlock_at": _optional_string(assignment.get("unlock_at"), "unlock_at"),
        "lock_at": _optional_string(assignment.get("lock_at"), "lock_at"),
        "points_possible": points_possible,
        "published": published,
        "html_url": _optional_string(assignment.get("html_url"), "html_url"),
        "submission_types": _string_list(
            assignment.get("submission_types"), "submission_types"
        ),
        "allowed_extensions": _string_list(
            assignment.get("allowed_extensions"), "allowed_extensions"
        ),
        "description_html": _optional_string(
            assignment.get("description"), "description"
        ),
        "description_text": html_to_text(assignment.get("description")),
        "submission": submission_summary(assignment.get("submission")),
    }

    all_dates = assignment.get("all_dates")
    if all_dates is not None:
        if not isinstance(all_dates, list) or any(
            not isinstance(item, dict) for item in all_dates
        ):
            raise AssignmentValidationError("all_dates must be a list of objects")
        result["all_dates"] = all_dates
    return result


def prepare_assignments(
    raw_assignments: list[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate assignments, reject duplicates, and retain diagnostics."""
    assignments: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    seen_ids: set[str] = set()

    for index, raw_assignment in enumerate(raw_assignments, start=1):
        try:
            assignment = compact_assignment(raw_assignment)
            assignment_id = assignment["id"]
            if assignment_id in seen_ids:
                raise AssignmentValidationError(
                    f"duplicate assignment id {assignment_id}"
                )
            seen_ids.add(assignment_id)
            assignments.append(assignment)
        except AssignmentValidationError as error:
            validation_errors.append(f"Assignment record #{index}: {error}")

    assignments.sort(key=sort_key)
    return assignments, validation_errors


def sort_key(assignment: dict[str, Any]) -> tuple[bool, str, str]:
    due_at = assignment.get("due_at")
    return (due_at is None, due_at or "", str(assignment.get("name") or ""))


def build_export(
    base_url: str,
    courses: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    assignment_count = sum(len(course["assignments"]) for course in courses)
    dated_count = sum(
        assignment["due_at"] is not None
        for course in courses
        for assignment in course["assignments"]
    )
    failed_count = sum(course["fetch_error"] is not None for course in courses)
    rejected_count = sum(course["rejected_assignment_count"] for course in courses)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canvas_base_url": base_url,
        "complete": complete,
        "summary": {
            "course_count": len(courses),
            "assignment_count": assignment_count,
            "assignments_with_due_dates": dated_count,
            "assignments_without_due_dates": assignment_count - dated_count,
            "failed_course_count": failed_count,
            "rejected_assignment_count": rejected_count,
        },
        "courses": courses,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON beside its destination and atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_name = file.name
            json.dump(value, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def partial_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(
            f"{output_path.stem}.partial{output_path.suffix}"
        )
    return output_path.with_name(f"{output_path.name}.partial")


def history_snapshot_path(
    history_dir: Path,
    generated_at: str,
    *,
    complete: bool,
) -> Path:
    """Return a sortable, collision-safe filename for one export snapshot."""
    try:
        generated_time = datetime.fromisoformat(generated_at)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid export timestamp: {generated_at!r}") from error
    if generated_time.tzinfo is None:
        generated_time = generated_time.replace(tzinfo=timezone.utc)

    timestamp = generated_time.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%S.%fZ"
    )
    status = "complete" if complete else "partial"
    candidate = history_dir / f"{timestamp}.{status}.json"
    collision_number = 2
    while candidate.exists():
        candidate = history_dir / (
            f"{timestamp}.{status}.{collision_number}.json"
        )
        collision_number += 1
    return candidate


def write_export_files(
    destination: Path,
    history_dir: Path,
    export: dict[str, Any],
) -> Path:
    """Archive an export, then update its complete or partial latest file."""
    snapshot = history_snapshot_path(
        history_dir,
        export["generated_at"],
        complete=export["complete"],
    )
    atomic_write_json(snapshot, export)
    atomic_write_json(destination, export)
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Canvas assignments for the courses in courses.json."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_CANVAS_BASE_URL,
        help="Canvas HTTPS origin (or set CANVAS_BASE_URL)",
    )
    parser.add_argument(
        "--login-url",
        help="interactive login URL (default: BASE_URL/login/saml)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="persistent Playwright profile (or set CANVAS_PROFILE_DIR)",
    )
    parser.add_argument(
        "--courses",
        type=Path,
        default=DEFAULT_COURSES_FILE,
        help=f"course JSON file (default: {DEFAULT_COURSES_FILE.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"successful output JSON file (default: {DEFAULT_OUTPUT_FILE.name})",
    )
    parser.add_argument(
        "--partial-output",
        type=Path,
        help="incomplete output path (default: OUTPUT with .partial before suffix)",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help=(
            "directory for timestamped snapshots "
            "(or set CANVAS_ASSIGNMENT_HISTORY_DIR)"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_MS / 1000,
        metavar="SECONDS",
        help="timeout for each Canvas API request (default: 60)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="maximum attempts for transient API failures (default: 4)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="hide the browser (requires a valid saved login)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_url = normalize_base_url(args.base_url)
        if args.request_timeout <= 0:
            raise ValueError("--request-timeout must be greater than zero")
        if args.max_attempts < 1:
            raise ValueError("--max-attempts must be at least 1")
        timeout_ms = int(args.request_timeout * 1000)

        output_path = args.output.resolve()
        incomplete_path = (
            args.partial_output.resolve()
            if args.partial_output
            else partial_output_path(output_path)
        )
        if output_path == incomplete_path:
            raise ValueError("--output and --partial-output must be different paths")
        history_dir = args.history_dir.resolve()
        courses = load_courses(args.courses.resolve(), base_url)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    if not courses:
        print(f"Error: no courses were found in {args.courses}")
        return 1

    login_url = args.login_url or f"{base_url}/login/saml"
    exported_courses: list[dict[str, Any]] = []

    try:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(args.profile_dir.resolve()),
                headless=args.headless,
            )
            try:
                ensure_canvas_login(
                    context,
                    base_url,
                    login_url,
                    headless=args.headless,
                    timeout_ms=timeout_ms,
                    max_attempts=args.max_attempts,
                )

                for index, course in enumerate(courses, start=1):
                    print(
                        f"[{index}/{len(courses)}] Fetching assignments for "
                        f"{course['name']}..."
                    )
                    fetch_error: str | None = None
                    validation_errors: list[str] = []
                    try:
                        raw_assignments = fetch_course_assignments(
                            context,
                            course["id"],
                            base_url,
                            timeout_ms=timeout_ms,
                            max_attempts=args.max_attempts,
                        )
                        assignments, validation_errors = prepare_assignments(
                            raw_assignments
                        )
                    except CanvasExtractorError as error:
                        assignments = []
                        fetch_error = str(error)
                        print(f"  Error: {fetch_error}")

                    if validation_errors:
                        print(
                            f"  Warning: rejected {len(validation_errors)} invalid "
                            "assignment record(s)."
                        )
                    exported_courses.append(
                        {
                            **course,
                            "assignment_count": len(assignments),
                            "rejected_assignment_count": len(validation_errors),
                            "fetch_error": fetch_error,
                            "validation_errors": validation_errors,
                            "assignments": assignments,
                        }
                    )
                    print(f"  Found {len(assignments)} valid assignments.")
            finally:
                context.close()
    except Exception as error:
        print(f"Error: {error}")
        return 1

    failed_count = sum(
        course["fetch_error"] is not None for course in exported_courses
    )
    rejected_count = sum(
        course["rejected_assignment_count"] for course in exported_courses
    )
    complete = failed_count == 0 and rejected_count == 0
    export = build_export(base_url, exported_courses, complete=complete)
    destination = output_path if complete else incomplete_path

    try:
        snapshot_path = write_export_files(destination, history_dir, export)
    except (OSError, ValueError) as error:
        print(f"Error: could not archive and write the export: {error}")
        return 1

    total = export["summary"]["assignment_count"]
    print(f"\nArchived this pull at {snapshot_path}")
    if complete:
        print(f"Saved {total} assignments to {destination}")
        return 0

    print(f"Saved an incomplete export with {total} assignments to {destination}")
    print(f"The last successful export at {output_path} was not replaced.")
    if failed_count:
        print(f"Warning: {failed_count} course(s) could not be fetched.")
    if rejected_count:
        print(f"Warning: {rejected_count} assignment record(s) were rejected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
