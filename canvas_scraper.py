#!/usr/bin/env python3
"""Export the Canvas data visible to the currently signed-in user.

This uses Canvas's JSON API through an authenticated Playwright browser context.
It does not ask for, print, or save a Canvas API token.  On the first run a
Chromium window opens so the user can complete their school's normal login.

Setup:
    python3 -m venv .venv
    .venv/bin/pip install "playwright>=1.52,<2"
    .venv/bin/playwright install chromium

Run:
    .venv/bin/python canvas_scraper.py --base-url https://school.instructure.com

The default base URL is SFSU Canvas.  Set CANVAS_BASE_URL or pass --base-url for
another school.  Run with --headless after the browser profile has a valid login.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlencode, urlparse


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = os.environ.get(
    "CANVAS_BASE_URL", "https://sfsu.instructure.com"
)
DEFAULT_PROFILE_DIR = Path(
    os.environ.get("CANVAS_PROFILE_DIR", PROJECT_DIR / ".canvas-profile")
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("CANVAS_OUTPUT_DIR", PROJECT_DIR / "extracted_canvas_data")
)
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_ATTEMPTS = 4
PER_PAGE = 100
MAX_RETRY_SECONDS = 60.0


class CanvasExportError(RuntimeError):
    """A user-actionable Canvas export failure."""


class CanvasHTTPError(CanvasExportError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class CanvasResponseError(CanvasExportError):
    """Canvas returned data that could not safely be processed."""


@dataclass(frozen=True)
class Resource:
    filename: str
    path: str
    query: tuple[tuple[str, str], ...] = ()


@dataclass
class ExportStats:
    request_count: int = 0
    resource_count: int = 0
    item_count: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)


COURSE_RESOURCES = (
    Resource(
        "assignments.json",
        "assignments",
        (
            ("include[]", "submission"),
            ("include[]", "score_statistics"),
            ("include[]", "all_dates"),
            ("include[]", "overrides"),
        ),
    ),
    Resource(
        "assignment_groups.json",
        "assignment_groups",
        (("include[]", "assignments"), ("include[]", "submission")),
    ),
    Resource(
        "modules.json",
        "modules",
        (("include[]", "items"), ("include[]", "content_details")),
    ),
    Resource("pages.json", "pages"),
    Resource(
        "discussion_topics.json",
        "discussion_topics",
        (("include[]", "all_dates"), ("include[]", "sections")),
    ),
    Resource("files.json", "files", (("include[]", "usage_rights"),)),
    Resource("folders.json", "folders"),
    Resource("quizzes.json", "quizzes"),
    Resource("sections.json", "sections"),
    Resource(
        "users.json",
        "users",
        (("include[]", "enrollments"), ("include[]", "avatar_url")),
    ),
    Resource(
        "enrollments.json",
        "enrollments",
        (("include[]", "avatar_url"), ("include[]", "current_points")),
    ),
    Resource("submissions.json", "students/submissions", (
        ("student_ids[]", "self"),
        ("include[]", "assignment"),
        ("include[]", "submission_comments"),
        ("include[]", "rubric_assessment"),
    )),
    Resource("tabs.json", "tabs"),
    Resource("rubrics.json", "rubrics"),
    Resource("outcome_groups.json", "outcome_groups"),
    Resource("groups.json", "groups"),
    Resource("collaborations.json", "collaborations"),
    Resource("conferences.json", "conferences"),
    Resource("external_tools.json", "external_tools"),
    Resource("features.json", "features"),
    Resource("grading_standards.json", "grading_standards"),
    Resource("activity_stream.json", "activity_stream"),
    Resource("activity_stream_summary.json", "activity_stream/summary"),
    Resource("todo.json", "todo"),
    Resource("progress.json", "users/self/progress"),
    Resource("settings.json", "settings"),
)


def normalize_base_url(value: str) -> str:
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
        raise ValueError("Canvas base URL has an invalid port") from error
    return f"https://{parsed.netloc.rstrip('/')}"


def _origin(url: str) -> tuple[str, str, int] | None:
    parsed = urlparse(url)
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return None
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError:
        return None
    return parsed.scheme.lower(), parsed.hostname.lower(), port


def is_safe_api_url(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme.lower() == "https"
        and parsed.fragment == ""
        and _origin(url) == _origin(base_url)
        and parsed.path.startswith("/api/v1/")
    )


def api_url(
    base_url: str,
    path: str,
    query: Iterable[tuple[str, str]] = (),
) -> str:
    path = path.lstrip("/")
    url = f"{base_url}/api/v1/{path}"
    values = list(query)
    if values:
        url += "?" + urlencode(values)
    return url


def _headers(value: Any) -> dict[str, str]:
    return {str(key).lower(): str(item) for key, item in dict(value).items()}


def _retry_delay(headers: dict[str, str], fallback: float) -> float:
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
        return min(MAX_RETRY_SECONDS, max(0.0, delay))
    return min(MAX_RETRY_SECONDS, fallback)


def request_json(
    request: Any,
    url: str,
    description: str,
    *,
    base_url: str,
    timeout_ms: int,
    max_attempts: int,
    stats: ExportStats | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Any, dict[str, str]]:
    """Fetch one JSON response with rate-limit and server-error retries."""
    if not is_safe_api_url(url, base_url):
        raise CanvasResponseError(f"Refusing unsafe API URL for {description}")

    for attempt in range(1, max_attempts + 1):
        if stats is not None:
            stats.request_count += 1
        try:
            response = request.get(url, timeout=timeout_ms, fail_on_status_code=False)
        except Exception as error:
            if attempt == max_attempts:
                raise CanvasExportError(
                    f"Request failed for {description} after {max_attempts} "
                    f"attempt(s): {error}"
                ) from error
            sleeper(min(MAX_RETRY_SECONDS, float(2 ** (attempt - 1))))
            continue

        wait: float | None = None
        try:
            headers = _headers(response.headers)
            if response.ok:
                try:
                    return response.json(), headers
                except Exception as error:
                    raise CanvasResponseError(
                        f"Canvas returned non-JSON data for {description}; "
                        "the browser login may have expired"
                    ) from error

            status = int(response.status)
            if (status == 429 or 500 <= status < 600) and attempt < max_attempts:
                wait = _retry_delay(headers, float(2 ** (attempt - 1)))
            else:
                body = " ".join(response.text()[:400].split())
                detail = f": {body}" if body and status not in {401, 403} else ""
                raise CanvasHTTPError(
                    status, f"Canvas returned HTTP {status} for {description}{detail}"
                )
        finally:
            response.dispose()

        if wait is not None:
            sleeper(wait)

    raise AssertionError("retry loop ended unexpectedly")


LINK_PART_RE = re.compile(r'<([^>]+)>\s*((?:;\s*[^,]+)*)')
REL_RE = re.compile(r'\brel\s*=\s*(?:"([^"]+)"|([^;,\s]+))', re.I)


def next_link(header: str | None) -> str | None:
    if not header:
        return None
    for match in LINK_PART_RE.finditer(header):
        rel_match = REL_RE.search(match.group(2))
        if rel_match and "next" in (rel_match.group(1) or rel_match.group(2)).split():
            return match.group(1)
    return None


def paginated_get(
    request: Any,
    base_url: str,
    path: str,
    description: str,
    *,
    query: Iterable[tuple[str, str]] = (),
    timeout_ms: int,
    max_attempts: int,
    stats: ExportStats,
) -> Any:
    """Collect Canvas list pagination while preserving non-list responses."""
    parameters = list(query) + [("per_page", str(PER_PAGE))]
    url = api_url(base_url, path, parameters)
    seen: set[str] = set()
    combined: list[Any] = []

    while True:
        if url in seen:
            raise CanvasResponseError(f"Canvas repeated a pagination URL for {description}")
        if not is_safe_api_url(url, base_url):
            raise CanvasResponseError(f"Canvas supplied an unsafe pagination URL for {description}")
        seen.add(url)
        payload, headers = request_json(
            request,
            url,
            description,
            base_url=base_url,
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
            stats=stats,
        )
        if not isinstance(payload, list):
            if len(seen) > 1:
                raise CanvasResponseError(
                    f"Canvas changed response type while paginating {description}"
                )
            return payload
        combined.extend(payload)
        following = next_link(headers.get("link"))
        if not following:
            return combined
        url = following


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as output:
            temporary = output.name
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def snapshot_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    candidate = root / stamp
    number = 2
    while candidate.exists():
        candidate = root / f"{stamp}.{number}"
        number += 1
    candidate.mkdir(parents=True)
    return candidate


def record_error(
    stats: ExportStats,
    *,
    course_id: str | None,
    resource: str,
    error: Exception,
) -> dict[str, Any]:
    item = {
        "course_id": course_id,
        "resource": resource,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, CanvasHTTPError):
        item["http_status"] = error.status
    stats.errors.append(item)
    return item


def export_resource(
    request: Any,
    base_url: str,
    course_id: str,
    destination: Path,
    resource: Resource,
    *,
    timeout_ms: int,
    max_attempts: int,
    stats: ExportStats,
) -> Any | None:
    label = resource.filename.removesuffix(".json")
    path = f"courses/{quote(course_id, safe='')}/{resource.path}"
    try:
        data = paginated_get(
            request,
            base_url,
            path,
            f"{label} for course {course_id}",
            query=resource.query,
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
            stats=stats,
        )
        atomic_write_json(destination / resource.filename, data)
        stats.resource_count += 1
        stats.item_count += len(data) if isinstance(data, list) else 1
        count = len(data) if isinstance(data, list) else 1
        print(f"    {label}: {count}")
        return data
    except CanvasExportError as error:
        details = record_error(
            stats, course_id=course_id, resource=label, error=error
        )
        atomic_write_json(
            destination / resource.filename,
            {"complete": False, "error": details},
        )
        print(f"    {label}: unavailable ({error})")
        return None


def export_details(
    request: Any,
    base_url: str,
    course_id: str,
    course_dir: Path,
    resource_name: str,
    records: Any,
    *,
    id_key: str,
    detail_path: Callable[[dict[str, Any]], str],
    detail_query: Iterable[tuple[str, str]] = (),
    timeout_ms: int,
    max_attempts: int,
    stats: ExportStats,
) -> None:
    """Export item-level data omitted from Canvas collection responses."""
    if not isinstance(records, list):
        return
    output_dir = course_dir / "details" / resource_name
    for record in records:
        if not isinstance(record, dict) or record.get(id_key) in (None, ""):
            continue
        identifier = str(record[id_key])
        try:
            payload = paginated_get(
                request,
                base_url,
                detail_path(record),
                f"{resource_name} detail {identifier} for course {course_id}",
                query=detail_query,
                timeout_ms=timeout_ms,
                max_attempts=max_attempts,
                stats=stats,
            )
            atomic_write_json(output_dir / f"{safe_filename(identifier)}.json", payload)
            stats.resource_count += 1
            stats.item_count += len(payload) if isinstance(payload, list) else 1
        except CanvasExportError as error:
            record_error(
                stats,
                course_id=course_id,
                resource=f"{resource_name}/{identifier}",
                error=error,
            )


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:160] or "item"


def special_course_resources(course_id: str) -> tuple[Resource, Resource]:
    # Canvas otherwise defaults these APIs to a narrow date window.
    return (
        Resource(
            "announcements.json",
            "announcements",
            (
                ("context_codes[]", f"course_{course_id}"),
                ("start_date", "1970-01-01T00:00:00Z"),
                ("end_date", "2100-01-01T00:00:00Z"),
                ("latest_only", "false"),
            ),
        ),
        Resource(
            "calendar_events.json",
            "calendar_events",
            (
                ("context_codes[]", f"course_{course_id}"),
                ("start_date", "1970-01-01T00:00:00Z"),
                ("end_date", "2100-01-01T00:00:00Z"),
            ),
        ),
    )


def export_course(
    request: Any,
    base_url: str,
    course: dict[str, Any],
    root: Path,
    *,
    timeout_ms: int,
    max_attempts: int,
    stats: ExportStats,
) -> None:
    course_id = str(course["id"])
    course_dir = root / "courses" / safe_filename(course_id)
    course_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(course_dir / "course_from_list.json", course)

    detail_query = (
        ("include[]", "syllabus_body"),
        ("include[]", "term"),
        ("include[]", "teachers"),
        ("include[]", "sections"),
        ("include[]", "total_scores"),
        ("include[]", "current_grading_period_scores"),
        ("include[]", "course_image"),
        ("include[]", "public_description"),
    )
    try:
        detail, _ = request_json(
            request,
            api_url(base_url, f"courses/{quote(course_id, safe='')}", detail_query),
            f"course details for {course_id}",
            base_url=base_url,
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
            stats=stats,
        )
        atomic_write_json(course_dir / "course.json", detail)
        stats.resource_count += 1
        stats.item_count += 1
    except CanvasExportError as error:
        details = record_error(
            stats, course_id=course_id, resource="course", error=error
        )
        atomic_write_json(course_dir / "course.json", {"complete": False, "error": details})

    pulled: dict[str, Any] = {}
    for resource in COURSE_RESOURCES:
        pulled[resource.filename] = export_resource(
            request,
            base_url,
            course_id,
            course_dir,
            resource,
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
            stats=stats,
        )

    # These endpoints live at the API root, not under /courses/:id.
    for resource in special_course_resources(course_id):
        try:
            data = paginated_get(
                request,
                base_url,
                resource.path,
                f"{resource.filename} for course {course_id}",
                query=resource.query,
                timeout_ms=timeout_ms,
                max_attempts=max_attempts,
                stats=stats,
            )
            atomic_write_json(course_dir / resource.filename, data)
            stats.resource_count += 1
            stats.item_count += len(data) if isinstance(data, list) else 1
            print(f"    {resource.filename.removesuffix('.json')}: {len(data) if isinstance(data, list) else 1}")
        except CanvasExportError as error:
            label = resource.filename.removesuffix(".json")
            details = record_error(
                stats, course_id=course_id, resource=label, error=error
            )
            atomic_write_json(
                course_dir / resource.filename,
                {"complete": False, "error": details},
            )

    course_prefix = f"courses/{quote(course_id, safe='')}"
    export_details(
        request,
        base_url,
        course_id,
        course_dir,
        "pages",
        pulled.get("pages.json"),
        id_key="url",
        detail_path=lambda item: f"{course_prefix}/pages/{quote(str(item['url']), safe='')}",
        timeout_ms=timeout_ms,
        max_attempts=max_attempts,
        stats=stats,
    )
    export_details(
        request,
        base_url,
        course_id,
        course_dir,
        "quiz_submissions",
        pulled.get("quizzes.json"),
        id_key="id",
        detail_path=lambda item: f"{course_prefix}/quizzes/{item['id']}/submissions",
        timeout_ms=timeout_ms,
        max_attempts=max_attempts,
        stats=stats,
    )
    export_details(
        request,
        base_url,
        course_id,
        course_dir,
        "module_items",
        pulled.get("modules.json"),
        id_key="id",
        detail_path=lambda item: f"{course_prefix}/modules/{item['id']}/items",
        detail_query=(("include[]", "content_details"),),
        timeout_ms=timeout_ms,
        max_attempts=max_attempts,
        stats=stats,
    )
    export_details(
        request,
        base_url,
        course_id,
        course_dir,
        "discussion_views",
        pulled.get("discussion_topics.json"),
        id_key="id",
        detail_path=lambda item: f"{course_prefix}/discussion_topics/{item['id']}/view",
        timeout_ms=timeout_ms,
        max_attempts=max_attempts,
        stats=stats,
    )
    export_details(
        request,
        base_url,
        course_id,
        course_dir,
        "quiz_questions",
        pulled.get("quizzes.json"),
        id_key="id",
        detail_path=lambda item: f"{course_prefix}/quizzes/{item['id']}/questions",
        timeout_ms=timeout_ms,
        max_attempts=max_attempts,
        stats=stats,
    )


def verify_login(
    context: Any,
    base_url: str,
    login_url: str,
    *,
    headless: bool,
    timeout_ms: int,
    max_attempts: int,
) -> dict[str, Any]:
    profile_url = api_url(base_url, "users/self/profile")

    def try_profile() -> dict[str, Any] | None:
        try:
            data, _ = request_json(
                context.request,
                profile_url,
                "current user profile",
                base_url=base_url,
                timeout_ms=timeout_ms,
                max_attempts=max_attempts,
            )
            return data if isinstance(data, dict) and data.get("id") else None
        except CanvasExportError:
            return None

    profile = try_profile()
    if profile is not None:
        return profile
    if headless:
        raise CanvasExportError(
            "the saved Canvas login is missing or expired; rerun without --headless"
        )

    page = context.pages[0] if context.pages else context.new_page()
    page.goto(login_url, wait_until="domcontentloaded", timeout=timeout_ms)
    print("\nComplete the Canvas login in the Chromium window.")
    input("Press Enter here after the Canvas dashboard is visible: ")
    profile = try_profile()
    if profile is None:
        raise CanvasExportError(
            "Canvas API authentication still failed. Make sure the dashboard "
            "is visible in the opened browser and try again."
        )
    return profile


def discover_courses(
    request: Any,
    base_url: str,
    *,
    timeout_ms: int,
    max_attempts: int,
    stats: ExportStats,
) -> list[dict[str, Any]]:
    query = (
        ("include[]", "term"),
        ("include[]", "teachers"),
        ("include[]", "total_scores"),
        ("include[]", "course_image"),
        ("include[]", "public_description"),
    )
    # Canvas accepts one enrollment_state at a time. Query every documented state
    # so concluded courses are not silently omitted from a "full" export.
    courses: list[dict[str, Any]] = []
    for enrollment_state in ("active", "invited_or_pending", "completed"):
        data = paginated_get(
            request,
            base_url,
            "courses",
            f"the current user's {enrollment_state} courses",
            query=query + (("enrollment_state", enrollment_state),),
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
            stats=stats,
        )
        if not isinstance(data, list):
            raise CanvasResponseError("Canvas returned a non-list course response")
        courses.extend(
            item for item in data if isinstance(item, dict) and item.get("id")
        )
    unique_courses: list[dict[str, Any]] = []
    seen: set[str] = set()
    for course in courses:
        course_id = str(course["id"])
        if course_id not in seen:
            seen.add(course_id)
            unique_courses.append(course)
    return unique_courses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open Canvas, reuse an authenticated browser profile, and export "
            "all course data that the signed-in account may access."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Canvas HTTPS origin (or set CANVAS_BASE_URL)",
    )
    parser.add_argument(
        "--login-url",
        help="school login URL (default: BASE_URL/login/saml)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="persistent Chromium profile (or set CANVAS_PROFILE_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="parent directory for timestamped exports (or set CANVAS_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--course-id",
        action="append",
        default=[],
        help="export only this course ID; may be repeated",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="do not show Chromium; requires an existing valid login",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        base_url = normalize_base_url(args.base_url)
        if args.request_timeout <= 0:
            raise ValueError("--request-timeout must be greater than zero")
        if args.max_attempts < 1:
            raise ValueError("--max-attempts must be at least 1")
        requested_ids = {str(int(value)) for value in args.course_id}
    except (TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Error: Playwright is not installed. See the setup commands at the "
            "top of canvas_scraper.py.",
            file=sys.stderr,
        )
        return 1

    timeout_ms = int(args.request_timeout * 1000)
    stats = ExportStats()
    export_dir: Path | None = None
    login_url = args.login_url or f"{base_url}/login/saml"

    try:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(args.profile_dir.resolve()),
                headless=args.headless,
            )
            try:
                profile = verify_login(
                    context,
                    base_url,
                    login_url,
                    headless=args.headless,
                    timeout_ms=timeout_ms,
                    max_attempts=args.max_attempts,
                )
                print(f"Authenticated as {profile.get('name', profile.get('id'))}.")
                courses = discover_courses(
                    context.request,
                    base_url,
                    timeout_ms=timeout_ms,
                    max_attempts=args.max_attempts,
                    stats=stats,
                )
                if requested_ids:
                    courses = [
                        course for course in courses
                        if str(course["id"]) in requested_ids
                    ]
                    missing = requested_ids - {str(course["id"]) for course in courses}
                    if missing:
                        raise CanvasExportError(
                            "requested course IDs were not returned by Canvas: "
                            + ", ".join(sorted(missing))
                        )
                if not courses:
                    raise CanvasExportError("Canvas returned no accessible courses")

                export_dir = snapshot_directory(args.output_dir.resolve())
                atomic_write_json(export_dir / "profile.json", profile)
                atomic_write_json(export_dir / "courses.json", courses)
                print(f"Exporting {len(courses)} course(s) to {export_dir}")
                for number, course in enumerate(courses, 1):
                    name = course.get("name") or course.get("course_code") or course["id"]
                    print(f"\n[{number}/{len(courses)}] {name} (course {course['id']})")
                    export_course(
                        context.request,
                        base_url,
                        course,
                        export_dir,
                        timeout_ms=timeout_ms,
                        max_attempts=args.max_attempts,
                        stats=stats,
                    )
            finally:
                context.close()
    except (CanvasExportError, OSError, KeyboardInterrupt) as error:
        print(f"\nError: {error}", file=sys.stderr)
        if export_dir is not None:
            atomic_write_json(export_dir / "errors.json", stats.errors)
        return 1
    except Exception as error:
        print(f"\nUnexpected error: {error}", file=sys.stderr)
        print(
            "If Chromium says the profile is in use, close the other browser "
            "window and run the command again.",
            file=sys.stderr,
        )
        return 1

    assert export_dir is not None
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canvas_base_url": base_url,
        "complete": not stats.errors,
        "course_count": len(courses),
        "request_count": stats.request_count,
        "resource_count": stats.resource_count,
        "item_count": stats.item_count,
        "error_count": len(stats.errors),
        "note": (
            "Canvas only returns data the signed-in account is permitted to see. "
            "An unavailable resource is recorded in errors.json and in its resource file."
        ),
    }
    atomic_write_json(export_dir / "metadata.json", metadata)
    atomic_write_json(export_dir / "errors.json", stats.errors)
    print(f"\nFinished. Export saved to {export_dir}")
    if stats.errors:
        print(
            f"Completed with {len(stats.errors)} unavailable resource(s); "
            "see errors.json."
        )
        return 2
    print("All requested resources were exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
