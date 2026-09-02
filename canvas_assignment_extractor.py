"""Export assignments for every Canvas course listed in ``courses.json``.

The script reuses the Playwright browser profile created by
``canvas_course_extractor.py``.  Its output is grouped by course and intentionally
uses small, explicit records that can later be indexed or passed to an LLM.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from playwright.sync_api import BrowserContext, sync_playwright


CANVAS_BASE_URL = "https://sfsu.instructure.com"
CANVAS_HOST = urlparse(CANVAS_BASE_URL).hostname
LOGIN_URL = f"{CANVAS_BASE_URL}/login/saml"
COURSES_URL = f"{CANVAS_BASE_URL}/courses"

PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = PROJECT_DIR / ".canvas-profile"
DEFAULT_COURSES_FILE = PROJECT_DIR / "courses.json"
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "assignments.json"
PER_PAGE = 100


class _HTMLTextExtractor(HTMLParser):
    """Turn a Canvas HTML description into compact, readable text."""

    BLOCK_TAGS = {
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "table",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts).replace("\xa0", " ")
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


def html_to_text(value: str | None) -> str | None:
    if not value:
        return None

    parser = _HTMLTextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text() or None


def load_courses(path: Path) -> list[dict[str, str]]:
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

        course_id = str(course.get("id", "")).strip()
        name = str(course.get("name", "")).strip()

        if not course_id.isdigit():
            raise ValueError(f"Course #{index + 1} has an invalid id: {course_id!r}")
        if course_id in seen_ids:
            continue

        seen_ids.add(course_id)
        courses.append(
            {
                "id": course_id,
                "name": name or f"Course {course_id}",
                "url": f"{CANVAS_BASE_URL}/courses/{course_id}",
            }
        )

    return courses


def is_logged_in_to_canvas(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname == CANVAS_HOST and not parsed.path.startswith("/login")


def ensure_canvas_login(context: BrowserContext) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(COURSES_URL, wait_until="domcontentloaded")

    if is_logged_in_to_canvas(page.url):
        print("Using the existing Canvas login.")
        return

    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    print("\nLog into SFSU Canvas in the browser window.")
    print("Wait until you can see your Canvas dashboard.")
    input("\nPress Enter after you're logged in... ")

    page.goto(COURSES_URL, wait_until="domcontentloaded")
    if not is_logged_in_to_canvas(page.url):
        raise RuntimeError("Canvas login is not complete.")


def fetch_course_assignments(
    context: BrowserContext, course_id: str
) -> list[dict[str, Any]]:
    """Fetch every page of assignments visible to the logged-in Canvas user."""
    assignments: list[dict[str, Any]] = []
    page_number = 1
    query = urlencode(
        [
            ("per_page", str(PER_PAGE)),
            ("page", str(page_number)),
            ("order_by", "due_at"),
            ("include[]", "submission"),
            ("include[]", "all_dates"),
        ]
    )
    url = f"{CANVAS_BASE_URL}/api/v1/courses/{course_id}/assignments?{query}"
    seen_urls: set[str] = set()

    while url not in seen_urls:
        seen_urls.add(url)
        response = context.request.get(url)

        if not response.ok:
            detail = response.text()[:500].strip()
            raise RuntimeError(
                f"Canvas API returned HTTP {response.status}"
                + (f": {detail}" if detail else "")
            )

        try:
            page_items = response.json()
        except json.JSONDecodeError as error:
            raise RuntimeError("Canvas returned a non-JSON response") from error

        if not isinstance(page_items, list):
            raise RuntimeError("Canvas returned an unexpected assignments response")

        assignments.extend(page_items)
        link_header = response.headers.get("link")
        next_match = (
            re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            if link_header
            else None
        )

        if next_match:
            next_url = next_match.group(1)
            parsed_next = urlparse(next_url)
            if parsed_next.hostname != CANVAS_HOST:
                raise RuntimeError("Canvas returned an unsafe pagination URL")
            url = next_url
            continue

        # Canvas normally supplies RFC 5988 Link headers. This fallback covers
        # proxies that strip that header but leave conventional page behavior.
        if link_header is not None or len(page_items) < PER_PAGE:
            break

        page_number += 1
        query = urlencode(
            [
                ("per_page", str(PER_PAGE)),
                ("page", str(page_number)),
                ("order_by", "due_at"),
                ("include[]", "submission"),
                ("include[]", "all_dates"),
            ]
        )
        url = f"{CANVAS_BASE_URL}/api/v1/courses/{course_id}/assignments?{query}"

    return assignments


def submission_summary(submission: Any) -> dict[str, Any] | None:
    if not isinstance(submission, dict):
        return None

    return {
        "workflow_state": submission.get("workflow_state"),
        "submitted_at": submission.get("submitted_at"),
        "graded_at": submission.get("graded_at"),
        "score": submission.get("score"),
        "grade": submission.get("grade"),
        "late": bool(submission.get("late", False)),
        "missing": bool(submission.get("missing", False)),
        "excused": bool(submission.get("excused", False)),
        "attempt": submission.get("attempt"),
    }


def compact_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    """Keep useful fields while avoiding the very large raw Canvas object."""
    result = {
        "id": str(assignment.get("id", "")),
        "name": assignment.get("name"),
        "due_at": assignment.get("due_at"),
        "unlock_at": assignment.get("unlock_at"),
        "lock_at": assignment.get("lock_at"),
        "points_possible": assignment.get("points_possible"),
        "published": bool(assignment.get("published", False)),
        "html_url": assignment.get("html_url"),
        "submission_types": assignment.get("submission_types") or [],
        "allowed_extensions": assignment.get("allowed_extensions") or [],
        "description_text": html_to_text(assignment.get("description")),
        "submission": submission_summary(assignment.get("submission")),
    }

    # Canvas supplies this when differentiated due dates are enabled and the API
    # honors include[]=all_dates. Retain it so an LLM can reason about overrides.
    if assignment.get("all_dates") is not None:
        result["all_dates"] = assignment["all_dates"]

    return result


def sort_key(assignment: dict[str, Any]) -> tuple[bool, str, str]:
    due_at = assignment.get("due_at")
    return (due_at is None, due_at or "", str(assignment.get("name") or ""))


def write_export(path: Path, courses: list[dict[str, Any]]) -> None:
    assignment_count = sum(len(course["assignments"]) for course in courses)
    dated_count = sum(
        assignment["due_at"] is not None
        for course in courses
        for assignment in course["assignments"]
    )
    failed_count = sum(course["fetch_error"] is not None for course in courses)

    export = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canvas_base_url": CANVAS_BASE_URL,
        "summary": {
            "course_count": len(courses),
            "assignment_count": assignment_count,
            "assignments_with_due_dates": dated_count,
            "assignments_without_due_dates": assignment_count - dated_count,
            "failed_course_count": failed_count,
        },
        "courses": courses,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(export, file, indent=2, ensure_ascii=False)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Canvas assignments for the courses in courses.json."
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
        help=f"output JSON file (default: {DEFAULT_OUTPUT_FILE.name})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="hide the browser (only works while the saved login is valid)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        courses = load_courses(args.courses.resolve())
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    if not courses:
        print(f"Error: no courses were found in {args.courses}")
        return 1

    exported_courses: list[dict[str, Any]] = []

    try:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                headless=args.headless,
            )

            try:
                ensure_canvas_login(context)

                for index, course in enumerate(courses, start=1):
                    print(
                        f"[{index}/{len(courses)}] Fetching assignments for "
                        f"{course['name']}..."
                    )
                    fetch_error = None

                    try:
                        raw_assignments = fetch_course_assignments(
                            context, course["id"]
                        )
                        assignments = [
                            compact_assignment(item)
                            for item in raw_assignments
                            if isinstance(item, dict)
                        ]
                        assignments.sort(key=sort_key)
                    except Exception as error:  # Continue to preserve other courses.
                        assignments = []
                        fetch_error = str(error)
                        print(f"  Error: {fetch_error}")

                    exported_courses.append(
                        {
                            **course,
                            "assignment_count": len(assignments),
                            "fetch_error": fetch_error,
                            "assignments": assignments,
                        }
                    )
                    print(f"  Found {len(assignments)} assignments.")
            finally:
                context.close()
    except Exception as error:
        print(f"Error: {error}")
        return 1

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_export(output_path, exported_courses)

    failed_count = sum(
        course["fetch_error"] is not None for course in exported_courses
    )
    total = sum(course["assignment_count"] for course in exported_courses)
    print(f"\nSaved {total} assignments to {output_path}")

    if failed_count:
        print(f"Warning: {failed_count} course(s) could not be fetched.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
