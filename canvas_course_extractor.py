import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


CANVAS_BASE_URL = "https://sfsu.instructure.com"
CANVAS_HOST = urlparse(CANVAS_BASE_URL).hostname
LOGIN_URL = f"{CANVAS_BASE_URL}/login/saml"
COURSES_URL = f"{CANVAS_BASE_URL}/courses"
CURRENT_COURSES_SELECTOR = (
    '#my_courses_table a[href*="/courses/"]'
)

PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = PROJECT_DIR / ".canvas-profile"
OUTPUT_FILE = PROJECT_DIR / "courses.json"


def is_logged_in_to_canvas(url):
    """Return whether *url* is inside the authenticated SFSU Canvas app."""
    parsed = urlparse(url)
    return (
        parsed.hostname == CANVAS_HOST
        and not parsed.path.startswith("/login")
    )


def display_url(url):
    """Remove query parameters before printing authentication URLs."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def find_courses(page):
    print("\nOpening the Canvas All Courses page...")

    # canvas.sfsu.edu is a public information page, not the Canvas app. The
    # authenticated app and its course links live on sfsu.instructure.com.
    page.goto(COURSES_URL, wait_until="domcontentloaded")

    if not is_logged_in_to_canvas(page.url):
        raise RuntimeError(
            "Canvas login is not complete. Log in through the browser window "
            "and wait for the Canvas dashboard before pressing Enter."
        )

    # Canvas can keep background network requests open, so waiting for
    # `networkidle` indefinitely is unreliable. Give its course list a short
    # opportunity to finish rendering instead.
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except PlaywrightTimeoutError:
        pass

    page.wait_for_selector("#content", state="attached", timeout=30_000)

    print("Current URL:", page.url)
    print("Page title:", page.title())

    # Canvas separates this page into current, future, and past enrollment
    # tables. Only inspect the current-enrollment table so completed courses
    # are not included in the output.
    links = page.locator(CURRENT_COURSES_SELECTOR)
    link_count = links.count()
    print(f"Found {link_count} current course links")

    courses = {}

    for i in range(link_count):
        link = links.nth(i)

        href = link.get_attribute("href")

        if not href:
            continue

        # Converts relative URLs such as:
        # /courses/12345
        #
        # into:
        # https://sfsu.instructure.com/courses/12345
        full_url = urljoin(page.url, href)

        path = urlparse(full_url).path

        # On the All Courses page, links to the course itself contain the
        # useful course name. Ignore nested links such as assignments.
        match = re.fullmatch(r"/courses/(\d+)/?", path)

        if not match:
            continue

        if urlparse(full_url).hostname != CANVAS_HOST:
            continue

        course_id = match.group(1)

        name = " ".join(link.inner_text().split())
        if not name:
            name = (
                link.get_attribute("aria-label")
                or link.get_attribute("title")
                or ""
            ).strip()

        # Create the entry if this is the first time we've
        # encountered the course.
        if course_id not in courses:
            courses[course_id] = {
                "id": course_id,
                "name": name or f"Course {course_id}",
                "url": f"{CANVAS_BASE_URL}/courses/{course_id}",
            }

        # Sometimes the first link for a course is an icon with
        # no text. If we later find a link containing the course
        # name, use that instead.
        elif (
            courses[course_id]["name"] == f"Course {course_id}"
            and name
        ):
            courses[course_id]["name"] = name

    return list(courses.values())


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
        )

        page = (
            browser.pages[0]
            if browser.pages
            else browser.new_page()
        )

        page.goto(LOGIN_URL, wait_until="domcontentloaded")

        print("\nLog into SFSU Canvas.")
        print("Wait until you can see your Canvas dashboard.")

        input("\nPress Enter after you're logged in... ")

        print("\nBrowser URL after login:")
        print(display_url(page.url))

        try:
            courses = find_courses(page)
        except RuntimeError as error:
            print(f"\nError: {error}")
            browser.close()
            return 1

        print(f"\nFound {len(courses)} courses:\n")

        for course in courses:
            print(
                f"{course['id']} - "
                f"{course['name']} - "
                f"{course['url']}"
            )

        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(courses, file, indent=4)

        print(f"\nSaved results to {OUTPUT_FILE}")

        input("\nPress Enter to close the browser...")

        browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
