import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright


CANVAS_URL = "https://canvas.sfsu.edu"
PROFILE_DIR = Path(".canvas-profile")


def find_courses(page):
    print("\nOpening Canvas dashboard...")

    page.goto(CANVAS_URL, wait_until="networkidle")
    page.wait_for_timeout(3000)

    print("Current URL:", page.url)
    print("Page title:", page.title())

    # Look through every link on the dashboard.
    links = page.locator("a")
    print(f"Found {links.count()} total links on dashboard")

    courses = {}

    for i in range(links.count()):
        link = links.nth(i)

        href = link.get_attribute("href")

        if not href:
            continue

        # Converts relative URLs such as:
        # /courses/12345
        #
        # into:
        # https://canvas.sfsu.edu/courses/12345
        full_url = urljoin(page.url, href)

        path = urlparse(full_url).path

        # Accept:
        #
        # /courses/12345
        # /courses/12345/
        # /courses/12345/assignments
        #
        # We'll deduplicate them by course ID.
        match = re.search(r"/courses/(\d+)(?:/|$)", path)

        if not match:
            continue

        course_id = match.group(1)

        try:
            name = link.inner_text().strip()
        except Exception:
            name = ""

        # Create the entry if this is the first time we've
        # encountered the course.
        if course_id not in courses:
            courses[course_id] = {
                "id": course_id,
                "name": name or f"Course {course_id}",
                "url": full_url.split(
                    f"/courses/{course_id}"
                )[0] + f"/courses/{course_id}",
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

        page.goto(CANVAS_URL)

        print("\nLog into SFSU Canvas.")
        print("Make sure you can see your Canvas dashboard.")

        input("\nPress Enter after you're logged in... ")

        print("\nBrowser URL after login:")
        print(page.url)

        courses = find_courses(page)

        print(f"\nFound {len(courses)} courses:\n")

        for course in courses:
            print(
                f"{course['id']} - "
                f"{course['name']} - "
                f"{course['url']}"
            )

        with open("courses.json", "w") as file:
            json.dump(courses, file, indent=4)

        print("\nSaved results to courses.json")

        input("\nPress Enter to close the browser...")

        browser.close()


if __name__ == "__main__":
    main()