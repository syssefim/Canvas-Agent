#!/usr/bin/env python3
"""Export only CSC 340 from Canvas.

This is a course-specific entry point for ``canvas_scraper.py``. It uses the
same authenticated Playwright profile, exports the same resources, and
downloads the same supported files, but limits the export to the Fall 2026
CSC 340 course (Canvas course ID 71774).

Run:
    .venv/bin/python csc340_scraper.py

All options supported by ``canvas_scraper.py`` remain available except
``--course-id``, which is fixed by this script. For example:
    .venv/bin/python csc340_scraper.py --headless
    .venv/bin/python csc340_scraper.py --skip-file-downloads
"""

from __future__ import annotations

import sys

import canvas_scraper


CSC_340_COURSE_ID = "71774"


def main(argv: list[str] | None = None) -> int:
    """Run the standard Canvas exporter for CSC 340 only."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(
        argument == "--course-id" or argument.startswith("--course-id=")
        for argument in arguments
    ):
        print(
            "Error: --course-id cannot be used with csc340_scraper.py; "
            f"this script is fixed to CSC 340 (course {CSC_340_COURSE_ID}).",
            file=sys.stderr,
        )
        return 1

    return canvas_scraper.main(
        ["--course-id", CSC_340_COURSE_ID, *arguments]
    )


if __name__ == "__main__":
    raise SystemExit(main())
