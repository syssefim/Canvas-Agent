# Canvas assignment exporter

This project discovers current SFSU Canvas courses with Playwright and exports
assignments through Canvas's authenticated REST API. The assignment exporter
reuses the browser profile created by `canvas_course_extractor.py`, so it does
not store an API token in the repository.

## Setup

Create and activate a virtual environment, install Playwright, and install its
Chromium browser:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/playwright install chromium
```

The saved browser profile, course list, assignment exports, partial exports, and
assignment history contain private data and are ignored by Git.

## Usage

First generate `courses.json` and complete the interactive Canvas login:

```bash
venv/bin/python canvas_course_extractor.py
```

Then export assignments:

```bash
venv/bin/python canvas_assignment_extractor.py
```

Once the saved login has been verified, subsequent runs can be headless:

```bash
venv/bin/python canvas_assignment_extractor.py --headless
```

Use `--help` to see configuration for the Canvas origin, login URL, browser
profile, input/output files, history directory, request timeout, and retry count.
`CANVAS_BASE_URL`, `CANVAS_PROFILE_DIR`, and
`CANVAS_ASSIGNMENT_HISTORY_DIR` provide environment-variable alternatives.

Every completed pull is archived under `assignment_history/` with a sortable UTC
timestamp and either `.complete.json` or `.partial.json` in its filename. No
snapshots are automatically deleted. Successful runs also atomically replace
`assignments.json`. If any course request fails or an assignment record does not
pass validation, the script exits with status 1, writes
`assignments.partial.json`, and leaves the last successful export unchanged.
Check the top-level `complete` field and summary counts before using an export
downstream.

Both `description_html` and a searchable `description_text` are retained. The
text conversion keeps safe links, image descriptions, list markers, and table
separators while excluding script, style, and SVG content.

## Tests

Run the standard-library `unittest` suite inside the virtual environment:

```bash
venv/bin/python -m unittest discover -s tests -v
```
