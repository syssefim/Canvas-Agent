"""Export Canvas course content for later, authorized dataset preparation.

The exporter uses the Playwright browser profile created by canvas_scraper.py,
so no Canvas API token needs to be stored in the project. It exports course
materials and metadata, but intentionally does not collect student rosters,
submissions, grades, or classmates' discussion replies.
"""

import argparse
import json
import re
import shutil
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse, urlunparse
from xml.etree import ElementTree

from playwright.sync_api import sync_playwright


CANVAS_BASE_URL = "https://sfsu.instructure.com"
CANVAS_HOST = urlparse(CANVAS_BASE_URL).hostname
LOGIN_URL = f"{CANVAS_BASE_URL}/login/saml"
COURSES_URL = f"{CANVAS_BASE_URL}/courses"

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_COURSES_FILE = PROJECT_DIR / "courses.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "canvas_export"
PROFILE_DIR = PROJECT_DIR / ".canvas-profile"

LINK_PART_RE = re.compile(r'<([^>]+)>;\s*rel="([^"]+)"')
UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")
TEXT_FILE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".csv", ".h", ".hpp", ".java",
    ".js", ".json", ".jsx", ".log", ".md", ".py", ".r", ".rst",
    ".sql", ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml",
}
HTML_FILE_SUFFIXES = {".htm", ".html"}


class ExportError(RuntimeError):
    """An expected, user-actionable export error."""


class HTMLTextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text converter for Canvas rich text."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
        "nav", "ol", "p", "pre", "section", "table", "tr", "ul",
    }
    IGNORED_TAGS = {"script", "style", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.IGNORED_TAGS:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        elif not self.ignored_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored_depth:
            self.parts.append(data)


def html_to_text(value):
    """Convert Canvas HTML into readable, whitespace-normalized plain text."""
    if not value:
        return ""

    parser = HTMLTextExtractor()
    parser.feed(str(value))
    lines = []
    for line in "".join(parser.parts).splitlines():
        normalized = " ".join(line.split())
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def safe_name(value, fallback="item"):
    """Create a short filename component without changing file extensions."""
    value = Path(str(value or "")).name.strip()
    value = UNSAFE_FILENAME_RE.sub("_", value).strip(" ._")
    return (value or fallback)[:180]


def strip_url_query(url):
    """Avoid persisting signed Canvas download parameters in exported JSON."""
    if not url:
        return url
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def is_canvas_page(url):
    parsed = urlparse(url)
    return (
        parsed.hostname == CANVAS_HOST
        and not parsed.path.startswith("/login")
    )


def canvas_api_url(path, params=None):
    url = f"{CANVAS_BASE_URL}/api/v1/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    return url


def _response_json(response, description):
    try:
        if not response.ok:
            body = response.text()[:300].replace("\n", " ")
            raise ExportError(
                f"Canvas returned HTTP {response.status} for {description}: {body}"
            )
        try:
            return response.json()
        except Exception as error:
            raise ExportError(
                f"Canvas returned non-JSON data for {description}. "
                "The login session may have expired."
            ) from error
    finally:
        response.dispose()


def api_get(request, path, params=None, description=None):
    """Fetch one Canvas API response using the authenticated browser cookies."""
    url = canvas_api_url(path, params)
    response = request.get(url, timeout=60_000, fail_on_status_code=False)
    return _response_json(response, description or path)


def _next_link(headers):
    for url, relation in LINK_PART_RE.findall(headers.get("link", "")):
        if relation == "next":
            return url
    return None


def api_get_all(request, path, params=None, description=None):
    """Fetch every page of a Canvas list endpoint."""
    query = dict(params or {})
    query.setdefault("per_page", 100)
    url = canvas_api_url(path, query)
    items = []

    while url:
        if urlparse(url).hostname != CANVAS_HOST:
            raise ExportError(f"Canvas returned an unexpected pagination URL: {url}")

        response = request.get(url, timeout=60_000, fail_on_status_code=False)
        headers = response.headers
        data = _response_json(response, description or path)
        if not isinstance(data, list):
            raise ExportError(f"Expected a list from Canvas for {description or path}")
        items.extend(data)
        url = _next_link(headers)

    return items


def read_courses(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExportError(
            f"{path} does not exist. Run canvas_scraper.py first."
        ) from error
    except json.JSONDecodeError as error:
        raise ExportError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(data, list) or not data:
        raise ExportError(f"{path} does not contain any courses.")

    courses = []
    seen_ids = set()
    for entry in data:
        course_id = str(entry.get("id", "")) if isinstance(entry, dict) else ""
        if not course_id.isdigit():
            raise ExportError(f"Invalid course entry in {path}: {entry!r}")
        if course_id not in seen_ids:
            courses.append({
                "id": course_id,
                "name": str(entry.get("name") or f"Course {course_id}"),
                "url": str(
                    entry.get("url")
                    or f"{CANVAS_BASE_URL}/courses/{course_id}"
                ),
            })
            seen_ids.add(course_id)
    return courses


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _office_xml_text(path):
    """Extract paragraph text from modern Word and PowerPoint files."""
    with zipfile.ZipFile(path) as archive:
        if path.suffix.lower() == ".docx":
            members = ["word/document.xml"]
        else:
            members = sorted(
                name for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            )

        paragraphs = []
        for member in members:
            root = ElementTree.fromstring(archive.read(member))
            for paragraph in root.iter():
                if paragraph.tag.rsplit("}", 1)[-1] != "p":
                    continue
                text = " ".join(
                    node.text.strip()
                    for node in paragraph.iter()
                    if node.tag.rsplit("}", 1)[-1] == "t"
                    and node.text
                    and node.text.strip()
                )
                if text:
                    paragraphs.append(text)
        return "\n".join(paragraphs)


def extract_file_text(path):
    """Return ``(text, extractor)`` or ``(None, reason)`` for a local file."""
    suffix = path.suffix.lower()

    if suffix in TEXT_FILE_SUFFIXES or suffix in HTML_FILE_SUFFIXES:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        if suffix in HTML_FILE_SUFFIXES:
            text = html_to_text(text)
        return text.strip(), "built-in text extractor"

    if suffix in {".docx", ".pptx"}:
        try:
            return _office_xml_text(path).strip(), "built-in Office extractor"
        except (KeyError, OSError, ElementTree.ParseError, zipfile.BadZipFile) as error:
            return None, f"Office extraction failed: {error}"

    if suffix == ".pdf":
        executable = shutil.which("pdftotext")
        if not executable:
            return None, "pdftotext is not installed"
        try:
            result = subprocess.run(
                [executable, "-layout", str(path), "-"],
                check=True,
                capture_output=True,
                timeout=120,
            )
            return result.stdout.decode("utf-8", errors="replace").strip(), "pdftotext"
        except (OSError, subprocess.SubprocessError) as error:
            return None, f"PDF extraction failed: {error}"

    return None, f"unsupported file type: {suffix or 'no extension'}"


def add_document(documents, course, kind, item, title_key, body_key, source_url=None):
    text = html_to_text(item.get(body_key))
    if not text:
        return
    documents.append({
        "course_id": str(course["id"]),
        "course_name": course.get("name"),
        "type": kind,
        "id": str(item.get("id") or item.get("url") or ""),
        "title": item.get(title_key) or f"Untitled {kind}",
        "source_url": strip_url_query(source_url or item.get("html_url")),
        "text": text,
    })


def build_documents(course, exported, course_dir=None):
    """Flatten text-bearing Canvas records into LLM-friendly JSONL records."""
    documents = []
    details = exported["course"]
    add_document(
        documents,
        course,
        "syllabus",
        details,
        "name",
        "syllabus_body",
        course.get("url"),
    )

    for page in exported.get("pages", []):
        add_document(documents, course, "page", page, "title", "body")
    for assignment in exported.get("assignments", []):
        add_document(
            documents, course, "assignment", assignment, "name", "description"
        )
    for topic in exported.get("discussion_topics", []):
        kind = "announcement" if topic.get("is_announcement") else "discussion"
        add_document(documents, course, kind, topic, "title", "message")
    for quiz in exported.get("quizzes", []):
        add_document(documents, course, "quiz", quiz, "title", "description")
    for event in exported.get("calendar_events", []):
        add_document(documents, course, "calendar_event", event, "title", "description")

    for module in exported.get("modules", []):
        item_lines = []
        for item in module.get("items", []):
            title = " ".join(str(item.get("title") or "").split())
            if title:
                item_lines.append(f"{item.get('type', 'Item')}: {title}")
        if item_lines:
            documents.append({
                "course_id": str(course["id"]),
                "course_name": course.get("name"),
                "type": "module",
                "id": str(module.get("id") or ""),
                "title": module.get("name") or "Untitled module",
                "source_url": strip_url_query(module.get("items_url")),
                "text": "\n".join(item_lines),
            })

    if course_dir:
        for item in exported.get("files", []):
            text_path = item.get("download", {}).get("text_path")
            if not text_path:
                continue
            text = (course_dir / text_path).read_text(encoding="utf-8").strip()
            if not text:
                continue
            documents.append({
                "course_id": str(course["id"]),
                "course_name": course.get("name"),
                "type": "course_file",
                "id": str(item.get("id") or ""),
                "title": item.get("display_name") or item.get("filename") or "File",
                "source_url": strip_url_query(item.get("url")),
                "text": text,
            })

    return documents


def download_files(request, files, destination):
    destination.mkdir(parents=True, exist_ok=True)
    text_destination = destination.parent / "file_text"
    exported_files = []

    for item in files:
        record = dict(item)
        download_url = record.get("url")
        record["url"] = strip_url_query(download_url)

        if item.get("locked") or item.get("hidden") or not download_url:
            record["download"] = {"status": "skipped", "reason": "not accessible"}
            exported_files.append(record)
            continue

        filename = safe_name(item.get("display_name") or item.get("filename"), "file")
        filename = f"{item.get('id', 'unknown')}_{filename}"
        target = destination / filename
        print(f"    Downloading {filename}")

        try:
            response = request.get(
                download_url,
                timeout=120_000,
                fail_on_status_code=False,
            )
            try:
                if not response.ok:
                    raise ExportError(f"HTTP {response.status}")
                target.write_bytes(response.body())
            finally:
                response.dispose()
            record["download"] = {
                "status": "downloaded",
                "path": f"files/{filename}",
                "bytes": target.stat().st_size,
            }
            text, extractor = extract_file_text(target)
            if text is not None:
                text_destination.mkdir(parents=True, exist_ok=True)
                text_target = text_destination / f"{filename}.txt"
                text_target.write_text(text + ("\n" if text else ""), encoding="utf-8")
                record["download"].update({
                    "text_status": "extracted",
                    "text_path": f"file_text/{filename}.txt",
                    "text_extractor": extractor,
                })
            else:
                record["download"].update({
                    "text_status": "unavailable",
                    "text_reason": extractor,
                })
        except Exception as error:
            record["download"] = {"status": "failed", "error": str(error)}

        exported_files.append(record)

    return exported_files


def export_course(request, course, output_root, download_course_files=True):
    course_id = course["id"]
    course_dir = output_root / f"{course_id}_{safe_name(course['name'], 'course')}"
    course_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting {course['name']} ({course_id})")

    exported = {"source": course, "errors": {}}

    print("  Fetching course details")
    exported["course"] = api_get(
        request,
        f"courses/{course_id}",
        {
            "include[]": ["syllabus_body", "term", "teachers", "course_image"],
        },
        "course details",
    )

    collection_specs = {
        "assignments": (f"courses/{course_id}/assignments", {}),
        "discussion_topics": (f"courses/{course_id}/discussion_topics", {}),
        "quizzes": (f"courses/{course_id}/quizzes", {}),
        "files": (f"courses/{course_id}/files", {}),
        "folders": (f"courses/{course_id}/folders", {}),
        "sections": (f"courses/{course_id}/sections", {}),
        "tabs": (f"courses/{course_id}/tabs", {}),
        "rubrics": (f"courses/{course_id}/rubrics", {}),
        "calendar_events": (
            "calendar_events",
            {"context_codes[]": f"course_{course_id}", "all_events": "true"},
        ),
    }

    for name, (path, params) in collection_specs.items():
        print(f"  Fetching {name.replace('_', ' ')}")
        try:
            exported[name] = api_get_all(request, path, params, name)
        except ExportError as error:
            exported[name] = []
            exported["errors"][name] = str(error)

    print("  Fetching pages")
    try:
        page_summaries = api_get_all(request, f"courses/{course_id}/pages")
        pages = []
        for page in page_summaries:
            page_url = quote(str(page["url"]), safe="")
            pages.append(api_get(request, f"courses/{course_id}/pages/{page_url}"))
        exported["pages"] = pages
    except (ExportError, KeyError) as error:
        exported["pages"] = []
        exported["errors"]["pages"] = str(error)

    print("  Fetching modules and module items")
    try:
        modules = api_get_all(request, f"courses/{course_id}/modules")
        for module in modules:
            module["items"] = api_get_all(
                request,
                f"courses/{course_id}/modules/{module['id']}/items",
                {"include[]": "content_details"},
            )
        exported["modules"] = modules
    except (ExportError, KeyError) as error:
        exported["modules"] = []
        exported["errors"]["modules"] = str(error)

    if download_course_files:
        exported["files"] = download_files(
            request,
            exported.get("files", []),
            course_dir / "files",
        )
    else:
        exported["files"] = [
            {**item, "url": strip_url_query(item.get("url"))}
            for item in exported.get("files", [])
        ]

    documents = build_documents(course, exported, course_dir)
    write_json(course_dir / "course_export.json", exported)
    with (course_dir / "documents.jsonl").open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(documents)} text documents to {course_dir}")
    return course_dir, documents, exported["errors"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Canvas course materials listed in courses.json."
    )
    parser.add_argument(
        "--courses",
        type=Path,
        default=DEFAULT_COURSES_FILE,
        help=f"course list from the scraper (default: {DEFAULT_COURSES_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"export directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="export file metadata without downloading file contents",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        courses = read_courses(args.courses.resolve())
    except ExportError as error:
        print(f"Error: {error}")
        return 1

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
        )
        try:
            page = browser.pages[0] if browser.pages else browser.new_page()
            page.goto(COURSES_URL, wait_until="domcontentloaded")

            if not is_canvas_page(page.url):
                page.goto(LOGIN_URL, wait_until="domcontentloaded")
                print("\nLog into SFSU Canvas in the browser window.")
                input("Press Enter after the Canvas dashboard appears... ")

            page.goto(COURSES_URL, wait_until="domcontentloaded")
            if not is_canvas_page(page.url):
                raise ExportError(
                    "Canvas login is not complete. Run the exporter again and "
                    "finish SSO before pressing Enter."
                )

            manifest = {"courses": [], "document_count": 0}
            all_documents = []
            for course in courses:
                try:
                    course_dir, documents, errors = export_course(
                        browser.request,
                        course,
                        output_root,
                        download_course_files=not args.skip_files,
                    )
                    all_documents.extend(documents)
                    manifest["courses"].append({
                        "id": course["id"],
                        "name": course["name"],
                        "status": "exported",
                        "path": str(course_dir.relative_to(output_root)),
                        "errors": errors,
                    })
                except Exception as error:
                    print(f"  Failed: {error}")
                    manifest["courses"].append({
                        "id": course["id"],
                        "name": course["name"],
                        "status": "failed",
                        "error": str(error),
                    })

            with (output_root / "documents.jsonl").open("w", encoding="utf-8") as file:
                for document in all_documents:
                    file.write(json.dumps(document, ensure_ascii=False) + "\n")
            manifest["document_count"] = len(all_documents)
            write_json(output_root / "manifest.json", manifest)
        except KeyboardInterrupt:
            print("\nExport cancelled.")
            return 130
        except (ExportError, EOFError) as error:
            print(f"\nError: {error}")
            return 1
        finally:
            try:
                browser.close()
            except Exception:
                pass

    print(f"\nExport complete: {output_root}")
    print(f"LLM-ready text: {output_root / 'documents.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
