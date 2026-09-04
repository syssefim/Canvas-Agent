import json
import tempfile
import unittest
from pathlib import Path

import canvas_scraper as scraper


BASE_URL = "https://school.instructure.com"


class FakeResponse:
    def __init__(self, data=None, *, status=200, headers=None, text=""):
        self.data = data
        self.status = status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        self.body_text = text
        self.disposed = False

    def json(self):
        return self.data

    def text(self):
        return self.body_text

    def dispose(self):
        self.disposed = True


class FakeRequest:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class UrlTests(unittest.TestCase):
    def test_normalizes_canvas_origin(self):
        self.assertEqual(scraper.normalize_base_url(BASE_URL + "/"), BASE_URL)

    def test_rejects_insecure_or_non_origin_base_url(self):
        for value in ("http://school.example", "https://school.example/canvas"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scraper.normalize_base_url(value)

    def test_api_url_keeps_repeated_query_parameters(self):
        url = scraper.api_url(
            BASE_URL,
            "courses",
            (("include[]", "term"), ("include[]", "teachers")),
        )
        self.assertIn("include%5B%5D=term", url)
        self.assertIn("include%5B%5D=teachers", url)

    def test_safe_api_url_rejects_cross_origin_and_non_api_paths(self):
        self.assertTrue(
            scraper.is_safe_api_url(BASE_URL + "/api/v1/courses", BASE_URL)
        )
        self.assertFalse(
            scraper.is_safe_api_url("https://attacker.example/api/v1/x", BASE_URL)
        )
        self.assertFalse(scraper.is_safe_api_url(BASE_URL + "/login", BASE_URL))


class PaginationTests(unittest.TestCase):
    def test_collects_link_header_pages(self):
        next_url = BASE_URL + "/api/v1/courses?page=2&opaque=a,b"
        first = FakeResponse(
            [{"id": 1}],
            headers={
                "Link": (
                    f'<{next_url}>; type="application/json"; rel="next", '
                    f'<{BASE_URL}/api/v1/courses?page=1>; rel="current"'
                )
            },
        )
        second = FakeResponse([{"id": 2}])
        request = FakeRequest([first, second])
        stats = scraper.ExportStats()

        data = scraper.paginated_get(
            request,
            BASE_URL,
            "courses",
            "courses",
            timeout_ms=1000,
            max_attempts=1,
            stats=stats,
        )

        self.assertEqual(data, [{"id": 1}, {"id": 2}])
        self.assertEqual(stats.request_count, 2)
        self.assertTrue(first.disposed)
        self.assertTrue(second.disposed)

    def test_rejects_cross_origin_next_link(self):
        request = FakeRequest(
            [
                FakeResponse(
                    [],
                    headers={
                        "link": '<https://attacker.example/api/v1/x>; rel="next"'
                    },
                )
            ]
        )
        with self.assertRaisesRegex(scraper.CanvasResponseError, "unsafe"):
            scraper.paginated_get(
                request,
                BASE_URL,
                "courses",
                "courses",
                timeout_ms=1000,
                max_attempts=1,
                stats=scraper.ExportStats(),
            )

    def test_retries_a_rate_limit_and_honors_retry_after(self):
        limited = FakeResponse(status=429, headers={"Retry-After": "2"})
        success = FakeResponse({"id": 1})
        request = FakeRequest([limited, success])
        sleeps = []
        data, _ = scraper.request_json(
            request,
            BASE_URL + "/api/v1/users/self/profile",
            "profile",
            base_url=BASE_URL,
            timeout_ms=1000,
            max_attempts=2,
            sleeper=sleeps.append,
        )
        self.assertEqual(data, {"id": 1})
        self.assertEqual(sleeps, [2.0])
        self.assertTrue(limited.disposed)

    def test_discovers_active_pending_and_completed_courses_without_duplicates(self):
        request = FakeRequest(
            [
                FakeResponse([{"id": 1, "name": "Active"}]),
                FakeResponse([{"id": 2, "name": "Pending"}]),
                FakeResponse(
                    [
                        {"id": 1, "name": "Duplicate"},
                        {"id": 3, "name": "Completed"},
                    ]
                ),
            ]
        )
        courses = scraper.discover_courses(
            request,
            BASE_URL,
            timeout_ms=1000,
            max_attempts=1,
            stats=scraper.ExportStats(),
        )
        self.assertEqual([course["id"] for course in courses], [1, 2, 3])
        self.assertIn("enrollment_state=active", request.calls[0][0])
        self.assertIn("enrollment_state=invited_or_pending", request.calls[1][0])
        self.assertIn("enrollment_state=completed", request.calls[2][0])


class OutputTests(unittest.TestCase):
    def test_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "data.json"
            scraper.atomic_write_json(path, {"hello": "world"})
            self.assertEqual(json.loads(path.read_text()), {"hello": "world"})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_snapshot_directory_never_reuses_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = scraper.snapshot_directory(root)
            second = scraper.snapshot_directory(root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_safe_filename_removes_path_characters(self):
        self.assertEqual(scraper.safe_filename("../../course / 42"), "course_42")


if __name__ == "__main__":
    unittest.main()
