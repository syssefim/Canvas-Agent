import json
import tempfile
import unittest
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError

import canvas_assignment_extractor as extractor


BASE_URL = "https://school.instructure.com"


class FakeResponse:
    def __init__(
        self,
        data=None,
        *,
        status=200,
        headers=None,
        text="",
        json_error=None,
    ):
        self.data = data
        self.status = status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        self.body_text = text
        self.json_error = json_error
        self.disposed = False

    def json(self):
        if self.json_error:
            raise self.json_error
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


class FakeContext:
    def __init__(self, responses):
        self.request = FakeRequest(responses)


def assignment(assignment_id=1, name="Assignment", **overrides):
    value = {
        "id": assignment_id,
        "name": name,
        "due_at": None,
        "unlock_at": None,
        "lock_at": None,
        "points_possible": 10,
        "published": True,
        "html_url": f"{BASE_URL}/assignments/{assignment_id}",
        "submission_types": ["online_upload"],
        "allowed_extensions": ["pdf"],
        "description": "<p>Instructions</p>",
        "submission": None,
        "all_dates": [],
    }
    value.update(overrides)
    return value


class ConfigurationTests(unittest.TestCase):
    def test_normalizes_https_origin(self):
        self.assertEqual(
            extractor.normalize_base_url(f"{BASE_URL}/"),
            BASE_URL,
        )

    def test_rejects_non_origin_or_insecure_base_url(self):
        for value in ("http://example.com", "https://example.com/canvas"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                extractor.normalize_base_url(value)

    def test_course_loading_uses_configured_origin_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "courses.json"
            path.write_text(
                '[{"id": 7, "name": " Course "}, {"id": "7"}]',
                encoding="utf-8",
            )
            self.assertEqual(
                extractor.load_courses(path, BASE_URL),
                [
                    {
                        "id": "7",
                        "name": "Course",
                        "url": f"{BASE_URL}/courses/7",
                    }
                ],
            )


class HtmlConversionTests(unittest.TestCase):
    def test_preserves_links_images_lists_and_tables_but_ignores_active_content(self):
        html = """
            <style>secret style</style><script>secret script</script>
            <p>Read <a href="https://example.com/doc">the guide</a>.</p>
            <img src="/images/chart.png" alt="Grade chart">
            <ul><li>First</li><li>Second</li></ul>
            <table><tr><th>Name</th><th>Value</th></tr></table>
            <a href="javascript:alert(1)">Unsafe</a>
        """
        text = extractor.html_to_text(html)
        self.assertIn("the guide (https://example.com/doc)", text)
        self.assertIn("Grade chart (/images/chart.png)", text)
        self.assertIn("- First", text)
        self.assertIn("Name | Value", text)
        self.assertIn("Unsafe", text)
        self.assertNotIn("javascript:", text)
        self.assertNotIn("secret", text)


class RequestTests(unittest.TestCase):
    def test_retries_rate_limit_honors_retry_after_and_disposes_responses(self):
        limited = FakeResponse(status=429, headers={"Retry-After": "3"})
        success = FakeResponse({"id": 4})
        request = FakeRequest([limited, success])
        sleeps = []

        data, _ = extractor.request_json(
            request,
            f"{BASE_URL}/api/test",
            "test request",
            max_attempts=2,
            sleeper=sleeps.append,
        )

        self.assertEqual(data, {"id": 4})
        self.assertEqual(sleeps, [3.0])
        self.assertTrue(limited.disposed)
        self.assertTrue(success.disposed)

    def test_retries_playwright_transport_errors(self):
        success = FakeResponse([])
        request = FakeRequest([PlaywrightError("network failure"), success])
        sleeps = []
        data, _ = extractor.request_json(
            request,
            f"{BASE_URL}/api/test",
            "test request",
            max_attempts=2,
            sleeper=sleeps.append,
        )
        self.assertEqual(data, [])
        self.assertEqual(sleeps, [1])

    def test_non_json_response_is_explicit_and_disposed(self):
        response = FakeResponse(json_error=ValueError("not json"))
        request = FakeRequest([response])
        with self.assertRaisesRegex(extractor.CanvasResponseError, "non-JSON"):
            extractor.request_json(
                request,
                f"{BASE_URL}/api/test",
                "test request",
                max_attempts=1,
            )
        self.assertTrue(response.disposed)


class AuthenticationTests(unittest.TestCase):
    def test_api_profile_verifies_authenticated_session(self):
        context = FakeContext([FakeResponse({"id": 42, "name": "Student"})])
        extractor.verify_canvas_api(context, BASE_URL, max_attempts=1)
        self.assertTrue(context.request.responses == [])

    def test_unauthorized_api_profile_is_reported_as_expired_login(self):
        context = FakeContext([FakeResponse(status=401, text="private response")])
        with self.assertRaisesRegex(
            extractor.CanvasAuthenticationError, "missing or expired"
        ):
            extractor.verify_canvas_api(context, BASE_URL, max_attempts=1)

    def test_html_instead_of_api_json_is_reported_as_authentication_failure(self):
        context = FakeContext(
            [FakeResponse(json_error=ValueError("HTML login page"))]
        )
        with self.assertRaisesRegex(
            extractor.CanvasAuthenticationError, "authenticated API response"
        ):
            extractor.verify_canvas_api(context, BASE_URL, max_attempts=1)


class PaginationTests(unittest.TestCase):
    def test_follows_extended_link_header_with_comma_in_opaque_url(self):
        next_url = f"{BASE_URL}/api/v1/courses/1/assignments?opaque=a,b"
        first = FakeResponse(
            [assignment(1)],
            headers={
                "Link": (
                    f'<{next_url}>; type="application/json"; rel="next", '
                    f'<{BASE_URL}/first>; rel="first"'
                )
            },
        )
        second = FakeResponse(
            [assignment(2)],
            headers={"link": f'<{next_url}>; rel="current"'},
        )
        context = FakeContext([first, second])

        result = extractor.fetch_course_assignments(
            context, "1", BASE_URL, max_attempts=1
        )

        self.assertEqual([item["id"] for item in result], [1, 2])
        self.assertEqual(len(context.request.calls), 2)

    def test_rejects_cross_origin_next_link(self):
        response = FakeResponse(
            [assignment()],
            headers={"link": '<https://attacker.example/next>; rel="next"'},
        )
        with self.assertRaisesRegex(extractor.CanvasResponseError, "unsafe"):
            extractor.fetch_course_assignments(
                FakeContext([response]), "1", BASE_URL, max_attempts=1
            )

    def test_repeated_next_link_is_an_error(self):
        first_url = extractor._assignments_url(BASE_URL, "1", 1)
        response = FakeResponse(
            [assignment()],
            headers={"link": f'<{first_url}>; rel="next"'},
        )
        with self.assertRaisesRegex(extractor.CanvasResponseError, "repeated"):
            extractor.fetch_course_assignments(
                FakeContext([response]), "1", BASE_URL, max_attempts=1
            )

    def test_falls_back_when_link_header_is_absent(self):
        first = FakeResponse([assignment(index) for index in range(1, 101)])
        second = FakeResponse([assignment(101)])
        context = FakeContext([first, second])
        result = extractor.fetch_course_assignments(
            context, "1", BASE_URL, max_attempts=1
        )
        self.assertEqual(len(result), 101)
        self.assertIn("page=2", context.request.calls[1][0])


class ValidationAndOutputTests(unittest.TestCase):
    def test_rejects_invalid_and_duplicate_records(self):
        valid, errors = extractor.prepare_assignments(
            [assignment(1), assignment(1), {"id": 2}, "not an object"]
        )
        self.assertEqual([item["id"] for item in valid], ["1"])
        self.assertEqual(len(errors), 3)
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_retains_original_html_and_useful_text(self):
        result = extractor.compact_assignment(
            assignment(description='<p><a href="https://example.com">Read</a></p>')
        )
        self.assertEqual(
            result["description_html"],
            '<p><a href="https://example.com">Read</a></p>',
        )
        self.assertEqual(
            result["description_text"], "Read (https://example.com)"
        )

    def test_atomic_write_produces_complete_json_without_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            extractor.atomic_write_json(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_partial_output_name(self):
        self.assertEqual(
            extractor.partial_output_path(Path("assignments.json")),
            Path("assignments.partial.json"),
        )

    def test_history_snapshot_names_include_timestamp_and_status(self):
        history_dir = Path("history")
        generated_at = "2026-09-02T22:45:28.123456+00:00"
        self.assertEqual(
            extractor.history_snapshot_path(
                history_dir, generated_at, complete=True
            ),
            history_dir / "2026-09-02T22-45-28.123456Z.complete.json",
        )
        self.assertEqual(
            extractor.history_snapshot_path(
                history_dir, generated_at, complete=False
            ),
            history_dir / "2026-09-02T22-45-28.123456Z.partial.json",
        )

    def test_write_export_files_archives_and_updates_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "assignments.json"
            history_dir = root / "history"
            export = {
                "generated_at": "2026-09-02T22:45:28.123456+00:00",
                "complete": True,
                "courses": [],
            }

            snapshot = extractor.write_export_files(
                destination, history_dir, export
            )

            self.assertEqual(json.loads(destination.read_text()), export)
            self.assertEqual(json.loads(snapshot.read_text()), export)
            self.assertEqual(snapshot.parent, history_dir)

    def test_history_snapshot_avoids_overwriting_a_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            history_dir = Path(directory)
            generated_at = "2026-09-02T22:45:28.123456+00:00"
            first = extractor.history_snapshot_path(
                history_dir, generated_at, complete=True
            )
            first.write_text("existing", encoding="utf-8")
            second = extractor.history_snapshot_path(
                history_dir, generated_at, complete=True
            )
            self.assertEqual(
                second.name,
                "2026-09-02T22-45-28.123456Z.complete.2.json",
            )


if __name__ == "__main__":
    unittest.main()
