import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import canvas_embeddings as embeddings


class FakeElement:
    def __init__(
        self,
        category,
        text="",
        *,
        element_id=None,
        **metadata,
    ):
        self.category = category
        self.text = text
        self.id = element_id
        self.metadata = SimpleNamespace(**metadata)


def make_record(number):
    text = f"record {number}"
    return embeddings.IndexRecord(
        source_file=f"course/{number}.html",
        element_type="Text",
        payload=text,
        content=({"type": "text", "text": text},),
    )


class DiscoveryTests(unittest.TestCase):
    def test_discovers_supported_files_recursively_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / "z-last.JPG",
                root / "a" / "lecture.PDF",
                root / "a" / "page.HTM",
                root / "middle.html",
                root / "a" / "ignore.json",
                root / "excluded.png",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            try:
                (root / "linked.pdf").symlink_to(root / "a" / "lecture.PDF")
            except (NotImplementedError, OSError):
                pass

            found = embeddings.discover_documents(
                root, excluded_paths=(root / "excluded.png",)
            )

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in found],
                ["a/lecture.PDF", "a/page.HTM", "middle.html", "z-last.JPG"],
            )

    def test_rejects_missing_and_non_directory_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing"
            regular_file = root / "document.pdf"
            regular_file.write_bytes(b"test")

            with self.assertRaises(FileNotFoundError):
                embeddings.discover_documents(missing)
            with self.assertRaises(NotADirectoryError):
                embeddings.discover_documents(regular_file)


class StructuralChunkTests(unittest.TestCase):
    def test_requests_semantic_chunking_and_retains_original_order(self):
        title = FakeElement("Title", "Week 1", element_id="title")
        image = FakeElement("Image", "Figure caption", element_id="image")
        narrative = FakeElement(
            "NarrativeText", "Explanation", element_id="narrative"
        )
        elements = [title, image, narrative]
        captured = {}

        def chunker(
            received,
            *,
            include_orig_elements,
            combine_text_under_n_chars,
            multipage_sections,
            max_characters,
            new_after_n_chars,
            overlap,
            overlap_all,
            isolate_table,
        ):
            captured.update(
                received=received,
                include_orig_elements=include_orig_elements,
                combine_text_under_n_chars=combine_text_under_n_chars,
                multipage_sections=multipage_sections,
                max_characters=max_characters,
                new_after_n_chars=new_after_n_chars,
                overlap=overlap,
                overlap_all=overlap_all,
                isolate_table=isolate_table,
            )
            # Model Unstructured's shallow original-element copies. Their IDs,
            # rather than object identity, connect them to document positions.
            originals = [
                FakeElement("Title", "Week 1", element_id="title"),
                FakeElement("Image", "Figure caption", element_id="image"),
                FakeElement(
                    "NarrativeText", "Explanation", element_id="narrative"
                ),
            ]
            return [
                FakeElement(
                    "CompositeElement",
                    "Week 1\nFigure caption\nExplanation",
                    element_id="chunk",
                    orig_elements=originals,
                )
            ]

        chunks = embeddings.build_structural_chunks(
            elements, max_characters=321, chunker=chunker
        )

        self.assertIs(captured["received"], elements)
        self.assertEqual(
            {key: value for key, value in captured.items() if key != "received"},
            {
                "include_orig_elements": True,
                "combine_text_under_n_chars": 0,
                "multipage_sections": False,
                "max_characters": 321,
                "new_after_n_chars": 321,
                "overlap": 0,
                "overlap_all": False,
                "isolate_table": False,
            },
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            [element.id for element in chunks[0].elements],
            ["title", "image", "narrative"],
        )
        self.assertEqual(chunks[0].position, 0)

    def test_restores_source_order_and_reinserts_dropped_visual(self):
        before = FakeElement("NarrativeText", "Before", element_id="before")
        image = FakeElement("Image", "", element_id="image")
        after = FakeElement("NarrativeText", "After", element_id="after")

        def chunker(received, **kwargs):
            del received, kwargs
            # Return chunks out of order and omit the empty-alt image, as older
            # Unstructured chunkers can do.
            return [
                FakeElement(
                    "CompositeElement",
                    "After",
                    element_id="late-chunk",
                    orig_elements=[
                        FakeElement(
                            "NarrativeText", "After", element_id="after"
                        )
                    ],
                ),
                FakeElement(
                    "CompositeElement",
                    "Before",
                    element_id="early-chunk",
                    orig_elements=[
                        FakeElement(
                            "NarrativeText", "Before", element_id="before"
                        )
                    ],
                ),
            ]

        chunks = embeddings.build_structural_chunks(
            [before, image, after], chunker=chunker
        )

        self.assertEqual([chunk.position for chunk in chunks], [0, 1, 2])
        self.assertEqual(
            [chunk.category for chunk in chunks],
            ["CompositeElement", "Image", "CompositeElement"],
        )
        self.assertIs(chunks[1].elements[0], image)

    def test_rejects_non_positive_character_ceiling(self):
        with self.assertRaisesRegex(ValueError, "max_characters"):
            embeddings.build_structural_chunks(
                [FakeElement("Text", "content")],
                max_characters=0,
                chunker=lambda elements, **kwargs: [],
            )


class RecordPreparationTests(unittest.TestCase):
    def test_interleaves_target_images_tables_and_narrative_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "course" / "lesson.html"
            source.parent.mkdir()
            source.write_text("<html></html>", encoding="utf-8")

            before = FakeElement("NarrativeText", "Before", element_id="before")
            image = FakeElement("Image", "Figure one", element_id="image")
            between = FakeElement(
                "NarrativeText", "Between", element_id="between"
            )
            table = FakeElement("Table", "A | B", element_id="table")
            after = FakeElement("NarrativeText", "After", element_id="after")
            chunk = embeddings.StructuralChunk(
                text="Before\nFigure one\nBetween\nA | B\nAfter",
                elements=(before, image, between, table, after),
                category="CompositeElement",
                position=0,
            )
            encoded = {
                "image": embeddings.EncodedImage(
                    "data:image/png;base64,SU1BR0U=", 100
                ),
                "table": embeddings.EncodedImage(
                    "data:image/png;base64,VEFCTEU=", 100
                ),
            }

            with mock.patch.object(
                embeddings,
                "_image_for_element",
                side_effect=lambda element, **kwargs: encoded.get(element.id),
            ):
                records = embeddings.prepare_records(source, root, [chunk])

        self.assertEqual(
            [(record.source_file, record.element_type) for record in records],
            [("course/lesson.html", "Image"), ("course/lesson.html", "Table")],
        )
        self.assertEqual(records[0].payload, encoded["image"].data_url)
        self.assertEqual(
            records[0].content,
            (
                {"type": "text", "text": "Before"},
                {
                    "type": "image_base64",
                    "image_base64": encoded["image"].data_url,
                },
                {
                    "type": "text",
                    "text": "Figure one\n\nBetween\n\nA | B\n\nAfter",
                },
            ),
        )
        self.assertEqual(records[1].payload, "A | B")
        self.assertEqual(
            records[1].content,
            (
                {
                    "type": "text",
                    "text": "Before\n\nFigure one\n\nBetween\n\nA | B",
                },
                {
                    "type": "image_base64",
                    "image_base64": encoded["table"].data_url,
                },
                {"type": "text", "text": "After"},
            ),
        )

    def test_isolated_visual_gets_neighboring_text_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "lesson.html"
            source.write_text("<html></html>", encoding="utf-8")
            visual = FakeElement("Image", "", element_id="visual")
            image = embeddings.EncodedImage(
                "data:image/jpeg;base64,SU1BR0U=", 25
            )
            chunks = [
                embeddings.StructuralChunk(
                    "Lead context",
                    (FakeElement("NarrativeText", "Lead context"),),
                    "CompositeElement",
                    0,
                ),
                embeddings.StructuralChunk("", (visual,), "Image", 1),
                embeddings.StructuralChunk(
                    "Follow context",
                    (FakeElement("NarrativeText", "Follow context"),),
                    "CompositeElement",
                    2,
                ),
            ]

            with mock.patch.object(
                embeddings, "_image_for_element", return_value=image
            ):
                records = embeddings.prepare_records(source, root, chunks)

        image_record = next(
            record for record in records if record.element_type == "Image"
        )
        self.assertEqual(
            image_record.content,
            (
                {"type": "text", "text": "Lead context"},
                {"type": "image_base64", "image_base64": image.data_url},
                {"type": "text", "text": "Follow context"},
            ),
        )

    def test_image_document_without_ocr_still_produces_a_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photo.png"
            source.write_bytes(b"not decoded because the helper is patched")
            image = embeddings.EncodedImage(
                "data:image/png;base64,UEhPVE8=", 64
            )

            with mock.patch.object(
                embeddings, "encode_image_file", return_value=image
            ) as encoder:
                records = embeddings.prepare_records(source, root, [])

        encoder.assert_called_once_with(source)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].element_type, "Image")
        self.assertEqual(records[0].payload, image.data_url)
        self.assertEqual(
            records[0].content,
            ({"type": "image_base64", "image_base64": image.data_url},),
        )


class VoyageEmbeddingTests(unittest.TestCase):
    def test_batches_records_and_passes_multimodal_matryoshka_options(self):
        class RecordingClient:
            def __init__(self):
                self.calls = []

            def multimodal_embed(self, **kwargs):
                self.calls.append(kwargs)
                marker = float(len(self.calls))
                return SimpleNamespace(
                    embeddings=[
                        [marker] * embeddings.EMBEDDING_DIMENSION
                        for unused in kwargs["inputs"]
                    ]
                )

        client = RecordingClient()
        records = [make_record(number) for number in range(5)]

        batches = list(embeddings.embed_records(client, records, batch_size=2))

        self.assertEqual([len(batch) for batch, unused in batches], [2, 2, 1])
        self.assertEqual([len(call["inputs"]) for call in client.calls], [2, 2, 1])
        self.assertEqual(
            client.calls[0]["inputs"][0], records[0].voyage_input()
        )
        for call in client.calls:
            self.assertEqual(call["model"], "voyage-multimodal-3")
            self.assertEqual(call["input_type"], "document")
            self.assertFalse(call["truncation"])
            self.assertEqual(call["output_dtype"], "float")
            self.assertEqual(call["output_dimension"], 1024)
        self.assertEqual(batches[0][1][0][0], 1.0)
        self.assertEqual(batches[-1][1][0][0], 3.0)

    def test_rejects_wrong_embedding_count_and_dimension(self):
        record = make_record(1)
        count_client = mock.Mock()
        count_client.multimodal_embed.return_value = SimpleNamespace(embeddings=[])
        with self.assertRaisesRegex(
            embeddings.CanvasEmbeddingError, "0 embeddings.*batch of 1"
        ):
            list(embeddings.embed_records(count_client, [record]))

        dimension_client = mock.Mock()
        dimension_client.multimodal_embed.return_value = SimpleNamespace(
            embeddings=[[0.0] * (embeddings.EMBEDDING_DIMENSION - 1)]
        )
        with self.assertRaisesRegex(
            embeddings.CanvasEmbeddingError, "1023 dimensions; expected 1024"
        ):
            list(embeddings.embed_records(dimension_client, [record]))

    def test_validate_embedding_rejects_non_numeric_and_non_finite_values(self):
        valid = [0.0] * embeddings.EMBEDDING_DIMENSION
        for invalid, message in ((True, "not numeric"), (math.nan, "not finite")):
            with self.subTest(invalid=invalid):
                vector = list(valid)
                vector[9] = invalid
                with self.assertRaisesRegex(
                    embeddings.CanvasEmbeddingError, message
                ):
                    embeddings.validate_embedding(vector)

    def test_empty_record_list_does_not_call_voyage(self):
        client = mock.Mock()
        self.assertEqual(list(embeddings.embed_records(client, [])), [])
        client.multimodal_embed.assert_not_called()


class KnnSearchTests(unittest.TestCase):
    def test_executes_l2_join_query_and_maps_results(self):
        class FakeCursor:
            def fetchall(self):
                return [
                    (7, "course/lesson.html", "Table", "A | B", 0.25),
                    (3, "course/photo.png", "Image", "data:image/png", 0.5),
                ]

        class FakeConnection:
            def __init__(self):
                self.executions = []
                self.closed = False

            def execute(self, sql, parameters):
                self.executions.append((sql, parameters))
                return FakeCursor()

            def close(self):
                self.closed = True

        class FakeSqliteVec:
            def __init__(self):
                self.vectors = []

            def serialize_float32(self, vector):
                self.vectors.append(vector)
                return b"query-vector"

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "canvas_embeddings.db"
            database_path.write_bytes(b"test placeholder")
            connection = FakeConnection()
            sqlite_vec = FakeSqliteVec()
            with mock.patch.object(
                embeddings,
                "connect_vector_database",
                return_value=(connection, sqlite_vec),
            ) as connect:
                results = embeddings.knn_search(
                    database_path,
                    [0.0] * embeddings.EMBEDDING_DIMENSION,
                    k=2,
                )

        connect.assert_called_once_with(database_path.resolve())
        self.assertTrue(connection.closed)
        self.assertEqual(len(sqlite_vec.vectors), 1)
        self.assertEqual(len(sqlite_vec.vectors[0]), 1024)
        self.assertEqual(len(connection.executions), 1)
        sql, parameters = connection.executions[0]
        normalized_sql = " ".join(sql.split())
        self.assertIn("vec_distance_L2(vectors.embedding, ?)", normalized_sql)
        self.assertIn(
            "JOIN canvas_metadata AS metadata ON metadata.rowid = vectors.rowid",
            normalized_sql,
        )
        self.assertIn("ORDER BY distance ASC LIMIT ?", normalized_sql)
        self.assertEqual(parameters, (b"query-vector", 2))
        self.assertEqual(
            results,
            [
                embeddings.SearchResult(
                    7, "course/lesson.html", "Table", "A | B", 0.25
                ),
                embeddings.SearchResult(
                    3,
                    "course/photo.png",
                    "Image",
                    "data:image/png",
                    0.5,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
