import base64
import unittest
from types import SimpleNamespace
from unittest import mock

import canvas_embeddings as embeddings
import canvas_query


def make_result(number, payload="Course content", element_type="Text"):
    return embeddings.SearchResult(
        rowid=number,
        source_file=f"courses/example/page-{number}.json",
        element_type=element_type,
        payload=payload,
        distance=number / 10,
    )


class EmbedQuestionTests(unittest.TestCase):
    def test_uses_matching_voyage_query_configuration(self):
        client = mock.Mock()
        client.multimodal_embed.return_value = SimpleNamespace(
            embeddings=[[0.25] * embeddings.EMBEDDING_DIMENSION]
        )

        vector = canvas_query.embed_question(client, "When is class?")

        self.assertEqual(len(vector), embeddings.EMBEDDING_DIMENSION)
        client.multimodal_embed.assert_called_once_with(
            inputs=[
                {"content": [{"type": "text", "text": "When is class?"}]}
            ],
            model=embeddings.MODEL_NAME,
            input_type="query",
            truncation=False,
            output_dtype="float",
            output_dimension=embeddings.EMBEDDING_DIMENSION,
        )

    def test_rejects_empty_question_and_missing_vector(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            canvas_query.embed_question(mock.Mock(), "  ")

        client = mock.Mock()
        client.multimodal_embed.return_value = SimpleNamespace(embeddings=[])
        with self.assertRaisesRegex(canvas_query.CanvasQueryError, "0 embeddings"):
            canvas_query.embed_question(client, "Question")


class ContextTests(unittest.TestCase):
    def test_numbers_sources_and_omits_base64_from_text_context(self):
        image_payload = "data:image/png;base64," + base64.b64encode(b"image").decode()
        context = canvas_query.build_context(
            [make_result(1), make_result(2, image_payload, "Image")]
        )

        self.assertIn("[1]", context)
        self.assertIn("Course content", context)
        self.assertIn("[2]", context)
        self.assertIn("[retrieved image attached separately]", context)
        self.assertNotIn(base64.b64encode(b"image").decode(), context)

    def test_respects_context_character_limit(self):
        context = canvas_query.build_context(
            [make_result(1, "x" * 1_000)], max_characters=100
        )
        self.assertLessEqual(len(context), 100)
        self.assertIn("[truncated]", context)

    def test_limit_includes_separators_between_sources(self):
        context = canvas_query.build_context(
            [make_result(1, "short"), make_result(2, "x" * 1_000)],
            max_characters=150,
        )
        self.assertLessEqual(len(context), 150)


class GenerateAnswerTests(unittest.TestCase):
    def test_sends_grounded_prompt_and_retrieved_image_to_gemini(self):
        image_payload = "data:image/png;base64," + base64.b64encode(b"image").decode()
        results = [make_result(1), make_result(2, image_payload, "Image")]
        client = mock.Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text="Grounded answer [1]"
        )

        answer = canvas_query.generate_answer(
            client,
            model="test-model",
            question="What happened?",
            results=results,
            max_output_tokens=200,
        )

        self.assertEqual(answer, "Grounded answer [1]")
        call = client.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "test-model")
        self.assertIn("What happened?", call.kwargs["contents"][0])
        self.assertIn("Course content", call.kwargs["contents"][0])
        self.assertEqual(call.kwargs["contents"][1], "Image attached for source [2]:")
        self.assertEqual(call.kwargs["config"].max_output_tokens, 200)


class AnswerQuestionTests(unittest.TestCase):
    def test_runs_embedding_search_and_generation_in_order(self):
        vector = [0.0] * embeddings.EMBEDDING_DIMENSION
        results = [make_result(1)]
        with (
            mock.patch.object(canvas_query, "embed_question", return_value=vector) as embed,
            mock.patch.object(embeddings, "knn_search", return_value=results) as search,
            mock.patch.object(canvas_query, "generate_answer", return_value="Answer [1]") as generate,
        ):
            answer, returned = canvas_query.answer_question(
                "My question",
                database_path=canvas_query.DEFAULT_DATABASE_PATH,
                voyage_client="voyage",
                gemini_client="gemini",
                model="test-model",
                k=3,
                max_context_characters=500,
                max_output_tokens=200,
            )

        self.assertEqual(answer, "Answer [1]")
        self.assertIs(returned, results)
        embed.assert_called_once_with("voyage", "My question")
        search.assert_called_once_with(canvas_query.DEFAULT_DATABASE_PATH, vector, k=3)
        generate.assert_called_once_with(
            "gemini",
            model="test-model",
            question="My question",
            results=results,
            max_context_characters=500,
            max_output_tokens=200,
        )


if __name__ == "__main__":
    unittest.main()
