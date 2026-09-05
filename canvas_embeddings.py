#!/usr/bin/env python3
"""Build a local multimodal vector index from exported Canvas documents.

The indexer recursively processes PDF, HTML, and image files under
``./extracted_canvas_data``.  Unstructured performs layout-aware partitioning
and title/page-aware chunking; the original elements are then reconstructed in
document order so text and visual content are sent to Voyage as interleaved
inputs.  The resulting 1024-dimensional float vectors are stored with their
raw payloads in a sqlite-vec database.

Install Python dependencies (native Poppler and Tesseract packages are also
needed for Unstructured's local high-resolution PDF/image processing):

    python3.13 -m venv .venv
    .venv/bin/pip install -r requirements-embeddings.txt
    .venv/bin/python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng

Create ``.env`` with ``VOYAGE_API_KEY=...``, then run:

    .venv/bin/python canvas_embeddings.py

The high-resolution Unstructured model may be downloaded on its first use.
Only local image references in HTML are read; remote image URLs are skipped.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import inspect
import io
import logging
import math
import mimetypes
import os
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote, urlsplit


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "extracted_canvas_data"
DEFAULT_DATABASE_PATH = PROJECT_DIR / "canvas_embeddings.db"
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"

# voyage-multimodal-3 has a native, fixed 1024-dimensional output. The current
# client accepts output_dimension explicitly (and we validate it below); unlike
# voyage-multimodal-3.5, this legacy model does not expose smaller Matryoshka
# dimensions.
MODEL_NAME = "voyage-multimodal-3"
EMBEDDING_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_CHARACTERS = 12_000
MAX_VOYAGE_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VOYAGE_IMAGE_PIXELS = 16_000_000
# Leave room in Voyage's per-input token budget for surrounding text. Voyage
# counts approximately one image token per 560 pixels.
TARGET_IMAGE_PIXELS = 8_000_000

PDF_SUFFIXES = frozenset({".pdf"})
HTML_SUFFIXES = frozenset({".html", ".htm"})
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".bmp", ".heic"}
)
SUPPORTED_SUFFIXES = PDF_SUFFIXES | HTML_SUFFIXES | IMAGE_SUFFIXES
VOYAGE_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
VISUAL_CATEGORIES = frozenset({"Image", "Table", "TableChunk"})

LOGGER = logging.getLogger("canvas_embeddings")


class CanvasEmbeddingError(RuntimeError):
    """A user-actionable indexing failure."""


class DependencyError(CanvasEmbeddingError):
    """A required optional or native dependency is unavailable."""


@dataclass(frozen=True)
class EncodedImage:
    """Voyage-ready image data and its decoded pixel count."""

    data_url: str
    pixel_count: int


@dataclass(frozen=True)
class StructuralChunk:
    """A semantic Unstructured chunk with its original ordered elements."""

    text: str
    elements: tuple[Any, ...]
    category: str
    position: int


@dataclass(frozen=True)
class IndexRecord:
    """One metadata row and one corresponding multimodal embedding input."""

    source_file: str
    element_type: str
    payload: str
    content: tuple[dict[str, str], ...]

    def voyage_input(self) -> dict[str, list[dict[str, str]]]:
        return {"content": [dict(part) for part in self.content]}


@dataclass(frozen=True)
class SearchResult:
    rowid: int
    source_file: str
    element_type: str
    payload: str
    distance: float


@dataclass
class IndexStats:
    discovered_files: int = 0
    processed_files: int = 0
    empty_files: int = 0
    records: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _element_category(element: Any) -> str:
    value = getattr(element, "category", None)
    return str(value or element.__class__.__name__)


def _element_text(element: Any) -> str:
    value = getattr(element, "text", "")
    return value if isinstance(value, str) else str(value or "")


def _metadata_value(element: Any, name: str) -> Any:
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _element_key(element: Any) -> str:
    """Return a stable key shared by Unstructured's shallow element copies."""
    try:
        element_id = getattr(element, "id")
    except Exception:
        element_id = None
    return str(element_id) if element_id else f"object:{id(element)}"


def discover_documents(
    root: Path, *, excluded_paths: Iterable[Path] = ()
) -> list[Path]:
    """Return supported, in-root regular files in deterministic order.

    Symlinks are deliberately ignored: an exported course document should not
    make the indexer traverse outside the selected local data directory.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {root}")

    excluded = {path.expanduser().resolve(strict=False) for path in excluded_paths}
    documents: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            LOGGER.warning("Skipping unreadable path %s: %s", path, error)
            continue
        if not resolved.is_file() or not resolved.is_relative_to(root):
            continue
        if resolved in excluded:
            continue
        documents.append(resolved)
    return sorted(documents, key=lambda path: path.relative_to(root).as_posix())


def partition_document(path: Path, languages: Sequence[str] = ()) -> list[Any]:
    """Partition one supported document with Unstructured."""
    common_kwargs: dict[str, Any] = {
        "filename": str(path),
        "extract_image_block_to_payload": True,
    }
    if path.suffix.lower() in HTML_SUFFIXES:
        try:
            from unstructured.partition.html import partition_html
        except ImportError as error:
            raise DependencyError(
                "Unstructured HTML dependencies are missing; install "
                "requirements-embeddings.txt"
            ) from error
        try:
            return list(
                partition_html(
                    **common_kwargs,
                    extract_image_block_types=["Image"],
                )
            )
        except LookupError as error:
            raise DependencyError(
                "Unstructured NLP data is missing; run `.venv/bin/python -m "
                "nltk.downloader punkt_tab averaged_perceptron_tagger_eng`"
            ) from error

    try:
        if path.suffix.lower() in PDF_SUFFIXES:
            from unstructured.partition.pdf import partition_pdf as partitioner
        else:
            from unstructured.partition.image import partition_image as partitioner
    except ImportError as error:
        raise DependencyError(
            "Unstructured PDF/image dependencies are missing; install "
            "requirements-embeddings.txt and the documented native packages"
        ) from error

    if languages:
        common_kwargs["languages"] = list(languages)
    try:
        return list(
            partitioner(
                **common_kwargs,
                strategy="hi_res",
                infer_table_structure=True,
                include_page_breaks=True,
                extract_image_block_types=["Image", "Table"],
            )
        )
    except LookupError as error:
        raise DependencyError(
            "Unstructured NLP/OCR data is missing; install the requested "
            "Tesseract language packs and run `.venv/bin/python -m "
            "nltk.downloader punkt_tab averaged_perceptron_tagger_eng`"
        ) from error
    except ImportError as error:
        raise DependencyError(
            "Unstructured local inference dependencies are incomplete; install "
            "requirements-embeddings.txt"
        ) from error


def build_structural_chunks(
    elements: Sequence[Any],
    *,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    chunker: Callable[..., Sequence[Any]] | None = None,
) -> list[StructuralChunk]:
    """Chunk by headings/pages while retaining original multimodal elements.

    The character ceiling is an API-safety backstop. Unstructured first groups
    complete semantic elements and only splits text when one individual element
    itself exceeds that ceiling.
    """
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    if not elements:
        return []

    if chunker is None:
        try:
            from unstructured.chunking.title import chunk_by_title
        except ImportError as error:
            raise DependencyError(
                "Unstructured chunking is unavailable; install "
                "requirements-embeddings.txt"
            ) from error
        chunker = chunk_by_title

    # Force IDs before chunking. Unstructured copies original elements into
    # orig_elements, and a pre-existing ID survives those copies.
    positions = {_element_key(element): index for index, element in enumerate(elements)}
    kwargs: dict[str, Any] = {
        "include_orig_elements": True,
        "combine_text_under_n_chars": 0,
        "multipage_sections": False,
        "max_characters": max_characters,
        "new_after_n_chars": max_characters,
        "overlap": 0,
        "overlap_all": False,
    }
    try:
        parameters = inspect.signature(chunker).parameters
    except (TypeError, ValueError):
        parameters = {}
    # New Unstructured versions can keep tables in the same semantic chunk as
    # adjacent prose. Older releases isolate them; record preparation below adds
    # nearest-neighbor paragraph context in that case.
    if "isolate_table" in parameters:
        kwargs["isolate_table"] = False

    raw_chunks = list(chunker(elements, **kwargs))
    groups: list[tuple[int, int, StructuralChunk]] = []
    covered_visuals: set[str] = set()
    for order, chunk in enumerate(raw_chunks):
        originals = _metadata_value(chunk, "orig_elements")
        if not isinstance(originals, (list, tuple)) or not originals:
            originals = [chunk]
        original_tuple = tuple(originals)
        original_positions = [
            positions[key]
            for element in original_tuple
            if (key := _element_key(element)) in positions
        ]
        position = min(original_positions, default=len(elements) + order)
        for element in original_tuple:
            if _element_category(element) in VISUAL_CATEGORIES:
                covered_visuals.add(_element_key(element))
        group = StructuralChunk(
            text=_element_text(chunk).strip(),
            elements=original_tuple,
            category=_element_category(chunk),
            position=position,
        )
        groups.append((position, order, group))

    # Empty-alt images can disappear when a chunker emits only text-bearing
    # chunks. Reinsert any dropped visual element in its original location.
    synthetic_order = len(raw_chunks)
    for position, element in enumerate(elements):
        if (
            _element_category(element) in VISUAL_CATEGORIES
            and _element_key(element) not in covered_visuals
        ):
            group = StructuralChunk(
                text=_element_text(element).strip(),
                elements=(element,),
                category=_element_category(element),
                position=position,
            )
            groups.append((position, synthetic_order, group))
            synthetic_order += 1

    groups.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in groups]


def _register_heif_opener() -> None:
    """Best-effort registration for HEIC images installed by Unstructured."""
    for module_name in ("pi_heif", "pillow_heif"):
        try:
            module = __import__(module_name, fromlist=["register_heif_opener"])
            register = getattr(module, "register_heif_opener", None)
            if register is not None:
                register()
                return
        except (ImportError, OSError):
            continue


def _canonical_image_mime(mime_type: str | None) -> str | None:
    if not mime_type:
        return None
    value = mime_type.split(";", 1)[0].strip().lower()
    if value in {"image/jpg", "image/pjpeg"}:
        return "image/jpeg"
    return value


def encode_image_bytes(data: bytes, mime_hint: str | None = None) -> EncodedImage:
    """Validate and, only when needed, normalize an image for Voyage."""
    if not data:
        raise ValueError("image payload is empty")
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as error:
        raise DependencyError("Pillow is required to validate image payloads") from error

    _register_heif_opener()
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("image has invalid dimensions")
            source_pixels = width * height
            if source_pixels > MAX_VOYAGE_IMAGE_PIXELS * 8:
                raise ValueError(
                    f"image is too large to safely decode ({source_pixels:,} pixels)"
                )
            detected_mime = _canonical_image_mime(
                Image.MIME.get(image.format or "") or mime_hint
            )
            needs_conversion = (
                detected_mime not in VOYAGE_IMAGE_MIME_TYPES
                or len(data) > MAX_VOYAGE_IMAGE_BYTES
                or source_pixels > TARGET_IMAGE_PIXELS
            )
            image.load()
            if not needs_conversion:
                encoded = base64.b64encode(data).decode("ascii")
                return EncodedImage(
                    f"data:{detected_mime};base64,{encoded}", source_pixels
                )

            image.seek(0)
            normalized = ImageOps.exif_transpose(image)
            if normalized.width * normalized.height > TARGET_IMAGE_PIXELS:
                scale = math.sqrt(
                    TARGET_IMAGE_PIXELS / (normalized.width * normalized.height)
                )
                size = (
                    max(1, int(normalized.width * scale)),
                    max(1, int(normalized.height * scale)),
                )
                normalized = normalized.resize(size, Image.Resampling.LANCZOS)

            if normalized.mode in {"RGBA", "LA"} or (
                normalized.mode == "P" and "transparency" in normalized.info
            ):
                rgba = normalized.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                normalized = background
            else:
                normalized = normalized.convert("RGB")

            output = io.BytesIO()
            normalized.save(output, format="JPEG", quality=92, optimize=True)
            normalized_bytes = output.getvalue()
            if len(normalized_bytes) > MAX_VOYAGE_IMAGE_BYTES:
                raise ValueError(
                    "normalized image still exceeds Voyage's 20 MB limit"
                )
            pixels = normalized.width * normalized.height
            encoded = base64.b64encode(normalized_bytes).decode("ascii")
            return EncodedImage(f"data:image/jpeg;base64,{encoded}", pixels)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"invalid or unsupported image: {error}") from error


def encode_image_file(path: Path) -> EncodedImage:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise OSError(f"cannot read image {path}: {error}") from error
    mime_type, _ = mimetypes.guess_type(path.name)
    return encode_image_bytes(data, mime_type)


def _decode_element_image(element: Any) -> EncodedImage | None:
    value = _metadata_value(element, "image_base64")
    if not isinstance(value, str) or not value.strip():
        return None
    encoded = value.strip()
    mime_type = _canonical_image_mime(_metadata_value(element, "image_mime_type"))
    if encoded.lower().startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("image data URL is not base64 encoded")
        mime_type = _canonical_image_mime(header[5:].split(";", 1)[0])
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("element contains invalid base64 image data") from error
    return encode_image_bytes(data, mime_type)


def _resolve_local_image_reference(
    reference: str, *, source_path: Path, input_root: Path
) -> Path | None:
    parsed = urlsplit(reference)
    if parsed.scheme.lower() in {"http", "https"} or (
        not parsed.scheme and parsed.netloc
    ):
        return None
    if parsed.scheme.lower() not in {"", "file"}:
        return None
    if parsed.scheme.lower() == "file" and parsed.netloc not in {"", "localhost"}:
        return None
    raw_path = Path(unquote(parsed.path))
    candidate = raw_path if raw_path.is_absolute() else source_path.parent / raw_path
    resolved = candidate.resolve(strict=True)
    root = input_root.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("HTML image reference resolves outside the input directory")
    if not resolved.is_file():
        raise ValueError("HTML image reference is not a regular file")
    return resolved


def _image_for_element(
    element: Any,
    *,
    source_path: Path,
    input_root: Path,
) -> EncodedImage | None:
    embedded = _decode_element_image(element)
    if embedded is not None:
        return embedded

    reference = _metadata_value(element, "image_path") or _metadata_value(
        element, "image_url"
    )
    if not isinstance(reference, str) or not reference.strip():
        return None
    try:
        resolved = _resolve_local_image_reference(
            reference.strip(), source_path=source_path, input_root=input_root
        )
    except (OSError, ValueError) as error:
        LOGGER.warning("Skipping image reference in %s: %s", source_path, error)
        return None
    if resolved is None:
        LOGGER.warning(
            "Skipping non-local image reference in %s: %s", source_path, reference
        )
        return None
    try:
        return encode_image_file(resolved)
    except (OSError, ValueError) as error:
        LOGGER.warning("Skipping unreadable image %s: %s", resolved, error)
        return None


def _append_text(parts: list[dict[str, str]], text: str) -> None:
    text = text.strip()
    if not text:
        return
    if parts and parts[-1]["type"] == "text":
        parts[-1]["text"] += "\n\n" + text
    else:
        parts.append({"type": "text", "text": text})


def _append_image(parts: list[dict[str, str]], image: EncodedImage) -> None:
    parts.append({"type": "image_base64", "image_base64": image.data_url})


def _neighbor_text(
    chunks: Sequence[StructuralChunk], index: int, direction: int
) -> str:
    cursor = index + direction
    while 0 <= cursor < len(chunks):
        candidate = chunks[cursor]
        # Avoid using another visual's extracted data as narrative context.
        if not any(
            _element_category(element) in VISUAL_CATEGORIES
            for element in candidate.elements
        ):
            return candidate.text
        cursor += direction
    return ""


@dataclass(frozen=True)
class _VisualItem:
    element: Any | None
    element_type: str
    image: EncodedImage | None
    text: str


def prepare_records(
    source_path: Path,
    input_root: Path,
    chunks: Sequence[StructuralChunk],
) -> list[IndexRecord]:
    """Create one retrievable row per text chunk or visual element.

    Visual rows receive a joint embedding of the image/table and surrounding
    prose. Keeping one visual per request stays within Voyage's image-token
    limit while preserving the requested singular ``element_type``/``payload``
    schema.
    """
    input_root = input_root.resolve()
    source_file = source_path.resolve().relative_to(input_root).as_posix()
    is_image_document = source_path.suffix.lower() in IMAGE_SUFFIXES
    source_image = encode_image_file(source_path) if is_image_document else None
    image_cache: dict[str, EncodedImage | None] = {}
    records: list[IndexRecord] = []

    for index, chunk in enumerate(chunks):
        visuals: list[_VisualItem] = []
        for element in chunk.elements:
            category = _element_category(element)
            if category not in VISUAL_CATEGORIES:
                continue
            cache_key = _element_key(element)
            if cache_key not in image_cache:
                try:
                    image_cache[cache_key] = _image_for_element(
                        element, source_path=source_path, input_root=input_root
                    )
                except (DependencyError, ValueError) as error:
                    LOGGER.warning(
                        "Skipping invalid visual element in %s: %s", source_path, error
                    )
                    image_cache[cache_key] = None
            image = image_cache[cache_key]
            if category in {"Table", "TableChunk"}:
                table_text = _element_text(element).strip()
                if chunk.category == "TableChunk" and chunk.text:
                    table_text = chunk.text
                visuals.append(_VisualItem(element, "Table", image, table_text))
            elif image is not None:
                visuals.append(
                    _VisualItem(element, "Image", image, _element_text(element).strip())
                )

        # An image file is itself visual content even when OCR/layout detection
        # yields no Image element. Pair it with every structural OCR chunk.
        if source_image is not None:
            visuals.insert(0, _VisualItem(None, "Image", source_image, ""))

        if not visuals:
            payload = chunk.text.strip()
            if payload:
                records.append(
                    IndexRecord(
                        source_file=source_file,
                        element_type="Text",
                        payload=payload,
                        content=({"type": "text", "text": payload},),
                    )
                )
            continue

        has_narrative_context = any(
            _element_category(element) not in VISUAL_CATEGORIES
            and bool(_element_text(element).strip())
            for element in chunk.elements
        )
        before = "" if has_narrative_context else _neighbor_text(chunks, index, -1)
        after = "" if has_narrative_context else _neighbor_text(chunks, index, 1)

        for visual in visuals:
            parts: list[dict[str, str]] = []
            _append_text(parts, before)
            if visual.element is None:
                _append_text(parts, chunk.text)
                assert visual.image is not None
                _append_image(parts, visual.image)
            else:
                for element in chunk.elements:
                    category = _element_category(element)
                    if element is visual.element:
                        if visual.element_type == "Table":
                            _append_text(parts, visual.text)
                            if visual.image is not None:
                                _append_image(parts, visual.image)
                        else:
                            assert visual.image is not None
                            _append_image(parts, visual.image)
                            _append_text(parts, visual.text)
                    elif category not in VISUAL_CATEGORIES:
                        _append_text(parts, _element_text(element))
                    else:
                        # Alt/caption-like text from other visuals remains useful
                        # context, but each embedding contains only its target image.
                        _append_text(parts, _element_text(element))
            _append_text(parts, after)

            if visual.element_type == "Image":
                assert visual.image is not None
                payload = visual.image.data_url
            else:
                payload = visual.text or (
                    visual.image.data_url if visual.image is not None else chunk.text
                )
            if parts and payload:
                records.append(
                    IndexRecord(
                        source_file=source_file,
                        element_type=visual.element_type,
                        payload=payload,
                        content=tuple(parts),
                    )
                )

    # A photograph with no OCR text produces no Unstructured chunks. It must
    # still be embedded and indexed as a visual-only document.
    if not chunks and source_image is not None:
        records.append(
            IndexRecord(
                source_file=source_file,
                element_type="Image",
                payload=source_image.data_url,
                content=(
                    {
                        "type": "image_base64",
                        "image_base64": source_image.data_url,
                    },
                ),
            )
        )
    return records


def validate_embedding(
    embedding: Sequence[Any], *, dimensions: int = EMBEDDING_DIMENSION
) -> list[float]:
    if len(embedding) != dimensions:
        raise CanvasEmbeddingError(
            f"Voyage returned {len(embedding)} dimensions; expected {dimensions}"
        )
    vector: list[float] = []
    for index, value in enumerate(embedding):
        if isinstance(value, bool):
            raise CanvasEmbeddingError(f"embedding value {index} is not numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise CanvasEmbeddingError(
                f"embedding value {index} is not numeric"
            ) from error
        if not math.isfinite(number):
            raise CanvasEmbeddingError(f"embedding value {index} is not finite")
        vector.append(number)
    return vector


def embed_records(
    client: Any,
    records: Sequence[IndexRecord],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[list[IndexRecord], list[list[float]]]]:
    """Yield aligned record/vector batches from the official Voyage client."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(records), batch_size):
        batch = list(records[start : start + batch_size])
        inputs = [record.voyage_input() for record in batch]
        try:
            response = client.multimodal_embed(
                inputs=inputs,
                model=MODEL_NAME,
                input_type="document",
                truncation=False,
                output_dtype="float",
                output_dimension=EMBEDDING_DIMENSION,
            )
        except TypeError as error:
            raise DependencyError(
                "the installed voyageai client is too old for explicit multimodal "
                "output dimensions; install requirements-embeddings.txt"
            ) from error
        except Exception as error:
            sources = ", ".join(sorted({record.source_file for record in batch}))
            raise CanvasEmbeddingError(
                f"Voyage embedding request failed for batch containing {sources}: {error}"
            ) from error
        embeddings = getattr(response, "embeddings", None)
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            actual = len(embeddings) if isinstance(embeddings, list) else "no"
            raise CanvasEmbeddingError(
                f"Voyage returned {actual} embeddings for a batch of {len(batch)}"
            )
        yield batch, [validate_embedding(embedding) for embedding in embeddings]


def _import_sqlite_vec() -> ModuleType:
    try:
        import sqlite_vec
    except ImportError as error:
        raise DependencyError(
            "sqlite-vec is not installed; install requirements-embeddings.txt"
        ) from error
    return sqlite_vec


def connect_vector_database(
    database_path: Path | str,
    *,
    sqlite_vec_module: ModuleType | Any | None = None,
) -> tuple[sqlite3.Connection, Any]:
    """Connect and load sqlite-vec, minimizing the extension-loading window."""
    sqlite_vec_module = sqlite_vec_module or _import_sqlite_vec()
    connection = sqlite3.connect(str(database_path), timeout=30.0)
    try:
        try:
            connection.enable_load_extension(True)
            sqlite_vec_module.load(connection)
        finally:
            connection.enable_load_extension(False)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection, sqlite_vec_module
    except Exception:
        connection.close()
        raise


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        CREATE TABLE canvas_metadata (
            rowid INTEGER PRIMARY KEY,
            source_file TEXT NOT NULL,
            element_type TEXT NOT NULL
                CHECK (element_type IN ('Text', 'Image', 'Table')),
            payload TEXT NOT NULL
        );

        CREATE INDEX canvas_metadata_source_idx
            ON canvas_metadata(source_file);
        CREATE INDEX canvas_metadata_type_idx
            ON canvas_metadata(element_type);

        CREATE VIRTUAL TABLE canvas_embeddings USING vec0(
            embedding FLOAT[{EMBEDDING_DIMENSION}]
        );
        """
    )


def insert_embedding_batch(
    connection: sqlite3.Connection,
    sqlite_vec_module: Any,
    records: Sequence[IndexRecord],
    embeddings: Sequence[Sequence[float]],
) -> None:
    if len(records) != len(embeddings):
        raise ValueError("record and embedding counts do not match")
    with connection:
        for record, embedding in zip(records, embeddings, strict=True):
            vector = validate_embedding(embedding)
            cursor = connection.execute(
                """
                INSERT INTO canvas_metadata(source_file, element_type, payload)
                VALUES (?, ?, ?)
                """,
                (record.source_file, record.element_type, record.payload),
            )
            rowid = cursor.lastrowid
            if rowid is None:
                raise CanvasEmbeddingError("SQLite did not return a metadata rowid")
            connection.execute(
                "INSERT INTO canvas_embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec_module.serialize_float32(vector)),
            )


def _create_voyage_client(api_key: str, *, timeout: float, max_retries: int) -> Any:
    try:
        import voyageai
    except ImportError as error:
        raise DependencyError(
            "voyageai is not installed; install requirements-embeddings.txt"
        ) from error
    return voyageai.Client(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def _load_api_key(env_path: Path) -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise DependencyError(
            "python-dotenv is not installed; install requirements-embeddings.txt"
        ) from error
    load_dotenv(dotenv_path=env_path, override=False)
    api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise CanvasEmbeddingError(
            f"VOYAGE_API_KEY is not set (checked the environment and {env_path})"
        )
    return api_key


def _temporary_database_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(source: Path, destination: Path) -> None:
    _fsync_file(source)
    os.replace(source, destination)
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def build_index(
    *,
    input_root: Path,
    database_path: Path,
    client: Any,
    batch_size: int,
    max_characters: int,
    languages: Sequence[str] = (),
) -> IndexStats:
    """Build a complete temporary index and atomically replace the final DB."""
    input_root = input_root.expanduser().resolve()
    database_path = database_path.expanduser().resolve()
    documents = discover_documents(input_root, excluded_paths=(database_path,))
    if not documents:
        raise CanvasEmbeddingError(
            f"no supported PDF, HTML, or image files found under {input_root}"
        )

    stats = IndexStats(discovered_files=len(documents))
    temporary_path = _temporary_database_path(database_path)
    connection: sqlite3.Connection | None = None
    try:
        connection, sqlite_vec_module = connect_vector_database(temporary_path)
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        initialize_schema(connection)

        pending: list[IndexRecord] = []

        def flush_pending() -> None:
            if not pending:
                return
            for record_batch, vector_batch in embed_records(
                client, pending, batch_size=batch_size
            ):
                insert_embedding_batch(
                    connection, sqlite_vec_module, record_batch, vector_batch
                )
                stats.records += len(record_batch)
            pending.clear()

        for number, path in enumerate(documents, 1):
            relative_path = path.relative_to(input_root).as_posix()
            LOGGER.info("[%d/%d] Processing %s", number, len(documents), relative_path)
            try:
                elements = partition_document(path, languages)
                chunks = build_structural_chunks(
                    elements, max_characters=max_characters
                )
                records = prepare_records(path, input_root, chunks)
            except (OSError, ValueError, DependencyError) as error:
                LOGGER.error("Could not process %s: %s", relative_path, error)
                stats.errors.append((relative_path, str(error)))
                continue
            except Exception as error:
                # Unstructured raises several backend-specific exceptions for
                # encrypted, malformed, or unreadable files. Continue with the
                # remaining course data while preserving the diagnostic.
                LOGGER.exception("Could not process %s", relative_path)
                stats.errors.append((relative_path, str(error)))
                continue

            if not records:
                LOGGER.warning("No indexable elements found in %s", relative_path)
                stats.empty_files += 1
                continue
            stats.processed_files += 1
            pending.extend(records)
            if len(pending) >= batch_size:
                flush_pending()

        flush_pending()
        if stats.records == 0:
            if stats.errors:
                LOGGER.error(
                    "Every file failed to process (%d/%d); grouped causes:",
                    len(stats.errors),
                    stats.discovered_files,
                )
                by_message: dict[str, list[str]] = {}
                for source_file, message in stats.errors:
                    by_message.setdefault(message, []).append(source_file)
                for message, files in by_message.items():
                    sample = ", ".join(files[:3])
                    remainder = len(files) - 3
                    if remainder > 0:
                        sample += f", and {remainder} more"
                    LOGGER.error("  %s (%s)", message, sample)
            raise CanvasEmbeddingError(
                "no embeddings were created; the existing database was left unchanged"
            )
        connection.execute("PRAGMA optimize")
        connection.commit()
        connection.close()
        connection = None
        _atomic_replace(temporary_path, database_path)
    finally:
        if connection is not None:
            connection.close()
        temporary_path.unlink(missing_ok=True)
        Path(f"{temporary_path}-journal").unlink(missing_ok=True)
        Path(f"{temporary_path}-wal").unlink(missing_ok=True)
        Path(f"{temporary_path}-shm").unlink(missing_ok=True)
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Structurally chunk local Canvas PDFs, HTML, and images; create "
            "Voyage multimodal embeddings; and store them in sqlite-vec."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory to scan recursively (default: ./extracted_canvas_data)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="output sqlite-vec database (default: ./canvas_embeddings.db)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="dotenv file containing VOYAGE_API_KEY (default: ./.env)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Voyage inputs per request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=DEFAULT_MAX_CHARACTERS,
        help=(
            "hard safety ceiling for a semantic chunk; whole Unstructured "
            f"elements are preserved when possible (default: {DEFAULT_MAX_CHARACTERS})"
        ),
    )
    parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="OCR language code; may be repeated (for example --language eng)",
    )
    parser.add_argument(
        "--api-timeout", type=float, default=120.0, metavar="SECONDS"
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        if args.max_characters <= 0:
            raise ValueError("--max-characters must be positive")
        if args.api_timeout <= 0:
            raise ValueError("--api-timeout must be positive")
        if args.max_retries < 1:
            raise ValueError("--max-retries must be at least 1")
        api_key = _load_api_key(args.env_file.expanduser().resolve())
        client = _create_voyage_client(
            api_key, timeout=args.api_timeout, max_retries=args.max_retries
        )
        stats = build_index(
            input_root=args.input_dir,
            database_path=args.database,
            client=client,
            batch_size=args.batch_size,
            max_characters=args.max_characters,
            languages=args.language,
        )
    except (CanvasEmbeddingError, OSError, ValueError, sqlite3.Error) as error:
        LOGGER.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("interrupted; the existing database was left unchanged")
        return 130

    print(
        f"Indexed {stats.records} element(s) from {stats.processed_files}/"
        f"{stats.discovered_files} file(s) into {args.database.resolve()}"
    )
    if stats.empty_files:
        print(f"Skipped {stats.empty_files} file(s) with no indexable content.")
    if stats.errors:
        print(f"Completed with {len(stats.errors)} unreadable/invalid file(s):")
        for source_file, message in stats.errors:
            print(f"  {source_file}: {message}")
        return 2
    return 0


def knn_search(
    database_path: Path | str,
    query_embedding: Sequence[Any],
    *,
    k: int = 5,
) -> list[SearchResult]:
    """Run exact KNN search using sqlite-vec's ``vec_distance_L2``.

    ``query_embedding`` should be a 1024-dimensional embedding produced with
    ``voyage-multimodal-3`` and ``input_type='query'``. This explicit scalar
    distance example performs an exact full scan; sqlite-vec's ``MATCH`` syntax
    can be substituted for larger indexes.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    vector = validate_embedding(query_embedding)
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"vector database does not exist: {path}")
    connection, sqlite_vec_module = connect_vector_database(path)
    try:
        query_blob = sqlite_vec_module.serialize_float32(vector)
        rows = connection.execute(
            """
            SELECT
                metadata.rowid,
                metadata.source_file,
                metadata.element_type,
                metadata.payload,
                vec_distance_L2(vectors.embedding, ?) AS distance
            FROM canvas_embeddings AS vectors
            JOIN canvas_metadata AS metadata
                ON metadata.rowid = vectors.rowid
            ORDER BY distance ASC
            LIMIT ?
            """,
            (query_blob, k),
        ).fetchall()
    finally:
        connection.close()
    return [
        SearchResult(
            rowid=int(row[0]),
            source_file=str(row[1]),
            element_type=str(row[2]),
            payload=str(row[3]),
            distance=float(row[4]),
        )
        for row in rows
    ]


if __name__ == "__main__":
    raise SystemExit(main())
