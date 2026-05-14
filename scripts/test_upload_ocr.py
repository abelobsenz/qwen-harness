"""Regression test for image-attachment OCR.

User-reported bug: uploading a PNG via the chat UI surfaced to the model
as an empty `<details>` block plus the filename, and the model tried to
`read_file <filename>` from disk (path that doesn't exist server-side).

Fix: `_extract_text_from_upload` now routes images through `_ocr_image`,
which shells out to tesseract and either inlines the OCR'd text or
returns a clear bracketed marker explaining the model can't see images
directly.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen_ui  # type: ignore  # noqa: E402


def _make_test_png(text: str = "Hello OCR World 2026") -> Path:
    """Render a small PNG with `text` so tesseract has something to read."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (520, 80), color="white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    d.text((10, 20), text, fill="black", font=font)
    out = Path("/tmp/qwen_test_ocr.png")
    img.save(out)
    return out


class UploadOCRTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.test_image = _make_test_png()
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"PIL unavailable: {e}")
        import shutil
        if not shutil.which("tesseract"):
            cls.has_tesseract = False
        else:
            cls.has_tesseract = True

    def test_image_extract_returns_nonempty(self) -> None:
        """An image must yield text — OCR text OR an `[image attachment]`
        marker — not the silent empty string that used to bubble up."""
        text, err = qwen_ui._extract_text_from_upload(
            self.test_image, ".png", "image/png")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0,
            "image extract must not return empty text; "
            "model would otherwise see just the filename")

    def test_image_ocr_or_marker(self) -> None:
        """If tesseract is available, the actual text should appear in
        the OCR output; otherwise the bracket-marker should explain why."""
        text, err = qwen_ui._extract_text_from_upload(
            self.test_image, ".png", "image/png")
        if self.has_tesseract and err is None:
            # We rendered "Hello OCR World" — tesseract should pick that
            # up. Be lenient: check for at least one of the words.
            low = text.lower()
            self.assertTrue(
                "ocr" in low or "hello" in low or "world" in low,
                f"OCR output didn't contain expected words: {text[:200]!r}",
            )
            self.assertIn("OCR'd from image attachment", text,
                "OCR output should carry the explanatory header so the "
                "model treats it as transcribed image content, not "
                "user-provided file content")
        else:
            # Should at least have an explanatory marker.
            self.assertIn("image attachment", text.lower())

    def test_merge_into_user_has_visible_content(self) -> None:
        """The chat-merge path must produce a non-empty `<details>` block.
        The original bug was the model seeing an empty code fence and
        trying read_file on the filename instead."""
        original = qwen_ui._read_upload_text
        def fake_read(_upload_id):
            return (
                "[OCR'd from image attachment: test.png]\n\nsome OCR'd text",
                {"path": "/tmp/test.png", "filename": "test.png",
                 "ext": ".png", "size": 117000, "extractor_error": None},
            )
        qwen_ui._read_upload_text = fake_read
        try:
            merged = qwen_ui._merge_attachments_into_last_user(
                [{"role": "user", "content": "what's in this?"}],
                [{"id": "abc123", "filename": "test.png"}])
        finally:
            qwen_ui._read_upload_text = original
        content = merged[0]["content"]
        self.assertIn("OCR'd from image", content,
            "merged user message should contain OCR'd text so the model "
            "doesn't try to read_file the filename")
        self.assertIn("what's in this?", content,
            "the original user prose must still be appended after the "
            "attachment block")


if __name__ == "__main__":
    unittest.main(verbosity=2)
