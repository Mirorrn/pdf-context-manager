"""PDF Document loader and content extractor."""

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""

    page_number: int
    text: str
    image_base64: str
    file_id: str

    @property
    def has_text(self) -> bool:
        """Check if the page has extractable text."""
        return bool(self.text.strip())


class PDFDocument:
    """Load and extract content from a PDF document."""

    def __init__(
        self,
        pdf_path: str | Path,
        dpi: int = 150,
        image_format: str = "PNG",
    ):
        """
        Initialize PDF document. 

        Args:
            pdf_path: Path to the PDF file.
            dpi: Resolution for page image conversion.
            image_format: Output format for images (PNG, JPEG).
        """
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self.image_format = image_format
        self._pages: list[PageContent] | None = None
        self._file_id: str | None = None

    @property
    def file_id(self) -> str:
        """Return the filename as the document identifier."""
        if self._file_id is None:
            self._file_id = self.pdf_path.name
        return self._file_id

    @property
    def pages(self) -> list[PageContent]:
        """Get all pages with extracted content (lazy loaded)."""
        if self._pages is None:
            self._pages = self._extract_all_pages()
        return self._pages

    @property
    def page_count(self) -> int:
        """Get the total number of pages."""
        return len(self.pages)

    def _extract_all_pages(self) -> list[PageContent]:
        """Extract text and images from all pages."""
        # Extract text using pypdf
        reader = PdfReader(self.pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            texts.append(text)

        # Convert pages to images
        images = convert_from_path(
            self.pdf_path,
            dpi=self.dpi,
            fmt=self.image_format.lower(),
        )

        # Combine into PageContent objects
        pages = []
        for i, (text, image) in enumerate(zip(texts, images)):
            page_number = i + 1
            image_base64 = self._image_to_base64(image)
            pages.append(
                PageContent(
                    page_number=page_number,
                    text=text,
                    image_base64=image_base64,
                    file_id=self.file_id,
                )
            )

        return pages

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=self.image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_page(self, page_number: int) -> PageContent:
        """
        Get a specific page by number (1-indexed).

        Args:
            page_number: Page number (starts at 1).

        Returns:
            PageContent for the requested page.

        Raises:
            IndexError: If page number is out of range.
        """
        if page_number < 1 or page_number > self.page_count:
            raise IndexError(
                f"Page {page_number} out of range. Document has {self.page_count} pages."
            )
        return self.pages[page_number - 1]
