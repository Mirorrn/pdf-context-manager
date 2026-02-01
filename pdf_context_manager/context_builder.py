"""Build multimodal LLM request context from PDF documents."""

import base64
from typing import Any

from pydantic_ai import BinaryContent

from .document import PDFDocument, PageContent


class ContextBuilder:
    """Construct multimodal LLM requests from PDF documents."""

    DEFAULT_SYSTEM_PROMPT = """You are a document analysis assistant. You have been provided with:
1. Extracted text from PDF pages (if available)
2. Images of each PDF page for visual analysis

Use both the text content and visual information to answer questions accurately.

## CRITICAL: Citation Requirements

You MUST cite EVERY piece of information you provide. This is non-negotiable.

### Citation Format
Use this exact format immediately after each fact:
- Text content: [p.X]
- Figure/image: [fig, p.X]
- Table: [table, p.X]

If multiple documents are provided, include the filename: [p.X, filename.pdf]

### Examples

CORRECT (every fact is cited):
"The study included 500 participants [p.3]. Results showed a 23% improvement [table, p.7] compared to the baseline shown in Figure 2 [fig, p.5]."

WRONG (missing citations - DO NOT DO THIS):
"The study included 500 participants. Results showed a 23% improvement compared to the baseline."

### Rules
1. NEVER state a fact without a citation
2. Place citation IMMEDIATELY after each fact, not at end of paragraph
3. If you cannot find a source for information, do not include it
4. When uncertain about the page, still provide your best estimate with the citation"""

    def __init__(
        self,
        system_prompt: str | None = None,
        include_text_layer: bool = True,
        image_detail: str = "high",
    ):
        """
        Initialize context builder.

        Args:
            system_prompt: Custom system prompt. If None, uses default.
            include_text_layer: Whether to include extracted text in context.
            image_detail: OpenAI image detail level ('low', 'high', 'auto').
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.include_text_layer = include_text_layer
        self.image_detail = image_detail
        self._documents: list[tuple[PDFDocument, str]] = []  # (document, display_name)
        self._name_counts: dict[str, int] = {}

    def add_document(self, document: PDFDocument) -> "ContextBuilder":
        """
        Add a PDF document to the context.

        Args:
            document: PDFDocument to include in the context.

        Returns:
            Self for method chaining.
        """
        base_name = document.file_id
        self._name_counts[base_name] = self._name_counts.get(base_name, 0) + 1
        count = self._name_counts[base_name]

        if count == 1:
            display_name = base_name
        else:
            display_name = f"{base_name} ({count})"

        self._documents.append((document, display_name))
        return self

    def _build_system_message(self) -> dict[str, str]:
        """Build the system message with document metadata."""
        metadata_parts = [self.system_prompt, "\n\n## Document Metadata\n"]

        for doc, display_name in self._documents:
            metadata_parts.append(f"\n### Document: {display_name}")
            metadata_parts.append(f"- Total pages: {doc.page_count}")
            metadata_parts.append(f"- Source file: {doc.pdf_path.name}\n")

            if self.include_text_layer:
                metadata_parts.append("#### Extracted Text Content:\n")
                for page in doc.pages:
                    if page.has_text:
                        metadata_parts.append(
                            f'Page {page.page_number} extracted text from {display_name}:\n"""\n{page.text}\n"""\n'
                        )
                    else:
                        metadata_parts.append(
                            f"Page {page.page_number} from {display_name}: [No extracted text - use image]\n"
                        )

        return {"role": "system", "content": "\n".join(metadata_parts)}

    def _build_image_content(
        self, page: PageContent, display_name: str
    ) -> list[dict[str, Any]]:
        """Build image content block for a page."""
        mime_type = "image/png"  # Default, could be made configurable
        return [
            {
                "type": "text",
                "text": f"Page {page.page_number} image from {display_name}:",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{page.image_base64}",
                    "detail": self.image_detail,
                },
            },
        ]

    def _build_user_message(self, question: str) -> dict[str, Any]:
        """Build the user message with images and question."""
        content: list[dict[str, Any]] = []

        # Add all page images with annotations
        for doc, display_name in self._documents:
            for page in doc.pages:
                content.extend(self._build_image_content(page, display_name))

        # Add the user's question
        content.append({"type": "text", "text": f"\n\nQuestion: {question}"})

        return {"role": "user", "content": content}

    def build_messages(self, question: str) -> list[dict[str, Any]]:
        """
        Build the complete message array for the LLM request.

        Args:
            question: The question to ask about the document(s).

        Returns:
            List of messages in OpenAI chat format.
        """
        if not self._documents:
            raise ValueError("No documents added. Use add_document() first.")

        return [
            self._build_system_message(),
            self._build_user_message(question),
        ]

    def build_request_payload(
        self,
        question: str,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Build the complete API request payload.

        Args:
            question: The question to ask about the document(s).
            model: The model to use (must support vision).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Complete request payload for OpenAI API.
        """
        return {
            "model": model,
            "messages": self.build_messages(question),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def build_message_history(self) -> list[Any]:
        """
        Build Pydantic AI message history from the documents.

        Args:
            question: The question to ask about the document(s).

        Returns:
            List of ModelMessage for Pydantic AI message_history parameter.
        """
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

        if not self._documents:
            raise ValueError("No documents added. Use add_document() first.")

        # Reuse existing system message builder
        system_message = self._build_system_message()

        # Build user content with images
        user_content: list[str | BinaryContent] = []
        for doc, display_name in self._documents:
            for page in doc.pages:
                user_content.append(f"Page {page.page_number} image from {display_name}:")
                user_content.append(
                    BinaryContent(
                        data=base64.b64decode(page.image_base64),
                        media_type="image/png",
                    )
                )

        user_content.append(f"\n\nQuestion: {question}")

        return [
            ModelRequest(
                parts=[
                    SystemPromptPart(content=system_message["content"]),
                    UserPromptPart(content=user_content),
                ]
            )
        ]
