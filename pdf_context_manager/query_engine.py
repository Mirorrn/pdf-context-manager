"""Query engine for sending PDF context to OpenAI."""

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .context_builder import ContextBuilder
from .document import PDFDocument


@dataclass
class QueryResult:
    """Result from a PDF query."""

    answer: str
    model: str
    usage: dict[str, int]
    finish_reason: str
    raw_response: Any

    @property
    def is_truncated(self) -> bool:
        """Check if the response was truncated due to token limits."""
        return self.finish_reason == "length"


class PDFQueryEngine:
    """Query PDF documents using OpenAI-compatible vision models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        include_text_layer: bool = True,
        image_detail: str = "high",
        dpi: int = 150,
        verbose: bool = False,
    ):
        """
        Initialize the query engine.

        Args:
            api_key: API key for the LLM provider.
            base_url: API base URL. Use "https://openrouter.ai/api/v1" for OpenRouter.
                      If None, uses OpenAI's default endpoint.
            model: Model to use (must support vision).
                   For OpenRouter, use format like "openai/gpt-4o" or "anthropic/claude-3.5-sonnet".
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            system_prompt: Custom system prompt for the context builder.
            include_text_layer: Whether to include extracted PDF text.
            image_detail: OpenAI image detail level ('low', 'high', 'auto').
            dpi: Resolution for PDF page image conversion.
            verbose: Print the request payload before sending (images truncated).
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.include_text_layer = include_text_layer
        self.image_detail = image_detail
        self.dpi = dpi
        self.verbose = verbose

    def _create_context_builder(self) -> ContextBuilder:
        """Create a configured context builder."""
        return ContextBuilder(
            system_prompt=self.system_prompt,
            include_text_layer=self.include_text_layer,
            image_detail=self.image_detail,
        )

    def _print_payload(self, payload: dict[str, Any]) -> None:
        """Print the request payload with truncated base64 images."""
        import copy
        import json

        truncated = copy.deepcopy(payload)

        for message in truncated.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # Truncate base64 data
                            prefix = url[:50]
                            item["image_url"]["url"] = f"{prefix}...[BASE64 TRUNCATED]"

        print("\n" + "=" * 60)
        print("REQUEST PAYLOAD (verbose mode)")
        print("=" * 60)
        print(json.dumps(truncated, indent=2))
        print("=" * 60 + "\n")

    def query(
        self,
        pdf_path: str,
        question: str,
    ) -> QueryResult:
        """
        Query a single PDF document.

        Args:
            pdf_path: Path to the PDF file.
            question: Question to ask about the document.

        Returns:
            QueryResult with the answer and metadata.
        """
        document = PDFDocument(pdf_path, dpi=self.dpi)
        builder = self._create_context_builder()
        builder.add_document(document)

        payload = builder.build_request_payload(
            question=question,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if self.verbose:
            self._print_payload(payload)

        response = self.client.chat.completions.create(**payload)

        return QueryResult(
            answer=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason or "unknown",
            raw_response=response,
        )

    def query_multiple(
        self,
        pdf_paths: list[str],
        question: str,
    ) -> QueryResult:
        """
        Query multiple PDF documents together.

        Args:
            pdf_paths: List of paths to PDF files.
            question: Question to ask about the documents.

        Returns:
            QueryResult with the answer and metadata.
        """
        builder = self._create_context_builder()

        for path in pdf_paths:
            document = PDFDocument(path, dpi=self.dpi)
            builder.add_document(document)

        payload = builder.build_request_payload(
            question=question,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if self.verbose:
            self._print_payload(payload)

        response = self.client.chat.completions.create(**payload)

        return QueryResult(
            answer=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason or "unknown",
            raw_response=response,
        )

    def query_document(
        self,
        document: PDFDocument,
        question: str,
    ) -> QueryResult:
        """
        Query a pre-loaded PDFDocument.

        Args:
            document: Pre-loaded PDFDocument instance.
            question: Question to ask about the document.

        Returns:
            QueryResult with the answer and metadata.
        """
        builder = self._create_context_builder()
        builder.add_document(document)

        payload = builder.build_request_payload(
            question=question,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if self.verbose:
            self._print_payload(payload)

        response = self.client.chat.completions.create(**payload)

        return QueryResult(
            answer=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason or "unknown",
            raw_response=response,
        )
