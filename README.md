# PDF Context Manager

Convert PDFs into multimodal LLM requests for vision-capable models. Query your PDFs using GPT-4o, Claude, or any OpenAI-compatible API.

## Features

- **Multimodal PDF Processing** - Extracts both text and renders page images for vision models
- **OpenAI & OpenRouter Support** - Works with GPT-4o, Claude, Mistral, and other vision models
- **Pydantic AI Integration** - Build conversational agents with PDF context
- **Multi-Document Queries** - Ask questions across multiple PDFs simultaneously
- **Automatic Citations** - Built-in prompting for page-level source citations

## Installation

```bash
pip install pdf-context-manager
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add pdf-context-manager
```

### System Dependencies

This package requires `poppler` for PDF rendering:

```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

## Quick Start

### Basic Query

```python
from pdf_context_manager import PDFQueryEngine

engine = PDFQueryEngine(
    api_key="your-openai-api-key",
    model="gpt-4o",
)

result = engine.query(
    pdf_path="document.pdf",
    question="What is the main topic of this document?",
)

print(result.answer)
print(f"Tokens used: {result.usage['total_tokens']}")
```

### Using OpenRouter

```python
engine = PDFQueryEngine(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-3.5-sonnet",  # or any vision model
)

result = engine.query("paper.pdf", "Summarize the key findings.")
```

### Multi-Document Comparison

```python
result = engine.query_multiple(
    pdf_paths=["report_q1.pdf", "report_q2.pdf"],
    question="Compare the quarterly results between these reports.",
)
```

## Advanced Usage

### Manual Context Building

For more control over the request, use `PDFDocument` and `ContextBuilder` directly:

```python
from pdf_context_manager import PDFDocument, ContextBuilder

# Load document with custom settings
doc = PDFDocument("document.pdf", dpi=200)

print(f"Pages: {doc.page_count}")
for page in doc.pages:
    status = "has text" if page.has_text else "image only"
    print(f"  Page {page.page_number}: {status}")

# Build context
builder = ContextBuilder(
    system_prompt="You are analyzing a technical document.",
    include_text_layer=True,
    image_detail="high",
)
builder.add_document(doc)

# Get the raw payload for custom API calls
payload = builder.build_request_payload(
    question="Summarize the key points.",
    model="gpt-4o",
    max_tokens=2048,
)
```

### Pydantic AI Integration

Build conversational agents that can discuss PDF content:

```python
from pydantic_ai import Agent
from pdf_context_manager import PDFDocument, ContextBuilder

# Load document
doc = PDFDocument("paper.pdf")
builder = ContextBuilder()
builder.add_document(doc)

# Build message history with document context
history = builder.build_message_history("What is this document about?")

# Create agent and chat
agent = Agent(model="openai:gpt-4o")
result = agent.run_sync("What is this document about?", message_history=history)
print(result.output)

# Follow-up questions maintain context
result2 = agent.run_sync(
    "What are the main conclusions?",
    message_history=result.all_messages()
)
```

### Interactive Chat Session

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pdf_context_manager import PDFDocument, ContextBuilder

# Setup
doc = PDFDocument("paper.pdf")
builder = ContextBuilder()
builder.add_document(doc)

model = OpenRouterModel(
    "anthropic/claude-3.5-sonnet",
    provider=OpenRouterProvider(api_key="your-api-key"),
)
agent = Agent(model=model)

# Interactive loop
history = builder.build_message_history("")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    result = agent.run_sync(user_input, message_history=history)
    print(f"Assistant: {result.output}")
    history = result.all_messages()
```

## Configuration Options

### PDFDocument

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pdf_path` | required | Path to the PDF file |
| `dpi` | 150 | Resolution for page image rendering |
| `image_format` | "PNG" | Output format (PNG, JPEG) |

### ContextBuilder

| Parameter | Default | Description |
|-----------|---------|-------------|
| `system_prompt` | (citation prompt) | Custom system instructions |
| `include_text_layer` | True | Include extracted PDF text |
| `image_detail` | "high" | OpenAI image detail (low/high/auto) |

### PDFQueryEngine

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | None | API key for LLM provider |
| `base_url` | None | Custom API endpoint (e.g., OpenRouter) |
| `model` | "gpt-4o" | Model name |
| `max_tokens` | 4096 | Max response tokens |
| `temperature` | 0.0 | Sampling temperature |
| `verbose` | False | Print request payloads for debugging |

## QueryResult

The `query()` methods return a `QueryResult` object:

```python
result = engine.query("doc.pdf", "What is this?")

result.answer        # The model's response text
result.model         # Model used
result.usage         # Token counts: prompt_tokens, completion_tokens, total_tokens
result.finish_reason # "stop" or "length"
result.is_truncated  # True if response was cut off
result.raw_response  # Full API response object
```

## Environment Variables

```bash
OPENAI_API_KEY=sk-...        # For OpenAI
OPENROUTER_API_KEY=sk-or-... # For OpenRouter
```

## License

MIT
