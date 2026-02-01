"""Example usage of pdf-context-manager."""

import os

from dotenv import load_dotenv

from pydantic_ai import Agent

from pdf_context_manager import PDFDocument, ContextBuilder, PDFQueryEngine

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def example_basic_query():
    """Basic example: query a PDF with a question using OpenAI."""
    engine = PDFQueryEngine(
        api_key=API_KEY,
        model="gpt-4o",
        include_text_layer=True,
        image_detail="high",
    )

    result = engine.query(
        pdf_path="document.pdf",
        question="What is the main topic of this document?",
    )

    print(f"Answer: {result.answer}")
    print(f"Tokens used: {result.usage['total_tokens']}")


def example_openrouter_query():
    """Example: query a PDF using OpenRouter."""
    engine = PDFQueryEngine(
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/ministral-14b-2512",  # OpenRouter model format
        include_text_layer=False,
        image_detail="auto",
        verbose=True,
    )

    result = engine.query(
        pdf_path="data/paper2.pdf",
        question="Was sagen die einzelnen Tabellen aus? Fasse die Ergebnisse zusammen und stelle sie in Bezug zueinander.",
    )

    print(f"Answer: {result.answer}")
    print(f"Tokens used: {result.usage['total_tokens']}")
    print(f"Finish reason: {result.finish_reason}")
    if result.is_truncated:
        print("WARNING: Response was truncated due to token limits")


def example_manual_context_building():
    """Advanced example: manually build context for inspection."""
    # Load document
    doc = PDFDocument("document.pdf", dpi=150)

    print(f"Document ID: {doc.file_id}")
    print(f"Pages: {doc.page_count}")

    # Check text extraction
    for page in doc.pages:
        status = "has text" if page.has_text else "image only"
        print(f"  Page {page.page_number}: {status}")

    # Build context manually
    builder = ContextBuilder(
        system_prompt="You are analyzing a technical document.",
        include_text_layer=True,
        image_detail="high",
    )
    builder.add_document(doc)

    # Get the payload (useful for debugging or custom API calls)
    payload = builder.build_request_payload(
        question="Summarize the key points.",
        model="gpt-4o",
        max_tokens=2048,
    )

    print(f"\nPayload model: {payload['model']}")
    print(f"Number of messages: {len(payload['messages'])}")


def example_multiple_documents():
    """Example: query multiple PDFs together."""
    engine = PDFQueryEngine(api_key=API_KEY, model="gpt-4o")

    result = engine.query_multiple(
        pdf_paths=["report_q1.pdf", "report_q2.pdf"],
        question="Compare the quarterly results between these two reports.",
    )

    print(f"Answer: {result.answer}")


def example_pydantic_ai_agent():
    """Example: use Pydantic AI agent with PDF message history."""
    # Load document and build context
    doc = PDFDocument("data/paper2.pdf")
    builder = ContextBuilder()
    builder.add_document(doc)

    # Build message history with initial question
    history = builder.build_message_history("What is this document about?")

    # Create agent and run with history
    agent = Agent(model="openai:gpt-4o")

    print("User: What is this document about?")
    result = agent.run_sync("What is this document about?", message_history=history)
    print(f"Assistant: {result.output}\n")

    # Follow-up question (pass updated history)
    print("User: Can you summarize the key findings?")
    result2 = agent.run_sync("Can you summarize the key findings?", message_history=result.all_messages())
    print(f"Assistant: {result2.output}\n")

    # Another follow-up
    print("User: What does the table on page 2 show?")
    result3 = agent.run_sync("What does the table on page 2 show?", message_history=result2.all_messages())
    print(f"Assistant: {result3.output}\n")


def example_pydantic_ai_openrouter():
    """Example: interactive chat with PDF using Pydantic AI and OpenRouter."""
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    # Load document and build context
    doc = PDFDocument("data/paper2.pdf")
    builder = ContextBuilder()
    builder.add_document(doc)

    # Create OpenRouter model
    model = OpenRouterModel(
        "mistralai/ministral-14b-2512",
        provider=OpenRouterProvider(api_key=API_KEY),
    )
    agent = Agent(model=model)

    print(f"Loaded: {doc.file_id} ({doc.page_count} pages)")
    print("Type 'quit' to exit.\n")

    # Initialize with document context (no question yet)
    history = builder.build_message_history("")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = agent.run_sync(user_input, message_history=history)
        print(f"Assistant: {result.output}\n")
        history = result.all_messages()


def main():
    """Run examples (requires actual PDF files)."""
    print("PDF Context Manager Examples")
    print("=" * 40)
    print()
    print("To use this package:")
    print()
    print("1. Basic query:")
    print("   from pdf_context_manager import PDFQueryEngine")
    print("   engine = PDFQueryEngine()")
    print('   result = engine.query("doc.pdf", "What is this about?")')
    print("   print(result.answer)")
    print()
    print("2. Manual context building:")
    print("   from pdf_context_manager import PDFDocument, ContextBuilder")
    print('   doc = PDFDocument("doc.pdf")')
    print("   builder = ContextBuilder()")
    print("   builder.add_document(doc)")
    print('   payload = builder.build_request_payload("Your question")')
    print()
    print("Set OPENAI_API_KEY environment variable before running queries.")
    print()
    print("3. Using OpenRouter:")
    print("   engine = PDFQueryEngine(")
    print('       base_url="https://openrouter.ai/api/v1",')
    print('       model="openai/gpt-4o",  # or anthropic/claude-3.5-sonnet')
    print("   )")
    example_pydantic_ai_openrouter()


if __name__ == "__main__":
    main()
