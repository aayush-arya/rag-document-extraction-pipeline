"""Question answering over the same hybrid retriever (with page citations)."""

from extraction.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template(
    """
You are a document extraction assistant.

Use ONLY the provided context to answer the user's question.
Quote ids, numbers and dates exactly as written. If the context contains
conflicting values, report both and say they conflict.

If the answer cannot be found in the context, say:
"Information not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)


def _label(doc) -> str:
    m = doc.metadata
    pages = m.get("pages") or [m.get("page")]
    where = f"p.{pages[0]}" if len(set(pages)) == 1 else f"p.{pages[0]}-{pages[-1]}"
    return f"[{m.get('title') or m.get('section') or 'text'}, {where}]"


def create_rag_chain(retriever, llm=None, k: int = 8):
    def run(query):
        documents = retriever.invoke_many([query], k=k)
        context = "\n\n".join(f"{_label(d)}\n{d.page_content}" for d in documents)
        messages = PROMPT.format_messages(context=context, question=query)
        response = (llm or get_llm()).invoke(messages)
        return {"answer": response.content, "documents": documents}

    return run
