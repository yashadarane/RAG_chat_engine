from __future__ import annotations

from core.schemas import ChatTurn, SearchMatch


SYSTEM_INSTRUCTIONS = (
    "You are a document-grounded RAG assistant. Answer only from the retrieved context. "
    "Do not use outside knowledge. If the answer is not in the context, say: "
    "'I could not find that in the uploaded documents.' Cite document and page references."
)


def build_grounded_prompt(query: str, history: list[ChatTurn], matches: list[SearchMatch]) -> str:
    context = "\n\n".join(
        f"[{idx}] {match.chunk.source_label}, chunk {match.chunk.chunk_id}, "
        f"similarity {match.similarity_score:.3f}\n{match.chunk.text}"
        for idx, match in enumerate(matches, start=1)
    )
    turns = "\n".join(f"{turn.role}: {turn.content}" for turn in history[-6:])
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Recent conversation:\n{turns or 'No previous turns.'}\n\n"
        f"Retrieved context:\n{context or 'No relevant context retrieved.'}\n\n"
        f"Question: {query}\n\n"
        "Grounded answer:"
    )
