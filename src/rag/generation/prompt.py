SYSTEM_PROMPT = """
You are a grounded Persian ecommerce assistant.

Your job is to answer the user's question using ONLY the supplied
real user comments.

Rules:
- Do not use outside knowledge.
- Do not add general cautions, recommendations, guarantees, assumptions, or statements about variability unless they are directly supported by the supplied comments.
- Do not invent product specifications or medical/technical claims.
- Every factual claim about user experience must be supported by the comments.
- If comments disagree, explicitly mention that the experiences are mixed.
- If the supplied evidence is insufficient, say so clearly.
- evidence_ids must contain ONLY comment IDs supplied in the context.
- Use only the smallest set of comments that actually supports the answer.
- Write the answer in natural Persian.
- Do not expose internal reasoning.

Return ONLY one JSON object in this exact shape:
{
  "answer": "...",
  "evidence_ids": [123, 456],
  "confidence": "high",
  "insufficient_evidence": false
}

confidence must be one of:
high, medium, low
""".strip()


def _clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def build_qa_prompt(
    query,
    retrieved_documents,
    max_context_chars=10_000,
    max_chars_per_comment=1_500,
):
    parts = []
    used_chars = 0

    for rank, row in enumerate(
        retrieved_documents.itertuples(
            index=False
        ),
        start=1,
    ):
        row_dict = row._asdict()

        comment_id = int(
            row_dict["id"]
        )

        rate = row_dict.get(
            "rate"
        )

        text = (
            row_dict.get(
                "search_text"
            )
            or row_dict.get(
                "body"
            )
            or ""
        )

        text = _clean_text(
            text
        )

        if len(text) > max_chars_per_comment:
            text = (
                text[
                    :max_chars_per_comment
                ]
                + "..."
            )

        block = (
            f"[COMMENT {rank}]\n"
            f"id: {comment_id}\n"
            f"rate: {rate}\n"
            f"text: {text}\n"
        )

        if (
            used_chars
            + len(block)
            > max_context_chars
        ):
            break

        parts.append(
            block
        )

        used_chars += len(
            block
        )

    context = "\n".join(
        parts
    )

    return f"""
User question:
{query}

Retrieved user comments:
{context}

Answer the question using only these comments.
""".strip()



def build_citation_repair_prompt(
    original_user_prompt,
    previous_payload,
    retrieved_ids,
    validation_errors,
):
    allowed_ids = ", ".join(
        str(int(x))
        for x in retrieved_ids
    )

    errors = "; ".join(
        str(error)
        for error in validation_errors
    )

    return f"""
The previous JSON response was invalid.

Validation errors:
{errors}

Allowed evidence IDs:
[{allowed_ids}]

Previous response:
{previous_payload}

Regenerate the answer from the SAME supplied comments.

Hard requirements:
- evidence_ids may contain ONLY IDs from the allowed list above.
- Do not invent or alter comment IDs.
- If the comments do not support the answer, set
  insufficient_evidence=true and use an empty evidence_ids list.
- Return ONLY the required JSON object.

Original task:
{original_user_prompt}
""".strip()
