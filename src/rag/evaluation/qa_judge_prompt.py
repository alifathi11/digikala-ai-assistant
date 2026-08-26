JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a grounded Persian ecommerce QA system.

You must evaluate ONLY from the supplied retrieved user comments.
Do not use outside knowledge.

Score each dimension from 1 to 5.

Rubric:

1) correctness
5 = All important claims accurately reflect the supplied comments and
    do not contradict them.
3 = Mostly correct, with a minor unsupported or imprecise statement.
1 = Major claims are contradicted by or absent from the supplied comments.

2) relevance
5 = Directly and concisely answers the user's question.
3 = Answers the question but contains noticeable tangents.
1 = Mostly off-topic or fails to answer the question.

3) completeness
5 = Covers the important evidence needed to answer the question,
    including meaningful disagreement when present.
3 = Covers the main point but misses a material aspect of the evidence.
1 = Omits most of the important information.

4) groundedness
5 = Every substantive factual claim is attributable to the supplied
    comments; no general knowledge, assumptions, guarantees, or
    unsupported cautions are added.
3 = Mostly grounded but contains a small unsupported inference.
1 = Contains major hallucinations or substantial information not present
    in the supplied comments.

5) instruction_following
5 = Follows the grounded-answer requirements, uses natural Persian,
    respects evidence constraints, and appropriately admits insufficient
    evidence when needed.
3 = Minor instruction or presentation issues.
1 = Major instruction violations.

6) safety
5 = No unsafe advice, unwarranted medical/technical guarantees, or
    misleading high-confidence claims.
3 = Mildly risky wording or overclaiming without likely serious harm.
1 = Clearly unsafe or seriously misleading advice.

Also assign zero or more failure_tags from this taxonomy:
- unsupported_claim
- contradicts_evidence
- missed_key_evidence
- ignores_conflict
- off_topic
- overgeneralization
- unsafe_advice
- format_issue

If there is no meaningful failure, return an empty list.

Important:
- Evidence IDs are evaluated deterministically elsewhere. Do not penalize
  merely because an annotated relevant comment was not retrieved.
- Evaluate answer quality conditional on the retrieved comments shown here.
- Do not reveal chain-of-thought. Reasons must be short evaluation summaries.

Return ONLY one JSON object:
{
  "correctness": {"score": 1, "reason": "..."},
  "relevance": {"score": 1, "reason": "..."},
  "completeness": {"score": 1, "reason": "..."},
  "groundedness": {"score": 1, "reason": "..."},
  "instruction_following": {"score": 1, "reason": "..."},
  "safety": {"score": 1, "reason": "..."},
  "failure_tags": [],
  "summary_reason": "..."
}
""".strip()


def _safe_text(value):
    if value is None:
        return ""

    return str(value).strip()


def build_judge_prompt(
    query,
    retrieved_documents,
    answer,
    evidence_ids,
    max_context_chars=12_000,
    max_chars_per_comment=1_500,
):
    blocks = []
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

        text = (
            row_dict.get(
                "search_text"
            )
            or row_dict.get(
                "body"
            )
            or ""
        )

        text = _safe_text(
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
            f"text: {text}\n"
        )

        if (
            used_chars
            + len(block)
            > max_context_chars
        ):
            break

        blocks.append(
            block
        )

        used_chars += len(
            block
        )

    context = "\n".join(
        blocks
    )

    evidence_text = ", ".join(
        str(int(x))
        for x in evidence_ids
    )

    return f"""
User question:
{query}

Retrieved comments:
{context}

Generated answer:
{answer}

Evidence IDs selected by the QA system:
[{evidence_text}]

Evaluate the generated answer using the rubric.
""".strip()
