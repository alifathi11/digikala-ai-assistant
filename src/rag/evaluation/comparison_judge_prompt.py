import json
import math


COMPARISON_JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a grounded Persian ecommerce product-comparison
system.

Evaluate ONLY from the supplied product metadata and retrieved user reviews.
Do not use outside knowledge.

Score each dimension from 1 to 5.

1) correctness
5 = Product-specific claims accurately reflect the supplied metadata/reviews;
    winners and comparisons do not contradict the evidence.
3 = Mostly correct with a minor imprecision or weak inference.
1 = Major product claims or winners contradict the supplied evidence.

2) groundedness
5 = Every substantive experiential claim is attributable to supplied reviews,
    and metadata claims are supported by supplied metadata. Unknowns are
    admitted when evidence is missing.
3 = Mostly grounded but contains a small unsupported inference.
1 = Major unsupported claims, invented features, or outside knowledge.

3) criterion_coverage
5 = Directly addresses all important criteria in the user's comparison request
    for every selected product.
3 = Covers the main criteria but misses one meaningful aspect.
1 = Misses most requested criteria or fails to compare products symmetrically.

4) conflict_handling
5 = Meaningful disagreement in reviews is represented as mixed/uncertain and
    not flattened into a false consensus. If no conflict is present, a normal
    evidence-consistent treatment also receives 5.
3 = Minor oversimplification of mixed evidence.
1 = Ignores or reverses important conflicting evidence.

5) recommendation_calibration
5 = Winner/recommendation strength matches evidence strength. Uses null/no
    winner when evidence is insufficient or trade-offs make a single winner
    indefensible.
3 = Direction is plausible but wording is somewhat overconfident or hesitant.
1 = Strong unjustified winner, wrong winner, or failure to admit insufficient
    evidence.

6) relevance
5 = Concise and directly useful for the user's requested decision.
3 = Useful but contains noticeable tangents or repetition.
1 = Mostly off-topic or fails to support the decision.

7) instruction_following
5 = Follows the structured grounded-comparison behavior and keeps evidence
    separated by product.
3 = Minor presentation/instruction issue.
1 = Major comparison-format or product-separation failure.

8) safety
5 = No unsafe advice, medical guarantees, or misleading certainty.
3 = Mild overstatement without likely serious harm.
1 = Clearly unsafe or seriously misleading advice.

Assign zero or more failure_tags from:
- unsupported_claim
- contradicts_evidence
- missed_criterion
- ignores_conflict
- overconfident_winner
- wrong_winner
- insufficient_evidence_mishandled
- cross_product_evidence
- off_topic
- format_issue
- unsafe_claim

Important:
- Citation ownership/ID validity is checked deterministically elsewhere.
- Evaluate response quality conditional on the retrieved evidence shown here.
- Do not penalize a system for not citing every retrieved review.
- Do not reveal chain-of-thought. Reasons must be short evaluation summaries.

Return ONLY one JSON object:
{
  "correctness": {"score": 1, "reason": "..."},
  "groundedness": {"score": 1, "reason": "..."},
  "criterion_coverage": {"score": 1, "reason": "..."},
  "conflict_handling": {"score": 1, "reason": "..."},
  "recommendation_calibration": {"score": 1, "reason": "..."},
  "relevance": {"score": 1, "reason": "..."},
  "instruction_following": {"score": 1, "reason": "..."},
  "safety": {"score": 1, "reason": "..."},
  "failure_tags": [],
  "summary_reason": "..."
}
""".strip()


DIMENSIONS = (
    "correctness",
    "groundedness",
    "criterion_coverage",
    "conflict_handling",
    "recommendation_calibration",
    "relevance",
    "instruction_following",
    "safety",
)


def _clean_text(
    value,
):
    if value is None:
        return ""

    try:
        if math.isnan(
            value
        ):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(
        value
    ).strip()


def _metadata_blocks(
    product_metadata,
):
    blocks = []

    for row in product_metadata.itertuples(
        index=False
    ):
        values = row._asdict()

        fields = [
            "title_fa",
            "Brand",
            "Category1",
            "Category2",
            "sub_category",
            "Price",
            "min_price_last_month",
            "Rate",
            "Rate_cnt",
        ]

        lines = [
            f"product_id: {int(values['id'])}"
        ]

        for field in fields:
            if field not in values:
                continue

            text = _clean_text(
                values.get(
                    field
                )
            )

            if text:
                lines.append(
                    f"{field}: {text}"
                )

        blocks.append(
            "\n".join(
                lines
            )
        )

    return "\n\n".join(
        blocks
    )


def _review_blocks(
    review_documents,
    max_context_chars=18_000,
    max_chars_per_review=900,
):
    if (
        review_documents is None
        or len(
            review_documents
        ) == 0
    ):
        return "No retrieved reviews."

    blocks = []
    used = 0

    for row in review_documents.itertuples(
        index=False
    ):
        product_id = int(
            getattr(
                row,
                "product_id",
            )
        )

        review_id = int(
            getattr(
                row,
                "id",
            )
        )

        text = (
            getattr(
                row,
                "body",
                None,
            )
            or getattr(
                row,
                "search_text",
                None,
            )
            or ""
        )

        text = _clean_text(
            text
        )

        if len(
            text
        ) > int(
            max_chars_per_review
        ):
            text = (
                text[
                    :int(
                        max_chars_per_review
                    )
                ]
                + "..."
            )

        block = (
            f"product_id={product_id} "
            f"review_id={review_id} "
            f"rate={getattr(row, 'rate', None)}\n"
            f"text: {text}"
        )

        if (
            used
            + len(
                block
            )
            > int(
                max_context_chars
            )
        ):
            break

        blocks.append(
            block
        )
        used += len(
            block
        )

    return "\n\n".join(
        blocks
    )


def build_comparison_judge_prompt(
    query,
    product_metadata,
    retrieved_reviews,
    generated_result,
    case_notes="",
    max_context_chars=18_000,
    max_chars_per_review=900,
):
    response = {
        "summary": generated_result.get(
            "summary",
            "",
        ),
        "criteria": generated_result.get(
            "criteria",
            [],
        ),
        "overall_winner_product_id": (
            generated_result.get(
                "overall_winner_product_id"
            )
        ),
        "overall_recommendation": (
            generated_result.get(
                "overall_recommendation",
                "",
            )
        ),
        "confidence": generated_result.get(
            "confidence"
        ),
        "insufficient_evidence": (
            generated_result.get(
                "insufficient_evidence"
            )
        ),
    }

    return f"""
User comparison request:
{query}

Selected product metadata:
{_metadata_blocks(product_metadata)}

Retrieved review evidence:
{_review_blocks(
    retrieved_reviews,
    max_context_chars=max_context_chars,
    max_chars_per_review=max_chars_per_review,
)}

Generated comparison JSON:
{json.dumps(
    response,
    ensure_ascii=False,
    indent=2,
)}

Evaluate the generated comparison using the rubric.
""".strip()
