
import json


ANALYTICS_JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a grounded Persian ecommerce Manager Analytics
assistant.

Evaluate ONLY from the supplied deterministic facts, scope, data-quality
policy, and generated answer. Do not use outside knowledge.

The system architecture computes all numbers in Python. Numeric faithfulness
and exact aggregation are checked deterministically elsewhere. Your task is to
evaluate semantic quality: whether the explanation correctly interprets those
facts and respects limitations.

Score every dimension from 1 to 5.

1) correctness
5 = Correctly interprets the supplied facts and comparisons; no inversion,
    wrong ranking, or misleading semantic claim.
3 = Mostly correct with one minor imprecision.
1 = Major interpretation error or contradiction with supplied facts.

2) groundedness
5 = Every substantive claim is supported by supplied facts/data-quality notes.
    It does not introduce outside market knowledge.
3 = Mostly grounded with a small unsupported inference.
1 = Major unsupported claims or invented context.

3) caveat_compliance
5 = Respects all relevant limitations. In particular:
    - brand catalog share is NOT called sales/revenue/market share;
    - brand analysis is qualified when coverage is limited;
    - historical price is not analyzed when disabled;
    - review_count is not presented as true market-wide popularity when
      review-volume ranking is disabled;
    - product ratings are qualified when rating coverage is limited;
    - zero/insufficient review coverage is admitted.
3 = Main limitation is respected but one caveat is weakly communicated.
1 = Violates an important data-policy limitation.

4) completeness
5 = Covers all important parts of the user's specific managerial question.
3 = Answers the main request but omits one useful aspect.
1 = Misses most of the requested analysis.

5) relevance
5 = Concise and directly useful for the manager's question.
3 = Useful but somewhat generic/repetitive.
1 = Mostly off-topic.

6) managerial_usefulness
5 = Converts the facts into a clear, decision-useful managerial interpretation
    without overclaiming.
3 = Correct but mostly restates metrics.
1 = Confusing or not useful for a manager.

7) instruction_following
5 = Follows the grounded Manager Analytics behavior, uses the provided scope,
    and preserves terminology such as "تعداد امتیاز ثبت‌شده" when relevant.
3 = Minor presentation/terminology issue.
1 = Major behavior or format failure.

Allowed failure_tags:
- unsupported_claim
- semantic_numeric_misinterpretation
- missed_requested_metric
- historical_price_used
- review_volume_overclaim
- brand_market_share_overclaim
- brand_coverage_ignored
- rating_coverage_ignored
- zero_review_mishandled
- comparison_direction_error
- off_topic
- format_issue

Important:
- Do NOT penalize literal numeric formatting; numeric rendering is checked
  separately.
- Do penalize semantic misuse of a correct number, e.g. calling rating_count
  "تعداد نظر" or product_share "سهم بازار".
- If a user asks for an unavailable metric, a calibrated refusal/explanation
  is the correct answer.
- Do not reveal chain-of-thought. Reasons must be brief evaluation summaries.

Return ONLY one JSON object:
{
  "correctness": {"score": 1, "reason": "..."},
  "groundedness": {"score": 1, "reason": "..."},
  "caveat_compliance": {"score": 1, "reason": "..."},
  "completeness": {"score": 1, "reason": "..."},
  "relevance": {"score": 1, "reason": "..."},
  "managerial_usefulness": {"score": 1, "reason": "..."},
  "instruction_following": {"score": 1, "reason": "..."},
  "failure_tags": [],
  "summary_reason": "..."
}
""".strip()


DIMENSIONS = (
    "correctness",
    "groundedness",
    "caveat_compliance",
    "completeness",
    "relevance",
    "managerial_usefulness",
    "instruction_following",
)


def build_analytics_judge_prompt(
    case,
    generated_result,
):
    facts = []

    for key, fact in (
        generated_result.get(
            "facts",
            {}
        ).items()
    ):
        facts.append(
            {
                "key": key,
                "label": fact.get(
                    "label"
                ),
                "value": fact.get(
                    "value"
                ),
                "display_value": (
                    fact.get(
                        "display_value"
                    )
                ),
                "unit": fact.get(
                    "unit"
                ),
                "caveat": fact.get(
                    "caveat"
                ),
            }
        )

    payload = {
        "case_id": case[
            "case_id"
        ],
        "case_type": case[
            "case_type"
        ],
        "question": case[
            "question"
        ],
        "scope": (
            generated_result.get(
                "context",
                {}
            ).get(
                "scope",
                {},
            )
        ),
        "policy_expectations": (
            case.get(
                "policy_expectations"
            )
            or []
        ),
        "case_notes": case.get(
            "notes",
            "",
        ),
        "deterministic_facts": facts,
        "data_quality": (
            generated_result.get(
                "context",
                {}
            ).get(
                "data_quality",
                {},
            )
        ),
        "supporting_tables": {
            "top_brands": (
                generated_result.get(
                    "context",
                    {}
                ).get(
                    "top_brands",
                    [],
                )
            ),
            "top_products_by_rating": (
                generated_result.get(
                    "context",
                    {}
                ).get(
                    "top_products_by_rating",
                    [],
                )
            ),
            "top_products_by_rating_count": (
                generated_result.get(
                    "context",
                    {}
                ).get(
                    "top_products_by_rating_count",
                    [],
                )
            ),
            "category_comparison": (
                generated_result.get(
                    "context",
                    {}
                ).get(
                    "category_comparison",
                    [],
                )
            ),
        },
        "generated": {
            "answer": (
                generated_result.get(
                    "answer"
                )
            ),
            "insights": [
                {
                    "title": value.get(
                        "title"
                    ),
                    "text": value.get(
                        "text"
                    ),
                }
                for value
                in (
                    generated_result.get(
                        "insights"
                    )
                    or []
                )
            ],
            "caveats": (
                generated_result.get(
                    "caveats"
                )
                or []
            ),
            "confidence": (
                generated_result.get(
                    "confidence"
                )
            ),
            "numeric_faithfulness_valid": (
                generated_result.get(
                    "numeric_faithfulness_valid"
                )
            ),
        },
    }

    return (
        "Evaluate this Manager Analytics response.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
