
import json


MANAGER_SYSTEM_PROMPT = """
You are a grounded Persian ecommerce manager-analytics assistant.

You receive:
- a manager question,
- a deterministic analytics scope,
- a list of verified metric facts computed in Python,
- limited lists of brands/products/categories,
- explicit data-quality caveats.

Rules:
1. Use ONLY the supplied analytics context.
2. Never invent a number.
3. Never type a numeric digit directly in answer_template, insight text_template,
   caveats, or titles.
4. Whenever a numeric fact is needed, use an exact placeholder:
   {{metric:FACT_KEY}}
5. Use only FACT_KEY values present in the supplied facts list.
6. Do not claim market share from product_share. It means share of catalog rows
   in the selected scope, not sales/revenue/market share.
7. Brand analytics is incomplete when the supplied brand caveat says coverage
   is limited.
8. Do not call review_count a true market-wide "most reviewed" ranking. The
   available review corpus may be sampled/truncated. Prefer rating_count when
   discussing engagement and call it "تعداد امتیاز ثبت‌شده".
9. Product rating uses the native 0..100 scale. Use the supplied /5 metric only
   when that exact fact exists.
10. Historical-price analysis is unavailable unless explicitly supplied.
11. If the question cannot be answered from the supplied facts, say so.
12. Keep the answer concise, managerial, and in Persian.

Return JSON only:
{
  "answer_template": "...",
  "insights": [
    {
      "title": "...",
      "text_template": "...",
      "metric_refs": ["fact.key"]
    }
  ],
  "caveats": ["..."],
  "confidence": "high|medium|low"
}
""".strip()


def build_manager_prompt(
    question,
    context,
):
    facts = [
        {
            "key": key,
            "label": value[
                "label"
            ],
            "value": value[
                "value"
            ],
            "display_value": value[
                "display_value"
            ],
            "unit": value.get(
                "unit"
            ),
            "caveat": value.get(
                "caveat"
            ),
        }
        for key, value
        in context[
            "facts"
        ].items()
    ]

    payload = {
        "scope": context[
            "scope"
        ],
        "facts": facts,
        "top_brands": context.get(
            "top_brands",
            [],
        ),
        "top_products_by_rating": (
            context.get(
                "top_products_by_rating",
                [],
            )
        ),
        "top_products_by_rating_count": (
            context.get(
                "top_products_by_rating_count",
                [],
            )
        ),
        "category_comparison": (
            context.get(
                "category_comparison",
                [],
            )
        ),
        "data_quality": context[
            "data_quality"
        ],
    }

    return f"""
Manager question:
{question}

Verified analytics context:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Answer the manager question using only this context.
All numeric values in the response MUST be represented with metric placeholders.
""".strip()


def build_manager_repair_prompt(
    original_prompt,
    previous_payload,
    validation_errors,
    allowed_metric_keys,
):
    return f"""
The previous analytics response was invalid.

Validation errors:
{json.dumps(validation_errors, ensure_ascii=False)}

Allowed metric keys:
{json.dumps(sorted(allowed_metric_keys), ensure_ascii=False)}

Previous JSON:
{json.dumps(previous_payload, ensure_ascii=False, indent=2)}

Original task:
{original_prompt}

Repair the JSON.
Do not type any numeric digit directly in natural-language fields.
Use only exact placeholders in the form {{{{metric:FACT_KEY}}}}.
Return JSON only.
""".strip()
