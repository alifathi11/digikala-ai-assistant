import math


COMPARISON_SYSTEM_PROMPT = """
You are a grounded Persian ecommerce product-comparison assistant.

You receive ONLY:
- the user's comparison request,
- product metadata for the selected products,
- retrieved real user reviews, kept separate for each product.

Your job is to compare ONLY those selected products.

Grounding rules:
- Do not use outside knowledge.
- Do not invent specifications, features, user experiences, medical claims,
  guarantees, or hidden product attributes.
- Metadata may support claims about title, brand, category, price, rating,
  rating count and other explicitly supplied metadata fields.
- Review-derived claims about real-world experience must cite review IDs.
- A review ID may be cited ONLY for the product that owns that review.
- If reviews disagree, describe the evidence as mixed.
- If a criterion cannot be established from supplied metadata/reviews, mark it
  unknown rather than guessing.
- Product popularity or rating alone does not prove experiential superiority.
- winner_product_id may be null when there is no defensible winner for a
  criterion.
- overall_winner_product_id may be null when evidence is insufficient or the
  answer depends on user priorities.
- Keep the explanation concise and natural in Persian.
- Do not expose internal reasoning.

Return ONLY one JSON object in this exact shape:
{
  "summary": "...",
  "criteria": [
    {
      "name": "...",
      "assessments": [
        {
          "product_id": 123,
          "stance": "positive",
          "text": "...",
          "evidence_ids": [1001]
        }
      ],
      "winner_product_id": 123,
      "winner_reason": "..."
    }
  ],
  "overall_winner_product_id": 123,
  "overall_recommendation": "...",
  "confidence": "high",
  "insufficient_evidence": false
}

Allowed stance values:
positive, mixed, negative, unknown

Allowed confidence values:
high, medium, low
""".strip()


def _clean_text(value):
    if value is None:
        return ""

    try:
        if math.isnan(value):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(value).strip()


def _format_metadata_block(
    row_dict,
):
    fields = [
        (
            "title",
            row_dict.get(
                "title_fa"
            ),
        ),
        (
            "brand",
            row_dict.get(
                "Brand"
            ),
        ),
        (
            "category1",
            row_dict.get(
                "Category1"
            ),
        ),
        (
            "category2",
            row_dict.get(
                "Category2"
            ),
        ),
        (
            "sub_category",
            row_dict.get(
                "sub_category"
            ),
        ),
        (
            "price",
            row_dict.get(
                "Price"
            ),
        ),
        (
            "min_price_last_month",
            row_dict.get(
                "min_price_last_month"
            ),
        ),
        (
            "rating",
            row_dict.get(
                "Rate"
            ),
        ),
        (
            "rating_count",
            row_dict.get(
                "Rate_cnt"
            ),
        ),
    ]

    return "\n".join(
        f"{name}: {_clean_text(value)}"
        for name, value
        in fields
    )


def build_comparison_prompt(
    query,
    product_metadata,
    review_documents,
    max_context_chars=18_000,
    max_chars_per_review=900,
):
    product_ids = (
        product_metadata[
            "id"
        ]
        .astype(int)
        .tolist()
    )

    reviews_by_product = {
        product_id: []
        for product_id
        in product_ids
    }

    if (
        review_documents is not None
        and len(
            review_documents
        )
        > 0
    ):
        for row in (
            review_documents
            .itertuples(
                index=False
            )
        ):
            product_id = int(
                getattr(
                    row,
                    "product_id",
                )
            )

            if (
                product_id
                not in reviews_by_product
            ):
                continue

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

            reviews_by_product[
                product_id
            ].append(
                {
                    "id": int(
                        getattr(
                            row,
                            "id",
                        )
                    ),
                    "rate": getattr(
                        row,
                        "rate",
                        None,
                    ),
                    "text": text,
                }
            )

    blocks = []
    used_chars = 0

    for index, row in enumerate(
        product_metadata.itertuples(
            index=False
        ),
        start=1,
    ):
        row_dict = row._asdict()

        product_id = int(
            row_dict[
                "id"
            ]
        )

        header = (
            f"[PRODUCT {index}]\n"
            f"product_id: {product_id}\n"
            + _format_metadata_block(
                row_dict
            )
            + "\nreviews:\n"
        )

        review_lines = []

        for review in (
            reviews_by_product[
                product_id
            ]
        ):
            line = (
                "- review_id="
                f"{review['id']}"
                f" | rate={review['rate']}"
                f" | text={review['text']}\n"
            )

            if (
                used_chars
                + len(header)
                + sum(
                    len(value)
                    for value
                    in review_lines
                )
                + len(line)
                > int(
                    max_context_chars
                )
            ):
                break

            review_lines.append(
                line
            )

        if not review_lines:
            review_lines = [
                "- no retrieved review evidence\n"
            ]

        block = (
            header
            + "".join(
                review_lines
            )
        )

        blocks.append(
            block
        )

        used_chars += len(
            block
        )

    return f"""
User comparison request:
{query}

Selected products and evidence:
{chr(10).join(blocks)}

Compare only these selected products.
For every criterion, include one assessment for every selected product.
Use review evidence IDs only for the product that owns each review.
""".strip()


def build_comparison_repair_prompt(
    original_user_prompt,
    previous_payload,
    product_ids,
    allowed_evidence_by_product,
    validation_errors,
):
    allowed = "\n".join(
        (
            f"product_id={int(product_id)}: "
            f"{sorted(int(value) for value in evidence_ids)}"
        )
        for product_id, evidence_ids
        in allowed_evidence_by_product.items()
    )

    errors = "; ".join(
        str(error)
        for error
        in validation_errors
    )

    return f"""
The previous comparison JSON was invalid.

Validation errors:
{errors}

Selected product IDs:
{[int(value) for value in product_ids]}

Allowed review evidence IDs by product:
{allowed}

Previous response:
{previous_payload}

Regenerate the comparison from the SAME supplied metadata and reviews.

Hard requirements:
- Use only selected product IDs.
- Every criterion must contain exactly one assessment for every selected product.
- An evidence ID may appear only under the product that owns it.
- Invalid/unknown evidence IDs must not be used.
- winner_product_id and overall_winner_product_id must be one of the selected
  product IDs or null.
- If evidence is insufficient, prefer null winners and unknown assessments.
- Return ONLY the required JSON object.

Original task:
{original_user_prompt}
""".strip()
