RERANKER_SYSTEM_PROMPT = """
You are a grounded product-search reranker for a Persian ecommerce system.

You receive:
- one user search/discovery query,
- a small list of candidate products,
- product metadata,
- zero or more retrieved real user reviews for each product.

Evaluate how well EACH candidate matches the user's request.

Use ONLY the supplied metadata and reviews.
Do not use outside knowledge.

Important rules:
- Product identity, category and explicit brand requirements may be supported
  by product metadata.
- Product TYPE is a hard relevance constraint when the query names one.
  Examples: if the user asks for "ضد آفتاب", a face wash, hair brush, shampoo,
  comb, serum, etc. is NOT a plausible match even if some words such as
  "پوست", "چرب", "ضد", or other generic attributes overlap.
- A clear product-type/category mismatch must receive match_score 0 or 1.
  Do not give score 2 merely because one generic attribute overlaps.
- Experiential requirements such as "سبک", "زود جذب", "باعث جوش نشه",
  "بوی تند نداشته باشه", "بادوام باشه" should be supported by user reviews
  when reviews are available.
- Semantic similarity alone is NOT positive evidence.
  A review saying the opposite of the user's preference is contradictory
  evidence and must lower the score.
- If reviews conflict, mark the evidence as "mixed".
- If there is no review evidence for an experiential requirement, do not
  invent support.
- Do not invent product IDs or review IDs.

For each candidate return:
- product_id
- match_score: integer 0..5
- evidence_status: one of support, mixed, contradict, none
- evidence_ids: only review IDs supplied for that same product
- reason: one short Persian sentence explaining the score

Scoring:
5 = Excellent match; explicit metadata/brand/category fit and, where needed,
    supporting review evidence.
4 = Strong match with minor uncertainty or incomplete experiential evidence.
3 = Plausible match but important preference evidence is missing or mixed.
2 = Weak but still the SAME general product type; important requested
    properties are missing.
1 = Same broad shopping area but substantial mismatch.
0 = Clearly irrelevant, wrong product type, or strongly contradicted.

Return ONLY one JSON object:
{
  "rankings": [
    {
      "product_id": 123,
      "match_score": 4,
      "evidence_status": "support",
      "evidence_ids": [111, 222],
      "reason": "..."
    }
  ]
}
""".strip()


def _safe_text(
    value,
):
    if value is None:
        return ""

    return str(
        value
    ).strip()


def build_reranker_prompt(
    query,
    candidates,
    review_map,
    max_reviews_per_product=2,
    max_review_chars=700,
):
    blocks = []

    for rank, row in enumerate(
        candidates.itertuples(
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

        title = _safe_text(
            row_dict.get(
                "title_fa"
            )
        )

        brand = _safe_text(
            row_dict.get(
                "Brand"
            )
        )

        category = _safe_text(
            row_dict.get(
                "Category1"
            )
        )

        category2 = _safe_text(
            row_dict.get(
                "Category2"
            )
        )

        reviews = (
            review_map.get(
                product_id,
                []
            )
        )[
            :int(
                max_reviews_per_product
            )
        ]

        review_lines = []

        for review in reviews:
            text = _safe_text(
                review.get(
                    "text"
                )
            )

            if len(text) > int(
                max_review_chars
            ):
                text = (
                    text[
                        :int(
                            max_review_chars
                        )
                    ]
                    + "..."
                )

            review_lines.append(
                (
                    "- review_id="
                    f"{int(review['id'])}: "
                    f"{text}"
                )
            )

        if not review_lines:
            review_lines = [
                "- no retrieved review evidence"
            ]

        block = (
            f"[CANDIDATE {rank}]\n"
            f"product_id: {product_id}\n"
            f"title: {title}\n"
            f"brand: {brand}\n"
            f"category1: {category}\n"
            f"category2: {category2}\n"
            "reviews:\n"
            + "\n".join(
                review_lines
            )
        )

        blocks.append(
            block
        )

    return f"""
User query:
{query}

Candidate products:
{chr(10).join(chr(10) + block for block in blocks)}

Score every candidate and return the required JSON object.
""".strip()
