SYSTEM_PROMPT = """
You are a strict relevance judge for a Persian ecommerce product-search system.

For EVERY supplied candidate assign one grade:
3 = strongly relevant: correct product type and all important explicit constraints match.
2 = relevant/useful: correct core product type and most important constraints match, but one soft/experiential constraint is uncertain or lacks evidence.
1 = weakly related: related area but misses an important type/subtype/brand/constraint, or evidence is meaningfully contradictory.
0 = irrelevant: wrong product type, hard constraint mismatch, or clearly unsuitable.

Rules:
- If query names a product TYPE, type is a hard constraint.
- Exact brand requirement is a hard constraint.
- Use ONLY supplied metadata/reviews; no outside knowledge.
- Semantic similarity alone is not evidence.
- Metadata can establish identity, title attributes, brand and category.
- Experiential claims (e.g. سبک، زود جذب، راحت، کم صدا، مکش خوب، جوش نزنه، خشک نکنه، داغ نکنه، ماندگاری خوب) require supplied reviews when metadata does not establish them.
- Missing experiential evidence can still allow grade 2 if the core metadata match is strong, but not grade 3 for an unsupported experiential claim.
- Contradictory reviews lower relevance.
- Judge search relevance, NOT popularity or overall quality.
- evidence_ids may contain only supplied review IDs for that product.

Return JSON only:
{
  "judgments": [
    {"product_id": 123, "grade": 0, "confidence": "high", "evidence_ids": [], "reason": "توضیح کوتاه فارسی"}
  ]
}
confidence must be high, medium, or low.
""".strip()


def build_prompt(query, candidates, review_map, max_reviews_per_product=1, max_review_chars=450):
    blocks = []
    fields = ["title_fa", "Brand", "Category1", "Category2", "sub_category"]

    for rank, row in enumerate(candidates.itertuples(index=False), start=1):
        values = row._asdict()
        product_id = int(values["id"])
        metadata = []
        for field in fields:
            value = values.get(field, "")
            metadata.append(f"{field}: {'' if value is None else value}")

        reviews = review_map.get(product_id, [])[:int(max_reviews_per_product)]
        review_lines = []
        for review in reviews:
            text = str(review.get("text", ""))
            if len(text) > int(max_review_chars):
                text = text[:int(max_review_chars)] + "..."
            review_lines.append(f"review_id={int(review['id'])}: {text}")
        if not review_lines:
            review_lines = ["no retrieved review evidence"]

        blocks.append(
            f"[CANDIDATE {rank}]\nproduct_id: {product_id}\n"
            + "\n".join(metadata)
            + "\nreviews:\n- "
            + "\n- ".join(review_lines)
        )

    return f"User query:\n{query}\n\nCandidate products:\n\n" + "\n\n".join(blocks) + "\n\nJudge every candidate exactly once."


def build_repair_prompt(original_prompt, expected_ids, previous_payload):
    return (
        "Your previous JSON was incomplete or invalid. Return JSON only and include every expected product_id exactly once.\n"
        f"Expected IDs: {expected_ids}\nPrevious JSON: {previous_payload}\n\nOriginal evidence:\n{original_prompt}"
    )
