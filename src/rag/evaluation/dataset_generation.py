EVAL_DATASET_SYSTEM_PROMPT = """
You create retrieval benchmark queries for a Persian ecommerce RAG system.

Use only the supplied product title and candidate comments.
Return structured JSON exactly as requested.
Do not use outside knowledge.
""".strip()


def build_eval_prompt(
    product_title,
    comments,
):
    comments_text = "\n\n".join(
        (
            f"ID: {comment['id']}\n"
            f"TEXT: {comment['text']}"
        )
        for comment in comments
    )

    return f"""
Product:
{product_title}

Candidate comments:
{comments_text}

Generate exactly 3 realistic Persian user queries.

Rules:
- Return exactly 3 queries.
- Each query must represent only ONE user intent.
- The 3 queries must cover different intents.
- Queries should sound like real Persian ecommerce questions/searches.
- Do not ask statistical questions.
- Every query must be answerable directly from the candidate comments.
- For each query, include ALL candidate comment IDs that directly answer it.
- Prefer queries for which 1 to 3 comments are directly relevant.
- If more than 3 comments answer an intent, make the query more specific.
- Do not select a comment merely because it is about the same product.
- relevant_ids must contain only IDs shown above.
- Do not invent facts that do not appear in the candidate comments.

Return ONLY one JSON object:
{{
  "queries": [
    {{"query": "...", "relevant_ids": [123]}},
    {{"query": "...", "relevant_ids": [456, 789]}},
    {{"query": "...", "relevant_ids": [101112]}}
  ]
}}
""".strip()


def validate_generated_queries(
    result,
    valid_ids,
    expected_queries=3,
):
    if not isinstance(
        result,
        dict,
    ):
        return False

    queries = result.get(
        "queries"
    )

    if not isinstance(
        queries,
        list,
    ):
        return False

    if len(
        queries
    ) != expected_queries:
        return False

    valid_ids = set(
        int(value)
        for value in valid_ids
    )

    seen_queries = set()

    for item in queries:
        if not isinstance(
            item,
            dict,
        ):
            return False

        query = item.get(
            "query"
        )

        relevant_ids = item.get(
            "relevant_ids"
        )

        if not isinstance(
            query,
            str,
        ):
            return False

        query = query.strip()

        if (
            not query
            or query in seen_queries
        ):
            return False

        seen_queries.add(
            query
        )

        if not isinstance(
            relevant_ids,
            list,
        ):
            return False

        if not (
            1
            <= len(relevant_ids)
            <= 3
        ):
            return False

        try:
            normalized = [
                int(value)
                for value in relevant_ids
            ]
        except (
            TypeError,
            ValueError,
        ):
            return False

        if len(
            normalized
        ) != len(
            set(normalized)
        ):
            return False

        if not set(
            normalized
        ).issubset(
            valid_ids
        ):
            return False

    return True
