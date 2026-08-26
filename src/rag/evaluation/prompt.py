def build_eval_prompt(
    product_title,
    comments
):
    comments_text = ""

    for c in comments:
        comments_text += (
            f"ID: {c['id']}\n"
            f"TEXT: {c['text']}\n\n"
        )

    prompt = f"""
You are creating a retrieval evaluation dataset for an ecommerce RAG system.

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

Return ONLY one JSON object in exactly this shape:

{{
  "queries": [
    {{
      "query": "...",
      "relevant_ids": [123]
    }},
    {{
      "query": "...",
      "relevant_ids": [456, 789]
    }},
    {{
      "query": "...",
      "relevant_ids": [101112]
    }}
  ]
}}
"""
    return prompt
