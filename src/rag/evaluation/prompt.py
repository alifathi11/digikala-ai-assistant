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
You are creating a retrieval evaluation dataset.

Product:
{product_title}


User comments:

{comments_text}


Generate 3 realistic Persian user queries.

Rules:

- Generate exactly 3 queries.
- Each query must represent only ONE user intent.
- Do not combine multiple questions.
- Do not ask statistical questions.
- Queries must look like real ecommerce search queries.
- Select only comments that directly answer the query.
- Select minimum 1 and maximum 3 relevant comment IDs.
- Do not select comments only because they are about the same product.


Return ONLY JSON:

[
 {{
   "query": "...",
   "relevant_ids": []
 }}
]

Important:
Before selecting relevant_ids, verify that each selected comment
actually contains information that answers the query.

Do not select comments only because they are about the same product.
"""

    return prompt