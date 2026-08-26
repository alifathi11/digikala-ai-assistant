def validate_result(
    result,
    valid_ids,
    expected_queries=3
):
    if not isinstance(result, dict):
        return False

    queries = result.get("queries")

    if not isinstance(queries, list):
        return False

    if len(queries) != expected_queries:
        return False

    valid_ids = set(valid_ids)
    seen_queries = set()

    for item in queries:
        if not isinstance(item, dict):
            return False

        query = item.get("query")
        ids = item.get("relevant_ids")

        if not isinstance(query, str):
            return False

        query = query.strip()

        if not query:
            return False

        if query in seen_queries:
            return False

        seen_queries.add(query)

        if not isinstance(ids, list):
            return False

        if not 1 <= len(ids) <= 3:
            return False

        if len(ids) != len(set(ids)):
            return False

        if not all(isinstance(x, int) for x in ids):
            return False

        if not set(ids).issubset(valid_ids):
            return False

    return True
