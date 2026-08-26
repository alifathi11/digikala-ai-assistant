def validate_result(
    result,
    valid_ids
):

    print("TYPE:", type(result))

    if not isinstance(result, dict):
        print("FAIL: not dict")
        return False


    if "query" not in result:
        print("FAIL: no query")
        return False


    if "relevant_ids" not in result:
        print("FAIL: no relevant_ids")
        return False


    ids = result["relevant_ids"]

    print("IDS:", ids)
    print("VALID:", valid_ids)


    if not isinstance(ids, list):
        print("FAIL: ids not list")
        return False


    if len(ids) == 0:
        print("FAIL: empty ids")
        return False


    subset = set(ids).issubset(
        set(valid_ids)
    )

    print("SUBSET:", subset)


    if not subset:
        print("FAIL: invalid ids")
        return False


    print("PASS")
    return True