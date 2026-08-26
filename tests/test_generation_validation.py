from src.rag.generation.validation import validate_grounded_response


def test_grounded_response_accepts_retrieved_citations():
    payload = {
        "answer": "پاسخ",
        "evidence_ids": [10],
        "confidence": "high",
        "insufficient_evidence": False,
    }

    valid, errors = validate_grounded_response(
        payload,
        retrieved_ids=[10, 11],
    )

    assert valid
    assert errors == []


def test_grounded_response_rejects_unknown_citation():
    payload = {
        "answer": "پاسخ",
        "evidence_ids": [999],
        "confidence": "high",
        "insufficient_evidence": False,
    }

    valid, errors = validate_grounded_response(
        payload,
        retrieved_ids=[10, 11],
    )

    assert not valid
    assert any(
        "evidence_not_retrieved" in error
        for error in errors
    )
