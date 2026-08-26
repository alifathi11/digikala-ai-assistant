import pandas as pd

from .product_search_judge_prompt import SYSTEM_PROMPT, build_prompt, build_repair_prompt


class ProductSearchRelevanceJudge:
    def __init__(self, generator, max_reviews_per_product=1, max_review_chars=450):
        self.generator = generator
        self.max_reviews_per_product = int(max_reviews_per_product)
        self.max_review_chars = int(max_review_chars)

    @staticmethod
    def _review_map(review_comments, product_ids, max_reviews_per_product):
        result = {int(product_id): [] for product_id in product_ids}
        if review_comments is None or len(review_comments) == 0:
            return result

        frame = review_comments.copy()
        frame["product_id"] = pd.to_numeric(frame["product_id"], errors="coerce")
        frame = frame[frame["product_id"].notna()].copy()
        frame["product_id"] = frame["product_id"].astype(int)
        frame = frame[frame["product_id"].isin(set(result))].copy()
        if "score" in frame.columns:
            frame = frame.sort_values("score", ascending=False)

        for row in frame.itertuples(index=False):
            product_id = int(getattr(row, "product_id"))
            if len(result[product_id]) >= int(max_reviews_per_product):
                continue
            body = getattr(row, "body", None) or getattr(row, "search_text", None) or ""
            result[product_id].append({"id": int(getattr(row, "id")), "text": str(body)})
        return result

    @staticmethod
    def _parse(payload, expected_ids, review_map):
        expected_ids = [int(value) for value in expected_ids]
        expected = set(expected_ids)
        rows = []
        seen = set()
        judgments = payload.get("judgments", []) if isinstance(payload, dict) else []

        for item in judgments:
            if not isinstance(item, dict):
                continue
            try:
                product_id = int(item.get("product_id"))
                grade = int(item.get("grade"))
            except (TypeError, ValueError):
                continue
            if product_id not in expected or product_id in seen:
                continue

            grade = max(0, min(3, grade))
            confidence = str(item.get("confidence", "low")).strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"

            allowed = {int(review["id"]) for review in review_map.get(product_id, [])}
            evidence_ids = []
            for value in item.get("evidence_ids", []):
                try:
                    review_id = int(value)
                except (TypeError, ValueError):
                    continue
                if review_id in allowed and review_id not in evidence_ids:
                    evidence_ids.append(review_id)

            rows.append({
                "product_id": product_id,
                "teacher_grade": grade,
                "teacher_confidence": confidence,
                "teacher_evidence_ids": evidence_ids,
                "teacher_reason": str(item.get("reason", "")).strip(),
            })
            seen.add(product_id)

        missing = [product_id for product_id in expected_ids if product_id not in seen]
        return pd.DataFrame(rows), missing

    def judge(self, query, candidates, review_comments):
        product_ids = candidates["id"].astype(int).tolist()
        review_map = self._review_map(review_comments, product_ids, self.max_reviews_per_product)
        prompt = build_prompt(
            query=query,
            candidates=candidates,
            review_map=review_map,
            max_reviews_per_product=self.max_reviews_per_product,
            max_review_chars=self.max_review_chars,
        )

        attempts = []
        first = self.generator.generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        attempts.append(first)
        rows, missing = self._parse(first["payload"], product_ids, review_map)

        if missing:
            repair = self.generator.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_repair_prompt(prompt, product_ids, first["payload"]),
            )
            attempts.append(repair)
            rows, missing = self._parse(repair["payload"], product_ids, review_map)

        if missing:
            raise ValueError(f"Relevance judge missed product IDs after retry: {missing}")

        telemetry = {
            "judge_retry_count": len(attempts) - 1,
            "model": attempts[-1]["model"],
            "latency_ms": float(sum(float(a["latency_ms"]) for a in attempts)),
            "prompt_tokens": int(sum(int(a["prompt_tokens"]) for a in attempts)),
            "completion_tokens": int(sum(int(a["completion_tokens"]) for a in attempts)),
            "total_tokens": int(sum(int(a["total_tokens"]) for a in attempts)),
        }
        costs = [a["estimated_cost_usd"] for a in attempts]
        telemetry["estimated_cost_usd"] = (
            sum(float(value) for value in costs) if all(value is not None for value in costs) else None
        )
        return rows, telemetry
