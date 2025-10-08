"""
Drop-in replacement for evaluator.py
- Honors Streamlit-selected fields when building prompts
- Adds locale support
- Implements question generation -> QA -> judgement pipeline
- Computes weighted coverage and returns a pandas DataFrame

Requires: prompt.py (v2) in the same project providing:
  QUESTION_GENERATION_PROMPT, QUESTION_QA_PROMPT, ANSWER_JUDGEMENT_PROMPT

App entrypoint expects:
  ProductFeedEvaluator(api_key, model, locale)
  .evaluate_products_batch(products_df, selected_fields, num_questions, batch_size, on_progress, debug)
"""
from __future__ import annotations

import json
import math
import re
import html
import time
from typing import Any, Dict, List, Tuple, Callable, Optional

import pandas as pd

from prompts import (
    QUESTION_GENERATION_PROMPT,
    QUESTION_QA_PROMPT,
    ANSWER_JUDGEMENT_PROMPT,
)

try:
    # New-style OpenAI SDK
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -------------------------
# Formatting helpers
# -------------------------
FIELD_LABELS = {
    "id": "ID",
    "title": "Title",
    "description": "Description",
    "link": "URL",
    "mobile_link": "Mobile URL",
    "image_link": "Image",
    "availability": "Availability",
    "availability_date": "Available From",
    "expiration_date": "Expires",
    "condition": "Condition",
    "price": "Price",
    "sale_price": "Sale Price",
    "sale_price_effective_date": "Sale Window",
    "brand": "Brand",
    "gtin": "GTIN",
    "mpn": "MPN",
    "identifier_exists": "Identifier Exists",
    "item_group_id": "Group",
    "google_product_category": "GPC",
    "product_type": "Type",
    "color": "Color",
    "size": "Size",
    "size_type": "Size Type",
    "size_system": "Size System",
    "material": "Material",
    "pattern": "Pattern",
    "age_group": "Age Group",
    "gender": "Gender",
    "adult": "Adult",
    "multipack": "Multipack",
    "is_bundle": "Bundle",
    "unit_pricing_measure": "Unit Measure",
    "unit_pricing_base_measure": "Unit Base",
    "shipping": "Shipping",
    "shipping_weight": "Ship Wt",
    "shipping_length": "Ship L",
    "shipping_width": "Ship W",
    "shipping_height": "Ship H",
    "shipping_label": "Ship Label",
    "tax": "Tax",
    "custom_label_0": "Custom 0",
    "custom_label_1": "Custom 1",
    "custom_label_2": "Custom 2",
    "custom_label_3": "Custom 3",
    "custom_label_4": "Custom 4",
    "included_destination": "Included Dest",
    "excluded_destination": "Excluded Dest",
    "shopping_ads_excluded_country": "Excluded Country",
    "loyalty_points": "Loyalty",
    "installment": "Installment",
    "subscription_cost": "Subscription",
}

PREFERRED_ORDER = [
    "title",
    "brand",
    "product_type",
    "google_product_category",
    "price",
    "availability",
    "color",
    "size",
    "material",
    "gtin",
    "mpn",
]

RE_JSON_PREFIX = re.compile(r"^[\s\S]*?(\[|\{)\s*")


def _clean_text(val: Any) -> str:
    if val is None:
        return ""
    s = html.unescape(str(val))
    s = re.sub(r"<[^>]+>", " ", s)  # strip tags
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_context(product: Dict[str, Any], selected_fields: List[str]) -> Tuple[str, str]:
    """Return (short_context_text, name_plus_context) built ONLY from selected_fields + link."""
    fields = list(dict.fromkeys(selected_fields + ["link"]))  # ensure link present, preserve order
    parts: List[str] = []
    for f in fields:
        v = _clean_text(product.get(f, ""))
        if not v:
            continue
        label = FIELD_LABELS.get(f, f.replace("_", " ").title())
        parts.append(f"{label}: {v}")

    name_plus_context = " | ".join(parts)

    preferred_labels = {FIELD_LABELS.get(k, k.replace('_',' ').title()) for k in PREFERRED_ORDER}
    preferred = [p for p in parts if p.split(":")[0] in preferred_labels]
    others = [p for p in parts if p not in preferred]
    short = " | ".join(preferred + others)
    if len(short) > 2000:
        short = short[:2000] + "…"
    return short, name_plus_context


# -------------------------
# Scoring
# -------------------------
REQUIRED_WEIGHT = 2.0
OPTIONAL_WEIGHT = 1.0


def _verdict_score(v: str) -> float:
    if v == "yes":
        return 1.0
    if v == "partial":
        return 0.5
    return 0.0


def compute_weighted_coverage(items: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    yes = partial = no = 0
    total_weight = 0.0
    achieved = 0.0
    for it in items:
        v = str(it.get("verdict", "no")).lower()
        req = bool(it.get("required", False))
        w = REQUIRED_WEIGHT if req else OPTIONAL_WEIGHT
        total_weight += w
        s = _verdict_score(v)
        achieved += s * w
        if v == "yes":
            yes += 1
        elif v == "partial":
            partial += 1
        else:
            no += 1
    pct = 0.0 if total_weight == 0 else (achieved / total_weight) * 100.0
    return pct, yes, partial, no


# -------------------------
# OpenAI client wrapper
# -------------------------
class _LLM:
    def __init__(self, api_key: str, model: str):
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not available. Install openai>=1.0.0.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def json_completion(self, prompt: str, max_retries: int = 2) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content or ""
                # try to isolate JSON if the model chatters
                text = text.strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Best-effort extraction of first JSON array/object
                    indices = [text.find("[{"), text.find("[\n{"), text.find("[")]
                    valid_indices = [i for i in indices if i != -1]
                    start = min(valid_indices) if valid_indices else -1
                    if start != -1:
                        snippet = text[start:]
                        # balance brackets roughly
                        # fallback: strip trailing code fences
                        snippet = snippet.strip().strip("`")
                        return json.loads(snippet)
                    raise
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        raise last_err  # type: ignore


# -------------------------
# Evaluator
# -------------------------
class ProductFeedEvaluator:
    def __init__(self, api_key: str, model: str, locale: str = "en-GB"):
        self.locale = locale
        self.llm = _LLM(api_key=api_key, model=model)

    # ---- internal steps
    def _gen_questions(self, product: Dict[str, Any], selected_fields: List[str], n: int) -> List[Dict[str, Any]]:
        short_ctx, _ = build_context(product, selected_fields)
        prompt = QUESTION_GENERATION_PROMPT.format(
            N=n,
            locale=self.locale,
            title=_clean_text(product.get("title", "")),
            brand=_clean_text(product.get("brand", "")),
            product_type=_clean_text(product.get("product_type", "")) or _clean_text(product.get("google_product_category", "")),
            short_context_text=short_ctx,
        )
        data = self.llm.json_completion(prompt)
        if isinstance(data, list):
            return data
        return []

    def _qa_questions(self, product: Dict[str, Any], selected_fields: List[str], n: int, candidate_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        short_ctx, _ = build_context(product, selected_fields)
        prompt = QUESTION_QA_PROMPT.format(
            N=n,
            locale=self.locale,
            title=_clean_text(product.get("title", "")),
            brand=_clean_text(product.get("brand", "")),
            product_type=_clean_text(product.get("product_type", "")) or _clean_text(product.get("google_product_category", "")),
            short_context_text=short_ctx,
            candidate_questions_json=json.dumps(candidate_questions, ensure_ascii=False),
        )
        data = self.llm.json_completion(prompt)
        if isinstance(data, list):
            return data
        return []

    def _judge_answers(self, product: Dict[str, Any], selected_fields: List[str], final_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        _, full_ctx = build_context(product, selected_fields)
        prompt = ANSWER_JUDGEMENT_PROMPT.format(
            name_plus_context=full_ctx,
            final_questions_json=json.dumps(final_questions, ensure_ascii=False),
        )
        data = self.llm.json_completion(prompt)
        if isinstance(data, list):
            return data
        return []

    # ---- public API
    def evaluate_products_batch(
        self,
        products_df: pd.DataFrame,
        selected_fields: List[str],
        num_questions: int,
        batch_size: int,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        total = len(products_df)

        def progress(i: int, url: str = "", msg: str = ""):
            if on_progress:
                on_progress({"processed": i, "total": total, "current_product_url": url, "message": msg})

        for i, (_, prod) in enumerate(products_df.iterrows(), start=1):
            product: Dict[str, Any] = prod.to_dict()
            product_url = _clean_text(product.get("link", ""))
            try:
                # Step 1: candidates
                candidates = self._gen_questions(product, selected_fields, num_questions)
                # Step 2: QA/refine
                final_qs = self._qa_questions(product, selected_fields, num_questions, candidates)
                # Step 3: judge
                judgements = self._judge_answers(product, selected_fields, final_qs)

                # Scoring
                coverage_pct, yes_c, partial_c, no_c = compute_weighted_coverage(judgements)
                unanswered_required = sum(1 for j in judgements if j.get("required") and j.get("verdict") in {"no", "partial"})

                # Build contexts for output
                short_ctx, name_plus_context = build_context(product, selected_fields)

                row = {
                    "product_url": product_url,
                    "name_plus_context": name_plus_context,
                    "questions_json": json.dumps([{"question": q.get("question"), "taxonomy": q.get("taxonomy"), "required": bool(q.get("required", False))} for q in final_qs], ensure_ascii=False),
                    "judgements_json": json.dumps(judgements, ensure_ascii=False),
                    "yes_count": yes_c,
                    "partial_count": partial_c,
                    "no_count": no_c,
                    "coverage_pct": round(coverage_pct, 1),
                    "unanswered_required_count": unanswered_required,
                    "context_fields": ",".join(selected_fields),
                }
                rows.append(row)
            except Exception as e:  # keep pipeline resilient
                if debug:
                    rows.append({
                        "product_url": product_url,
                        "name_plus_context": "",
                        "questions_json": "[]",
                        "judgements_json": json.dumps([{"error": str(e)}]),
                        "yes_count": 0,
                        "partial_count": 0,
                        "no_count": 0,
                        "coverage_pct": 0.0,
                        "unanswered_required_count": 0,
                        "context_fields": ",".join(selected_fields),
                    })
            finally:
                progress(i, product_url)

        return pd.DataFrame(rows)
