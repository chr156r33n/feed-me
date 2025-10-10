"""
Enhanced Product Feed Evaluator with batch saving and resume capabilities.
- Saves intermediate results to avoid data loss on large runs
- Supports resuming from where evaluation left off
- Implements proper error handling and logging
- Optimizes memory usage for large datasets

Requires: prompts.py (v2) in the same project providing:
  QUESTION_GENERATION_PROMPT, QUESTION_QA_PROMPT, ANSWER_JUDGEMENT_PROMPT

App entrypoint expects:
  ProductFeedEvaluator(api_key, model, locale)
  .evaluate_products_batch(products_df, selected_fields, num_questions, batch_size, on_progress, debug)
"""
from __future__ import annotations

import json
import os
import re
import html
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, Dict, List, Tuple, Callable, Optional
from pathlib import Path

import pandas as pd
from monitoring import create_monitoring_suite

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
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class RateLimitedError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class QuotaExceededError(RuntimeError):
    pass


def _looks_like_rate_limit(message: str) -> bool:
    m = (message or "").lower()
    return (
        "429" in m
        or "rate limit" in m
        or "too many requests" in m
        or ("please try again" in m and "seconds" in m)
    )


def _looks_like_quota(message: str) -> bool:
    m = (message or "").lower()
    return (
        "quota" in m
        or "exceeded your current quota" in m
        or "insufficient_quota" in m
    )


def _clean_text(val: Any) -> str:
    if val is None:
        return ""
    s = html.unescape(str(val))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_context(product: Dict[str, Any], selected_fields: List[str]) -> Tuple[str, str]:
    fields = list(dict.fromkeys(selected_fields + ["link"]))
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


class _LLM:
    def __init__(self, api_key: str, model: str):
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not available. Install openai>=1.0.0.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def json_completion(self, prompt: str, max_retries: int = 3) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60,  # 60 second timeout per request
                )
                text = resp.choices[0].message.content or ""
                text = text.strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    indices = [text.find("[{"), text.find("[\n{"), text.find("[")]
                    valid_indices = [i for i in indices if i != -1]
                    start = min(valid_indices) if valid_indices else -1
                    if start != -1:
                        snippet = text[start:]
                        snippet = snippet.strip().strip("`")
                        return json.loads(snippet)
                    raise
            except Exception as e:
                last_err = e
                wait_time = min(2.0 * (2 ** attempt), 30)  # Exponential backoff, max 30s
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        # Classification for better upstream handling
        msg = str(last_err or "")
        if _looks_like_rate_limit(msg):
            raise RateLimitedError(msg)
        if _looks_like_quota(msg):
            raise QuotaExceededError(msg)
        raise last_err  # type: ignore


class BatchSaver:
    """Handles saving and loading batch results to/from disk."""
    
    def __init__(self, output_dir: str = "output"):
        # Use a more isolated directory structure to avoid Streamlit file watcher
        self.output_dir = Path(output_dir) / "batches"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Maintain an active session file to enable safe resume across app reruns
        self._active_session_file = self.output_dir / "session_active.json"
        self.timestamp = self._load_or_create_session_timestamp()

    def _load_or_create_session_timestamp(self) -> str:
        """Reuse active session timestamp if present, otherwise create new."""
        try:
            if self._active_session_file.exists():
                with open(self._active_session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    ts = data.get('timestamp')
                    if ts:
                        return ts
        except Exception:
            pass
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with open(self._active_session_file, 'w', encoding='utf-8') as f:
                json.dump({'timestamp': ts}, f)
        except Exception:
            pass
        return ts
        
    def get_batch_file(self, batch_num: int) -> Path:
        """Get the file path for a specific batch."""
        return self.output_dir / f"batch_{batch_num:04d}_{self.timestamp}.json"
    
    def get_progress_file(self) -> Path:
        """Get the file path for progress tracking."""
        return self.output_dir / f"progress_{self.timestamp}.json"
    
    def save_batch(self, batch_num: int, results: List[Dict[str, Any]]) -> None:
        """Save a batch of results to disk."""
        batch_file = self.get_batch_file(batch_num)
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved batch {batch_num} with {len(results)} results to {batch_file}")
    
    def load_batch(self, batch_num: int) -> List[Dict[str, Any]]:
        """Load a batch of results from disk."""
        batch_file = self.get_batch_file(batch_num)
        if not batch_file.exists():
            return []
        with open(batch_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_progress(self, progress_data: Dict[str, Any]) -> None:
        """Save progress information."""
        progress_file = self.get_progress_file()
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def load_progress(self) -> Optional[Dict[str, Any]]:
        """Load progress information."""
        progress_file = self.get_progress_file()
        if not progress_file.exists():
            return None
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_batch_files(self) -> List[Path]:
        """Get all batch files for the current session."""
        pattern = f"batch_*_{self.timestamp}.json"
        return list(self.output_dir.glob(pattern))
    
    def consolidate_results(self) -> List[Dict[str, Any]]:
        """Load and consolidate all batch results."""
        all_results = []
        batch_files = sorted(self.get_all_batch_files())
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_results = json.load(f)
                    all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Failed to load batch file {batch_file}: {e}")
        
        return all_results
    
    def list_batch_sessions(self) -> List[Dict[str, Any]]:
        """List all batch processing sessions."""
        sessions = []
        
        # Group files by timestamp
        timestamp_groups = {}
        for file_path in self.output_dir.glob("batch_*.json"):
            parts = file_path.stem.split("_")
            if len(parts) >= 3:
                timestamp = "_".join(parts[2:])  # Everything after batch_XXXX_
                if timestamp not in timestamp_groups:
                    timestamp_groups[timestamp] = []
                timestamp_groups[timestamp].append(file_path)
        
        for timestamp, files in timestamp_groups.items():
            # Sort files by batch number
            files.sort(key=lambda x: int(x.stem.split("_")[1]))
            
            # Get progress info
            progress_file = self.output_dir / f"progress_{timestamp}.json"
            progress_data = None
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                except:
                    pass
            
            sessions.append({
                'timestamp': timestamp,
                'batch_files': files,
                'progress_data': progress_data,
                'total_batches': len(files)
            })
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)

    def finalize_session(self) -> None:
        """Mark the session complete so future runs start fresh."""
        try:
            if self._active_session_file.exists():
                self._active_session_file.unlink()
        except Exception:
            pass


class ProductFeedEvaluator:
    def __init__(self, api_key: str, model: str, locale: str = "en-GB", output_dir: str = "output"):
        self.locale = locale
        self.llm = _LLM(api_key=api_key, model=model)
        self.batch_saver = BatchSaver(output_dir)
        
        # Initialize monitoring suite
        self.monitor, self.rate_limiter, self.memory_monitor, self.health_checker = create_monitoring_suite(output_dir)

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

    def evaluate_products_batch(
        self,
        products_df: pd.DataFrame,
        selected_fields: List[str],
        num_questions: int,
        batch_size: int,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        debug: bool = False,
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate products with batch saving and resume capability.
        
        Args:
            products_df: DataFrame of products to evaluate
            selected_fields: Fields to include in context
            num_questions: Number of questions per product
            batch_size: Number of products to process before saving
            on_progress: Progress callback function
            debug: Include error rows in output
            resume: Whether to resume from previous progress
        """
        total = len(products_df)
        start_index = 0
        
        # Check for existing progress if resuming
        if resume:
            progress_data = self.batch_saver.load_progress()
            if progress_data:
                start_index = progress_data.get('last_processed', 0)
                logger.info(f"Resuming from product {start_index + 1}/{total}")
        
        def progress(i: int, url: str = "", msg: str = ""):
            if on_progress:
                on_progress({"processed": i, "total": total, "current_product_url": url, "message": msg})
            
            # Save progress less frequently to prevent Streamlit file watcher triggers
            # Only save every 100 products or at the end of batches
            if i % 100 == 0 or (i % batch_size == 0 and i > 0):
                try:
                    self.batch_saver.save_progress({
                        'last_processed': i,
                        'total': total,
                        'timestamp': datetime.now().isoformat(),
                        'batch_size': batch_size,
                        'selected_fields': selected_fields,
                        'num_questions': num_questions
                    })
                except Exception as e:
                    # Don't let progress saving errors interrupt evaluation
                    logger.warning(f"Failed to save progress: {e}")

        # Concurrency settings: bounded parallelism with rate limiter gating per call
        concurrency = max(1, min(4, batch_size // 10 if batch_size >= 10 else 1))
        logger.info(f"Using bounded concurrency: {concurrency}")

        # Prepare tasks
        tasks: List[Tuple[int, Dict[str, Any]]] = []
        for i, (_, prod) in enumerate(products_df.iterrows(), start=1):
            if i <= start_index:
                continue
            tasks.append((i, prod.to_dict()))

        current_batch: List[Dict[str, Any]] = []
        batch_num = 0
        processed_count = 0
        abort_event = threading.Event()

        def process_one(idx: int, product: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]], str]:
            product_url_local = _clean_text(product.get("link", ""))

            def call_with_rate_limit(fn: Callable[[], Any]) -> Any:
                # Gate each API call
                attempt = 0
                while True:
                    if abort_event.is_set():
                        raise RuntimeError("Aborted")
                    if not self.rate_limiter.can_make_request():
                        wait_time_local = self.rate_limiter.get_wait_time()
                        time.sleep(max(0.1, wait_time_local))
                        continue
                    try:
                        self.rate_limiter.record_request()
                        return fn()
                    except RateLimitedError:
                        cooldown_local = min(2.0 * (2 ** attempt), 60.0)
                        time.sleep(cooldown_local)
                        attempt += 1
                        if attempt >= 5:
                            raise
                    except QuotaExceededError:
                        # Bubble up to stop everything
                        raise

            try:
                # Step 1: candidates
                candidates = call_with_rate_limit(lambda: self._gen_questions(product, selected_fields, num_questions))
                # Step 2: refine
                final_qs = call_with_rate_limit(lambda: self._qa_questions(product, selected_fields, num_questions, candidates))
                # Step 3: judge
                judgements = call_with_rate_limit(lambda: self._judge_answers(product, selected_fields, final_qs))

                coverage_pct, yes_c, partial_c, no_c = compute_weighted_coverage(judgements)
                unanswered_required = sum(1 for j in judgements if j.get("required") and j.get("verdict") in {"no", "partial"})
                _, name_plus_context = build_context(product, selected_fields)
                row = {
                    "product_url": product_url_local,
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
                return idx, row, product_url_local
            except Exception as e:
                # Log and optionally return error row
                logger.error(f"Error processing product {idx} ({product_url_local}): {e}")
                self.monitor.log_error(
                    product_index=idx,
                    product_url=product_url_local,
                    error=e,
                    context={'selected_fields': selected_fields,'num_questions': num_questions,'batch_size': batch_size}
                )
                if isinstance(e, QuotaExceededError):
                    abort_event.set()
                    raise
                if debug:
                    return idx, {
                        "product_url": product_url_local,
                        "name_plus_context": "",
                        "questions_json": "[]",
                        "judgements_json": json.dumps([{ "error": str(e) }]),
                        "yes_count": 0,
                        "partial_count": 0,
                        "no_count": 0,
                        "coverage_pct": 0.0,
                        "unanswered_required_count": 0,
                        "context_fields": ",".join(selected_fields),
                    }, product_url_local
                return idx, None, product_url_local

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {executor.submit(process_one, idx, prod): idx for idx, prod in tasks}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    finished_idx, row, url_for_msg = fut.result()
                except QuotaExceededError as qee:
                    progress(processed_count, "", msg="Quota exceeded. Stopping evaluation.")
                    # Save current partial batch and progress
                    if current_batch:
                        self.batch_saver.save_batch(batch_num, current_batch)
                    self.batch_saver.save_progress({
                        'last_processed': start_index + processed_count,
                        'total': total,
                        'timestamp': datetime.now().isoformat(),
                        'reason': 'quota_exceeded'
                    })
                    raise
                except Exception as e:
                    # Already logged inside; just continue
                    row = None
                    url_for_msg = ""
                processed_count += 1

                # Health check periodically based on completions
                if processed_count % 50 == 0:
                    health = self.health_checker.check_health()
                    if self.health_checker.should_pause_evaluation():
                        logger.warning(f"Pausing evaluation due to health issues: {health['issues']}")
                        if current_batch:
                            self.batch_saver.save_batch(batch_num, current_batch)
                        self.batch_saver.save_progress({
                            'last_processed': start_index + processed_count,
                            'total': total,
                            'timestamp': datetime.now().isoformat(),
                            'health_issues': health['issues']
                        })
                        raise Exception(f"Evaluation paused due to health issues: {health['issues']}")

                # Progress update
                self.monitor.log_progress(processed_count, total, url_for_msg)
                progress(processed_count, url_for_msg)

                if row is not None:
                    current_batch.append(row)
                    if len(current_batch) >= batch_size:
                        self.batch_saver.save_batch(batch_num, current_batch)
                        batch_num += 1
                        current_batch = []
                        logger.info(f"Saved batch {batch_num - 1}, processed {processed_count}/{total} products")
        
        # Save any remaining products in the final batch
        if current_batch:
            self.batch_saver.save_batch(batch_num, current_batch)
            logger.info(f"Saved final batch {batch_num} with {len(current_batch)} results")
        
        # Consolidate all results
        logger.info("Consolidating all batch results...")
        all_results = self.batch_saver.consolidate_results()
        # Exactly-once: remove duplicates by product_url if present
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for row in all_results:
            key = row.get("product_url")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(row)
        
        # Clean up progress file
        progress_file = self.batch_saver.get_progress_file()
        if progress_file.exists():
            progress_file.unlink()
        
        # Finalize session so next run starts a fresh session id
        self.batch_saver.finalize_session()
        logger.info(f"Evaluation complete. Total results: {len(deduped)} (deduped)")
        return pd.DataFrame(deduped)

    def resume_evaluation(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Resume evaluation from saved batches."""
        logger.info("Resuming evaluation from saved batches...")
        all_results = self.batch_saver.consolidate_results()
        return pd.DataFrame(all_results)
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """Get comprehensive evaluation status and health information."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'batch_info': {},
            'health_status': {},
            'error_summary': {},
            'memory_info': {}
        }
        
        # Get batch information
        try:
            sessions = self.batch_saver.list_batch_sessions()
            if sessions:
                latest_session = sessions[0]
                status['batch_info'] = {
                    'latest_session': latest_session['timestamp'],
                    'total_batches': latest_session['total_batches'],
                    'progress_data': latest_session.get('progress_data')
                }
        except Exception as e:
            status['batch_info']['error'] = str(e)
        
        # Get health status
        try:
            status['health_status'] = self.health_checker.check_health()
        except Exception as e:
            status['health_status']['error'] = str(e)
        
        # Get error summary
        try:
            status['error_summary'] = self.monitor.get_error_summary()
        except Exception as e:
            status['error_summary']['error'] = str(e)
        
        # Get memory information
        try:
            status['memory_info'] = self.memory_monitor.check_memory()
        except Exception as e:
            status['memory_info']['error'] = str(e)
        
        return status