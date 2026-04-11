"""
Deterministic graders for SupportEnv tasks.

Each grader inspects the agent's action_history against ground-truth data
and returns (score, breakdown, feedback) where score is in (0.0, 1.0).

Task 1 — Classification:  category match (0.50) + priority match (0.40) + efficiency (0.10)
Task 2 — Extraction:      entity coverage (0.60) + action coverage (0.30) + no hallucination (0.10)
                          (extra entity keys, wrong entity values, and bogus actions reduce the
                          hallucination component)
Task 3 — Resolution:      keyword coverage (0.30) + step coverage (0.30) + tone (0.25) +
                           length (0.10) + non-empty steps (0.05)
                          (keyword stuffing / low lexical diversity and unordered or bogus steps
                           reduce keyword and step components)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple


SCORE_EPSILON = 0.01


def _strict_score(score: float) -> float:
    """Map any score into the strict open interval (0, 1)."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = SCORE_EPSILON

    # Guard against NaN, which would bypass numeric comparisons.
    if math.isnan(value):
        value = SCORE_EPSILON

    value = min(max(value, SCORE_EPSILON), 1.0 - SCORE_EPSILON)
    return round(value, 4)


def grade_task(
    task_id: str, episode_state: Dict[str, Any]
) -> Tuple[float, Dict[str, float], str]:
    if task_id == "task1":
        return _grade_classification(episode_state)
    elif task_id == "task2":
        return _grade_extraction(episode_state)
    elif task_id == "task3":
        return _grade_resolution(episode_state)
    return _strict_score(0.01), {}, "Unknown task"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_action_of_type(
    history: List[Dict[str, Any]], action_type: str
) -> Optional[Dict[str, Any]]:
    """Return the last action matching *action_type*, or None."""
    for action in reversed(history):
        if action.get("action_type") == action_type:
            return action
    return None


def _normalize(s: Any) -> str:
    return str(s).strip().lower() if s is not None else ""


def _token_diversity_ratio(text: str) -> float:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _ordered_step_matches(gt_steps: List[str], pred_steps: List[str]) -> int:
    """Count ground-truth steps matched in order as a subsequence of predicted steps."""
    pred_norm = [_normalize(s) for s in pred_steps if str(s).strip()]
    idx = 0
    matched = 0
    for gs in gt_steps:
        target = _normalize(gs)
        while idx < len(pred_norm) and pred_norm[idx] != target:
            idx += 1
        if idx < len(pred_norm) and pred_norm[idx] == target:
            matched += 1
            idx += 1
    return matched


# ---------------------------------------------------------------------------
# Task 1 — Classification
# ---------------------------------------------------------------------------

def _grade_classification(ep: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Score breakdown:
      category_correct  0.50 — exact match
      priority_correct  0.40 — exact match
      efficiency        0.10 — 1 step = full, degrades linearly
    """
    gt = ep["ticket_data"]["ground_truth"]
    history = ep.get("action_history", [])

    breakdown: Dict[str, float] = {
        "baseline": 0.01,
        "category_correct": 0.0,
        "priority_correct": 0.0,
        "efficiency": 0.0,
    }

    classify_action = _last_action_of_type(history, "classify")
    if classify_action is None:
        return _strict_score(0.0), breakdown, "No classify action found."

    # Category
    if _normalize(classify_action.get("category")) == _normalize(gt["category"]):
        breakdown["category_correct"] = 0.49

    # Priority
    if _normalize(classify_action.get("priority")) == _normalize(gt["priority"]):
        breakdown["priority_correct"] = 0.40

    # Efficiency: full marks if classified in 1 step, degrades linearly
    max_steps = ep.get("max_steps", 3)
    steps_used = ep.get("step_number", max_steps)
    if steps_used <= 1:
        breakdown["efficiency"] = 0.09
    else:
        breakdown["efficiency"] = round(max(0.0, 0.09 * (1 - (steps_used - 1) / max_steps)), 4)

    score = _strict_score(sum(breakdown.values()))
    parts = ", ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
    return score, breakdown, f"Task 1: {parts}"


# ---------------------------------------------------------------------------
# Task 2 — Information Extraction
# ---------------------------------------------------------------------------

def _grade_extraction(ep: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Score breakdown:
      entity_coverage   0.60 — fraction of ground-truth entities matched
      action_coverage   0.30 — fraction of required actions matched
      no_hallucination  0.10 — penalty for extra entities not in ground truth
    """
    gt = ep["ticket_data"]["ground_truth"]
    history = ep.get("action_history", [])

    breakdown: Dict[str, float] = {
        "baseline": 0.01,
        "entity_coverage": 0.0,
        "action_coverage": 0.0,
        "no_hallucination": 0.09,  # start with full marks, deduct
    }

    extract_action = _last_action_of_type(history, "extract")
    if extract_action is None:
        breakdown["no_hallucination"] = 0.0
        return _strict_score(0.01), breakdown, "No extract action found."

    # --- Entity coverage ---
    gt_entities: Dict[str, Any] = gt.get("entities", {})
    pred_entities: Dict[str, Any] = extract_action.get("extracted_entities") or {}

    if gt_entities:
        matched = 0
        wrong_values = 0
        for key, gt_val in gt_entities.items():
            pred_val = pred_entities.get(key)
            if pred_val is None:
                continue
            if _entity_matches(gt_val, pred_val):
                matched += 1
            else:
                wrong_values += 1
        breakdown["entity_coverage"] = round(0.59 * matched / len(gt_entities), 4)
        if wrong_values:
            misinfo_penalty = min(0.08, wrong_values * 0.025)
            breakdown["no_hallucination"] = round(
                max(0.0, breakdown["no_hallucination"] - misinfo_penalty), 4
            )

    # --- Action coverage ---
    gt_actions: List[str] = gt.get("required_actions", [])
    pred_actions: List[str] = extract_action.get("required_actions") or []
    pred_actions_lower = [_normalize(a) for a in pred_actions]
    gt_actions_lower = [_normalize(ga) for ga in gt_actions]

    if gt_actions:
        matched_actions = sum(
            1 for ga in gt_actions if _normalize(ga) in pred_actions_lower
        )
        breakdown["action_coverage"] = round(0.30 * matched_actions / len(gt_actions), 4)

        extra_actions = [
            pa for pa in pred_actions_lower if pa and pa not in set(gt_actions_lower)
        ]
        if extra_actions:
            halluc_action_penalty = min(0.09, len(extra_actions) * 0.03)
            breakdown["no_hallucination"] = round(
                max(0.0, breakdown["no_hallucination"] - halluc_action_penalty), 4
            )

    # --- No hallucination ---
    if pred_entities and gt_entities:
        extra_keys = set(pred_entities.keys()) - set(gt_entities.keys())
        if extra_keys:
            penalty = min(len(extra_keys) * 0.02, 0.09)
            breakdown["no_hallucination"] = round(max(0.0, breakdown["no_hallucination"] - penalty), 4)
    score = _strict_score(sum(breakdown.values()))
    parts = ", ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
    return score, breakdown, f"Task 2: {parts}"

def _entity_matches(gt_val: Any, pred_val: Any) -> bool:
    """Flexible entity comparison — handles strings, lists, and numbers."""
    if isinstance(gt_val, list) and isinstance(pred_val, list):
        gt_set = {_normalize(v) for v in gt_val}
        pred_set = {_normalize(v) for v in pred_val}
        return gt_set == pred_set
    return _normalize(gt_val) == _normalize(pred_val)


# ---------------------------------------------------------------------------
# Task 3 — Resolution Generation
# ---------------------------------------------------------------------------

def _grade_resolution(ep: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Score breakdown:
      keyword_coverage  0.30 — fraction of required keywords found in response
      step_coverage     0.30 — fraction of required resolution steps matched
      tone_compliance   0.25 — apology / urgency / timeline adherence
      length_adequate   0.10 — response meets minimum length
      no_empty_steps    0.05 — all resolution steps are non-empty
    """
    gt = ep["ticket_data"]["ground_truth"]
    history = ep.get("action_history", [])

    breakdown: Dict[str, float] = {
        "baseline": 0.01,
        "keyword_coverage": 0.0,
        "step_coverage": 0.0,
        "tone_compliance": 0.0,
        "length_adequate": 0.0,
        "no_empty_steps": 0.04,  # assume pass unless empty steps found
    }

    respond_action = _last_action_of_type(history, "respond")
    if respond_action is None:
        breakdown["no_empty_steps"] = 0.0
        return _strict_score(0.01), breakdown, "No respond action found."

    response_text: str = respond_action.get("response_text") or ""
    resolution_steps: List[str] = respond_action.get("resolution_steps") or []
    response_lower = response_text.lower()

    diversity = _token_diversity_ratio(response_text)
    diversity_floor = 0.34
    diversity_scale = (
        1.0 if diversity >= diversity_floor else max(0.12, diversity / diversity_floor)
    )

    # --- Keyword coverage ---
    required_keywords: List[str] = gt.get("required_keywords", [])
    if required_keywords:
        matched_kw = sum(1 for kw in required_keywords if kw.lower() in response_lower)
        breakdown["keyword_coverage"] = round(
            0.29 * (matched_kw / len(required_keywords)) * diversity_scale, 4
        )

    # --- Step coverage ---
    gt_steps: List[str] = gt.get("required_resolution_steps", [])
    if gt_steps:
        matched_steps = _ordered_step_matches(gt_steps, resolution_steps)
        gt_step_set = {_normalize(gs) for gs in gt_steps}
        extra_steps = sum(
            1
            for ps in resolution_steps
            if str(ps).strip() and _normalize(ps) not in gt_step_set
        )
        step_quality = matched_steps / len(gt_steps)
        extra_penalty = min(0.18, extra_steps * 0.04)
        raw_step = max(0.0, 0.30 * step_quality - extra_penalty)
        if diversity_scale < 1.0:
            raw_step *= max(0.2, diversity_scale)
        breakdown["step_coverage"] = round(raw_step, 4)

    # --- Tone compliance ---
    tone_req = gt.get("tone_requirements", {})
    tone_checks = 0
    tone_pass = 0
    if tone_req.get("must_apologize"):
        tone_checks += 1
        apology_words = ["apolog", "sorry", "regret", "sincerely"]
        if any(w in response_lower for w in apology_words):
            tone_pass += 1
    if tone_req.get("must_acknowledge_urgency"):
        tone_checks += 1
        urgency_words = ["urgent", "immediately", "priority", "asap", "right away", "as soon as"]
        if any(w in response_lower for w in urgency_words):
            tone_pass += 1
    if tone_req.get("must_provide_timeline"):
        tone_checks += 1
        timeline_words = ["within", "hours", "minutes", "by end of", "shortly", "today", "tomorrow", "timeline", "expect"]
        if any(w in response_lower for w in timeline_words):
            tone_pass += 1
    if tone_checks > 0:
        breakdown["tone_compliance"] = round(0.25 * tone_pass / tone_checks, 4)
    else:
        breakdown["tone_compliance"] = 0.25  # no tone requirements = full marks

    # --- Length adequate ---
    min_len = gt.get("expected_response_length_min", 80)
    if len(response_text) >= min_len:
        breakdown["length_adequate"] = round(0.10 * min(1.0, diversity_scale**0.5), 4)

    # --- Non-empty steps ---
    if not resolution_steps or any(not s.strip() for s in resolution_steps):
        breakdown["no_empty_steps"] = 0.0

    score = _strict_score(sum(breakdown.values()))
    parts = ", ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
    return score, breakdown, f"Task 3: {parts}"
