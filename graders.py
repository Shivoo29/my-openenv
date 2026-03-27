
"""
Deterministic graders for all three SupportEnv tasks.

All graders return a score in [0.0, 1.0] and a dict of per-criterion
partial scores.  No external calls are made — scoring is purely rule-based
so results are fully reproducible.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from data import TASK_META


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_keyword(text: str, keyword: str) -> bool:
    """Return True if keyword (normalised) appears anywhere in text."""
    return _normalise(keyword) in _normalise(text)


# ---------------------------------------------------------------------------
# Task 1 — Ticket Classification
# ---------------------------------------------------------------------------

def grade_task1(
    ticket_index: int,
    action_history: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """
    Grade a Task-1 episode (classification).

    Scoring:
      category_correct  : 0.50
      priority_correct  : 0.40
      acted_efficiently : 0.10  (submitted within max_steps without wasted loops)

    Returns (score, breakdown, feedback)
    """
    tickets = TASK_META["task1"]["tickets"]
    gt = tickets[ticket_index]["ground_truth"]

    category_score = 0.0
    priority_score = 0.0

    # Find the last classify action the agent submitted
    last_classify = None
    for act in reversed(action_history):
        if act.get("action_type") == "classify":
            last_classify = act
            break

    if last_classify is not None:
        if last_classify.get("category", "").lower() == gt["category"]:
            category_score = 0.50
        elif _normalise(last_classify.get("category", "")) in _normalise(gt["category"]):
            category_score = 0.25  # partial credit for substring match

        if last_classify.get("priority", "").lower() == gt["priority"]:
            priority_score = 0.40
        elif _is_adjacent_priority(
            last_classify.get("priority", "").lower(), gt["priority"]
        ):
            priority_score = 0.20  # partial credit for one level off

    # Efficiency bonus: agent didn't fill max steps with redundant classify attempts
    classify_count = sum(
        1 for a in action_history if a.get("action_type") == "classify"
    )
    efficiency_score = 0.10 if classify_count <= 2 else 0.0

    total = category_score + priority_score + efficiency_score
    breakdown = {
        "category_correct": category_score,
        "priority_correct": priority_score,
        "efficiency": efficiency_score,
    }
    feedback = _task1_feedback(gt, last_classify, breakdown)
    return round(total, 4), breakdown, feedback


def _is_adjacent_priority(pred: str, gold: str) -> bool:
    """One priority level off gets partial credit."""
    levels = ["low", "medium", "high", "critical"]
    if pred not in levels or gold not in levels:
        return False
    return abs(levels.index(pred) - levels.index(gold)) == 1


def _task1_feedback(gt, last_classify, breakdown) -> str:
    parts = []
    if last_classify is None:
        return "No classify action found in episode history."
    if breakdown["category_correct"] == 0.50:
        parts.append(f"Category '{gt['category']}' correctly identified.")
    elif breakdown["category_correct"] > 0:
        parts.append(f"Category partially correct (got substring match).")
    else:
        parts.append(
            f"Category incorrect: expected '{gt['category']}', "
            f"got '{last_classify.get('category')}'."
        )
    if breakdown["priority_correct"] == 0.40:
        parts.append(f"Priority '{gt['priority']}' correctly identified.")
    elif breakdown["priority_correct"] > 0:
        parts.append("Priority off by one level (partial credit).")
    else:
        parts.append(
            f"Priority incorrect: expected '{gt['priority']}', "
            f"got '{last_classify.get('priority')}'."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Task 2 — Information Extraction
# ---------------------------------------------------------------------------

def grade_task2(
    ticket_index: int,
    action_history: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """
    Grade a Task-2 episode (information extraction).

    Scoring:
      entity_coverage  : 0.60  (fraction of required entities correctly extracted)
      action_coverage  : 0.30  (fraction of required actions identified)
      no_hallucination : 0.10  (no extra entities that don't exist in source)

    Returns (score, breakdown, feedback)
    """
    tickets = TASK_META["task2"]["tickets"]
    gt = tickets[ticket_index]["ground_truth"]
    required_entities: Dict[str, Any] = gt["entities"]
    required_actions: List[str] = gt["required_actions"]

    # Find the last extract action
    last_extract = None
    for act in reversed(action_history):
        if act.get("action_type") == "extract":
            last_extract = act
            break

    if last_extract is None:
        return 0.0, {"entity_coverage": 0.0, "action_coverage": 0.0, "no_hallucination": 0.0}, \
               "No extract action found in episode history."

    extracted: Dict[str, Any] = last_extract.get("extracted_entities") or {}
    submitted_actions: List[str] = last_extract.get("required_actions") or []

    # --- Entity scoring ---
    entity_hits = 0
    entity_notes = []
    for key, expected_val in required_entities.items():
        pred_val = extracted.get(key)
        if pred_val is None:
            entity_notes.append(f"Missing: {key}")
            continue
        if isinstance(expected_val, list):
            # For list values, check set overlap (case-insensitive)
            pred_set = {_normalise(str(v)) for v in (pred_val if isinstance(pred_val, list) else [pred_val])}
            gold_set = {_normalise(str(v)) for v in expected_val}
            overlap = len(pred_set & gold_set) / len(gold_set)
            entity_hits += overlap
            if overlap < 1.0:
                entity_notes.append(f"Partial match for '{key}': {overlap:.0%}")
        else:
            if _normalise(str(pred_val)) == _normalise(str(expected_val)):
                entity_hits += 1
            elif _normalise(str(expected_val)) in _normalise(str(pred_val)):
                entity_hits += 0.5
                entity_notes.append(f"Partial match for '{key}'")
            else:
                entity_notes.append(
                    f"Wrong value for '{key}': expected '{expected_val}', got '{pred_val}'"
                )

    entity_coverage = entity_hits / max(len(required_entities), 1)

    # --- Action scoring ---
    gold_action_set = {a.lower() for a in required_actions}
    pred_action_set = {a.lower() for a in submitted_actions}
    action_hits = len(gold_action_set & pred_action_set)
    # Partial credit: normalised keyword match
    for gold_act in gold_action_set:
        if gold_act not in pred_action_set:
            for pred_act in pred_action_set:
                if _normalise(gold_act) in _normalise(pred_act) or \
                   _normalise(pred_act) in _normalise(gold_act):
                    action_hits += 0.5
                    break
    action_coverage = min(action_hits / max(len(required_actions), 1), 1.0)

    # --- Hallucination penalty ---
    # Extra keys in extracted that don't exist in required_entities
    extra_keys = set(extracted.keys()) - set(required_entities.keys())
    hallucination_score = 0.10 if len(extra_keys) <= 2 else 0.0

    total = (
        entity_coverage * 0.60
        + action_coverage * 0.30
        + hallucination_score
    )

    breakdown = {
        "entity_coverage": round(entity_coverage * 0.60, 4),
        "action_coverage": round(action_coverage * 0.30, 4),
        "no_hallucination": hallucination_score,
    }
    feedback_parts = [
        f"Entity coverage: {entity_coverage:.0%} ({entity_hits:.1f}/{len(required_entities)}).",
        f"Action coverage: {action_coverage:.0%} ({action_hits:.1f}/{len(required_actions)}).",
    ]
    if entity_notes:
        feedback_parts.append("Issues: " + "; ".join(entity_notes[:3]))
    feedback = " ".join(feedback_parts)

    return round(total, 4), breakdown, feedback


# ---------------------------------------------------------------------------
# Task 3 — Resolution Generation
# ---------------------------------------------------------------------------

def grade_task3(
    ticket_index: int,
    action_history: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """
    Grade a Task-3 episode (resolution generation).

    Scoring:
      keyword_coverage   : 0.30  (required keywords present in response_text)
      step_coverage      : 0.30  (required steps present in resolution_steps)
      tone_compliance    : 0.25  (apology / urgency / timeline requirements)
      length_adequate    : 0.10  (response_text meets minimum word count)
      no_empty_steps     : 0.05  (no empty or single-word steps)

    Returns (score, breakdown, feedback)
    """
    tickets = TASK_META["task3"]["tickets"]
    gt = tickets[ticket_index]["ground_truth"]
    required_keywords: List[str] = gt["required_keywords"]
    required_steps: List[str] = gt["required_resolution_steps"]
    tone: Dict[str, bool] = gt["tone_requirements"]
    min_length: int = gt["expected_response_length_min"]

    # Find the last respond or resolve action
    last_response_act = None
    for act in reversed(action_history):
        if act.get("action_type") in ("respond", "resolve"):
            last_response_act = act
            break

    if last_response_act is None:
        return 0.0, {
            "keyword_coverage": 0.0,
            "step_coverage": 0.0,
            "tone_compliance": 0.0,
            "length_adequate": 0.0,
            "no_empty_steps": 0.0,
        }, "No respond/resolve action found in episode history."

    response_text: str = last_response_act.get("response_text") or ""
    resolution_steps: List[str] = last_response_act.get("resolution_steps") or []

    # --- Keyword coverage ---
    kw_hits = sum(
        1 for kw in required_keywords if _contains_keyword(response_text, kw)
    )
    keyword_coverage = kw_hits / max(len(required_keywords), 1)

    # --- Step coverage ---
    step_hits = 0.0
    for gold_step in required_steps:
        # A step is "covered" if any resolution_step string contains all
        # main words from the gold step label (split on underscores)
        main_words = gold_step.replace("_", " ").split()
        for pred_step in resolution_steps:
            matches = sum(1 for w in main_words if _contains_keyword(pred_step, w))
            if matches == len(main_words):
                step_hits += 1
                break
            elif matches >= max(1, len(main_words) - 1):
                step_hits += 0.5
                break
    step_coverage = min(step_hits / max(len(required_steps), 1), 1.0)

    # --- Tone compliance ---
    tone_score = _check_tone(response_text, tone)

    # --- Length adequacy ---
    word_count = len(response_text.split())
    length_score = 0.10 if word_count >= min_length else round(
        0.10 * word_count / min_length, 4
    )

    # --- Step quality ---
    empty_step_count = sum(
        1 for s in resolution_steps if len(s.strip().split()) <= 1
    )
    no_empty_steps_score = 0.05 if empty_step_count == 0 else 0.0

    total = (
        keyword_coverage * 0.30
        + step_coverage * 0.30
        + tone_score
        + length_score
        + no_empty_steps_score
    )

    breakdown = {
        "keyword_coverage": round(keyword_coverage * 0.30, 4),
        "step_coverage": round(step_coverage * 0.30, 4),
        "tone_compliance": round(tone_score, 4),
        "length_adequate": round(length_score, 4),
        "no_empty_steps": no_empty_steps_score,
    }
    feedback = (
        f"Keywords: {kw_hits}/{len(required_keywords)} found. "
        f"Steps: {step_hits:.1f}/{len(required_steps)} covered. "
        f"Tone score: {tone_score:.2f}/0.25. "
        f"Word count: {word_count} (min {min_length})."
    )

    return round(total, 4), breakdown, feedback


_APOLOGY_WORDS = [
    "sorry", "apologise", "apologize", "apologies", "regret",
    "inconvenience", "unfortunately",
]
_URGENCY_WORDS = [
    "urgent", "urgently", "immediately", "priority", "asap",
    "right away", "as soon as possible",
]
_TIMELINE_WORDS = [
    "within", "hour", "minute", "today", "by", "shortly",
    "soon", "moment", "immediately", "right away",
]


def _check_tone(text: str, tone: Dict[str, bool]) -> float:
    """Return tone compliance score (max 0.25)."""
    criteria = [
        ("must_apologize", _APOLOGY_WORDS),
        ("must_acknowledge_urgency", _URGENCY_WORDS),
        ("must_provide_timeline", _TIMELINE_WORDS),
    ]
    required = [k for k, _ in criteria if tone.get(k, False)]
    if not required:
        return 0.25  # nothing required → full tone score

    per_criterion = 0.25 / len(required)
    score = 0.0
    for key, words in criteria:
        if not tone.get(key, False):
            continue
        if any(_contains_keyword(text, w) for w in words):
            score += per_criterion
    return round(score, 4)


# ---------------------------------------------------------------------------
# Public grader dispatcher
# ---------------------------------------------------------------------------

def grade_episode(
    task_id: str,
    ticket_index: int,
    action_history: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """
    Dispatch to the correct grader based on task_id.

    Returns (score: float, breakdown: dict, feedback: str)
    """
    if task_id == "task1":
        return grade_task1(ticket_index, action_history)
    elif task_id == "task2":
        return grade_task2(ticket_index, action_history)
    elif task_id == "task3":
        return grade_task3(ticket_index, action_history)
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}")
