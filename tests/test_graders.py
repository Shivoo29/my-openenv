"""Regression tests for deterministic graders (anti-gaming and hallucination)."""

from __future__ import annotations

from copy import deepcopy

from data import get_tickets
from graders import grade_task


def _ep_task3(ticket_index: int, response_text: str, resolution_steps: list[str]) -> dict:
    ticket = get_tickets("task3")[ticket_index]
    return {
        "ticket_data": ticket,
        "action_history": [
            {
                "action_type": "respond",
                "response_text": response_text,
                "resolution_steps": resolution_steps,
            }
        ],
        "step_number": 2,
        "max_steps": 8,
    }


def test_task3_keyword_stuffing_scores_low() -> None:
    gt = get_tickets("task3")[0]["ground_truth"]
    kws = gt["required_keywords"]
    spam = " ".join(["blah"] * 200 + kws * 15)
    score, breakdown, _ = grade_task(
        "task3",
        _ep_task3(0, spam, list(gt["required_resolution_steps"])),
    )
    assert score < 0.35, breakdown


def test_task3_wrong_step_order_hurts() -> None:
    gt = get_tickets("task3")[0]["ground_truth"]
    steps = list(reversed(gt["required_resolution_steps"]))
    text = (
        "We sincerely apologize for the trouble. We understand this is urgent "
        "and will act immediately. You can expect an update within 24 hours. "
        + " ".join(gt["required_keywords"])
    )
    score_ordered, _, _ = grade_task(
        "task3",
        _ep_task3(0, text, list(gt["required_resolution_steps"])),
    )
    score_reversed, _, _ = grade_task("task3", _ep_task3(0, text, steps))
    assert score_reversed < score_ordered


def test_task2_extra_required_actions_reduce_score() -> None:
    ticket = get_tickets("task2")[0]
    clean = {
        "action_type": "extract",
        "extracted_entities": deepcopy(ticket["ground_truth"]["entities"]),
        "required_actions": list(ticket["ground_truth"]["required_actions"]),
    }
    dirty = {
        "action_type": "extract",
        "extracted_entities": deepcopy(ticket["ground_truth"]["entities"]),
        "required_actions": list(ticket["ground_truth"]["required_actions"])
        + ["delete_database", "wipe_production"],
    }
    ep_clean = {
        "ticket_data": ticket,
        "action_history": [clean],
        "step_number": 1,
        "max_steps": 5,
    }
    ep_dirty = {
        "ticket_data": ticket,
        "action_history": [dirty],
        "step_number": 1,
        "max_steps": 5,
    }
    s_clean, _, _ = grade_task("task2", ep_clean)
    s_dirty, br_dirty, _ = grade_task("task2", ep_dirty)
    assert s_dirty < s_clean
    assert br_dirty["no_hallucination"] < 0.09


def test_task2_wrong_entity_value_hurts() -> None:
    ticket = get_tickets("task2")[0]
    entities = deepcopy(ticket["ground_truth"]["entities"])
    entities["account_id"] = "ACC-WRONG"
    action = {
        "action_type": "extract",
        "extracted_entities": entities,
        "required_actions": list(ticket["ground_truth"]["required_actions"]),
    }
    ep = {"ticket_data": ticket, "action_history": [action], "step_number": 1, "max_steps": 5}
    score, breakdown, _ = grade_task("task2", ep)
    assert breakdown["entity_coverage"] < 0.59
    assert breakdown["no_hallucination"] < 0.09
    assert score < 0.95
