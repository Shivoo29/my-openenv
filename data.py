"""
SupportEnv — Customer Support Ticket Triage data.

Task 1 (easy)   — Ticket Classification
Task 2 (medium) — Information Extraction
Task 3 (hard)   — Resolution Generation
"""
from __future__ import annotations
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# TASK 1 — Ticket Classification
#   Agent must choose: category + priority
#   Categories: billing | technical | account | feature_request | complaint | general
#   Priorities: low | medium | high | critical
# ---------------------------------------------------------------------------

TASK1_TICKETS: List[Dict[str, Any]] = [
    {
        "ticket_id": "T1-001",
        "subject": "Double-charged on my Pro subscription this month",
        "body": (
            "Hi, I noticed I was billed $49.99 twice on my credit card "
            "on March 3rd for my Pro subscription. My account email is "
            "alice@example.com. This is really frustrating — please issue "
            "a refund for the duplicate charge as soon as possible."
        ),
        "customer_tier": "pro",
        "account_age_days": 420,
        "previous_tickets": 0,
        "attachments": [],
        "ground_truth": {
            "category": "billing",
            "priority": "high",
        },
    },
    {
        "ticket_id": "T1-002",
        "subject": "API rate limit keeps hitting 429 even though we're Enterprise",
        "body": (
            "Our production service has been receiving HTTP 429 errors from "
            "your API for the past 2 hours. We're on the Enterprise plan which "
            "should give us 10 000 req/min. Current usage is ~3 000 req/min "
            "so we're well within limits. This is causing a production outage "
            "for our customers. Ticket urgency: CRITICAL."
        ),
        "customer_tier": "enterprise",
        "account_age_days": 730,
        "previous_tickets": 3,
        "attachments": ["rate_limit_screenshot.png"],
        "ground_truth": {
            "category": "technical",
            "priority": "critical",
        },
    },
    {
        "ticket_id": "T1-003",
        "subject": "Can I change my account email address?",
        "body": (
            "Hello, I recently changed jobs and want to update the email "
            "associated with my account from oldname@corp.com to "
            "newname@newcorp.com. Can you walk me through how to do this? "
            "There's no rush — just want to get it sorted at some point."
        ),
        "customer_tier": "free",
        "account_age_days": 60,
        "previous_tickets": 0,
        "attachments": [],
        "ground_truth": {
            "category": "account",
            "priority": "low",
        },
    },
    {
        "ticket_id": "T1-004",
        "subject": "Would love a dark mode option in the dashboard",
        "body": (
            "Hi team, long-time Pro user here. I spend a lot of time in the "
            "dashboard and it would be fantastic if you could add a dark mode "
            "toggle. Many other SaaS tools have this now and it really reduces "
            "eye strain. Would be great to see this in a future release!"
        ),
        "customer_tier": "pro",
        "account_age_days": 900,
        "previous_tickets": 1,
        "attachments": [],
        "ground_truth": {
            "category": "feature_request",
            "priority": "low",
        },
    },
    {
        "ticket_id": "T1-005",
        "subject": "Your customer service is absolutely terrible",
        "body": (
            "I submitted a ticket three weeks ago about a bug in your export "
            "feature and nobody has responded. This is completely unacceptable. "
            "I am paying for a Pro subscription and I expect timely support. "
            "If this is not resolved in 48 hours I will be requesting a full "
            "refund and posting a review online."
        ),
        "customer_tier": "pro",
        "account_age_days": 200,
        "previous_tickets": 2,
        "attachments": [],
        "ground_truth": {
            "category": "complaint",
            "priority": "high",
        },
    },
]

# ---------------------------------------------------------------------------
# TASK 2 — Information Extraction
#   Agent must populate extracted_entities and required_actions.
#   Ground truth defines exact entity keys and accepted values.
# ---------------------------------------------------------------------------

TASK2_TICKETS: List[Dict[str, Any]] = [
    {
        "ticket_id": "T2-001",
        "subject": "Refund request – incorrect charge on invoice #INV-20240312",
        "body": (
            "Dear Support, I was incorrectly charged $199.00 on invoice "
            "#INV-20240312 dated 2024-03-12. My account ID is ACC-78234. "
            "The charge should have been $99.00 as per our agreed annual plan. "
            "Please issue a refund of $100.00 to my card on file and send a "
            "corrected invoice. My name is Robert Chen."
        ),
        "customer_tier": "pro",
        "account_age_days": 540,
        "previous_tickets": 1,
        "attachments": ["invoice_INV-20240312.pdf"],
        "ground_truth": {
            "entities": {
                "customer_name": "Robert Chen",
                "account_id": "ACC-78234",
                "invoice_number": "INV-20240312",
                "incorrect_amount": "199.00",
                "correct_amount": "99.00",
                "refund_amount": "100.00",
            },
            "required_actions": [
                "issue_refund",
                "send_corrected_invoice",
            ],
        },
    },
    {
        "ticket_id": "T2-002",
        "subject": "SSO login broken after company domain migration",
        "body": (
            "Hi, our company just migrated from acme-old.com to acme-new.com. "
            "Since the migration last Tuesday (2024-03-19), our 45 users cannot "
            "log in via SSO. The error message is: 'SAML assertion domain "
            "mismatch'. Our org ID is ORG-5512. We need this fixed ASAP as "
            "nobody can access the platform. Contact: Maria Gonzalez, "
            "IT Director."
        ),
        "customer_tier": "enterprise",
        "account_age_days": 1100,
        "previous_tickets": 5,
        "attachments": [],
        "ground_truth": {
            "entities": {
                "contact_name": "Maria Gonzalez",
                "contact_role": "IT Director",
                "org_id": "ORG-5512",
                "old_domain": "acme-old.com",
                "new_domain": "acme-new.com",
                "error_message": "SAML assertion domain mismatch",
                "affected_users": "45",
            },
            "required_actions": [
                "update_sso_domain_config",
                "verify_saml_settings",
                "notify_affected_users",
            ],
        },
    },
    {
        "ticket_id": "T2-003",
        "subject": "Need to add 3 seats to our Team plan before end of quarter",
        "body": (
            "Hello, I'm the account admin for Brightfield Analytics "
            "(account: ACC-11099). We currently have 12 seats on Team plan "
            "and need to add 3 more seats before March 31st (end of our fiscal "
            "quarter). The 3 new team members are: "
            "dev1@brightfield.io, dev2@brightfield.io, dev3@brightfield.io. "
            "Please prorate the cost. — James Park, VP Engineering"
        ),
        "customer_tier": "pro",
        "account_age_days": 820,
        "previous_tickets": 3,
        "attachments": [],
        "ground_truth": {
            "entities": {
                "contact_name": "James Park",
                "contact_role": "VP Engineering",
                "account_id": "ACC-11099",
                "company_name": "Brightfield Analytics",
                "current_seats": "12",
                "seats_to_add": "3",
                "deadline": "March 31st",
                "new_users": [
                    "dev1@brightfield.io",
                    "dev2@brightfield.io",
                    "dev3@brightfield.io",
                ],
            },
            "required_actions": [
                "add_seats",
                "send_prorated_invoice",
                "provision_new_users",
            ],
        },
    },
    {
        "ticket_id": "T2-004",
        "subject": "Data export stuck at 0% for 6 hours – export ID EXP-990021",
        "body": (
            "Our scheduled data export (ID: EXP-990021) has been stuck at 0% "
            "for over 6 hours. This export contains 90 days of transaction "
            "data for our compliance report due tomorrow. Account: ACC-30041, "
            "region: eu-west-1. We need this export completed or the raw data "
            "sent via secure link by 09:00 UTC tomorrow. If the export cannot "
            "be fixed, please escalate to engineering. — Priya Sharma"
        ),
        "customer_tier": "enterprise",
        "account_age_days": 650,
        "previous_tickets": 7,
        "attachments": [],
        "ground_truth": {
            "entities": {
                "contact_name": "Priya Sharma",
                "export_id": "EXP-990021",
                "account_id": "ACC-30041",
                "region": "eu-west-1",
                "data_range": "90 days",
                "deadline": "09:00 UTC tomorrow",
            },
            "required_actions": [
                "investigate_export_job",
                "fix_or_restart_export",
                "escalate_to_engineering",
            ],
        },
    },
    {
        "ticket_id": "T2-005",
        "subject": "GDPR deletion request for former employee account",
        "body": (
            "We need to permanently delete all data associated with a former "
            "employee's account as per GDPR Article 17 (right to erasure). "
            "The account to delete: user ID USR-88821, email "
            "j.doe@departed.co.uk. Our DPO is Claire Lambert "
            "(dpo@ourenterprise.com). Legal reference: GDPR-REQ-2024-03. "
            "Please confirm deletion in writing within 30 days. "
            "Org ID: ORG-7740."
        ),
        "customer_tier": "enterprise",
        "account_age_days": 1400,
        "previous_tickets": 2,
        "attachments": ["gdpr_deletion_form.pdf"],
        "ground_truth": {
            "entities": {
                "user_id": "USR-88821",
                "user_email": "j.doe@departed.co.uk",
                "org_id": "ORG-7740",
                "dpo_name": "Claire Lambert",
                "dpo_email": "dpo@ourenterprise.com",
                "legal_reference": "GDPR-REQ-2024-03",
                "legal_basis": "GDPR Article 17",
            },
            "required_actions": [
                "delete_user_data",
                "send_written_confirmation",
                "log_gdpr_request",
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# TASK 3 — Resolution Generation
#   Agent must submit a response_text + resolution_steps.
#   Ground truth contains required keywords and accepted resolution steps.
#   Scoring is deterministic (keyword matching + step coverage).
# ---------------------------------------------------------------------------

TASK3_TICKETS: List[Dict[str, Any]] = [
    {
        "ticket_id": "T3-001",
        "subject": "Cannot reset my password — reset email never arrives",
        "body": (
            "Hi, I've tried resetting my password 5 times today but the reset "
            "email never arrives. I've checked my spam folder. My account email "
            "is user@example.com. I need access to my account urgently as I "
            "have a demo with a client in 2 hours."
        ),
        "customer_tier": "pro",
        "account_age_days": 280,
        "previous_tickets": 0,
        "attachments": [],
        "ground_truth": {
            "required_keywords": [
                "password",
                "reset",
                "email",
                "spam",
                "whitelist",
            ],
            "required_resolution_steps": [
                "verify_email_delivery",
                "check_spam_filters",
                "manual_password_reset",
                "follow_up_confirmation",
            ],
            "tone_requirements": {
                "must_apologize": True,
                "must_acknowledge_urgency": True,
                "must_provide_timeline": True,
            },
            "expected_response_length_min": 80,
        },
    },
    {
        "ticket_id": "T3-002",
        "subject": "Webhook payloads stopped arriving after updating secret key",
        "body": (
            "We updated our webhook secret key yesterday in the dashboard and "
            "now none of our webhook endpoints are receiving payloads. Our "
            "endpoint is https://hooks.our-app.com/receive. We've verified the "
            "endpoint is up and responding 200. It seems like the new secret "
            "is not being used to sign payloads. Account: ACC-50039."
        ),
        "customer_tier": "pro",
        "account_age_days": 510,
        "previous_tickets": 4,
        "attachments": [],
        "ground_truth": {
            "required_keywords": [
                "webhook",
                "secret",
                "signature",
                "HMAC",
                "regenerate",
            ],
            "required_resolution_steps": [
                "verify_webhook_config",
                "regenerate_webhook_secret",
                "update_endpoint_verification",
                "test_delivery",
            ],
            "tone_requirements": {
                "must_apologize": False,
                "must_acknowledge_urgency": False,
                "must_provide_timeline": True,
            },
            "expected_response_length_min": 100,
        },
    },
    {
        "ticket_id": "T3-003",
        "subject": "Annual invoice shows wrong VAT number — urgent for accounting",
        "body": (
            "Our annual invoice (INV-2024-ANN-00567) shows VAT number "
            "DE123456789 but our correct VAT number is DE987654321. This "
            "invoice needs to be corrected immediately as our accounting team "
            "needs it for Q1 closing (deadline: end of today). "
            "Company: Müller GmbH, Account: ACC-20987."
        ),
        "customer_tier": "enterprise",
        "account_age_days": 1200,
        "previous_tickets": 8,
        "attachments": [],
        "ground_truth": {
            "required_keywords": [
                "VAT",
                "invoice",
                "corrected",
                "accounting",
                "reissue",
            ],
            "required_resolution_steps": [
                "update_vat_number",
                "reissue_invoice",
                "send_corrected_invoice",
                "confirm_receipt",
            ],
            "tone_requirements": {
                "must_apologize": True,
                "must_acknowledge_urgency": True,
                "must_provide_timeline": True,
            },
            "expected_response_length_min": 80,
        },
    },
    {
        "ticket_id": "T3-004",
        "subject": "Team member accidentally deleted our production dataset",
        "body": (
            "One of our team members accidentally deleted dataset "
            "DS-PROD-77. This dataset had 6 months of customer analytics data "
            "that we haven't backed up externally. Is there any way to restore "
            "it? We are on Enterprise plan. Org: ORG-3302. Please help "
            "immediately — this data is critical for our quarterly review."
        ),
        "customer_tier": "enterprise",
        "account_age_days": 960,
        "previous_tickets": 2,
        "attachments": [],
        "ground_truth": {
            "required_keywords": [
                "restore",
                "backup",
                "recovery",
                "dataset",
                "retention",
            ],
            "required_resolution_steps": [
                "check_retention_policy",
                "attempt_data_recovery",
                "escalate_to_engineering",
                "provide_recovery_status",
            ],
            "tone_requirements": {
                "must_apologize": True,
                "must_acknowledge_urgency": True,
                "must_provide_timeline": True,
            },
            "expected_response_length_min": 100,
        },
    },
    {
        "ticket_id": "T3-005",
        "subject": "Need formal SLA documentation for enterprise procurement",
        "body": (
            "Our procurement team is finalizing contracts and requires formal "
            "SLA documentation for your Enterprise plan. Specifically we need: "
            "uptime guarantee percentage, support response time commitments, "
            "incident severity classification, and escalation procedures. "
            "Contact: procurement@bigcorp.io. Org: ORG-8855."
        ),
        "customer_tier": "enterprise",
        "account_age_days": 30,
        "previous_tickets": 0,
        "attachments": [],
        "ground_truth": {
            "required_keywords": [
                "SLA",
                "uptime",
                "response time",
                "documentation",
                "enterprise",
            ],
            "required_resolution_steps": [
                "provide_sla_document",
                "highlight_enterprise_terms",
                "connect_with_account_manager",
                "confirm_receipt",
            ],
            "tone_requirements": {
                "must_apologize": False,
                "must_acknowledge_urgency": False,
                "must_provide_timeline": True,
            },
            "expected_response_length_min": 80,
        },
    },
]

# ---------------------------------------------------------------------------
# Task-level metadata
# ---------------------------------------------------------------------------

TASK_META: Dict[str, Dict[str, Any]] = {
    "task1": {
        "name": "Ticket Classification",
        "description": (
            "Classify each support ticket by category and priority. "
            "Correct classification routes the ticket to the right team and "
            "sets appropriate response SLAs."
        ),
        "difficulty": "easy",
        "max_steps": 3,
        "tickets": TASK1_TICKETS,
        "available_actions": ["classify", "submit"],
    },
    "task2": {
        "name": "Information Extraction",
        "description": (
            "Extract structured entities (IDs, names, amounts, dates) from "
            "tickets and identify the list of required actions needed to "
            "resolve each case. Accuracy drives downstream automation."
        ),
        "difficulty": "medium",
        "max_steps": 5,
        "tickets": TASK2_TICKETS,
        "available_actions": ["extract", "submit"],
    },
    "task3": {
        "name": "Resolution Generation",
        "description": (
            "Generate a complete, professional customer-facing response "
            "plus an ordered list of resolution steps for each ticket. "
            "Responses are graded on keyword coverage (with lexical diversity checks), "
            "ordered resolution-step coverage, tone adherence, and minimum length."
        ),
        "difficulty": "hard",
        "max_steps": 8,
        "tickets": TASK3_TICKETS,
        "available_actions": ["respond", "submit"],
    },
}


def get_tickets(task_id: str) -> List[Dict[str, Any]]:
    """Return the ticket list for a given task."""
    return TASK_META[task_id]["tickets"]


def get_task_meta(task_id: str) -> Dict[str, Any]:
    """Return task metadata (without ticket ground truth exposed to agent)."""
    meta = dict(TASK_META[task_id])
    # Strip ground_truth from tickets before returning to agents
    safe_tickets = []
    for t in meta["tickets"]:
        safe_t = {k: v for k, v in t.items() if k != "ground_truth"}
        safe_tickets.append(safe_t)
    meta["tickets"] = safe_tickets
    return meta