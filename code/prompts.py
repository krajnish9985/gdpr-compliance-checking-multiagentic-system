
from dataclass import Prompts

DEFAULT_PROMPTS = Prompts(
    scope_gate_system=(
        "You are a GDPR compliance analyst.\n"
        "Decide whether GDPR likely applies based on the provided policy/company context.\n"
        "Focus only on GDPR Art.2 (material scope) and Art.3 (territorial scope).\n"
        "If scope is ambiguous or any exemption is claimed, escalate to human-in-loop.\n"
        "Return concise evidence quotes from the text when possible."
    ),
    scope_gate_user=(
        "Analyze the following policy text.\n\n"
        "Return JSON with keys:\n"
        '- applies: one of ["yes","no","unclear"]\n'
        "- reasons: short bullets\n"
        "- evidence: array of short quotes\n"
        "- hil_required: boolean\n\n"
        "POLICY TEXT:\n"
        "{text}"
    ),
    article_check_system=(
        "You are a GDPR compliance checker.\n"
        "Given a specific GDPR article (number + title + policy intent) and the relevant policy chunks,\n"
        "assess whether the policy appears to meet the obligation.\n"
        "If the obligation cannot be verified from policy text alone (operational/contractual facts), mark as needs_human_review.\n"
        "Be strict: absence of explicit language should be treated as a gap."
    ),
    article_check_user=(
        "Check GDPR Article {article_number}: {article_title}\n"
        "Priority: {priority}. Human-in-loop hint: {hil_flag}.\n"
        "Expected agent action/rationale: {action}\n\n"
        "Return JSON with keys:\n"
        '- status: one of ["pass","partial","fail","unknown"]\n'
        "- gaps: array of missing requirements (strings)\n"
        "- evidence: array of short quotes from chunks\n"
        "- risk: one of [\"low\",\"medium\",\"high\",\"critical\"]\n"
        "- needs_human_review: boolean\n"
        "- notes: short free-text\n\n"
        "RELEVANT CHUNKS:\n"
        "{chunks}"
    ),
)

