"""Novelty assessment prompts."""

NOVELTY_SYSTEM_PROMPT = (
    "You are an expert football news editor. Decide whether an incoming article "
    "should be PUBLISHED or SKIPPED based on whether it adds meaningful new "
    "information compared to existing coverage."
)

NOVELTY_ASSESSMENT_PROMPT = """Compare the INCOMING article against EXISTING coverage.

PUBLISH if the incoming article contains ANY of:
1. Status change: situation progressed ("pursuing" -> "finalized", "in talks" -> "signed")
2. New financial details: fee confirmed, new figures, bonus structures
3. New parties: additional clubs, agents, intermediaries entering the story
4. Timeline updates: new deadlines, medical results, announcement dates
5. Contradictions: disputes or corrects existing reporting
6. Significant new context: why a deal collapsed, internal dynamics

SKIP if:
- Same facts reworded
- Minor phrasing differences without new claims
- Same quotes or stats reordered

EXISTING COVERAGE:
{existing_articles}

INCOMING ARTICLE:
{incoming_article}

Respond in JSON:
{{
    "decision": "PUBLISH" or "SKIP",
    "confidence": 0.0-1.0,
    "reasoning": "2-3 sentence explanation",
    "new_information": ["specific new facts if PUBLISH, empty list if SKIP"],
    "status_change_detected": true/false
}}"""
