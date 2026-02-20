"""Reranker prompt."""

RERANKER_PROMPT = """You are a relevance assessor for a football article/news retrieval system.

Rate how relevant the CANDIDATE article is to the QUERY article on a scale of 0.0 to 1.0:

- 1.0: Same story, same entities, directly comparable
- 0.7-0.9: Same topic/player, closely related context
- 0.4-0.6: Some overlap (same club, same transfer window) but different story
- 0.1-0.3: Loosely related (same league, same position)
- 0.0: Completely unrelated

QUERY:
{query}

CANDIDATE:
{candidate}

Respond in JSON:
{{"relevance": 0.0-1.0, "reason": "brief explanation"}}"""
