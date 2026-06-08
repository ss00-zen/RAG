import os
import re
import json
from rag.logger import logger
from rag.llm import call_litellm


def _call_litellm(prompt):
    selected_model = os.getenv("LITELLM_MODEL")
    if not selected_model:
        raise EnvironmentError("LITELLM_MODEL must be set for query rewrite/expansion.")

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite or expand search queries for document retrieval. Preserve the meaning and keep the outputs concise. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    return call_litellm(
        messages,
        model=selected_model,
        temperature=0.2,
        max_tokens=512,
    )


def _clean_line(line):
    line = line.strip().strip('"“”')
    line = re.sub(r'^[\*\-\d\.\)\s]+', '', line).strip()
    return line


def _basic_valid(line):
    if not line:
        return False
    if len(line) < 5:
        return False
    if len(line) > 200:
        return False
    bad_prefixes = (
        "here are", "original query", "alternatively",
        "query:", "answer:", "explanation:"
    )
    if line.lower().startswith(bad_prefixes):
        return False
    return True


def rewrite_query(query):
    prompt = f"""
Rewrite the following search query to improve retrieval.
Preserve meaning. Keep it concise.
Return JSON only:
{{"query": "<rewritten query>"}}

Query: {query}
"""
    try:
        response = _call_litellm(prompt)
        data = json.loads(response)
        rewritten = _clean_line(data.get("query", ""))
        if _basic_valid(rewritten) and rewritten.lower() != query.lower():
            logger.info("Query rewritten from %r to %r", query, rewritten)
            return rewritten
    except Exception as e:
        logger.warning("Rewrite failed: %s", e)

    logger.info("Using original query")
    return query


def expand_query(query):
    prompt = f"""
Generate 3 diverse search queries for document retrieval.
Include:
1) close rewrite
2) keyword-style query
3) broader semantic variant

Return JSON only:
{{"queries": ["q1", "q2", "q3"]}}

Original query: {query}
"""
    try:
        response = _call_litellm(prompt)
        data = json.loads(response)
        queries = data.get("queries", [])
        sanitized = []
        for q in queries:
            q = _clean_line(q)
            if _basic_valid(q) and q.lower() != query.lower() and q not in sanitized:
                sanitized.append(q)

        if sanitized:
            result = [query] + sanitized[:3]
            logger.info("Expanded query into %s alternatives", len(result) - 1)
            return result
    except Exception as e:
        logger.warning("Expand failed: %s", e)

    logger.warning("Expand query returned no valid alternatives; falling back to original")
    return [query]