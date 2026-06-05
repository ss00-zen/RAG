import os
import re
import json
import requests
from rag.logger import logger


def _call_nvidia(prompt):
    api_key = os.getenv("NVIDIA_API_KEY")
    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


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
        response = _call_nvidia(prompt)
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
        response = _call_nvidia(prompt)
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