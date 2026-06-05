import os
import re
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def _clean_line(line):
    line = line.strip().strip('"“”')
    line = re.sub(r'^[\*\-\d\.\)\s]+', '', line).strip()
    return line


def _contains_original_terms(candidate, original_query):
    original_terms = {
        w.lower() for w in re.findall(r"\w+", original_query)
        if len(w) > 2
    }
    candidate_terms = {w.lower() for w in re.findall(r"\w+", candidate)}
    return bool(original_terms & candidate_terms)


def _is_redundant_query_line(candidate, original_query):
    normalized = candidate.lower()
    original_lower = original_query.lower().strip()
    if not original_lower:
        return False

    if normalized.count(original_lower) > 1:
        return True

    # Reject if the candidate contains the original query plus extra text but is still overly redundant.
    if original_lower in normalized and len(normalized) > len(original_lower) + 30:
        return True

    return False


def _is_valid_query_line(line, original_query):
    if not line:
        return False

    normalized = line.lower()
    invalid_prefixes = [
        "here are", "to improve retrieval", "rewrite this query", "generate 3", "return exactly",
        "original query", "alternatively", "this query", "the following", "question", "answer",
        "i can", "i will", "please", "if you", "otherwise",
        "query:" , "sql", "select", "from", "where", "insert", "update", "delete",
        "join", "table", "database", "description", "keywords", "content like"
    ]

    if any(normalized.startswith(prefix) for prefix in invalid_prefixes):
        return False

    if any(keyword in normalized for keyword in ["select ", " from ", " where ", " insert ", " update ", " delete "]):
        return False

    if _is_redundant_query_line(line, original_query):
        return False

    if len(line) < 15:
        return False

    if len(line) > 120 and "?" not in line:
        return False

    if not _contains_original_terms(line, original_query):
        return False

    return True


def _extract_query(generated, original_query):
    generated = generated.strip()
    if not generated:
        return original_query

    # Prefer an explicit query after a Query: label.
    for match in re.finditer(r'(?im)^query:\s*(.+)$', generated):
        candidate = _clean_line(match.group(1))
        if _is_valid_query_line(candidate, original_query):
            return candidate

    # Prefer a quoted query if present.
    quotes = re.findall(r'["“](.+?)["”]', generated)
    for quote in quotes:
        candidate = _clean_line(quote)
        if _is_valid_query_line(candidate, original_query):
            return candidate

    # Otherwise use the first valid non-boilerplate line.
    for line in generated.splitlines():
        candidate = _clean_line(line)
        if _is_valid_query_line(candidate, original_query):
            return candidate

    return original_query


def _sanitize_query(generated, original_query):
    candidate = _extract_query(generated, original_query)
    if candidate != original_query:
        logger.debug("Sanitized generated query. original=%r generated=%r result=%r", original_query, generated, candidate)
    return candidate


# ✅ Rewrite
def rewrite_query(query):
    prompt = f"""
Rewrite this query to improve retrieval. Return only the rewritten query on a single line with no explanation.

Query: {query}
"""
    response = _call_nvidia(prompt)
    rewritten = _sanitize_query(response, query)
    if rewritten != query:
        logger.info("Query rewritten from %r to %r", query, rewritten)
    else:
        logger.info("Query rewrite returned no improvement; using original query")
    return rewritten


# ✅ Expand
def expand_query(query):
    prompt = f"""
Generate 3 alternative search queries to retrieve relevant documents for the following query. Return exactly 3 alternatives, one per line, with no additional commentary.

Original query: {query}
"""
    response = _call_nvidia(prompt)
    expanded = [q.strip() for q in response.split("\n") if q.strip()]
    sanitized = []
    for q in expanded:
        sanitized_q = _sanitize_query(q, query)
        if sanitized_q and sanitized_q not in sanitized and sanitized_q != query:
            sanitized.append(sanitized_q)

    if not sanitized:
        logger.warning("Expand query returned no valid alternatives; falling back to original query")
        return [query]

    result = [query] + sanitized[:3]
    logger.info("Expanded query into %s alternatives", len(result) - 1)
    return result
