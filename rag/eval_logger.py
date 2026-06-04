import json
import os
from datetime import datetime
from datasets import Dataset

LOG_FILE = "eval_logs.jsonl"


def log_interaction(query, answer, docs):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": query,
        "answer": answer,
        "contexts": [d.page_content for d in docs]
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ✅ ✅ ADD THIS FUNCTION BELOW 👇

def load_logs(limit=None):
    """
    Load logs from file (optionally last N records)
    """

    if not os.path.exists(LOG_FILE):
        return []

    logs = []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            logs.append(json.loads(line))

    # ✅ return last N logs if limit is given
    if limit:
        return logs[-limit:]

    return logs




def build_dataset(logs):

    return Dataset.from_dict({
        "question": [l["question"] for l in logs],
        "answer": [l["answer"] for l in logs],
        "contexts": [l["contexts"] for l in logs],
    })
