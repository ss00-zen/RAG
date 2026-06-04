from datasets import Dataset

def build_dataset(logs):

    return Dataset.from_dict({
        "question": [l["question"] for l in logs],
        "answer": [l["answer"] for l in logs],
        "contexts": [l["contexts"] for l in logs],
    })
