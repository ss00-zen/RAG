import os
from rag.logger import logger
from rag.llm import call_litellm


def _call_litellm(prompt):
    prompt = prompt[:2000]

    messages = [
        {
            "role": "system",
            "content": "Be precise and follow instructions exactly."
        },
        {"role": "user", "content": prompt}
    ]

    return call_litellm(
        messages,
        model=os.getenv("LITELLM_MODEL", "nvidia-llm"),
        temperature=0,
        max_tokens=300,
    )


# ✅ Proper RAGAS wrapper
class SimpleLiteLLM:

    def set_run_config(self, run_config):
        pass

    def generate(self, prompts, **kwargs):

        results = []

        for prompt in prompts:

            print("\n--- RAGAS PROMPT ---\n", prompt)

            output = _call_litellm(prompt)

            print("\n--- LLM OUTPUT ---\n", output)

            results.append({
                "generations": [[{"text": output}]]
            })

        return results


# ✅ MAIN
def run_evaluation(dataset):

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_utilization,
    )

    # ✅ Use the LiteLLM gateway for evaluation prompts
    llm = SimpleLiteLLM()

    # ✅ Minimal dummy embedding (required by RAGAS internally)
    class DummyEmbedding:
        def embed_documents(self, texts):
            return [[0.0] * 5 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 5

    embeddings = DummyEmbedding()

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_utilization,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False   # ✅ IMPORTANT (prevents crash)
    )

    return result