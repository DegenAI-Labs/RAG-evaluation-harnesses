import evaluate as hf_evaluate

_compute = None


def _get_compute():
    global _compute
    if _compute is None:
        _compute = hf_evaluate.load("code_eval")
    return _compute


def pass_at_k(references: list, predictions: list, k: list = None):
    """Per-problem pass@k from HF `code_eval` (references + candidate completions)."""
    if k is None:
        k = [1]
    if isinstance(k, int):
        k = [k]
    res = _get_compute().compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    out = res[0]
    if isinstance(out, dict):
        key = f"pass@{k[0]}"
        if key not in out:
            key = next(iter(out))
        return float(out[key])
    return float(out)


def build_predictions(resps: list[list], docs: list[dict]) -> list[list]:
    """Prefix HumanEval completions with the prompt; pass through loglikelihood tuples (perplexity pair)."""
    out = []
    for resp, doc in zip(resps, docs):
        if resp and isinstance(resp[0], tuple):
            out.append(resp)
            continue
        out.append([doc["prompt"] + r for r in resp])
    return out


def build_predictions_instruct(
    resps: list[list], docs: list[dict]
) -> list[list]:
    out = []
    for resp, doc in zip(resps, docs):
        if resp and isinstance(resp[0], tuple):
            out.append(resp)
            continue
        out.append(
            [
                doc["prompt"]
                + (r if r.find("```") == -1 else r[: r.find("```")])
                for r in resp
            ]
        )
    return out
