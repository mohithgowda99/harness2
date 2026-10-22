"""Continuum exploratory probe benchmark. Standard library; zero paid services.

Rules are hidden from agents. Inference is done on unqueried cases only.
This is NOT neural training, agent consciousness, AGI, or evidence of transfer.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

INPUTS = tuple((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))


def rules():
    return {
        "and": lambda x: int(all(x)),
        "or": lambda x: int(any(x)),
        "xor": lambda x: x[0] ^ x[1] ^ x[2],
        "majority": lambda x: int(sum(x) >= 2),
        "nand": lambda x: int(not all(x)),
        "nor": lambda x: int(not any(x)),
        "first": lambda x: x[0],
        "second": lambda x: x[1],
        "third": lambda x: x[2],
        "not_first": lambda x: 1 - x[0],
        "pair_xor": lambda x: x[0] ^ x[1],
        "pair_and": lambda x: x[1] & x[2],
    }


def survivors(observations):
    """Candidate rules consistent with observed pairs. No hidden labels accessed."""
    return [f for f in rules().values() if all(f(x) == y for x, y in observations)]


def choose_probe(observations, strategy, rng):
    observed = {x for x, _ in observations}
    remaining = [x for x in INPUTS if x not in observed]
    if not remaining:
        raise ValueError("probe budget exhausted")
    if strategy == "random":
        return rng.choice(remaining)
    if strategy != "active":
        raise ValueError(f"unknown strategy: {strategy}")
    candidates = survivors(observations)
    if not candidates:
        # Detect model-class mismatch; do not fake certainty.
        return rng.choice(remaining)
    rng.shuffle(remaining)  # reproducible tie-breaking
    def disagreement(x):
        p = sum(f(x) for f in candidates) / len(candidates)
        return 0 if p in (0, 1) else -p * math.log2(p) - (1-p) * math.log2(1-p)
    return max(remaining, key=disagreement)


def symbolic_predict(observations, heldout):
    candidates = survivors(observations)
    if not candidates:
        return [0 for _ in heldout], True
    return [int(sum(f(x) for f in candidates) * 2 >= len(candidates)) for x in heldout], False


def ollama_predict(observations, heldout, model):
    """Optional real local model. No paid API and no outbound endpoint setting."""
    instruction = (
        "Infer the deterministic rule mapping 3 binary input bits to one output bit. "
        "Predict values for queries. Return ONLY a JSON array of 0/1 integers "
        f"with exactly {len(heldout)} elements."
    )
    payload = {
        "model": model, "stream": False,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps({"examples": observations, "queries": heldout})},
        ],
        "options": {"temperature": 0, "num_predict": 120},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        reply = json.load(response)["message"]["content"].strip()
    # Some models wrap JSON in a fenced code block; retain strict output validation.
    if reply.startswith("```"):
        reply = reply.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(reply)
    if not isinstance(result, list) or len(result) != len(heldout) or any(
        type(y) is not int or y not in (0, 1) for y in result
    ):
        raise ValueError(f"Malformed Ollama prediction: {reply[:160]}")
    return result


def run(trials=24, budget=3, seed=42, backend="symbolic", model="qwen3:0.6b"):
    if trials <= 0 or not 0 < budget < len(INPUTS):
        raise ValueError("trials must be positive and budget between 1 and 7")
    if backend not in ("symbolic", "ollama"):
        raise ValueError("backend must be symbolic or ollama")
    rule_names = tuple(rules())
    experiments = {}
    for strategy in ("random", "active"):
        total_correct = 0
        total_tested = 0
        mismatches = 0
        episodes = []
        for trial in range(trials):
            # Same hidden world on both strategies; no world label enters agent context.
            world_rng = random.Random(seed + trial * 199)
            name = rule_names[world_rng.randrange(len(rule_names))]
            world = rules()[name]
            rng = random.Random(seed * 1709 + trial * 17 + (strategy == "active"))
            obs = []
            for _ in range(budget):
                x = choose_probe(obs, strategy, rng)
                obs.append((x, world(x)))
            unknown = [x for x in INPUTS if x not in {v for v, _ in obs}]
            if backend == "ollama":
                pred = ollama_predict(obs, unknown, model)
                miss = not survivors(obs)
            else:
                pred, miss = symbolic_predict(obs, unknown)
            labels = [world(x) for x in unknown]  # EVALUATOR ONLY
            total_correct += sum(a == b for a, b in zip(pred, labels))
            total_tested += len(unknown)
            mismatches += int(miss)
            episodes.append({"trial": trial, "observations": obs,
                             "queries": unknown, "predictions": pred,
                             "heldout_correct": sum(a == b for a, b in zip(pred, labels)),
                             "heldout_count": len(unknown)})
        experiments[strategy] = {
            "accuracy": round(total_correct / total_tested, 4),
            "correct": total_correct, "tested": total_tested,
            "model_class_mismatch_trials": mismatches,
            "episodes": episodes,
        }
    return {"version": "continuum-probe-v1", "backend": backend,
            "model": model if backend == "ollama" else None,
            "seed": seed, "trials": trials, "query_budget_per_task": budget,
            "evaluation": "held-out inputs, same worlds and query budgets",
            "limitations": "Synthetic known rule grammar; no weight updates or cross-family transfer.",
            "strategies": experiments}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["symbolic", "ollama"], default="symbolic")
    p.add_argument("--model", default="qwen3:0.6b")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = run(args.trials, args.budget, args.seed, args.backend, args.model)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: {key: v[key] for key in ("accuracy", "correct", "tested", "model_class_mismatch_trials")}
               for k, v in result["strategies"].items()}
    print(json.dumps({"backend": result["backend"], "trials": args.trials,
                      "budget": args.budget, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
