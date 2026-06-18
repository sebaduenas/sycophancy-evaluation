"""
Read Inspect eval logs (.eval) and flatten to a tidy CSV (one row per item x cell)
for analysis in R. Run after you have executed the four cells of the eval:

  inspect eval sycophancy_eval.py@sycophancy --model <M> -T language=en -T condition=control
  inspect eval sycophancy_eval.py@sycophancy --model <M> -T language=en -T condition=pressure
  inspect eval sycophancy_eval.py@sycophancy --model <M> -T language=es -T condition=control
  inspect eval sycophancy_eval.py@sycophancy --model <M> -T language=es -T condition=pressure
  (repeat for each model)

  python export_results.py            # reads ./logs, writes ./results.csv
"""
import csv
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

LOG_DIR = Path("logs")
OUT = Path("results.csv")

FIELDS = [
    "model", "item_id", "epoch", "lang_pair_id", "language", "condition", "topic",
    "t1", "t2", "gold", "t1_correct", "t2_correct", "capitulation", "flipped",
]


def main() -> None:
    rows = []
    for info in list_eval_logs(str(LOG_DIR)):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples:
            continue
        model = log.eval.model
        for s in log.samples:
            if not s.scores:
                continue
            score = next(iter(s.scores.values()))
            val = score.value if isinstance(score.value, dict) else {}
            md = score.metadata or {}
            rows.append({
                "model": model,
                "item_id": s.id,
                "epoch": getattr(s, "epoch", 1),
                "lang_pair_id": md.get("lang_pair_id"),
                "language": md.get("language"),
                "condition": md.get("condition"),
                "topic": md.get("topic"),
                "t1": md.get("t1"),
                "t2": md.get("t2"),
                "gold": md.get("gold"),
                "t1_correct": val.get("t1_correct"),
                "t2_correct": val.get("t2_correct"),
                "capitulation": val.get("capitulation"),
                "flipped": val.get("flipped"),
            })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
