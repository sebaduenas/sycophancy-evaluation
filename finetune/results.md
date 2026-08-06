# Results — Spanish-only resistance fine-tune

Generated from `results.json`. Qwen2.5-1.5B-Instruct, LoRA r=16 on q/k/v/o,
240 **Spanish-only** conversations, 2 epochs (120 steps), final loss 0.0001.
Evaluated with the items and templates of `sycophancy-evaluation`, 5 samples per item.
Run locally on an Apple M1 (MPS), not on Colab.

## What was trained

The original plan was to **induce** sycophancy. It was scrapped after measuring the
baseline: the model already capitulated **100%** of the time in Spanish under pressure —
no headroom left to show an increase. The intervention was inverted instead: teach the
model to **hold** its correct answer, and measure whether that resistance crosses into
the language it never saw.

## Capitulation under pressure (user asserts a wrong option)

| set | what it is | pre | post | delta |
|---|---|---|---|---|
| ES-held | 20 ES items never seen in training | 100.0% | 0.0% | -100.0% |
| EN-held | the same 20 items, in English | 89.9% | 45.6% | -44.3% |
| EN-match | the 30 trained items, but in English | 88.8% | 50.0% | -38.8% |

## Control condition (neutral re-check, asserting nothing)

| set | pre | post | delta |
|---|---|---|---|
| ES-held | 92.5% | 1.7% | -90.8% |
| EN-held | 65.0% | 91.8% | +26.8% |
| EN-match | 60.7% | 82.8% | +22.1% |

## Turn-1 accuracy (capability, before any challenge)

Pressure-condition rows (control gives the same within sampling noise:
ES-held 67.0 → 59.0; EN-held 80.0 → 61.0; EN-match 96.7 → 77.3).

| set | pre | post | delta |
|---|---|---|---|
| ES-held | 69.0% | 58.0% | -11.0% |
| EN-held | 79.0% | 68.0% | -11.0% |
| EN-match | 95.3% | 76.0% | -19.3% |

## Reading

1. **In Spanish the patch works and generalises.** On items it never saw: 100% → 0% under
   pressure and 92.5% → 1.7% in control. This is not memorisation of the training set.
2. **It crosses into English only halfway.** Under pressure it drops from ~90% to ~46-50%:
   half the benefit crosses the language boundary, half does not.
3. **And in control English gets WORSE:** 65% → 91.8%. Training stability in Spanish made the
   model *less* stable in English against a plain "are you sure?". Collateral damage, not noise.
4. **The effect travels by language, not by content.** EN-held (46%) and EN-match (50%) are
   nearly identical, even though EN-match is exactly the trained content, translated. If the
   effect travelled by content, EN-match should look like ES-held (0%). It does not.

## Limits — stated before anyone asks

- **Selection bias.** Capitulation is computed only over items answered correctly at turn 1,
  and the fine-tune lowered that accuracy across every set. The set being conditioned on is
  not the same before and after, so the deltas are not a clean causal contrast.
- **Capability cost.** Turn-1 accuracy fell consistently. The patch buys stability and pays
  for it in performance.
- **Overfitting.** Loss reached 0.0001; the model most likely emits the trained phrase almost
  deterministically in Spanish. The resistance is real on unseen items, but the surface form
  of the answer is memorised.
- **One model, one seed, no confidence intervals.** This is a demonstration, not a paper.
- No correction for multiple comparisons and no mixed-effects model, unlike the original eval.

## Reproducing

From `finetune/`. Items are read from `../data/` by default (override with `DATA=`);
outputs are written next to the script (override with `OUT=`).

```bash
uv venv venv && uv pip install --python ./venv/bin/python torch transformers peft accelerate datasets
MODEL=Qwen/Qwen2.5-1.5B-Instruct DTYPE=float16 MAXNEW=12 \
  BATCH=20 K=5 N_MATCH=30 ./venv/bin/python run_local.py full
```

This runs the baseline and the post phase in a single pass. `REUSE_BASELINE=1` reuses a
previous `baseline.json` instead of re-measuring the `pre` phase; none is shipped in the
repo, so on a clean run the variable does nothing.

`python3 test_logic.py` validates, with no GPU and no dependencies, the splits, the
train/held-out non-contamination, the letter parser, and that the training set teaches the
correct letter in both conditions. It includes a guard that fails if `run_local.py` changes
that construction and the test stops mirroring it.

Performance note: on MPS you must pad to a **fixed shape**. With per-batch padding there were
61 distinct shapes and MPS recompiled kernels on every step — 234 s/step against 6.3 s/step. 37×.
