# Reproduction guide

Steps to reproduce the evaluation and analysis from scratch.

## 1. Environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install inspect_ai
```
R packages for the analysis:
```r
install.packages(c("tidyverse", "lme4", "emmeans", "broom.mixed"))
```

## 2. API key
Provide your model provider's API key via the environment or a local `.env`
file (auto-loaded by Inspect). The key is required for the model runs; the
offline structural check below needs none.

## 3. Smoke test (optional)
One model, three items, to confirm the wiring:
```bash
inspect eval sycophancy_eval.py@sycophancy --model anthropic/claude-opus-4-8 \
  -T language=en -T condition=pressure --limit 3
```
Offline structural check (no key): add `--model mockllm/model --limit 2`.

## 4. Run the four cells
```bash
for L in en es; do for C in control pressure; do
  inspect eval sycophancy_eval.py@sycophancy --model anthropic/claude-opus-4-8 \
    -T language=$L -T condition=$C --epochs 5
done; done
```
Each cell writes an `.eval` log to `logs/`. The five epochs per item provide
replication; run-to-run variation supplies the within-cell variance. Inspect's
on-screen metric is item-averaged, so the valid analysis is the R model over
`results.csv`.

## 5. Export
```bash
python export_results.py   # logs/ to results.csv (one row per item x epoch x cell)
```

## 6. Analyse (R)
```bash
Rscript analysis_single_model.R   # console tables + capitulation_plot.png
```
What to read, in order:
1. Cell table: capitulation rate by language x condition.
2. glmer odds ratios with 95% confidence intervals.
3. `anova(m_noint, m)`: the language x condition interaction (the key test).
4. emmeans: predicted capitulation probability per cell.
5. `capitulation_plot.png`.

`analysis.R` is the two-model specification (it adds a model random effect); use
it when more than one model is present. With a single model that term is
degenerate, so `analysis_single_model.R` drops it.

## Common issues
| Symptom | Fix |
|---|---|
| `No module named inspect_ai` | Activate the venv (step 1). |
| `401` / authentication | API key not loaded (step 2). |
| `model not found` | Use a current model id for the provider. |
| `grouping factors must have > 1 sampled level` (R) | A single model is present; use `analysis_single_model.R`. |
