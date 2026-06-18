# Cross-Lingual Sycophancy Eval

A small, rigorous evaluation that measures **regressive sycophancy** (a model
abandoning a correct answer under social pressure) and tests whether it is
**worse in Spanish than in English**, a gap the literature flags and the EU AI
Office's multilingual mandate cares about. Built on UK AISI **Inspect**.

## Why this project
Published sycophancy work is overwhelmingly English-only, yet the EU operates in
24 official languages. If a model abandons correct answers more readily in one
language than another, then robustness measured in English doesn't transfer, and
that gap is invisible to English-only evaluation.

This project measures it directly. It pairs a multi-turn Inspect task (custom
solver and scorer) with a randomised within-item design that causally identifies
the pressure effect, a mixed-effects logistic model with item-level partial
pooling, and an English/Spanish item bank validated by a native speaker. The
write-up frames the result for a regulator audience, around EU AI Act Art. 5 and
Arts. 92 to 93.

## Files
- `sycophancy_eval.py`: the Inspect task (two-turn challenge, capitulation outcome).
- `data/items_en.jsonl`, `data/items_es.jsonl`: parallel English/Spanish item banks.
- `export_results.py`: flattens Inspect `.eval` logs into `results.csv`.
- `analysis.R`: mixed-effects model, odds ratios, interaction test, item analysis, plot.
- `analysis_single_model.R`: single-model variant (drops the model random effect).
- `analysis_confound.R`: robustness of the language effect to topic and stimulus-length adjustment.
- `report.md`: the regulator-facing write-up.

## Setup
```bash
pip install inspect_ai
# R: install.packages(c("tidyverse","lme4","emmeans","broom.mixed"))
```
Provide your model provider's API key in the environment (or a local `.env`)
before running.

## Run (the four cells)
```bash
cd sycophancy-eval
for L in en es; do for C in control pressure; do
  inspect eval sycophancy_eval.py@sycophancy \
    --model anthropic/claude-opus-4-8 -T language=$L -T condition=$C --epochs 5
done; done

inspect view                     # browse the logs interactively
python export_results.py         # produces results.csv
Rscript analysis_single_model.R  # console output + capitulation_plot.png
```
Offline structural check (no key): `inspect eval sycophancy_eval.py@sycophancy --model mockllm/model --limit 2`
(needs network for the tokenizer download; the logic is tested independently.)

## Design notes
- **The control condition** separates the challenge effect (plain second-guessing) from social pressure, so the contrast measures capitulation to a false belief rather than mere instability.
- **Within-item randomisation** identifies the effect causally, without confounding by item difficulty.
- **Mixed-effects** rather than pooled proportions: random item intercepts (psychometric partial pooling) account for difficulty and give a cleaner estimate.
- This proxies one mechanism behind Art. 5 manipulation. It is not a legal finding, and the report says so.

## Scope
One construct (regressive sycophancy), one new axis (language), done rigorously.
Larger extensions (24 languages, human subjects, higher-stakes domains,
adversarial multi-turn pressure) are noted as next steps in the report rather
than attempted here.
