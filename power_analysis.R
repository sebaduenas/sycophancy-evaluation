# ============================================================
# Power / precision for the cross-lingual sycophancy INTERACTION
# Outcome: capitulation (binary), CONDITIONAL on correct-at-T1.
# Estimand: language x condition interaction on log-odds of capitulation.
# Model (matches analysis.R): capitulation ~ condition*language + model + (1|item)
# Tool: simr  (simulation-based power for GLMMs; Green & MacLeod 2016).
# ============================================================
# install.packages(c("lme4","simr"))
library(lme4); library(simr)
set.seed(2025)

## ---- 1. INPUTS you choose (probabilities are CONDITIONAL on T1-correct) ----
p_en_ctrl <- 0.10   # baseline capitulation, neutral re-check, English
p_en_pres <- 0.30   # under social pressure, English
p_es_ctrl <- 0.12   # baseline, Spanish (small language main effect)
p_es_pres <- 0.45   # under pressure, Spanish  <-- this gap drives the interaction
tau       <- 0.7    # SD of item random intercept (logit scale); item heterogeneity
p_T1      <- 0.85   # P(correct at T1): capitulation only defined then
n_models  <- 2
k_reps    <- 1      # samples per cell (1 = temperature 0; >1 REQUIRES temp > 0)
K_items   <- 50     # INFORMATIVE items per cell (your current bank = 50)

logit <- qlogis
b0 <- logit(p_en_ctrl)
bC <- logit(p_en_pres) - b0                                   # pressure effect (en)
bL <- logit(p_es_ctrl) - b0                                   # language main effect
bI <- (logit(p_es_pres) - logit(p_es_ctrl)) - bC              # INTERACTION (log-odds)
cat(sprintf("Assumed interaction: log-odds = %.3f  (OR = %.2f)\n", bI, exp(bI)))

## ---- 2. Design skeleton (the informative, T1-correct rows) ----
## NOTE: K_items = informative items. Raw items needed = K_items / p_T1.
skeleton <- expand.grid(
  item      = factor(seq_len(K_items)),
  language  = factor(c("en", "es")),
  condition = factor(c("control", "pressure")),
  model     = factor(seq_len(n_models)),
  rep       = seq_len(k_reps)
)

## ---- 3. Assemble the GLMM with assumed parameters ----
## Interaction is written LAST so the fixef vector order is unambiguous:
## (Intercept), conditionpressure, languagees, model2.., conditionpressure:languagees
fe <- c(b0, bC, bL, rep(0.15, n_models - 1), bI)   # model main effect = 0.15 (tweak if needed)
model <- makeGlmer(
  y ~ condition + language + model + condition:language + (1 | item),
  family = binomial, fixef = fe, VarCorr = tau^2, data = skeleton
)
# If makeGlmer complains about VarCorr, use: VarCorr = list(item = matrix(tau^2))

## ---- 4. Power at your CURRENT N ----
sim_int <- powerSim(model,
  test = fixed("conditionpressure:languagees", "z"), nsim = 500)
print(sim_int)

## ---- 5. Power CURVE: sweep number of items ----
model_big <- extend(model, along = "item", n = 200)   # allow up to 200 items
pc <- powerCurve(model_big,
  test  = fixed("conditionpressure:languagees", "z"),
  along = "item", breaks = c(30, 50, 80, 120, 160, 200), nsim = 200)
print(pc); plot(pc)

## ---- 6. Replication lever ----
## Set k_reps <- 5 (and use temperature > 0 in the eval) and re-run 2-4.
## For a large cross-lingual gap, replication reaches high power with far fewer items.

## ---- 7. Precision (complement to significance) ----
## Fit one large simulated dataset and read the interaction OR + 95% CI width:
# d <- doSim(model); m <- doFit(d, model); exp(confint(m, method = "Wald")["conditionpressure:languagees", ])

## ===================== TAKEAWAYS =====================
## - The MAIN pressure effect is well-powered at K=50; the INTERACTION is the hard
##   target (interactions need ~4x the N of a same-size main effect; Gelman).
## - Rates are CONDITIONAL on T1-correct -> K here = informative items;
##   raw items needed = K / p_T1 (e.g., 80 informative / 0.85 ~ 95 raw).
## - Modest gap (OR~1.5): no audit-scale N detects it -> frame as ESTIMATION
##   (report interaction OR + 95% CI) and flag a larger study as future work.
## - Large gap (OR>=~3): ~80-100 items (k=1), or ~30-50 items with k=5 replicates,
##   reach 80% power. Buying power via replication (temp>0) is usually cheaper
##   than more items -- but only when responses are stochastic at temp>0.
## NB: a fast cross-check used cluster-robust logistic (GEE-style); it is slightly
## anticonservative with few item-clusters, so trust this glmer-based curve for
## final planning and pre-registration.
