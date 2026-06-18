# =====================================================================
# Cross-lingual sycophancy: analysis of capitulation
# Outcome: capitulation (1 = correct at T1, incorrect at T2)
# Design: 2 (condition: control/pressure) x 2 (language: en/es), within-item
# Estimand: sycophantic capitulation = (pressure - control) contrast,
#           and whether it differs by language (interaction).
# =====================================================================
# install.packages(c("tidyverse","lme4","emmeans","broom.mixed"))
library(tidyverse)
library(lme4)
library(emmeans)
library(broom.mixed)

df <- read_csv("results.csv") %>%
  mutate(
    condition = factor(condition, levels = c("control", "pressure")),
    language  = factor(language,  levels = c("en", "es")),
    model     = factor(model),
    item      = factor(lang_pair_id)
  )

# ---------------------------------------------------------------------
# 0. Descriptive: capitulation rate per cell (the headline table)
# ---------------------------------------------------------------------
cell_tab <- df %>%
  group_by(model, language, condition) %>%
  summarise(
    n          = n(),
    capit_rate = mean(capitulation, na.rm = TRUE),
    se         = sqrt(capit_rate * (1 - capit_rate) / n),
    .groups    = "drop"
  )
print(cell_tab)

# Baseline comparison à la Fanous et al. (2025): two-proportion test,
# pressure vs control, pooled across language (sanity check vs the model below)
with(df, {
  p <- table(condition, capitulation)
  print(prop.test(p[, "1"], rowSums(p)))
})

# ---------------------------------------------------------------------
# 1. Mixed-effects logistic model
#    Random intercepts for item (psychometric/IRT-style partial pooling)
#    and model. This improves on the pooled z-tests in prior work by
#    accounting for item difficulty and repeated measures.
# ---------------------------------------------------------------------
m <- glmer(
  capitulation ~ condition * language + (1 | item) + (1 | model),
  data = df, family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)
summary(m)

# Odds ratios + 95% CI (Wald)
or_tab <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
print(or_tab)

# Likelihood-ratio test for the language x condition interaction
m_noint <- update(m, . ~ condition + language + (1 | item) + (1 | model))
print(anova(m_noint, m))   # is sycophancy effect language-dependent?

# ---------------------------------------------------------------------
# 2. Estimated marginal effects: predicted capitulation probability
#    per cell, and the pressure effect within each language
# ---------------------------------------------------------------------
emm <- emmeans(m, ~ condition | language, type = "response")
print(emm)
print(contrast(emm, "revpairwise"))          # pressure - control, per language
print(pairs(emmeans(m, ~ language | condition, type = "response")))  # es - en, per condition

# ---------------------------------------------------------------------
# 3. Item-level analysis (which items drive the effect)
# ---------------------------------------------------------------------
item_tab <- df %>%
  group_by(item, topic) %>%
  summarise(capit_rate = mean(capitulation), n = n(), .groups = "drop") %>%
  arrange(desc(capit_rate))
print(item_tab)

# ---------------------------------------------------------------------
# 4. Plot for the report
# ---------------------------------------------------------------------
p <- ggplot(cell_tab, aes(condition, capit_rate, fill = language)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_errorbar(aes(ymin = pmax(0, capit_rate - 1.96 * se),
                    ymax = capit_rate + 1.96 * se),
                position = position_dodge(0.8), width = 0.2) +
  facet_wrap(~ model) +
  scale_y_continuous("Capitulation rate", labels = scales::percent) +
  labs(x = "Challenge condition", fill = "Language",
       title = "Regressive sycophancy by language and challenge condition") +
  theme_minimal(base_size = 12)
ggsave("capitulation_plot.png", p, width = 8, height = 5, dpi = 150)

# ---------------------------------------------------------------------
# NOTE ON SPARSITY: if capitulation is rare, glmer may warn about
# singular fit / complete separation. Fallbacks:
#   - penalised likelihood: logistf::logistf (Firth)
#   - Bayesian: brms::brm(..., family = bernoulli) with weakly-informative
#     priors (e.g. normal(0, 1.5) on log-odds) -> stable estimates + credible
#     intervals, and a cleaner story for a regulator audience.
# Report CI WIDTH explicitly: this is an audit-scale pilot, not a population
# estimate. Pre-register N per cell to hit a target CI width before scaling.
# =====================================================================
