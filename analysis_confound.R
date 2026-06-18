# =====================================================================
# Is the Spanish capitulation gap a language effect, or a topic / prose-
# difficulty confound? Language is observational here (unlike the
# randomised challenge), so the marginal EN/ES gap could ride on the
# topics and the longer prose the Spanish items carry. This script
# measures those confounders and adjusts for them.
#
#   1. Adjust the language effect for topic.
#   2. Adjust it for stimulus length (a proxy for translation-induced
#      difficulty: Spanish prose runs longer).
#   3. Item-level: does how much an item lengthens in translation
#      predict its capitulation gap?
# =====================================================================
library(tidyverse)
library(lme4)
library(broom.mixed)
library(jsonlite)

# --- stimulus length per item, per language, from the item banks ---
read_lengths <- function(path) {
  stream_in(file(path), verbose = FALSE) %>%
    transmute(item_id = id,
              qlen = nchar(question),
              qwords = lengths(strsplit(question, "\\s+")))
}
lens <- bind_rows(read_lengths("data/items_en.jsonl"),
                  read_lengths("data/items_es.jsonl"))

df <- read_csv("results.csv", show_col_types = FALSE) %>%
  left_join(lens, by = "item_id") %>%
  mutate(language  = factor(language,  levels = c("en", "es")),
         condition = factor(condition, levels = c("control", "pressure")),
         topic     = factor(topic),
         item      = factor(lang_pair_id),
         qlen_z    = as.numeric(scale(qlen)))

ctl <- glmerControl(optimizer = "bobyqa")
lang_or <- function(m, lab) {
  t <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE, exponentiate = TRUE) %>%
       filter(term == "languagees")
  cat(sprintf("%-26s OR(es vs en) = %.2f  CI[%.2f, %.2f]  p = %.4f\n",
              lab, t$estimate, t$conf.low, t$conf.high, t$p.value))
}

cat("Spanish stimulus length vs English (mean chars):\n")
print(df %>% group_by(language) %>% summarise(mean_chars = mean(qlen), .groups = "drop"))

cat("\n=== Language effect (Spanish vs English) under successive adjustments ===\n")
lang_or(glmer(capitulation ~ language + condition + (1 | item),
              df, binomial, control = ctl), "1. unadjusted")
lang_or(glmer(capitulation ~ language + condition + topic + (1 | item),
              df, binomial, control = ctl), "2. + topic")
lang_or(glmer(capitulation ~ language + condition + qlen_z + (1 | item),
              df, binomial, control = ctl), "3. + stimulus length")
lang_or(glmer(capitulation ~ language + condition + topic + qlen_z + (1 | item),
              df, binomial, control = ctl), "4. + topic + length")

cat("\n=== Item level: capitulation gap (ES - EN) vs length inflation (ES / EN) ===\n")
it <- df %>%
  group_by(item, language) %>%
  summarise(cap = mean(capitulation), qlen = first(qlen), .groups = "drop") %>%
  pivot_wider(names_from = language, values_from = c(cap, qlen)) %>%
  mutate(gap = cap_es - cap_en, lenratio = qlen_es / qlen_en)
ct <- suppressWarnings(cor.test(it$gap, it$lenratio, method = "spearman"))
cat(sprintf("Spearman rho = %.3f  p = %.3f  (n = %d items)\n",
            ct$estimate, ct$p.value, nrow(it)))
cat("rho near zero / non-significant => translation length inflation does not\n",
    "explain the capitulation gap, i.e. the effect is not merely a length artefact.\n")
