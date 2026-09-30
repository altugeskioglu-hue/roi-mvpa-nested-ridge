# ===========================================================================
# Permutation null distributions of cross-validated R^2
#
# Reads the outputs of scripts/05_run_permutation_test.py and draws one panel
# per model:
#   solid red     observed CV R^2
#   dashed dark   null 95th percentile
#   dotted grey   R^2 = 0
#
# Requires: ggplot2 (>= 3.4)
# Usage:    Rscript figures/plot_permutation_null.R results/permutation
# ===========================================================================

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
perm_dir <- if (length(args) >= 1) args[1] else "results/permutation"

nulls <- read.csv(file.path(perm_dir, "permutation_null_CV_R2.csv"))
psum  <- read.csv(file.path(perm_dir, "permutation_summary.csv"))

BLUE <- "#4A7EBB"; RED <- "#B0362A"; DARK <- "#333333"; GREY <- "#8C8C8C"
key_levels <- c("Observed CV R2", "Null 95th percentile", "R2 = 0")

models <- psum$model
d <- do.call(rbind, lapply(models, function(m)
  data.frame(model = m, r2 = nulls[[paste0(m, "_CV_R2")]])))

lines <- do.call(rbind, lapply(models, function(m) {
  s <- psum[psum$model == m, ]
  data.frame(model = m,
             x = c(s$observed_CV_R2, s$null_95th_percentile_CV_R2, 0),
             key = factor(key_levels, levels = key_levels))
}))

labels <- do.call(rbind, lapply(models, function(m) {
  s <- psum[psum$model == m, ]
  data.frame(model = m, label = sprintf(
    "Observed CV R2 = %.3f\nNominal p = %.3f; FWER p = %.3f",
    s$observed_CV_R2, s$p_nominal, s$p_FWER_max_stat))
}))

fig <- ggplot(d, aes(x = r2)) +
  geom_histogram(bins = 45, fill = BLUE, colour = "white", linewidth = 0.2) +
  geom_vline(data = lines, aes(xintercept = x, colour = key, linetype = key),
             linewidth = 0.7) +
  geom_label(data = labels, aes(x = -Inf, y = Inf, label = label),
             hjust = -0.03, vjust = 1.1, size = 3, fill = "white",
             label.size = 0, inherit.aes = FALSE) +
  facet_wrap(~ model, nrow = 1) +
  scale_colour_manual(values = setNames(c(RED, DARK, GREY), key_levels)) +
  scale_linetype_manual(values = setNames(c("solid", "dashed", "dotted"),
                                          key_levels)) +
  # Headroom above the tallest bar keeps the p-value label off the data.
  scale_y_continuous(expand = expansion(mult = c(0, 0.35))) +
  labs(x = expression("Permuted cross-validated " * R^2), y = "Frequency") +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0, face = "bold"))

out <- file.path(perm_dir, "permutation_null.png")
ggsave(out, fig, width = 4.5 * length(models), height = 3.6, dpi = 300)
cat("Saved", out, "\n")
