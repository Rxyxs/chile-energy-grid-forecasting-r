# Figuras de la serie horaria REAL y del pronostico con machine learning.
# Lee data/real/sen_hourly_real.csv.gz y las tablas de R/11_ml_day_ahead.R.
#
#   Rscript R/12_real_figures.R

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readr)
  library(ggplot2)
  library(jsonlite)
})
Sys.setlocale("LC_TIME", "C")   # ejes de fecha en ingles, como el resto de las figuras

FIG <- "output/figures"
TAB <- "output/tables"
TZ_CL <- "America/Santiago"
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)

save <- function(p, name, w, h) {
  ggsave(file.path(FIG, name), p, width = w, height = h, dpi = 150, bg = "white")
  cat("  ->", name, "\n")
}

d <- read_csv("data/real/sen_hourly_real.csv.gz", show_col_types = FALSE) |>
  mutate(local = with_tz(ts_utc, TZ_CL), date = as.Date(local, tz = TZ_CL), hour = hour(local))

# 1. Ocho anos hora a hora: cada columna es un dia, cada fila una hora.
blackout <- d |> filter(anomaly, date == as.Date("2025-02-25"))
p1 <- ggplot(d, aes(date, hour, fill = gen_mw / 1000)) +
  geom_raster() +
  scale_fill_viridis_c(name = "GW", option = "magma", na.value = "#27c1d9") +
  scale_y_continuous(breaks = c(0, 6, 12, 18, 23), expand = c(0, 0)) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y", expand = c(0, 0)) +
  annotate("label", x = as.Date("2025-02-25"), y = 6, label = "25 Feb 2025\nnational blackout",
           size = 2.7, hjust = 1.05, fill = "white") +
  labs(title = "Chile's grid, every hour from 2018 to 2025 (real gross generation, SEN)",
       subtitle = "Each column is a day, each row an hour of the local day. Growth, winter peaks, the evening peak fading and a midday one rising after 2022.",
       x = NULL, y = "Hour of day (local)") +
  theme_minimal(base_size = 10) + theme(panel.grid = element_blank(), plot.title = element_text(face = "bold"))
save(p1, "real_sen_hourly_heatmap.png", 11, 4.6)

# 2. Perfil diario medio por ano.
prof <- read_csv(file.path(TAB, "real_daily_profile_by_year.csv"), show_col_types = FALSE)
p2 <- ggplot(prof, aes(hour, mean_mw / 1000, colour = factor(year))) +
  geom_line(linewidth = 0.9) +
  scale_colour_viridis_d(name = NULL, option = "viridis", end = 0.92) +
  scale_x_continuous(breaks = seq(0, 23, 3)) +
  labs(title = "The daily peak moved from 9 pm to early afternoon after 2022",
       subtitle = "Mean gross generation by local hour, one line per year",
       x = "Hour of day (local)", y = "GW") +
  theme_minimal(base_size = 10) + theme(plot.title = element_text(face = "bold"))
save(p2, "real_daily_profile_by_year.png", 8, 4.6)

# 3. Dos semanas de prueba (2025): real, P50 del objetivo elegido en 2024 con su
#    banda P10-P90 calibrada por CQR, y naive semanal.
summ <- fromJSON(file.path(TAB, "ml_day_ahead_summary.json"))
tag <- if (summ$chosen_target == "relative") "lgb_rel_" else "lgb_"
pr <- read_csv(file.path(TAB, "ml_day_ahead_predictions_2025.csv.gz"), show_col_types = FALSE) |>
  mutate(local = with_tz(ts_utc, TZ_CL), p50 = .data[[paste0(tag, "q50")]],
         lo = .data[[paste0(tag, "cqr_lo")]], hi = .data[[paste0(tag, "cqr_hi")]])
win <- pr |> filter(local >= as.POSIXct("2025-06-16", tz = TZ_CL), local < as.POSIXct("2025-06-30", tz = TZ_CL))
p3 <- ggplot(win, aes(local)) +
  geom_ribbon(aes(ymin = lo / 1000, ymax = hi / 1000), fill = "#1f4e79", alpha = 0.18) +
  geom_line(aes(y = naive_168 / 1000, colour = "Same hour last week"), linewidth = 0.5, linetype = "22") +
  geom_line(aes(y = p50 / 1000, colour = "LightGBM P50 (band: P10-P90, conformal)"), linewidth = 0.8) +
  geom_line(aes(y = actual / 1000, colour = "Actual"), linewidth = 0.8) +
  scale_colour_manual(name = NULL, values = c("Actual" = "black", "LightGBM P50 (band: P10-P90, conformal)" = "#1f4e79",
                                              "Same hour last week" = "#c0392b")) +
  labs(title = "Day-ahead forecast on unseen 2025 data, two winter weeks",
       subtitle = "Every input is at least 48 hours old; the model never saw 2025",
       x = NULL, y = "GW") +
  theme_minimal(base_size = 10) + theme(legend.position = "top", plot.title = element_text(face = "bold"))
save(p3, "ml_forecast_two_weeks.png", 10, 4.6)

# 4. Comparacion de modelos y perdidas en 2025.
tab <- read_csv(file.path(TAB, "ml_day_ahead_metrics.csv"), show_col_types = FALSE) |>
  filter(!grepl("trainonly", model))
loss <- c(l2 = "L2", l1 = "L1", huber = "Huber", q50 = "quantile 0.5", asym_l2 = "asymmetric L2 (under x2)")
lab_of <- function(m) {
  if (m == "naive_168") return("Naive: same hour last week")
  if (m == "naive_4w") return("Naive: mean of last 4 weeks")
  if (m == "stl_ets") return("STL + ETS (classical, refit daily)")
  rel <- grepl("^lgb_rel_", m)
  sprintf("LightGBM, %s, %s target", loss[sub("^lgb_(rel_)?", "", m)], if (rel) "relative" else "MW")
}
tab$label <- vapply(tab$model, lab_of, "")
tab$label <- factor(tab$label, levels = tab$label[order(-tab$mape)])
tab$group <- ifelse(grepl("^lgb_rel_", tab$model), "relative", ifelse(grepl("^lgb_", tab$model), "absolute", "baseline"))
long <- bind_rows(tab |> transmute(label, group, metric = "MAPE, %", value = mape),
                  tab |> transmute(label, group, metric = "Operational cost (under-forecast x2), MW", value = cost))
p4 <- ggplot(long, aes(value, label, fill = group)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = ifelse(value < 20, sprintf("%.2f", value), sprintf("%.0f", value))), hjust = -0.1, size = 3) +
  facet_wrap(~metric, scales = "free_x") +
  scale_fill_manual(values = c(relative = "#1f4e79", absolute = "#6e8ca0", baseline = "#c9ced4"), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(title = "Test year 2025: gradient boosting against classical and naive forecasts",
       subtitle = "Same features for every loss; hyperparameters, number of trees and target form chosen on 2024",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) + theme(plot.title = element_text(face = "bold"))
save(p4, "ml_model_comparison.png", 12, 6)

# 5. Validacion cruzada por anos.
cv <- read_csv(file.path(TAB, "ml_yearly_cv.csv"), show_col_types = FALSE)
cvl <- bind_rows(cv |> transmute(test_year, model = "LightGBM, MW target", mape = absolute_mape),
                 cv |> transmute(test_year, model = "LightGBM, relative target", mape = relative_mape),
                 cv |> transmute(test_year, model = "Same hour last week", mape = naive_mape))
p5 <- ggplot(cvl, aes(factor(test_year), mape, fill = model)) +
  geom_col(position = position_dodge(0.8), width = 0.75) +
  geom_text(aes(label = sprintf("%.2f", mape)), position = position_dodge(0.8), vjust = -0.4, size = 2.7) +
  scale_fill_manual(values = c("LightGBM, MW target" = "#6e8ca0", "LightGBM, relative target" = "#1f4e79",
                               "Same hour last week" = "#c9ced4"), name = NULL) +
  labs(title = "Rolling-origin validation: train up to Y-2, choose trees on Y-1, test on Y",
       x = "Test year", y = "MAPE, %") +
  theme_minimal(base_size = 10) + theme(legend.position = "top", plot.title = element_text(face = "bold"))
save(p5, "ml_yearly_cv.png", 8, 4.2)

# 6. El archivo cambio de definicion: suma de las horas contra el total mensual oficial.
hm <- d |> mutate(ym = format(local, "%Y-%m")) |> group_by(ym) |>
  summarise(hourly_gwh = sum(gen_mw, na.rm = TRUE) / 1000, .groups = "drop")
off <- read_csv("data/real/sen_generacion_mensual.csv", show_col_types = FALSE) |>
  transmute(ym = month, official_gwh = total_mwh / 1000)
gap <- inner_join(hm, off, by = "ym") |>
  mutate(gap_pct = 100 * (hourly_gwh / official_gwh - 1), month = as.Date(paste0(ym, "-01")))
write_csv(gap, file.path(TAB, "cen_hourly_vs_official_monthly.csv"))
p6 <- ggplot(gap, aes(month, gap_pct, fill = month >= as.Date("2025-01-01"))) +
  geom_col(width = 25) +
  geom_hline(yintercept = 0, colour = "grey30", linewidth = 0.3) +
  scale_fill_manual(values = c(`FALSE` = "#6e8ca0", `TRUE` = "#c0392b"), labels = c("2018-2024 sheets", "2025 sheet"), name = NULL) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title = "The hourly file changed definition in 2025",
       subtitle = "Sum of the hourly values against the official monthly total: they disagree in 2018 and again through 2023-2024; the 2025 sheet matches it exactly",
       x = NULL, y = "Hourly sum vs. official monthly, %") +
  theme_minimal(base_size = 10) + theme(legend.position = "top", plot.title = element_text(face = "bold"))
save(p6, "real_file_definition_break.png", 10, 4.2)
