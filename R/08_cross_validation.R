# Evaluacion de precision de todos los modelos: (1) un holdout fijo por resolucion
# (14 dias horarios, 28 dias diarios) con MAPE/RMSE/MASE via fabletools::accuracy(),
# y (2) validacion cruzada de origen movil (rolling-origin, "stretch_tsibble") sobre
# la serie diaria, computacionalmente mas liviana que repetir la busqueda ARIMA
# horaria en cada origen -- se documenta explicitamente esta decision de alcance.
#
# Nota sobre el backtest de SARIMAX: los regresores exogenos (solar_mw, wind_mw) se
# usan con sus valores REALES conocidos del periodo de holdout (no un pronostico de
# ellos), aislando asi la habilidad de pronostico de la dinamica ARIMA de la demanda
# neta del error de un eventual pronostico meteorologico separado.

library(fable)
library(fabletools)
library(dplyr)
library(tsibble)

#' Evalua un conjunto de modelos ya ajustados contra un test set con accuracy().
#' `full_ts` debe ser la serie completa (train + test) -- MASE necesita el tramo de
#' entrenamiento para calcular el error de referencia del naive estacional, un
#' `test_ts` solo no basta (queda NaN si se le pasa solo el tramo de test).
evaluate_holdout <- function(fit, test_ts, full_ts) {
  fc <- fabletools::forecast(fit, new_data = test_ts)
  fabletools::accuracy(fc, full_ts) %>%
    select(.model, .type, MAPE, RMSE, MASE) %>%
    arrange(MAPE)
}

#' Corre validacion cruzada de origen movil (rolling-origin) sobre la serie diaria.
run_daily_rolling_cv <- function(daily_ts, init_size = 550, step = 14, horizon = 7) {
  cv_data <- daily_ts %>%
    stretch_tsibble(.init = init_size, .step = step)

  cv_fits <- cv_data %>%
    model(
      ets = ETS(demand_mw_mean),
      snaive_week = SNAIVE(demand_mw_mean ~ lag("week"))
    )

  cv_fc <- cv_fits %>% forecast(h = horizon)

  fabletools::accuracy(cv_fc, daily_ts) %>%
    select(.model, .type, MAPE, RMSE, MASE) %>%
    arrange(MAPE)
}

if (sys.nframe() == 0) {
  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)

  # ---- Holdout horario: ARIMA + Fourier vs SNAIVE ----
  hourly_ts <- readRDS("data/hourly_tsibble.rds")
  test_hourly <- readRDS("data/test_hourly_tsibble.rds")
  arima_fit <- readRDS("output/models/arima_fourier_fit.rds")

  cat("=== Holdout horario (14 dias) -- Demanda ===\n")
  arima_accuracy <- evaluate_holdout(arima_fit, test_hourly, hourly_ts)
  print(arima_accuracy)
  readr::write_csv(arima_accuracy, "output/tables/accuracy_hourly_demand.csv")

  # ---- Holdout horario: SARIMAX vs SNAIVE (demanda neta) ----
  sarimax_fit <- readRDS("output/models/sarimax_fit.rds")

  cat("\n=== Holdout horario (14 dias) -- Demanda neta (SARIMAX) ===\n")
  sarimax_accuracy <- evaluate_holdout(sarimax_fit, test_hourly, hourly_ts)
  print(sarimax_accuracy)
  readr::write_csv(sarimax_accuracy, "output/tables/accuracy_hourly_net_demand.csv")

  # ---- Holdout diario: ETS vs SNAIVE semanal ----
  daily_ts_full <- readRDS("data/daily_tsibble.rds")
  test_daily <- readRDS("data/test_daily_tsibble.rds")
  ets_fit <- readRDS("output/models/ets_fit.rds")

  cat("\n=== Holdout diario (28 dias) -- Demanda diaria promedio ===\n")
  ets_accuracy <- evaluate_holdout(ets_fit, test_daily, daily_ts_full)
  print(ets_accuracy)
  readr::write_csv(ets_accuracy, "output/tables/accuracy_daily_demand.csv")

  # ---- Validacion cruzada de origen movil (diaria) ----
  cat("\n=== Validacion cruzada de origen movil (diaria, horizonte 7 dias) ===\n")
  cv_accuracy <- run_daily_rolling_cv(daily_ts_full)
  print(cv_accuracy)
  readr::write_csv(cv_accuracy, "output/tables/accuracy_rolling_cv_daily.csv")
}
