# Evaluacion de precision de todos los modelos: (1) un holdout fijo por resolucion
# (14 dias horarios, 28 dias diarios) con MAPE/RMSE/MASE via fabletools::accuracy(),
# y (2) validacion cruzada de origen movil (rolling-origin, "stretch_tsibble") sobre
# la serie diaria Y (desde este cambio) tambien sobre la serie horaria, con un
# numero de origenes acotado por costo computacional -- se documenta explicitamente
# esta decision de alcance (ver run_hourly_rolling_cv() mas abajo).
#
# Nota sobre el backtest de SARIMAX: los regresores exogenos (solar_mw, wind_mw) se
# usan con sus valores REALES conocidos del periodo de holdout (no un pronostico de
# ellos), aislando asi la habilidad de pronostico de la dinamica ARIMA de la demanda
# neta del error de un eventual pronostico meteorologico separado.

library(fable)
library(fabletools)
library(forecast)
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

#' MAPE/RMSE/MASE manuales para un objeto `forecast::tbats` (no es un mable de
#' fable, asi que fabletools::accuracy() no aplica directamente). MASE usa el
#' error absoluto medio de la persistencia estacional (lag 24) sobre el propio
#' tramo de entrenamiento como referencia, la misma definicion que fabletools
#' usa internamente para series estacionales.
evaluate_tbats_holdout <- function(tbats_fit, train_demand, test_demand, horizon) {
  fc <- forecast::forecast(tbats_fit, h = horizon)
  pred <- as.numeric(fc$mean)
  actual <- test_demand

  mape <- mean(abs((actual - pred) / actual)) * 100
  rmse <- sqrt(mean((actual - pred)^2))

  naive_seasonal_error <- mean(abs(diff(train_demand, lag = 24)))
  mase <- mean(abs(actual - pred)) / naive_seasonal_error

  tibble::tibble(.model = "tbats", .type = "Test", MAPE = mape, RMSE = rmse, MASE = mase)
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

#' Validacion cruzada de origen movil sobre la serie HORARIA -- extension pedida
#' explicitamente sobre la version anterior de este proyecto, que la aplicaba
#' solo a la serie diaria "por costo computacional (reajustar ARIMA/SARIMAX
#' horario en cada origen es significativamente mas lento)". Se mantiene esa
#' decision de alcance, ahora acotada explicitamente: ventana de entrenamiento
#' FIJA (no creciente, a diferencia de stretch_tsibble) de `init_size` horas,
#' `n_origins` origenes espaciados `step` horas, horizonte de `horizon` horas --
#' calibrado para que el costo total (n_origins ajustes de ARIMA+Fourier
#' completos) sea de minutos, no horas, en esta maquina. Ventana fija (no
#' expansiva) deliberadamente: as[i] compara todos los origenes con la MISMA
#' cantidad de historia, la misma logica que motiva la ventana fija de
#' rolling_stability en el proyecto hermano en Python
#' (chile-energy-grid-forecasting), no una decision distinta entre los dos.
run_hourly_rolling_cv <- function(hourly_ts, init_size = 12000, step = 1000, horizon = 48, n_origins = 5) {
  n <- nrow(hourly_ts)
  results <- list()

  for (i in seq_len(n_origins)) {
    train_end <- init_size + (i - 1) * step
    test_end <- min(train_end + horizon, n)
    if (train_end + 1 > n) break

    train_window <- hourly_ts %>% slice((train_end - init_size + 1):train_end)
    test_window <- hourly_ts %>% slice((train_end + 1):test_end)
    if (nrow(test_window) == 0) break

    t0 <- Sys.time()
    fit <- train_window %>%
      model(arima_fourier = ARIMA(demand_mw ~ pdq(d = 1) + fourier(period = "day", K = 12) +
                                     fourier(period = "week", K = 2) + PDQ(0, 0, 0)))
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")

    fc <- fabletools::forecast(fit, new_data = test_window)
    acc <- fabletools::accuracy(fc, hourly_ts) %>%
      select(.model, .type, MAPE, RMSE, MASE) %>%
      mutate(origin = i, train_end_row = train_end, tiempo_ajuste_s = round(elapsed, 1))

    cat(sprintf("  Origen %d/%d (fin de train en fila %d, %.0fs de ajuste): MAPE=%.2f%% RMSE=%.1f\n",
                i, n_origins, train_end, elapsed, acc$MAPE, acc$RMSE))
    results[[i]] <- acc
  }

  dplyr::bind_rows(results)
}

if (sys.nframe() == 0) {
  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)

  # ---- Holdout horario: ARIMA + Fourier vs SNAIVE ----
  hourly_ts <- readRDS("data/hourly_tsibble.rds")
  test_hourly <- readRDS("data/test_hourly_tsibble.rds")
  train_hourly <- readRDS("data/train_hourly_tsibble.rds")
  arima_fit <- readRDS("output/models/arima_fourier_fit.rds")

  cat("=== Holdout horario (14 dias) -- Demanda ===\n")
  arima_accuracy <- evaluate_holdout(arima_fit, test_hourly, hourly_ts)
  print(arima_accuracy)

  cat("\n=== Holdout horario (14 dias) -- Demanda, TBATS ===\n")
  tbats_fit <- readRDS("output/models/tbats_fit.rds")
  tbats_accuracy <- evaluate_tbats_holdout(
    tbats_fit, train_hourly$demand_mw, test_hourly$demand_mw, horizon = nrow(test_hourly)
  )
  print(tbats_accuracy)

  arima_accuracy <- bind_rows(arima_accuracy, tbats_accuracy) %>% arrange(MAPE)
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

  # ---- Validacion cruzada de origen movil (horaria, ARIMA + Fourier K=12) ----
  cat("\n=== Validacion cruzada de origen movil (horaria, ventana fija, horizonte 48h) ===\n")
  cv_accuracy_hourly <- run_hourly_rolling_cv(hourly_ts)
  print(cv_accuracy_hourly %>% select(.model, .type, MAPE, RMSE, MASE, origin))
  readr::write_csv(cv_accuracy_hourly, "output/tables/accuracy_rolling_cv_hourly.csv")
}
