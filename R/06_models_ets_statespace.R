# Modelo de espacio de estados ETS (Error-Tendencia-Estacionalidad) sobre la demanda
# diaria agregada. ETS es una familia de modelos de espacio de estados innovadores
# (innovations state space models) -- no un ARIMA disfrazado -- y aqui se aplica a
# la resolucion diaria porque maneja de forma nativa una sola estacionalidad
# (semanal, periodo 7), a diferencia de la demanda horaria que tiene dos periodos
# estacionales simultaneos (mejor abordados via Fourier + ARIMA en 05).

library(fable)
library(fabletools)
library(dplyr)

#' Ajusta ETS (busqueda automatica de componentes error/tendencia/estacionalidad)
#' y un SNAIVE semanal como baseline sobre la demanda diaria promedio.
fit_ets_daily <- function(train_ts) {
  train_ts %>%
    model(
      ets = ETS(demand_mw_mean),
      snaive_week = SNAIVE(demand_mw_mean ~ lag("week"))
    )
}

if (sys.nframe() == 0) {
  daily_ts <- readRDS("data/daily_tsibble.rds")

  horizonte_test_dias <- 28  # ultimas 4 semanas como holdout
  n <- nrow(daily_ts)
  train_ts <- daily_ts %>% slice(1:(n - horizonte_test_dias))
  test_ts <- daily_ts %>% slice((n - horizonte_test_dias + 1):n)

  cat("Ajustando ETS (demanda diaria promedio)...\n")
  ets_fit <- fit_ets_daily(train_ts)

  cat("\nComponentes seleccionados automaticamente:\n")
  print(ets_fit %>% select(ets) %>% fabletools::report())

  cat("\nCriterios de informacion:\n")
  print(fabletools::glance(ets_fit) %>% select(.model, AIC, AICc, BIC))

  dir.create("output/models", showWarnings = FALSE, recursive = TRUE)
  saveRDS(ets_fit, "output/models/ets_fit.rds")
  saveRDS(train_ts, "data/train_daily_tsibble.rds")
  saveRDS(test_ts, "data/test_daily_tsibble.rds")

  cat(sprintf("\nTrain: %d filas, Test (holdout): %d filas\n", nrow(train_ts), nrow(test_ts)))
}
