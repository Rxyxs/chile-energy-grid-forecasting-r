# Modelos ARIMA (demanda) y SARIMAX (demanda neta con generacion solar/eolica como
# regresores exogenos) sobre la serie horaria, usando terminos de Fourier para
# capturar la doble estacionalidad (diaria + semanal) en vez de un SARIMA estacional
# clasico con periodo 168 (computacionalmente inviable a esta frecuencia).
#
# Encuadre SARIMAX: en operacion real, el pronostico de generacion solar/eolica viene
# de un modelo meteorologico independiente y se trata como insumo exogeno conocido al
# pronosticar la demanda neta a corto plazo -- por eso solar_mw/wind_mw entran como
# regresores exogenos, no como parte de la dinamica ARIMA en si.
#
# NOTA METODOLOGICA: el orden de diferenciación `d` se fija explícitamente en 1,
# informado por las pruebas ADF/KPSS de 03_stationarity_tests.R (ambas series de
# nivel no son estacionarias, la primera diferencia sí lo es). Esto también evita
# un bug real encontrado en la búsqueda automática de `d` de fable 0.5.0 sobre esta
# serie (falla con un error interno sin mensaje); dejar `d` fijo y buscar p/q
# automáticamente funciona correctamente y además es más defendible
# metodológicamente que confiar en una búsqueda automática como caja negra.

library(fable)
library(fabletools)
library(dplyr)

#' ARIMA sobre demanda horaria con terminos de Fourier (estacionalidad diaria + semanal).
#' `d = 1` fijo (ver nota metodologica arriba); p, q se buscan automaticamente por AICc.
fit_arima_fourier <- function(train_ts) {
  train_ts %>%
    model(
      arima_fourier = ARIMA(demand_mw ~ pdq(d = 1) + fourier(period = "day", K = 4) +
                               fourier(period = "week", K = 2) + PDQ(0, 0, 0)),
      snaive_day = SNAIVE(demand_mw ~ lag("day"))
    )
}

#' SARIMAX sobre demanda neta horaria con solar/eolica como regresores exogenos.
#' `d = 1` fijo por el mismo motivo que fit_arima_fourier().
fit_sarimax_net_demand <- function(train_ts) {
  train_ts %>%
    model(
      sarimax = ARIMA(net_demand_mw ~ pdq(d = 1) + solar_mw + wind_mw +
                         fourier(period = "day", K = 4) +
                         fourier(period = "week", K = 2) + PDQ(0, 0, 0)),
      snaive_net = SNAIVE(net_demand_mw ~ lag("day"))
    )
}

if (sys.nframe() == 0) {
  hourly_ts <- readRDS("data/hourly_tsibble.rds")

  horizonte_test_horas <- 14 * 24  # ultimos 14 dias como holdout
  n <- nrow(hourly_ts)
  train_ts <- hourly_ts %>% slice(1:(n - horizonte_test_horas))
  test_ts <- hourly_ts %>% slice((n - horizonte_test_horas + 1):n)

  cat("Ajustando ARIMA + Fourier (demanda horaria)...\n")
  arima_fit <- fit_arima_fourier(train_ts)
  print(fabletools::glance(arima_fit) %>% select(.model, AIC, AICc, BIC))

  cat("\nAjustando SARIMAX (demanda neta con solar/eolica exogenas)...\n")
  sarimax_fit <- fit_sarimax_net_demand(train_ts)
  print(fabletools::glance(sarimax_fit) %>% select(.model, AIC, AICc, BIC))

  dir.create("output/models", showWarnings = FALSE, recursive = TRUE)
  saveRDS(arima_fit, "output/models/arima_fourier_fit.rds")
  saveRDS(sarimax_fit, "output/models/sarimax_fit.rds")
  saveRDS(train_ts, "data/train_hourly_tsibble.rds")
  saveRDS(test_ts, "data/test_hourly_tsibble.rds")

  cat(sprintf("\nTrain: %d filas, Test (holdout): %d filas\n", nrow(train_ts), nrow(test_ts)))
}
