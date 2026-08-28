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
#
# NOTA METODOLOGICA 2 -- K=12 en el Fourier diario, y por que no un termino
# estacional PDQ(period=24) ni D=1 (diferenciacion estacional en 24): el modelo
# original (K=4 diario) dejaba un pico de autocorrelacion residual notable en el
# lag 24 (ACF~0.41, ver reports/02_Residual_Autocorrelation_Fourier.Rmd para el
# diagnostico completo). Se probaron 5 alternativas antes de fijar esta:
#   1. K=4 -> K=12 diario (Nyquist, base completa para un ciclo de 24 horas):
#      ACF(lag24) baja de 0.413 a 0.373 -- mejora real pero parcial.
#   2. Aumentar K semanal (2 -> 6/12/20): EMPEORA el ACF(lag24) (hasta 0.469) --
#      descartado, no es la via.
#   3. log(demand_mw) en vez de nivel: sin mejora (ACF(lag24)=0.402).
#   4. PDQ(P=1,D=0,Q=1,period=24) o D=1 explicito en 24, sin Fourier diario:
#      fable::ARIMA() falla con "no se encontro un modelo ARIMA apropiado"
#      (raices caracteristicas numericamente inestables) -- la MISMA clase de
#      falla ya documentada en la nota metodologica de arriba sobre el bug de
#      busqueda automatica de `d`, ahora tambien en la busqueda estacional a
#      periodo 24. Un limite real de fable::ARIMA() en esta serie, no un error
#      de especificacion.
# Verificacion independiente: el residuo "verdadero" (demand_mw menos el
# componente determinista EXACTO usado por el generador -- tendencia, dia de
# semana, estacionalidad anual, forma diaria) tiene ACF(lag24)=0.004 -- practi-
# camente ruido blanco. Esto confirma que el proceso generador NO tiene
# autocorrelacion genuina en 24 horas: el pico residual es enteramente un
# artefacto de la maquinaria "regresion con errores ARIMA" de fable sobre esta
# serie, no una propiedad real de los datos. K=12 diario es la mejora mas
# grande lograble dentro de esa maquinaria sin toparse con el bug de arriba;
# TBATS (fit_tbats_demand(), un framework de espacio de estados enteramente
# distinto, sin regresores de Fourier ni diferenciacion) logra una reduccion
# mayor (ACF(lag24)=0.289) precisamente porque no depende de esa maquinaria.

library(fable)
library(fabletools)
library(forecast)
library(dplyr)

#' ARIMA sobre demanda horaria con terminos de Fourier (estacionalidad diaria + semanal).
#' `d = 1` fijo (ver nota metodologica arriba); p, q se buscan automaticamente por AICc.
#' K=12 en el termino diario (base de Fourier completa para periodo 24 -- ver nota
#' metodologica 2 sobre por que K=12 y no una alternativa estacional).
fit_arima_fourier <- function(train_ts) {
  train_ts %>%
    model(
      arima_fourier = ARIMA(demand_mw ~ pdq(d = 1) + fourier(period = "day", K = 12) +
                               fourier(period = "week", K = 2) + PDQ(0, 0, 0)),
      snaive_day = SNAIVE(demand_mw ~ lag("day"))
    )
}

#' SARIMAX sobre demanda neta horaria con solar/eolica como regresores exogenos.
#' `d = 1` fijo por el mismo motivo que fit_arima_fourier(); K=12 diario por la
#' misma nota metodologica 2.
fit_sarimax_net_demand <- function(train_ts) {
  train_ts %>%
    model(
      sarimax = ARIMA(net_demand_mw ~ pdq(d = 1) + solar_mw + wind_mw +
                         fourier(period = "day", K = 12) +
                         fourier(period = "week", K = 2) + PDQ(0, 0, 0)),
      snaive_net = SNAIVE(net_demand_mw ~ lag("day"))
    )
}

#' TBATS (Trigonometric, Box-Cox, ARMA errors, Trend, Seasonal -- De Livera,
#' Hyndman & Snyder, 2011) sobre demanda horaria, con estacionalidad multiple
#' (diaria=24h, semanal=168h) declarada explicitamente via `msts`. A diferencia
#' de ARIMA+Fourier, TBATS modela cada componente estacional como un ESTADO que
#' evoluciona en el tiempo (no una regresion de amplitud fija) y estima su
#' propia transformacion Box-Cox y su propio termino ARMA(p,q) de corrección de
#' errores -- un mecanismo estructuralmente distinto al de fable::ARIMA(), que
#' es precisamente por qué reduce el ACF(lag24) donde la via Fourier no puede
#' (ver nota metodologica 2 arriba). No soporta regresores exogenos (no hay
#' analogo SARIMAX aqui), asi que se ajusta solo sobre demand_mw, no net_demand_mw.
#' Nota de costo computacional: ~7-8 minutos sobre ~2 años de datos horarios en
#' esta maquina -- notablemente mas lento que ARIMA+Fourier (~2 min).
fit_tbats_demand <- function(train_ts) {
  demand_msts <- forecast::msts(train_ts$demand_mw, seasonal.periods = c(24, 168))
  forecast::tbats(demand_msts, use.parallel = FALSE)
}

if (sys.nframe() == 0) {
  hourly_ts <- readRDS("data/hourly_tsibble.rds")

  horizonte_test_horas <- 14 * 24  # ultimos 14 dias como holdout
  n <- nrow(hourly_ts)
  train_ts <- hourly_ts %>% slice(1:(n - horizonte_test_horas))
  test_ts <- hourly_ts %>% slice((n - horizonte_test_horas + 1):n)

  cat("Ajustando ARIMA + Fourier (demanda horaria, K=12 diario)...\n")
  arima_fit <- fit_arima_fourier(train_ts)
  print(fabletools::glance(arima_fit) %>% select(.model, AIC, AICc, BIC))

  cat("\nAjustando SARIMAX (demanda neta con solar/eolica exogenas, K=12 diario)...\n")
  sarimax_fit <- fit_sarimax_net_demand(train_ts)
  print(fabletools::glance(sarimax_fit) %>% select(.model, AIC, AICc, BIC))

  cat("\nAjustando TBATS (demanda horaria, esto tarda varios minutos)...\n")
  tbats_fit <- fit_tbats_demand(train_ts)
  print(tbats_fit)

  dir.create("output/models", showWarnings = FALSE, recursive = TRUE)
  saveRDS(arima_fit, "output/models/arima_fourier_fit.rds")
  saveRDS(sarimax_fit, "output/models/sarimax_fit.rds")
  saveRDS(tbats_fit, "output/models/tbats_fit.rds")
  saveRDS(train_ts, "data/train_hourly_tsibble.rds")
  saveRDS(test_ts, "data/test_hourly_tsibble.rds")

  cat(sprintf("\nTrain: %d filas, Test (holdout): %d filas\n", nrow(train_ts), nrow(test_ts)))
}
