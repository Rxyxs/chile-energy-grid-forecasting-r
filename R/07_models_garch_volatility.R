# Modelo GARCH(1,1) sobre los shocks de generacion eolica -- la variabilidad de la
# generacion renovable (nubosidad, rachas de viento) es precisamente lo que genera
# riesgo de estabilidad de red (rampas rapidas de demanda neta cuando el sol/viento
# caen). El componente eolico se SIMULO explicitamente como un proceso GARCH(1,1) en
# 01_generate_synthetic_data.R, asi que aqui se verifica que rugarch puede recuperar
# ese clustering de volatilidad real a partir de los datos observados.

library(rugarch)
library(dplyr)

#' Ajusta un GARCH(1,1) estandar (media constante, innovaciones normales) sobre una
#' serie de retornos/shocks.
fit_garch11 <- function(shock_series) {
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "norm"
  )
  ugarchfit(spec = spec, data = shock_series, solver = "hybrid")
}

if (sys.nframe() == 0) {
  sen_data <- readr::read_csv("data/sen_hourly_synthetic.csv", show_col_types = FALSE)

  # Los shocks eolicos (desviacion del nivel base horario esperado) son la serie de
  # interes -- reconstruimos una aproximacion restando la componente diurna promedio.
  wind_by_hour <- sen_data %>%
    mutate(hour_of_day = lubridate::hour(datetime)) %>%
    group_by(hour_of_day) %>%
    mutate(wind_hour_mean = mean(wind_mw)) %>%
    ungroup()

  wind_shock <- wind_by_hour$wind_mw - wind_by_hour$wind_hour_mean

  cat("Ajustando GARCH(1,1) sobre los shocks de generacion eolica...\n")
  garch_fit <- fit_garch11(wind_shock)

  cat("\nCoeficientes estimados:\n")
  print(rugarch::coef(garch_fit))

  cat("\nPersistencia (alpha1 + beta1):\n")
  coefs <- rugarch::coef(garch_fit)
  persistencia <- unname(coefs["alpha1"] + coefs["beta1"])
  cat(sprintf("%.4f (valores cercanos a 1 = alta persistencia/clustering de volatilidad)\n", persistencia))

  dir.create("output/models", showWarnings = FALSE, recursive = TRUE)
  saveRDS(garch_fit, "output/models/garch_wind_fit.rds")

  # Pronostico de volatilidad condicional a 48 horas
  garch_forecast <- rugarch::ugarchforecast(garch_fit, n.ahead = 48)
  sigma_forecast <- as.numeric(rugarch::sigma(garch_forecast))

  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(
    tibble::tibble(hora_adelante = 1:48, sigma_pronosticada_mw = round(sigma_forecast, 2)),
    "output/tables/garch_volatility_forecast.csv"
  )

  cat(sprintf(
    "\nVolatilidad condicional pronosticada: %.1f MW (h+1) -> %.1f MW (h+48)\n",
    sigma_forecast[1], sigma_forecast[48]
  ))
}
