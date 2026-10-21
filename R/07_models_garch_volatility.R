# Modelo GARCH(1,1) sobre los cambios horarios de la generacion eolica -- la variabilidad
# de la generacion renovable (rachas de viento, frentes) es precisamente lo que genera
# riesgo de estabilidad de red (rampas rapidas de demanda neta).
#
# Con la eolica anclada a viento real (ERA5), la desviacion del nivel respecto de su media
# horaria esta fuertemente autocorrelacionada (el viento persiste por horas), y un GARCH
# ajustado sobre ese nivel degenera (alpha1 ~ 0.92, beta1 ~ 0.07): esta modelando la
# persistencia del nivel, no la de la volatilidad. Por eso se ajusta sobre la primera
# diferencia (cambio hora a hora, la "rampa"), que es la innovacion casi no
# autocorrelacionada que un GARCH supone.

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
  sen_data <- readr::read_csv("data/sen_hourly.csv", show_col_types = FALSE)

  # Rampas eolicas: cambio de la generacion eolica de una hora a la siguiente (MW/h).
  wind_shock <- diff(sen_data$wind_mw)
  cat(sprintf("Autocorrelacion lag 1 -- nivel: %.3f | cambio horario: %.3f\n",
              acf(sen_data$wind_mw, plot = FALSE)$acf[2], acf(wind_shock, plot = FALSE)$acf[2]))

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
    "\nVolatilidad condicional de las rampas eolicas: %.1f MW/h (h+1) -> %.1f MW/h (h+48)\n",
    sigma_forecast[1], sigma_forecast[48]
  ))
}
