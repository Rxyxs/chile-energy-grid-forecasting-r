# Graficos finales: diagnostico de residuos (ACF + histograma), pronosticos con
# bandas de incertidumbre, y volatilidad GARCH pronosticada.

library(fable)
library(fabletools)
library(feasts)
library(ggtime)
library(ggplot2)
library(dplyr)

#' Diagnostico de residuos de un modelo (ACF + histograma + serie de tiempo).
plot_residual_diagnostics <- function(fit, model_name, path) {
  p <- fit %>%
    select(all_of(model_name)) %>%
    feasts::gg_tsresiduals() +
    theme_minimal(base_size = 10)
  ggsave(path, plot = p, width = 9, height = 7, dpi = 150)
  p
}

#' Grafico de pronostico con banda de incertidumbre: pronostica `horizon` periodos
#' hacia adelante desde el modelo ya ajustado, y muestra solo las ultimas
#' `display_history` observaciones del set de ENTRENAMIENTO como contexto (para que
#' el pronostico sea visible en vez de perderse en 2 años de datos horarios, y para
#' que la historia mostrada no se traslape con el periodo pronosticado).
plot_forecast_with_bands <- function(fit, model_name, train_ts, response_col, horizon, display_history, path, titulo) {
  fc <- fit %>% select(all_of(model_name)) %>% fabletools::forecast(h = horizon)

  historia <- train_ts %>% tail(display_history)

  p <- fc %>%
    autoplot(historia, level = c(80, 95)) +
    labs(title = titulo, y = response_col, x = NULL) +
    theme_minimal(base_size = 11)

  ggsave(path, plot = p, width = 10, height = 5, dpi = 150)
  p
}

#' Grafico de la volatilidad condicional pronosticada por el GARCH(1,1) eolico.
plot_garch_volatility_forecast <- function(csv_path, png_path) {
  vol <- readr::read_csv(csv_path, show_col_types = FALSE)

  p <- vol %>%
    ggplot(aes(x = hora_adelante, y = sigma_pronosticada_mw)) +
    geom_line(color = "#8A5A2C", linewidth = 0.7) +
    geom_point(size = 1.2, color = "#8A5A2C") +
    labs(
      title = "Volatilidad condicional pronosticada -- generación eólica",
      subtitle = "GARCH(1,1) sobre los shocks eólicos horarios",
      x = "Horas hacia adelante", y = "Sigma pronosticada (MW)"
    ) +
    theme_minimal(base_size = 11)

  ggsave(png_path, plot = p, width = 8, height = 4.5, dpi = 150)
  p
}

if (sys.nframe() == 0) {
  dir.create("output/figures", showWarnings = FALSE, recursive = TRUE)

  hourly_ts <- readRDS("data/hourly_tsibble.rds")
  daily_ts <- readRDS("data/daily_tsibble.rds")
  arima_fit <- readRDS("output/models/arima_fourier_fit.rds")
  ets_fit <- readRDS("output/models/ets_fit.rds")

  cat("Generando diagnostico de residuos (ARIMA)...\n")
  plot_residual_diagnostics(arima_fit, "arima_fourier", "output/figures/residuals_arima.png")

  cat("Generando diagnostico de residuos (ETS)...\n")
  plot_residual_diagnostics(ets_fit, "ets", "output/figures/residuals_ets.png")

  cat("Generando pronostico con bandas (ARIMA, contexto 10 dias + 14 dias pronosticados)...\n")
  train_hourly <- readRDS("data/train_hourly_tsibble.rds")
  plot_forecast_with_bands(
    arima_fit, "arima_fourier", train_hourly, "demand_mw",
    horizon = 14 * 24, display_history = 10 * 24,
    path = "output/figures/forecast_arima_demand.png",
    titulo = "Pronóstico de demanda horaria -- ARIMA + Fourier (bandas 80%/95%)"
  )

  cat("Generando pronostico con bandas (ETS, contexto 90 dias + 28 dias pronosticados)...\n")
  train_daily <- readRDS("data/train_daily_tsibble.rds")
  plot_forecast_with_bands(
    ets_fit, "ets", train_daily, "demand_mw_mean",
    horizon = 28, display_history = 90,
    path = "output/figures/forecast_ets_daily.png",
    titulo = "Pronóstico de demanda diaria promedio -- ETS(M,A,M) (bandas 80%/95%)"
  )

  cat("Generando grafico de volatilidad GARCH...\n")
  plot_garch_volatility_forecast(
    "output/tables/garch_volatility_forecast.csv",
    "output/figures/garch_wind_volatility_forecast.png"
  )

  cat("Todos los graficos guardados en output/figures/\n")
}
