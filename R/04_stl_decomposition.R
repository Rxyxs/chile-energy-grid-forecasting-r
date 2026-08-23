# Descomposicion STL con estacionalidad multiple (diaria + semanal) sobre la serie
# horaria de demanda -- separa tendencia, ambos componentes estacionales y el
# remanente (residuo), usando feasts::STL().

library(feasts)
library(fabletools)
library(ggtime)   # autoplot() para objetos dcmp_ts / tbl_cf (movido aqui desde fabletools/feasts)
library(ggplot2)
library(dplyr)

#' Ajusta una descomposicion STL con estacionalidad diaria y semanal.
fit_stl_decomposition <- function(hourly_ts) {
  hourly_ts %>%
    model(STL(demand_mw ~ season(period = "day") + season(period = "week"), robust = TRUE))
}

#' Genera y guarda el grafico de la descomposicion STL.
plot_stl_decomposition <- function(stl_model, path) {
  p <- stl_model %>%
    generics::components() %>%
    autoplot() +
    labs(
      title = "Descomposición STL de la demanda horaria del SEN",
      subtitle = "Tendencia + estacionalidad diaria + estacionalidad semanal + remanente"
    ) +
    theme_minimal(base_size = 11)

  ggsave(path, plot = p, width = 10, height = 8, dpi = 150)
  p
}

#' Grafico de detalle (2 semanas) de la demanda horaria -- a resolucion de 2 años la
#' descomposicion STL completa se ve como una banda solida; este grafico muestra la
#' forma real de la curva diaria de doble punta y el contraste semana/fin de semana.
plot_two_week_zoom <- function(hourly_ts, path) {
  ventana <- hourly_ts %>%
    dplyr::filter(datetime >= max(datetime) - lubridate::days(14))

  p <- ventana %>%
    ggplot(aes(x = datetime, y = demand_mw)) +
    geom_line(color = "#2C5F8A", linewidth = 0.5) +
    labs(
      title = "Demanda horaria -- detalle de las últimas 2 semanas",
      subtitle = "Curva de doble punta (mañana / noche) y menor demanda en fines de semana",
      x = NULL, y = "Demanda (MW)"
    ) +
    theme_minimal(base_size = 11)

  ggsave(path, plot = p, width = 10, height = 4, dpi = 150)
  p
}

#' Genera y guarda los graficos ACF y PACF de la demanda horaria y su remanente STL.
plot_acf_pacf <- function(hourly_ts, stl_components, path_prefix) {
  p_acf_raw <- hourly_ts %>%
    ACF(demand_mw, lag_max = 72) %>%
    autoplot() +
    labs(title = "ACF -- Demanda horaria (nivel)") +
    theme_minimal(base_size = 11)
  ggsave(paste0(path_prefix, "_acf_demand.png"), p_acf_raw, width = 8, height = 4, dpi = 150)

  p_pacf_raw <- hourly_ts %>%
    PACF(demand_mw, lag_max = 72) %>%
    autoplot() +
    labs(title = "PACF -- Demanda horaria (nivel)") +
    theme_minimal(base_size = 11)
  ggsave(paste0(path_prefix, "_pacf_demand.png"), p_pacf_raw, width = 8, height = 4, dpi = 150)

  p_acf_remainder <- stl_components %>%
    ACF(remainder, lag_max = 72) %>%
    autoplot() +
    labs(title = "ACF -- Remanente de la descomposición STL") +
    theme_minimal(base_size = 11)
  ggsave(paste0(path_prefix, "_acf_remainder.png"), p_acf_remainder, width = 8, height = 4, dpi = 150)

  invisible(NULL)
}

if (sys.nframe() == 0) {
  hourly_ts <- readRDS("data/hourly_tsibble.rds")

  stl_model <- fit_stl_decomposition(hourly_ts)
  stl_components <- generics::components(stl_model)

  dir.create("output/figures", showWarnings = FALSE, recursive = TRUE)
  plot_stl_decomposition(stl_model, "output/figures/stl_decomposition.png")
  plot_two_week_zoom(hourly_ts, "output/figures/demand_two_week_zoom.png")
  plot_acf_pacf(hourly_ts, stl_components, "output/figures/stl")

  saveRDS(stl_model, "output/models/stl_model.rds")

  varianzas <- stl_components %>%
    as_tibble() %>%
    summarise(across(c(trend, season_day, season_week, remainder), \(x) var(x, na.rm = TRUE)))
  cat("Varianza de cada componente STL:\n")
  print(varianzas)
}
