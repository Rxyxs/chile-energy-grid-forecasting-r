# Construye estructuras tsibble (tidy time series) a partir de los datos crudos:
# una serie horaria (para ARIMA/SARIMAX y GARCH) y una serie diaria agregada
# (para el modelo de espacio de estados ETS).

library(dplyr)
library(tsibble)
library(lubridate)

#' Convierte el tibble horario crudo en un tsibble horario con indice regular.
build_hourly_tsibble <- function(sen_data) {
  sen_data %>%
    as_tsibble(index = datetime) %>%
    fill_gaps() %>%
    mutate(
      hour_of_day = hour(datetime),
      day_of_week = wday(datetime, label = TRUE, week_start = 1),
      is_weekend = day_of_week %in% c("sáb", "dom")
    )
}

#' Agrega la serie horaria a resolucion diaria (suma de demanda/generacion del dia).
build_daily_tsibble <- function(hourly_tsibble) {
  hourly_tsibble %>%
    as_tibble() %>%
    mutate(date = as_date(datetime)) %>%
    group_by(date) %>%
    summarise(
      demand_mw_mean = mean(demand_mw, na.rm = TRUE),
      demand_mw_total_gwh = sum(demand_mw, na.rm = TRUE) / 1000,
      solar_mw_mean = mean(solar_mw, na.rm = TRUE),
      wind_mw_mean = mean(wind_mw, na.rm = TRUE),
      net_demand_mw_mean = mean(net_demand_mw, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    as_tsibble(index = date)
}

if (sys.nframe() == 0) {
  source("R/01_generate_synthetic_data.R")

  sen_data <- readr::read_csv("data/sen_hourly_synthetic.csv", show_col_types = FALSE)
  hourly_ts <- build_hourly_tsibble(sen_data)
  daily_ts <- build_daily_tsibble(hourly_ts)

  cat(sprintf("Serie horaria: %d filas, valores NA tras fill_gaps: %d\n", nrow(hourly_ts), sum(is.na(hourly_ts$demand_mw))))
  cat(sprintf("Serie diaria agregada: %d filas\n", nrow(daily_ts)))

  saveRDS(hourly_ts, "data/hourly_tsibble.rds")
  saveRDS(daily_ts, "data/daily_tsibble.rds")
}
