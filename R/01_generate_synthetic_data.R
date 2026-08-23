# Genera datos horarios sinteticos pero fisicamente/estadisticamente realistas para
# el Sistema Electrico Nacional (SEN) de Chile: demanda, generacion solar, generacion
# eolica y demanda neta (demanda - solar - eolica).
#
# Todas las cifras son ILUSTRATIVAS (ordenes de magnitud realistas del SEN), no datos
# operacionales reales del Coordinador Electrico Nacional (CEN). El componente eolico
# se simula explicitamente como un proceso GARCH(1,1) -- no ruido iid -- para que el
# modelo GARCH ajustado en 07_models_garch_volatility.R tenga clustering de volatilidad
# genuino que recuperar, no una señal inventada que el modelo no podria aprender.

library(dplyr)
library(lubridate)
library(tibble)

#' Forma horaria de doble punta (manana moderada, noche alta) tipica de un sistema
#' electrico con mezcla residencial/comercial/industrial.
daily_shape <- function(hour) {
  base <- 0.55
  manana <- 0.20 * exp(-((hour - 11)^2) / (2 * 2.5^2))
  noche <- 0.45 * exp(-((hour - 20)^2) / (2 * 2^2))
  raw <- base + manana + noche
  raw / mean(raw[unique(hour)])
}

#' Simula un proceso GARCH(1,1) simple: retorna una lista con los shocks (epsilon)
#' y la volatilidad condicional (sigma) para n pasos.
simulate_garch11 <- function(n, omega, alpha, beta, seed) {
  set.seed(seed)
  sigma2 <- numeric(n)
  eps <- numeric(n)
  sigma2[1] <- omega / (1 - alpha - beta)  # varianza incondicional como punto de partida
  eps[1] <- sqrt(sigma2[1]) * rnorm(1)
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * eps[t - 1]^2 + beta * sigma2[t - 1]
    eps[t] <- sqrt(sigma2[t]) * rnorm(1)
  }
  list(eps = eps, sigma = sqrt(sigma2))
}

#' Genera el dataset horario completo del SEN sintetico.
#'
#' @param start_date Fecha de inicio (Date)
#' @param end_date Fecha de termino (Date, inclusive)
#' @param seed Semilla para reproducibilidad
#' @return tibble con datetime, demand_mw, solar_mw, wind_mw, net_demand_mw
generate_sen_data <- function(start_date, end_date, seed = 42) {
  set.seed(seed)

  datetime <- seq(
    from = as.POSIXct(start_date, tz = "America/Santiago"),
    to = as.POSIXct(end_date, tz = "America/Santiago") + hours(23),
    by = "hour"
  )
  n <- length(datetime)

  hour_of_day <- hour(datetime)
  day_of_year <- yday(datetime)
  weekday <- wday(datetime, week_start = 1)  # 1 = lunes ... 7 = domingo
  years_elapsed <- as.numeric(difftime(datetime, datetime[1], units = "days")) / 365.25

  # ---- Demanda ----
  base_demand_mw <- 8000
  annual_growth_rate <- 0.03
  trend_factor <- (1 + annual_growth_rate)^years_elapsed

  weekday_factor <- case_when(
    weekday == 6 ~ 0.90,   # sabado
    weekday == 7 ~ 0.84,   # domingo
    TRUE ~ 1.00
  )

  winter_peak_doy <- 195  # aprox. mediados de julio (invierno hemisferio sur)
  annual_amplitude <- 0.12
  annual_factor <- 1 + annual_amplitude * cos(2 * pi * (day_of_year - winter_peak_doy) / 365.25)

  daily_factor <- daily_shape(hour_of_day)

  demand_noise <- rnorm(n, mean = 0, sd = base_demand_mw * 0.02)

  demand_mw <- base_demand_mw * trend_factor * weekday_factor * annual_factor * daily_factor + demand_noise

  # ---- Generacion solar ----
  solar_capacity_mw <- 3500
  # Duracion del dia aproximada (horas de luz) segun estacion -- mas larga en verano
  # (dic-feb, hemisferio sur), mas corta en invierno (jun-ago). Aproximacion ilustrativa,
  # no un calculo astronomico preciso.
  daylight_hours <- 12 + 2.2 * cos(2 * pi * (day_of_year - 355) / 365.25)
  sunrise <- 12 - daylight_hours / 2
  sunset <- 12 + daylight_hours / 2
  is_daylight <- hour_of_day >= floor(sunrise) & hour_of_day <= ceiling(sunset)

  solar_angle <- pmax(0, sin(pi * (hour_of_day - sunrise) / (sunset - sunrise)))

  # Indice de nubosidad: proceso AR(1) entre ~0.3 (muy nublado) y 1.0 (cielo despejado)
  cloud_index <- numeric(n)
  cloud_index[1] <- 0.85
  cloud_phi <- 0.85
  for (t in 2:n) {
    innovation <- rnorm(1, mean = 0, sd = 0.06)
    cloud_index[t] <- 0.85 + cloud_phi * (cloud_index[t - 1] - 0.85) + innovation
  }
  cloud_index <- pmin(1, pmax(0.25, cloud_index))

  solar_mw <- ifelse(
    is_daylight,
    solar_capacity_mw * solar_angle * cloud_index,
    0
  )
  solar_mw <- pmax(0, solar_mw)

  # ---- Generacion eolica (proceso GARCH(1,1) explicito) ----
  wind_capacity_mw <- 2500
  wind_base_factor <- 0.35 + 0.05 * cos(2 * pi * (hour_of_day - 3) / 24)  # levemente mas viento de madrugada

  garch <- simulate_garch11(n, omega = 0.02, alpha = 0.15, beta = 0.80, seed = seed + 1)
  # Escalar los shocks GARCH (en unidades estandarizadas) a fraccion de capacidad eolica
  wind_shock_scaled <- garch$eps * 0.12

  wind_mw <- wind_capacity_mw * pmin(1, pmax(0, wind_base_factor + wind_shock_scaled))

  # ---- Demanda neta ----
  min_must_run_mw <- 1200  # piso ilustrativo de generacion minima (hidro/termica de base)
  net_demand_mw <- pmax(min_must_run_mw, demand_mw - solar_mw - wind_mw)

  tibble(
    datetime = datetime,
    demand_mw = round(demand_mw, 1),
    solar_mw = round(solar_mw, 1),
    wind_mw = round(wind_mw, 1),
    net_demand_mw = round(net_demand_mw, 1),
    wind_volatility_sigma = round(garch$sigma, 4)
  )
}

if (sys.nframe() == 0) {
  # Solo se ejecuta si el script se corre directamente (Rscript), no al hacer source().
  end_date <- Sys.Date()
  start_date <- end_date - lubridate::years(2)

  sen_data <- generate_sen_data(start_date, end_date, seed = 42)

  dir.create("data", showWarnings = FALSE)
  readr::write_csv(sen_data, "data/sen_hourly_synthetic.csv")

  cat(sprintf("Filas generadas: %d\n", nrow(sen_data)))
  cat(sprintf("Rango de fechas: %s a %s\n", min(sen_data$datetime), max(sen_data$datetime)))
  cat(sprintf("Demanda promedio: %.0f MW\n", mean(sen_data$demand_mw)))
  cat(sprintf("Guardado en: data/sen_hourly_synthetic.csv\n"))
}
