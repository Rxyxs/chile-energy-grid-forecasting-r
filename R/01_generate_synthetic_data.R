# Construye la serie horaria nacional del Sistema Electrico Nacional (SEN) de Chile:
# demanda, generacion solar, generacion eolica y demanda neta (demanda - solar - eolica).
#
# Por defecto (real = TRUE) la serie queda anclada a datos reales publicos, versionados
# en data/real/:
#   - Clima horario real (Open-Meteo, reanalisis ERA5) en 5 subestaciones del SEN:
#     irradiancia (GHI) y viento a 100 m. La forma hora a hora de la solar y la eolica
#     sale de ese clima.
#   - Generacion bruta mensual real del SEN por tecnologia (datos.gob.cl, CNE): la
#     energia solar, eolica y total de cada mes se fija exactamente a la cifra real.
# Lo que ninguna fuente publica entrega sin cuenta institucional -- el perfil hora a
# hora de la demanda -- sigue simulado y se escala mes a mes a la generacion total real.
#
# Con real = FALSE se obtiene el simulador original: cifras ilustrativas y un
# componente eolico simulado explicitamente como un proceso GARCH(1,1).

library(dplyr)
library(lubridate)
library(tibble)

REAL_WEATHER_PATH <- "data/real/weather_hourly_open_meteo.csv.gz"
REAL_GENERATION_PATH <- "data/real/sen_generacion_mensual.csv"

# Peso de cada subestacion en la forma horaria nacional: la solar se concentra en el
# norte (Atacama/Antofagasta) y la eolica en el sur (Biobio) y el norte grande.
SOLAR_WEIGHTS <- c(Crucero_220kV = 0.35, Cardones_220kV = 0.30, Quillota_220kV = 0.10,
                   Alto_Jahuel_220kV = 0.15, Charrua_220kV = 0.10)
WIND_WEIGHTS <- c(Crucero_220kV = 0.35, Cardones_220kV = 0.10, Quillota_220kV = 0.05,
                  Alto_Jahuel_220kV = 0.05, Charrua_220kV = 0.45)
WIND_CUT_IN_MS <- 3.0
WIND_RATED_MS <- 12.5
WIND_INSTALLED_MW <- 5500

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

#' Fraccion de la capacidad nominal que entrega una turbina a una velocidad de viento:
#' cero bajo el arranque, rampa hasta la velocidad nominal y plana sobre ella.
wind_power_fraction <- function(speed_ms) {
  frac <- ((speed_ms - WIND_CUT_IN_MS) / (WIND_RATED_MS - WIND_CUT_IN_MS))^1.2
  frac[speed_ms <= WIND_CUT_IN_MS] <- 0
  frac[speed_ms >= WIND_RATED_MS] <- 1
  frac
}

#' Escala una serie horaria para que la energia de cada mes (suma de MW x 1 h) sea
#' igual a la energia real de ese mes. `month_key` es "AAAA-MM" por fila.
scale_to_monthly_energy <- function(hourly, month_key, real_mwh_by_month) {
  sums <- tapply(hourly, month_key, sum)
  factor <- real_mwh_by_month[names(sums)] / sums
  factor[!is.finite(factor)] <- 1
  as.numeric(hourly * unname(factor[month_key]))
}

#' Lee el clima real y lo deja en una fila por hora local con el promedio ponderado
#' nacional de la GHI y de la fraccion de potencia eolica.
#' Las rutas de data/real/ son relativas a la raiz del repo; los tests corren desde
#' tests/testthat, asi que se prueba tambien dos niveles arriba.
resolve_repo_path <- function(path) {
  for (candidate in c(path, file.path("..", "..", path))) if (file.exists(candidate)) return(candidate)
  stop(sprintf("No se encontro %s", path))
}

load_real_weather_national <- function(path = REAL_WEATHER_PATH) {
  w <- readr::read_csv(resolve_repo_path(path), show_col_types = FALSE)
  w$wind_frac <- wind_power_fraction(w$wind_speed_ms)
  w$solar_w <- unname(SOLAR_WEIGHTS[w$node])
  w$wind_w <- unname(WIND_WEIGHTS[w$node])
  w %>%
    mutate(local_hour = format(timestamp, "%Y-%m-%d %H")) %>%
    group_by(local_hour) %>%
    summarise(ghi = sum(ghi_w_m2 * solar_w), wind_frac = sum(wind_frac * wind_w), .groups = "drop")
}

load_real_generation <- function(path = REAL_GENERATION_PATH) {
  readr::read_csv(resolve_repo_path(path), show_col_types = FALSE,
                  col_types = readr::cols(month = "c", .default = "d"))
}

#' Genera el dataset horario nacional del SEN.
#'
#' @param start_date Fecha de inicio (Date)
#' @param end_date Fecha de termino (Date, inclusive)
#' @param seed Semilla para reproducibilidad
#' @param real TRUE para anclar a datos reales (clima horario y energia mensual);
#'   requiere que el rango este cubierto por data/real/. FALSE = simulador original.
#' @return tibble con datetime, demand_mw, solar_mw, wind_mw, net_demand_mw
generate_sen_data <- function(start_date, end_date, seed = 42, real = TRUE) {
  set.seed(seed)

  datetime <- seq(
    from = as.POSIXct(paste(start_date, "00:00:00"), tz = "America/Santiago"),
    to = as.POSIXct(paste(end_date, "23:00:00"), tz = "America/Santiago"),
    by = "hour"
  )
  n <- length(datetime)

  hour_of_day <- hour(datetime)
  day_of_year <- yday(datetime)
  weekday <- wday(datetime, week_start = 1)  # 1 = lunes ... 7 = domingo
  years_elapsed <- as.numeric(difftime(datetime, datetime[1], units = "days")) / 365.25

  # ---- Demanda: perfil horario simulado ----
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

  # ---- Generacion solar (simulada; se reemplaza si real = TRUE) ----
  solar_capacity_mw <- 3500
  daylight_hours <- 12 + 2.2 * cos(2 * pi * (day_of_year - 355) / 365.25)
  sunrise <- 12 - daylight_hours / 2
  sunset <- 12 + daylight_hours / 2
  is_daylight <- hour_of_day >= floor(sunrise) & hour_of_day <= ceiling(sunset)

  solar_angle <- pmax(0, sin(pi * (hour_of_day - sunrise) / (sunset - sunrise)))

  cloud_index <- numeric(n)
  cloud_index[1] <- 0.85
  cloud_phi <- 0.85
  for (t in 2:n) {
    innovation <- rnorm(1, mean = 0, sd = 0.06)
    cloud_index[t] <- 0.85 + cloud_phi * (cloud_index[t - 1] - 0.85) + innovation
  }
  cloud_index <- pmin(1, pmax(0.25, cloud_index))

  solar_mw <- pmax(0, ifelse(is_daylight, solar_capacity_mw * solar_angle * cloud_index, 0))

  # ---- Generacion eolica (simulada como GARCH(1,1); se reemplaza si real = TRUE) ----
  wind_capacity_mw <- 2500
  wind_base_factor <- 0.35 + 0.05 * cos(2 * pi * (hour_of_day - 3) / 24)
  garch <- simulate_garch11(n, omega = 0.02, alpha = 0.15, beta = 0.80, seed = seed + 1)
  wind_mw <- wind_capacity_mw * pmin(1, pmax(0, wind_base_factor + garch$eps * 0.12))

  if (real) {
    local_hour <- format(datetime, "%Y-%m-%d %H")
    month_key <- format(datetime, "%Y-%m")
    weather <- load_real_weather_national()
    idx <- match(local_hour, weather$local_hour)
    if (mean(is.na(idx)) > 0.01) {
      stop("El rango pedido no esta cubierto por el clima real de data/real/ (use real = FALSE).")
    }
    ghi <- weather$ghi[idx]
    wind_frac <- weather$wind_frac[idx]
    # Las horas que el cambio de horario deja sin dato se interpolan.
    ghi <- approx(seq_len(n), ghi, xout = seq_len(n), rule = 2)$y
    wind_frac <- approx(seq_len(n), wind_frac, xout = seq_len(n), rule = 2)$y

    gen <- load_real_generation()
    real_solar <- setNames(gen$solar_mwh, gen$month)
    real_wind <- setNames(gen$wind_mwh, gen$month)
    real_total <- setNames(gen$total_mwh, gen$month)
    if (!all(unique(month_key) %in% gen$month)) {
      stop("El rango pedido no esta cubierto por la generacion mensual real de data/real/.")
    }

    solar_mw <- scale_to_monthly_energy(pmax(0, ghi), month_key, real_solar)
    wind_mw <- scale_to_monthly_energy(pmax(1e-6, wind_frac), month_key, real_wind)
    # Tope en la capacidad eolica instalada del SEN (~5,3 GW a fines de 2025): el escalamiento mensual
    # puede empujar algunas horas de viento fuerte por sobre lo fisicamente posible.
    wind_mw <- pmin(wind_mw, WIND_INSTALLED_MW)
    demand_mw <- scale_to_monthly_energy(pmax(1, demand_mw), month_key, real_total)
  }

  # ---- Demanda neta ----
  min_must_run_mw <- 1200  # piso ilustrativo de generacion minima (hidro/termica de base)
  net_demand_mw <- pmax(min_must_run_mw, demand_mw - solar_mw - wind_mw)

  out <- tibble(
    datetime = datetime,
    demand_mw = round(demand_mw, 1),
    solar_mw = round(solar_mw, 1),
    wind_mw = round(wind_mw, 1),
    net_demand_mw = round(net_demand_mw, 1)
  )
  if (!real) out$wind_volatility_sigma <- round(garch$sigma, 4)
  out
}

if (sys.nframe() == 0) {
  # Solo se ejecuta si el script se corre directamente (Rscript), no al hacer source().
  # Ventana fija cubierta por las series reales de data/real/.
  start_date <- as.Date("2024-01-01")
  end_date <- as.Date("2025-12-31")

  sen_data <- generate_sen_data(start_date, end_date, seed = 42, real = TRUE)

  dir.create("data", showWarnings = FALSE)
  readr::write_csv(sen_data, "data/sen_hourly.csv")

  cat(sprintf("Filas generadas: %d\n", nrow(sen_data)))
  cat(sprintf("Rango de fechas: %s a %s\n", min(sen_data$datetime), max(sen_data$datetime)))
  cat(sprintf("Demanda promedio: %.0f MW | solar promedio: %.0f MW | eolica promedio: %.0f MW\n",
              mean(sen_data$demand_mw), mean(sen_data$solar_mw), mean(sen_data$wind_mw)))
  cat("Guardado en: data/sen_hourly.csv\n")
}
