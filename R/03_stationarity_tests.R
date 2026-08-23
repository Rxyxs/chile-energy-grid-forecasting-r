# Pruebas de estacionariedad: ADF (Augmented Dickey-Fuller, H0 = raiz unitaria /
# no estacionaria) y KPSS (H0 = estacionaria) -- se usan juntas porque tienen
# hipotesis nulas opuestas, lo que da una lectura mas robusta que cualquiera de
# las dos por separado.

library(tseries)
library(feasts)
library(fabletools)
library(dplyr)

#' Corre ADF + KPSS sobre un vector numerico y devuelve un resumen de una fila.
#'
#' @param x vector numerico (la serie a testear)
#' @param series_name nombre descriptivo para el reporte
test_stationarity <- function(x, series_name) {
  x <- x[!is.na(x)]

  adf <- suppressWarnings(tseries::adf.test(x, alternative = "stationary"))
  kpss <- suppressWarnings(tseries::kpss.test(x, null = "Level"))

  tibble::tibble(
    series = series_name,
    adf_statistic = round(unname(adf$statistic), 3),
    adf_p_value = round(adf$p.value, 4),
    adf_conclusion = if (adf$p.value < 0.05) "Estacionaria (rechaza H0)" else "No estacionaria (no rechaza H0)",
    kpss_statistic = round(unname(kpss$statistic), 3),
    kpss_p_value = round(kpss$p.value, 4),
    kpss_conclusion = if (kpss$p.value < 0.05) "No estacionaria (rechaza H0)" else "Estacionaria (no rechaza H0)"
  )
}

#' Corre las pruebas sobre las series clave del proyecto (horaria y diaria).
run_all_stationarity_tests <- function(hourly_ts, daily_ts) {
  dplyr::bind_rows(
    test_stationarity(hourly_ts$demand_mw, "Demanda horaria (nivel)"),
    test_stationarity(diff(hourly_ts$demand_mw), "Demanda horaria (1ra diferencia)"),
    test_stationarity(hourly_ts$net_demand_mw, "Demanda neta horaria (nivel)"),
    test_stationarity(daily_ts$demand_mw_mean, "Demanda diaria promedio (nivel)"),
    test_stationarity(diff(daily_ts$demand_mw_mean), "Demanda diaria promedio (1ra diferencia)")
  )
}

if (sys.nframe() == 0) {
  hourly_ts <- readRDS("data/hourly_tsibble.rds")
  daily_ts <- readRDS("data/daily_tsibble.rds")

  resultados <- run_all_stationarity_tests(hourly_ts, daily_ts)
  print(resultados)

  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(resultados, "output/tables/stationarity_tests.csv")
}
