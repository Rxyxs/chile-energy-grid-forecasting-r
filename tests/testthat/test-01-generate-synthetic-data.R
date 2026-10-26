library(testthat)
library(dplyr)

source(file.path("..", "..", "R", "01_generate_synthetic_data.R"), chdir = FALSE)

test_that("simulate_garch11 recovers a stationary, positive volatility process", {
  sim <- simulate_garch11(n = 2000, omega = 0.05, alpha = 0.15, beta = 0.80, seed = 1)

  expect_length(sim$eps, 2000)
  expect_length(sim$sigma, 2000)
  expect_true(all(sim$sigma > 0))
  expect_true(all(is.finite(sim$eps)))
  # alpha + beta < 1 (stationarity) implies bounded unconditional variance
  expect_lt(var(sim$eps), 10)
})

test_that("simulate_garch11 is reproducible given the same seed", {
  sim_a <- simulate_garch11(n = 500, omega = 0.05, alpha = 0.15, beta = 0.80, seed = 7)
  sim_b <- simulate_garch11(n = 500, omega = 0.05, alpha = 0.15, beta = 0.80, seed = 7)

  expect_equal(sim_a$eps, sim_b$eps)
  expect_equal(sim_a$sigma, sim_b$sigma)
})

test_that("daily_shape produces a double-peak profile normalized to mean 1", {
  hours <- 0:23
  shape <- daily_shape(hours)

  expect_length(shape, 24)
  expect_true(all(shape > 0))
  expect_equal(mean(shape), 1, tolerance = 1e-3)
  # Evening peak (~20h) must exceed early-morning trough (~4h)
  expect_gt(shape[hours == 20], shape[hours == 4])
})

test_that("generate_sen_data returns a well-formed hourly tibble with expected invariants", {
  sen <- generate_sen_data(as.Date("2023-01-01"), as.Date("2023-01-07"), seed = 42, real = FALSE)

  expect_s3_class(sen, "tbl_df")
  expect_equal(nrow(sen), 7 * 24)
  expect_true(all(c("datetime", "demand_mw", "solar_mw", "wind_mw", "net_demand_mw") %in% names(sen)))

  expect_true(all(sen$demand_mw > 0))
  expect_true(all(sen$solar_mw >= 0))
  expect_true(all(sen$wind_mw >= 0))

  # Net demand is demand minus renewable generation (floored at a minimum
  # must-run level), by construction. Each column is rounded independently
  # to 1 decimal before storage, so a small per-row tolerance is expected.
  gross_net <- sen$demand_mw - sen$solar_mw - sen$wind_mw
  expect_true(all(sen$net_demand_mw >= gross_net - 0.2))
  expect_true(all(sen$net_demand_mw == round(sen$net_demand_mw, 1)))

  # Solar must be zero at night (around 02:00 local) given the generator's daylight window
  night_rows <- sen[format(sen$datetime, "%H") == "02", ]
  expect_true(all(night_rows$solar_mw == 0))
})

test_that("generate_sen_data is reproducible given the same seed", {
  sen_a <- generate_sen_data(as.Date("2023-01-01"), as.Date("2023-01-03"), seed = 99, real = FALSE)
  sen_b <- generate_sen_data(as.Date("2023-01-01"), as.Date("2023-01-03"), seed = 99, real = FALSE)

  expect_equal(sen_a, sen_b)
})


test_that("real mode matches the real monthly SEN energy and stays physically bounded", {
  sen <- generate_sen_data(as.Date("2024-01-01"), as.Date("2024-03-31"), seed = 42, real = TRUE)
  expect_equal(nrow(sen), 91 * 24)
  month <- format(sen$datetime, "%Y-%m", tz = "America/Santiago")
  gen <- load_real_generation()
  real <- gen[gen$month %in% unique(month), ]
  solar_sum <- tapply(sen$solar_mw, month, sum)[real$month]
  demand_sum <- tapply(sen$demand_mw, month, sum)[real$month]
  wind_sum <- tapply(sen$wind_mw, month, sum)[real$month]
  expect_equal(as.numeric(solar_sum), real$solar_mwh, tolerance = 1e-3)
  expect_equal(as.numeric(demand_sum), real$total_mwh, tolerance = 1e-3)
  expect_equal(as.numeric(wind_sum), real$wind_mwh, tolerance = 0.02)  # el tope de capacidad recorta algunas horas
  expect_true(all(sen$wind_mw <= WIND_INSTALLED_MW))
  expect_true(all(sen$solar_mw >= 0))
  expect_true(all(sen$net_demand_mw >= 1200))
})

test_that("real mode refuses a range that the real series do not cover", {
  expect_error(generate_sen_data(as.Date("2019-01-01"), as.Date("2019-01-02"), real = TRUE))
})

test_that("demand_source = 'cen' uses the real hourly CEN series and flags the blackout", {
  sen <- generate_sen_data(as.Date("2025-02-20"), as.Date("2025-03-02"), real = TRUE, demand_source = "cen")
  r <- readr::read_csv(resolve_repo_path("data/real/sen_hourly_real.csv.gz"), show_col_types = FALSE)
  key <- format(lubridate::with_tz(sen$datetime, "UTC"), "%Y-%m-%d %H")
  real <- r$gen_mw[match(key, format(r$ts_utc, "%Y-%m-%d %H"))]
  ok <- !sen$demand_imputed
  expect_equal(sen$demand_mw[ok], round(real[ok], 1))
  expect_true(sum(sen$demand_imputed) >= 15)              # el apagon del 25 de febrero
  expect_true(all(sen$demand_mw > 4000))   # la recuperacion del 26 de febrero baja a ~4.200 MW reales
})
