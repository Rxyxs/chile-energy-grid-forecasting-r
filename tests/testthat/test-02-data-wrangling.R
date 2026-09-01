library(testthat)
library(dplyr)
library(tsibble)

source(file.path("..", "..", "R", "01_generate_synthetic_data.R"), chdir = FALSE)
source(file.path("..", "..", "R", "02_data_wrangling.R"), chdir = FALSE)

test_that("build_hourly_tsibble produces a regular, gap-filled tsibble with derived columns", {
  sen <- generate_sen_data(as.Date("2023-01-01"), as.Date("2023-01-10"), seed = 42)
  hourly_ts <- build_hourly_tsibble(sen)

  expect_s3_class(hourly_ts, "tbl_ts")
  expect_true(all(c("hour_of_day", "day_of_week", "is_weekend") %in% names(hourly_ts)))
  expect_equal(nrow(hourly_ts), 10 * 24)
  expect_true(all(hourly_ts$hour_of_day %in% 0:23))
  expect_type(hourly_ts$is_weekend, "logical")
})

test_that("build_hourly_tsibble fills gaps when rows are missing from the source data", {
  sen <- generate_sen_data(as.Date("2023-01-01"), as.Date("2023-01-05"), seed = 42)
  sen_with_gap <- sen[-c(10, 11), ]  # drop two consecutive hours

  hourly_ts <- build_hourly_tsibble(sen_with_gap)

  expect_equal(nrow(hourly_ts), 5 * 24)  # fill_gaps() restores the regular grid
  expect_equal(sum(is.na(hourly_ts$demand_mw)), 2)
})

test_that("build_daily_tsibble aggregates hourly data to one row per day with correct totals", {
  sen <- generate_sen_data(as.Date("2023-01-01"), as.Date("2023-01-05"), seed = 42)
  hourly_ts <- build_hourly_tsibble(sen)
  daily_ts <- build_daily_tsibble(hourly_ts)

  expect_s3_class(daily_ts, "tbl_ts")
  expect_true(all(c(
    "demand_mw_mean", "demand_mw_total_gwh", "solar_mw_mean",
    "wind_mw_mean", "net_demand_mw_mean"
  ) %in% names(daily_ts)))

  # Cross-check the aggregation independently, using the exact same grouping
  # logic (as_date on the hourly tsibble) that build_daily_tsibble applies,
  # rather than assuming a fixed number of calendar days (tz conversion can
  # shift day boundaries).
  expected <- hourly_ts %>%
    as_tibble() %>%
    mutate(date = lubridate::as_date(datetime)) %>%
    group_by(date) %>%
    summarise(
      demand_mw_mean = mean(demand_mw, na.rm = TRUE),
      demand_mw_total_gwh = sum(demand_mw, na.rm = TRUE) / 1000,
      .groups = "drop"
    )

  expect_equal(nrow(daily_ts), nrow(expected))
  expect_equal(as.numeric(daily_ts$demand_mw_mean), expected$demand_mw_mean, tolerance = 1e-8)
  expect_equal(as.numeric(daily_ts$demand_mw_total_gwh), expected$demand_mw_total_gwh, tolerance = 1e-8)
})
