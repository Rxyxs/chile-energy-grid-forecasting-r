library(testthat)
library(dplyr)

source(file.path("..", "..", "R", "11_ml_day_ahead.R"), chdir = FALSE)

test_that("metrics: perfect forecast and the asymmetric cost", {
  y <- c(100, 200, 300)
  m <- metrics(y, y, scale = 10)
  expect_equal(m$mape, 0)
  expect_equal(m$cost, 0)
  under <- metrics(y, y - 10, scale = 10)
  over <- metrics(y, y + 10, scale = 10)
  expect_equal(under$cost, UNDER_WEIGHT * over$cost)
  expect_equal(over$mase, 1)
})

test_that("pinball loss is minimised at the right quantile", {
  set.seed(1)
  y <- rnorm(5000)
  qs <- seq(-2, 2, by = 0.05)
  best <- qs[which.min(sapply(qs, function(q) pinball(y, q, 0.9)))]
  expect_equal(best, unname(quantile(y, 0.9)), tolerance = 0.1)
})

test_that("asymmetric L2 doubles the gradient when under-forecasting", {
  dtrain <- lightgbm::lgb.Dataset(matrix(1:4, ncol = 1), label = c(10, 10, 10, 10))
  lightgbm::lgb.Dataset.construct(dtrain)
  g <- asym_l2(c(8, 12, 10, 9), dtrain)
  expect_equal(g$grad, c(UNDER_WEIGHT * -2, 2, 0, UNDER_WEIGHT * -1))
  expect_equal(g$hess, c(UNDER_WEIGHT, 1, 1, UNDER_WEIGHT))
})

test_that("every lag feature is at least 48 hours old", {
  lags <- as.integer(sub("lag_", "", grep("^lag_", FEATURES, value = TRUE)))
  expect_true(all(lags >= 48))
  ts <- seq(as.POSIXct("2024-01-01", tz = "UTC"), by = "hour", length.out = 24 * 40)
  d <- tibble(ts_utc = ts, gen_mw = seq_along(ts), holiday = FALSE, temp_c = 20,
              anomaly = FALSE, missing = FALSE) |>
    mutate(local = lubridate::with_tz(ts_utc, TZ_CL), date_local = as.Date(local, tz = TZ_CL),
           year = lubridate::year(local))
  f <- make_features(d)
  i <- 900
  expect_equal(f$lag_48[i], f$gen_mw[i - 48])
  expect_equal(f$roll24_lag48[i], mean(f$gen_mw[(i - 48 - 23):(i - 48)]))
})

test_that("the relative target divides by last week's mean and drops the trend", {
  df <- as.data.frame(matrix(1, nrow = 2, ncol = length(FEATURES), dimnames = list(NULL, FEATURES)))
  df$roll168_lag48 <- c(1000, 2000)
  df$lag_48 <- c(900, 2200)
  df$gen_mw <- c(1100, 1800)
  a <- design(df, "relative")
  expect_false("trend" %in% colnames(a$X))
  expect_equal(unname(a$X[, "lag_48"]), c(0.9, 1.1))
  expect_equal(a$y, c(1.1, 0.9))
  expect_equal(a$y * a$scale, df$gen_mw)
})

test_that("the conformal margin makes the calibration interval cover as promised", {
  set.seed(3)
  y <- rnorm(2000)
  m <- cqr_margin(y, rep(-0.5, 2000), rep(0.5, 2000), coverage = 0.8)
  expect_gte(mean(y >= -0.5 - m & y <= 0.5 + m), 0.8)
  expect_lt(mean(y >= -0.5 - m & y <= 0.5 + m), 0.81)
})
