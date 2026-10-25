library(testthat)
library(dplyr)
library(lubridate)

source(file.path("..", "..", "R", "10_real_hourly_data.R"), chdir = FALSE)

# Hoja con el formato del XLSX del CEN: dos filas de encabezado y fecha como serial de Excel.
fake_sheet <- function(dates, values_by_day, with_dia = TRUE) {
  rows <- list()
  for (i in seq_along(dates)) {
    v <- values_by_day[[i]]
    serial <- as.numeric(as.Date(dates[i]) - as.Date("1899-12-30"))
    rows[[i]] <- if (with_dia) data.frame(a = serial, b = 1, c = seq_along(v), e = v)
                 else data.frame(a = serial, c = seq_along(v), e = v)
  }
  body <- do.call(rbind, rows)
  body[] <- lapply(body, as.character)
  header <- if (with_dia) c("fecha", "dia", "hora", "SEN") else c("fecha", "hora", "SEN")
  top <- as.data.frame(t(c(rep(NA, ncol(body) - 1), "Generacion Horaria [MWh/h]")), stringsAsFactors = FALSE)
  names(top) <- names(body)
  hdr <- as.data.frame(t(header), stringsAsFactors = FALSE)
  names(hdr) <- names(body)
  rbind(top, hdr, body)
}

test_that("a normal day maps rows to consecutive local hours", {
  out <- parse_cen_sheet(fake_sheet("2024-01-10", list(1:24 * 100)))
  local <- with_tz(out$ts_utc, TZ_CL)
  expect_equal(hour(local), 0:23)
  expect_equal(out$gen_mw, 1:24 * 100)
})

test_that("the 23-hour spring-forward day drops the non-existent first row", {
  # 2024-09-08: Chile pasa de 00:00 a 01:00. El archivo trae 24 filas, la primera en 0.
  out <- parse_cen_sheet(fake_sheet("2024-09-08", list(c(0, 2:24 * 100))))
  expect_equal(nrow(out), 23)
  expect_false(any(out$gen_mw == 0))
  expect_equal(hour(with_tz(out$ts_utc[1], TZ_CL)), 1)
  expect_true(all(diff(as.numeric(out$ts_utc)) == 3600))
})

test_that("the 2025 sheet's alphabetical hour order is undone", {
  true_profile <- 1:24 * 10
  published <- true_profile[as.integer(sort(as.character(1:24)))]   # orden alfabetico 1,10,11,...
  out <- parse_cen_sheet(fake_sheet("2025-01-15", list(published), with_dia = FALSE))
  expect_equal(out$gen_mw, true_profile)
})

test_that("clean_real_hourly flags a blackout and fills the grid", {
  ts <- seq(as.POSIXct("2024-01-01 03:00", tz = "UTC"), by = "hour", length.out = 24 * 40)
  gen <- 9000 + 1000 * sin(2 * pi * (seq_along(ts) %% 24) / 24)
  gen[500:505] <- c(50, 60, 70, 100, 200, 300)          # apagon
  d <- tibble(ts_utc = ts[-700], gen_mw = gen[-700])     # una hora que el archivo no trae
  out <- clean_real_hourly(d)
  expect_equal(nrow(out), length(ts))
  expect_equal(sum(out$anomaly), 6)
  expect_true(out$missing[700])
  expect_true(all(is.na(out$gen_mw[500:505])))
})
