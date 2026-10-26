library(testthat)

source(file.path("..", "..", "R", "13_cen_daily_reports.R"), chdir = FALSE)

# Lineas reales del informe del 1 de julio de 2025 (pdftools::pdf_text), recortadas.
report_2025 <- paste(
  "Bess          4.84       1.95%       4.84       0.83     483.13        0.77   Embalse  39.57 67.94%",
  "Eólica       18.58 7.47% 18.58 32.81 -43.37                            5.75   Pasada   18.67 32.06%",
  "Geotérmica 0.00 0.00% 0.00 0.90 -100.00                                0.07",
  "Hidráulica  58.10 23.36% 58.10 89.65 -35.19                            10.60",
  "Solar       44.65 17.95% 44.65 44.23 0.95                              9.53",
  "Térmica    122.60 49.28% 122.60 67.55 81.50                            16.82",
  "                    Eólica 18577.25 28586.90 -35.02 2320.54 11.10",
  "Solar 44653.57 47461.50 -5.93 4108.29 8.43",
  "Eólica       18.58 25.03% 18.58 32.81 -43.37",
  sep = "\n"
)

test_that("parse_report_text lee el GWh del dia de cada tecnologia", {
  v <- parse_report_text(report_2025)
  expect_equal(v$bess, 4.84)
  expect_equal(v$eolica, 18.58)
  expect_equal(v$geotermica, 0)
  expect_equal(v$hidraulica, 58.10)
  expect_equal(v$solar, 44.65)
  expect_equal(v$termica, 122.60)
})

test_that("parse_report_text ignora el eje del grafico y las filas OTR en MWh", {
  text <- paste("Solar 1 4 7 10 13 16 19 22",
                "Solar 51346.91 57299.00 -10.39 14731.74 22.29",
                "Solar       51.35 23.13% 567.76 527.38 7.66", sep = "\n")
  expect_equal(parse_report_text(text)$solar, 51.35)
})

test_that("parse_report_text separa columnas pegadas por el PDF", {
  expect_equal(parse_report_text("Hidráulica 105.5042.95% 610.13")$hidraulica, 105.50)
})

test_that("parse_number_cl entiende el formato de 2024 y el de 2025", {
  expect_equal(parse_number_cl("105.51"), 105.51)
  expect_equal(parse_number_cl("1.126,57"), 1126.57)
  expect_equal(parse_number_cl("62,35"), 62.35)
  expect_true(is.na(parse_number_cl(character(0))))
})

test_that("report_urls prueba la carpeta del mes y la del mes siguiente", {
  u <- report_urls("2025-01-30")
  expect_true(any(grepl("/2025/01/Resumen-Ejecutivo-de-Operacion-30-01-2025.pdf$", u)))
  expect_true(any(grepl("/2025/02/Resumen-Ejecutivo-de-Operacion-30-01-2025-V1.pdf$", u)))
})
