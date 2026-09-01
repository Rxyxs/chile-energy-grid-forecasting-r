# Punto de entrada de testthat para el proyecto (no un paquete R formal).
# Ejecutar con: Rscript tests/testthat.R

library(testthat)

test_check_dir <- function() {
  test_dir(
    file.path("tests", "testthat"),
    reporter = "summary"
  )
}

if (sys.nframe() == 0) {
  test_check_dir()
}
