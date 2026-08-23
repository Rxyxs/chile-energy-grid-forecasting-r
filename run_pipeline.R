# Orquestador: ejecuta el pipeline completo de punta a punta en un solo comando.
#
# Cada paso corre como un subproceso `Rscript` independiente (no via source()) --
# de ese modo cada script se ejecuta exactamente igual que cuando se corre solo
# (Rscript R/0X_....R), incluyendo su bloque `if (sys.nframe() == 0)`, que solo se
# activa a nivel de tope de un proceso Rscript real y NO se dispara si el archivo
# se cargara con source() desde otro script.
#
# Ejecutar desde la raiz del repositorio con:
#   Rscript run_pipeline.R

steps <- c(
  "R/01_generate_synthetic_data.R",
  "R/02_data_wrangling.R",
  "R/03_stationarity_tests.R",
  "R/04_stl_decomposition.R",
  "R/05_models_arima_sarimax.R",
  "R/06_models_ets_statespace.R",
  "R/07_models_garch_volatility.R",
  "R/08_cross_validation.R",
  "R/09_generate_plots.R"
)

rscript_bin <- file.path(R.home("bin"), "Rscript")

for (i in seq_along(steps)) {
  cat(sprintf("\n%s\n>> %d/%d %s\n%s\n", strrep("=", 70), i, length(steps), steps[i], strrep("=", 70)))
  status <- system2(rscript_bin, args = shQuote(steps[i]))
  if (status != 0) {
    stop(sprintf("Paso fallido: %s (codigo %d)", steps[i], status))
  }
}

cat("\nPipeline completo. Resultados en data/, output/models/, output/tables/, output/figures/\n")
