# Instala (si falta) el set de paquetes usado en todo el proyecto.
# Ejecutar una vez con: Rscript R/00_setup.R

# La libreria del sistema (Program Files) no es escribible sin permisos de admin --
# se usa una libreria de usuario en su lugar (patron estandar de R en Windows).
user_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
}
.libPaths(c(user_lib, .libPaths()))

required_packages <- c(
  "tidyverse",   # dplyr, ggplot2, tidyr, purrr, lubridate, etc.
  "tsibble",     # estructura de datos de series de tiempo ordenadas (tidy)
  "fable",       # modelos de pronostico (ARIMA, ETS) sobre tsibble
  "feasts",      # features, ACF/PACF, STL, tests de raiz unitaria (KPSS)
  "fabletools",  # utilidades compartidas de fable/feasts (accuracy, glance, etc.)
  "ggtime",      # autoplot() para objetos dcmp_ts/tbl_cf (movido aqui en versiones recientes)
  "rugarch",     # modelos GARCH para volatilidad
  "tseries"      # test de Dickey-Fuller aumentado (ADF)
)

installed <- rownames(installed.packages())
missing <- setdiff(required_packages, installed)

if (length(missing) > 0) {
  message("Instalando paquetes faltantes: ", paste(missing, collapse = ", "))
  install.packages(missing, lib = user_lib, repos = "https://cloud.r-project.org")
} else {
  message("Todos los paquetes requeridos ya estan instalados.")
}

invisible(lapply(required_packages, function(pkg) {
  library(pkg, character.only = TRUE)
}))

message("Setup completo. Version de R: ", R.version.string)
