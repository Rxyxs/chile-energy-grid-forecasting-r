# Generacion DIARIA REAL por tecnologia, desde los informes publicos del Coordinador.
#
# El Coordinador publica cada dia un "Resumen Ejecutivo de Operacion" en PDF. Su
# grafico horario por tecnologia es una imagen, pero la tabla "Generacion por
# Fuente Total del SEN" viene como texto: GWh del dia para BESS (baterias),
# eolica, geotermica, hidraulica, solar y termica. Este script recorre los
# informes de un rango de fechas (una consulta por segundo), extrae esa tabla y
# deja una fila por dia en data/real/cen_daily_by_technology.csv.
#
# Sirve para dos cosas que el archivo horario total no permite:
#   - solar y eolica diarias reales, en vez de modeladas desde el clima;
#   - poner a prueba la explicacion del cambio de definicion del archivo
#     horario en 2025 con la descarga diaria de las baterias.
#
#   Rscript R/13_cen_daily_reports.R            # 2024-01-01 a 2025-12-31, ~20 min

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

DAILY_TECH_PATH <- "data/real/cen_daily_by_technology.csv"
PDF_CACHE <- "data/real/cen_reports"
TECHS <- c(Bess = "bess", "E.lica" = "eolica", "Geot.rmica" = "geotermica",
           "Hidr.ulica" = "hidraulica", Solar = "solar", "T.rmica" = "termica")

#' URLs candidatas del informe de un dia: el archivo se sube en la carpeta del mes
#' del dia o del mes siguiente, a veces con sufijo de version.
report_urls <- function(date) {
  date <- as.Date(date)
  name <- sprintf("Resumen-Ejecutivo-de-Operacion-%s", format(date, "%d-%m-%Y"))
  months <- unique(c(format(date, "%Y/%m"), format(date + 3, "%Y/%m")))
  suffixes <- c("", "-V1", "-V1-1", "-V2", "-1")
  as.vector(outer(months, suffixes, function(m, s)
    sprintf("https://www.coordinador.cl/wp-content/uploads/%s/%s%s.pdf", m, name, s)))
}

#' Los informes cambian de formato numerico: "105.51" en 2024 y "1.126,57" en 2025.
parse_number_cl <- function(x) {
  if (length(x) == 0) return(NA_real_)
  if (grepl(",", x)) x <- sub(",", ".", gsub(".", "", x, fixed = TRUE), fixed = TRUE)
  as.numeric(x)
}

#' Texto del informe (la tabla esta en la pagina 2) -> GWh del dia por tecnologia. La fila de cada
#' tecnologia empieza con su nombre y el primer numero es el valor del dia.
parse_report_text <- function(text) {
  lines <- strsplit(text, "\n")[[1]]
  out <- list()
  for (pat in names(TECHS)) {
    # Las filas de la tabla traen la participacion en %; el eje del grafico, que
    # tambien empieza con el nombre de la tecnologia ("Solar ... 1 4 7 10"), no.
    hit <- grep(sprintf("^\\s*%s\\s+[0-9]", pat), lines, value = TRUE)
    hit <- hit[grepl("%", hit)]
    if (length(hit) == 0) next
    # La misma tecnologia aparece tambien en la tabla de operacion en tiempo real,
    # en MWh; el valor del dia en GWh es el primero bajo 1.000. A veces el PDF
    # pega dos columnas ("105.5042.95%"), asi que se toma el primer numero con
    # exactamente dos decimales.
    vals <- vapply(hit, function(h) {
      rest <- sub(sprintf("^\\s*%s", pat), "", h)
      parse_number_cl(regmatches(rest, regexpr("[0-9]{1,3}[.,][0-9]{2}", rest)))
    }, numeric(1))
    vals <- vals[!is.na(vals) & vals < 1000]
    if (length(vals)) out[[TECHS[[pat]]]] <- unname(vals[1])
  }
  out
}

fetch_report <- function(date, cache = PDF_CACHE, pause = 1) {
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  local <- file.path(cache, sprintf("%s.pdf", format(as.Date(date))))
  if (!file.exists(local)) {
    for (u in report_urls(date)) {
      ok <- tryCatch({
        suppressWarnings(download.file(u, local, mode = "wb", quiet = TRUE,
                                       headers = c("User-Agent" = "Mozilla/5.0")))
        TRUE
      }, error = function(e) FALSE)
      Sys.sleep(pause)
      if (ok && file.exists(local) && file.size(local) > 10000 &&
          identical(readBin(local, "raw", 4), charToRaw("%PDF"))) break
      if (file.exists(local)) file.remove(local)
    }
  }
  if (!file.exists(local)) return(NULL)
  vals <- tryCatch(parse_report_text(paste(pdftools::pdf_text(local), collapse = "\n")), error = function(e) list())
  if (length(vals) == 0) return(NULL)
  as_tibble(c(list(date = as.Date(date)), vals))
}

build_daily_by_technology <- function(from = "2024-01-01", to = "2025-12-31") {
  dates <- seq(as.Date(from), as.Date(to), by = "day")
  rows <- lapply(seq_along(dates), function(i) {
    if (i %% 50 == 0) cat(sprintf("  %s (%d/%d)\n", dates[i], i, length(dates)))
    fetch_report(dates[i])
  })
  bind_rows(rows) |> arrange(date)
}

#' Vertimiento horario REAL del SEN por tecnologia, desde el dataset de Reyes Cerda
#' y Cea Morales (Zenodo 10.5281/zenodo.21198817, CC BY 4.0): curtailment por
#' central y hora 2022-2026, cuadrado contra los totales anuales del CEN. Se
#' agrega a una fila por hora con el MWh vertido de solar y de eolica.
CURTAILMENT_URL <- "https://zenodo.org/records/21198817/files/curtailment_hourly.parquet?download=1"
CURTAILMENT_PATH <- "data/real/curtailment_hourly_sen.csv.gz"

build_curtailment <- function(parquet = "data/real/zenodo_curtailment/curtailment_hourly.parquet") {
  if (!file.exists(parquet)) {
    dir.create(dirname(parquet), recursive = TRUE, showWarnings = FALSE)
    download.file(CURTAILMENT_URL, parquet, mode = "wb", quiet = TRUE)
  }
  nanoparquet::read_parquet(parquet) |>
    filter(tecnologia %in% c("Solar", "Eólica")) |>
    mutate(tech = ifelse(tecnologia == "Solar", "solar_curtailed_mwh", "wind_curtailed_mwh")) |>
    group_by(fecha, hora, tech) |>
    summarise(mwh = sum(mwh), .groups = "drop") |>
    tidyr::pivot_wider(names_from = tech, values_from = mwh, values_fill = 0) |>
    arrange(fecha, hora)
}

if (sys.nframe() == 0) {
  cur <- build_curtailment()
  write_csv(cur, CURTAILMENT_PATH)
  cat(sprintf("vertimiento: %d horas, solar %.0f GWh, eolica %.0f GWh\n", nrow(cur),
              sum(cur$solar_curtailed_mwh) / 1000, sum(cur$wind_curtailed_mwh) / 1000))
  d <- build_daily_by_technology()
  write_csv(d, DAILY_TECH_PATH)
  cat(sprintf("%d dias con informe de %d; faltan %d\n", nrow(d),
              length(seq(as.Date("2024-01-01"), as.Date("2025-12-31"), by = "day")),
              length(seq(as.Date("2024-01-01"), as.Date("2025-12-31"), by = "day")) - nrow(d)))
}
