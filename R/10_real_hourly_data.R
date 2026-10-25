# Serie horaria REAL del SEN, 2018-2025: lo que faltaba en el resto del pipeline.
#
# El Coordinador Electrico Nacional publica un historico de generacion bruta
# horaria (MWh/h) desde el ano 2000 en un XLSX publico, una hoja por ano
# (SIC y SING por separado hasta 2017; SEN desde 2018, tras la interconexion).
# En un sistema aislado como el SEN, la generacion bruta horaria es la demanda
# mas las perdidas y los consumos propios: la mejor serie publica de carga
# horaria real sin cuenta institucional.
#
# Se le suman, por hora, la temperatura real de Santiago (Open-Meteo, reanalisis
# ERA5; la Region Metropolitana concentra cerca del 40% del consumo) y los
# feriados nacionales (Nager.Date).
#
#   Rscript R/10_real_hourly_data.R    # escribe data/real/sen_hourly_real.csv.gz

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readr)
  library(readxl)
  library(jsonlite)
})

CEN_URL <- "https://www.coordinador.cl/wp-content/uploads/2026/05/CEN-hist_gen_de_energia_por_hora_26.xlsx"
CEN_XLSX <- "data/real/cen/CEN-hist_gen_de_energia_por_hora_26.xlsx"
REAL_HOURLY_PATH <- "data/real/sen_hourly_real.csv.gz"
YEARS <- 2018:2025
TZ_CL <- "America/Santiago"

#' Lee una hoja anual del XLSX del CEN. Las etiquetas de hora no son confiables
#' (el dia del cambio de horario tiene una hora 25, y en 2025 la hora 24 aparece
#' como 0), asi que cada fila se ubica por su ORDEN dentro del dia: la k-esima
#' fila de una fecha es la medianoche local de esa fecha mas k-1 horas reales.
#' Asi los dias de 23 y de 25 horas quedan bien sin interpretar las etiquetas.
parse_cen_sheet <- function(raw) {
  names(raw) <- NULL
  df <- as.data.frame(raw, stringsAsFactors = FALSE)
  header_row <- which(df[[1]] == "fecha")[1]
  body <- df[(header_row + 1):nrow(df), , drop = FALSE]
  value <- suppressWarnings(as.numeric(body[[ncol(body)]]))
  fecha <- as.Date(suppressWarnings(as.numeric(body[[1]])), origin = "1899-12-30")
  keep <- !is.na(fecha) & !is.na(value)
  # La hoja 2025 (la unica sin columna "dia") trae las horas de cada dia en
  # orden ALFABETICO, aunque las etiquetas digan 1, 2, 3...: la fila 2 es la hora
  # 10, la 12 es la hora 2. Se ve en el apagon del 25 de febrero, que en el orden
  # publicado tiene una hora normal en medio de horas en cero, y en el perfil
  # medio del ano, con el minimo a media tarde. Se reordena por la hora verdadera.
  lexical <- !any(df[header_row, ] %in% "dia")
  out <- tibble(fecha = fecha[keep], gen_mw = value[keep]) |>
    group_by(fecha) |>
    mutate(pos = row_number(),
           true_hour = if (lexical) as.integer(sort(as.character(seq_len(n())))[pos]) else pos) |>
    arrange(fecha, true_hour, .by_group = FALSE) |>
    mutate(k = true_hour - 1L) |>
    ungroup() |>
    select(fecha, gen_mw, k)
  # roll_dst: el dia que empieza el horario de verano la medianoche local no
  # existe (el reloj salta de 00:00 a 01:00); su primer instante real es la 01:00.
  local_start <- function(f) force_tz(as.POSIXct(f, tz = "UTC"), TZ_CL, roll_dst = c("boundary", "post"))
  midnight <- local_start(out$fecha)
  day_hours <- as.numeric(difftime(local_start(out$fecha + 1), midnight, units = "hours"))
  n_rows <- ave(out$k, out$fecha, FUN = length)
  # El dia de 23 horas viene publicado con 24 filas: la primera es la hora que
  # no existe (00:00-01:00), con valor 0. Se descarta y las demas se corren.
  extra <- pmax(n_rows - day_hours, 0)
  out |>
    mutate(midnight = midnight, extra = extra) |>
    filter(k >= extra) |>
    mutate(ts_utc = with_tz(midnight + hours(k - extra), "UTC")) |>
    select(ts_utc, gen_mw)
}

read_cen_hourly <- function(path = CEN_XLSX, years = YEARS) {
  bind_rows(lapply(as.character(years), function(y) {
    parse_cen_sheet(suppressMessages(read_excel(path, sheet = y, col_names = FALSE)))
  })) |>
    distinct(ts_utc, .keep_all = TRUE) |>
    arrange(ts_utc)
}

fetch_santiago_temperature <- function(start = "2018-01-01", end = "2025-12-31") {
  url <- sprintf(paste0("https://archive-api.open-meteo.com/v1/archive?latitude=-33.45&longitude=-70.66",
                        "&start_date=%s&end_date=%s&hourly=temperature_2m&timezone=UTC"), start, end)
  j <- fromJSON(url)
  tibble(ts_utc = ymd_hm(j$hourly$time, tz = "UTC"), temp_c = j$hourly$temperature_2m)
}

fetch_holidays <- function(years = YEARS) {
  bind_rows(lapply(years, function(y) {
    h <- fromJSON(sprintf("https://date.nager.at/api/v3/PublicHolidays/%d/CL", y))
    tibble(fecha = as.Date(h$date), feriado = h$localName)
  })) |> distinct(fecha, .keep_all = TRUE)
}

#' Grilla horaria completa y marca de horas anomalas.
#'
#' Una hora es anomala si su generacion cae bajo la mitad de la mediana de esa
#' misma hora en los 14 dias que la rodean: eso aisla el apagon nacional del
#' 25 de febrero de 2025 y los ceros sueltos del archivo sin tocar la variacion
#' normal. Las horas que el archivo no trae (un dia de 25 horas publicado con 24
#' filas, alguna fila suelta) quedan como faltantes. Ninguna hora anomala o
#' faltante entra al entrenamiento ni a la evaluacion.
clean_real_hourly <- function(d, window_days = 14, ratio = 0.5) {
  d <- d |> filter(!is.na(ts_utc))
  grid <- tibble(ts_utc = seq(min(d$ts_utc), max(d$ts_utc), by = "hour"))
  out <- grid |> left_join(d, by = "ts_utc")
  hour_local <- hour(with_tz(out$ts_utc, TZ_CL))
  ref <- rep(NA_real_, nrow(out))
  for (h in 0:23) {
    idx <- which(hour_local == h)
    x <- out$gen_mw[idx]
    ref[idx] <- stats::runmed(ifelse(is.na(x), stats::median(x, na.rm = TRUE), x),
                              k = 2 * window_days + 1, endrule = "constant")
  }
  out |>
    mutate(missing = is.na(gen_mw),
           anomaly = !missing & gen_mw < ratio * ref,
           gen_mw = ifelse(anomaly, NA_real_, gen_mw))
}

build_real_hourly <- function() {
  if (!file.exists(CEN_XLSX)) {
    dir.create(dirname(CEN_XLSX), recursive = TRUE, showWarnings = FALSE)
    download.file(CEN_URL, CEN_XLSX, mode = "wb")
  }
  gen <- read_cen_hourly()
  temp <- fetch_santiago_temperature()
  hol <- fetch_holidays()
  clean_real_hourly(gen) |>
    left_join(temp, by = "ts_utc") |>
    mutate(fecha_local = as.Date(with_tz(ts_utc, TZ_CL), tz = TZ_CL),
           holiday = fecha_local %in% hol$fecha) |>
    select(ts_utc, gen_mw, temp_c, holiday, missing, anomaly)
}

if (sys.nframe() == 0) {
  d <- build_real_hourly()
  dir.create(dirname(REAL_HOURLY_PATH), recursive = TRUE, showWarnings = FALSE)
  write_csv(d, REAL_HOURLY_PATH)
  cat(sprintf("%d horas, %s a %s; faltantes %d; anomalas %d; temperatura faltante %d; horas feriado %d\n",
              nrow(d), format(min(d$ts_utc)), format(max(d$ts_utc)), sum(d$missing), sum(d$anomaly),
              sum(is.na(d$temp_c)), sum(d$holiday)))
}
