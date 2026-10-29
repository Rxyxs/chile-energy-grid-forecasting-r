# Pronostico del dia siguiente sobre la serie horaria REAL del SEN (2018-2025),
# con gradient boosting, distintas funciones de perdida y dos formas del objetivo.
#
# Protocolo
#   - Dia siguiente honesto: toda variable del pasado tiene al menos 48 horas de
#     antiguedad, lo que alcanza para pronosticar todo el dia D con lo que se
#     sabe a medianoche del dia D-1 (horizontes de 1 a 48 horas).
#   - Entrenamiento 2018-2023, validacion 2024 (early stopping, hiperparametros,
#     forma del objetivo y calibracion de intervalos), prueba 2025 sin tocar.
#     El modelo final se reentrena con 2018-2024 con los arboles elegidos.
#   - Las horas anomalas (apagon del 25-feb-2025) y faltantes no entran ni al
#     entrenamiento ni a la evaluacion, y todos los modelos se evaluan sobre
#     exactamente las mismas horas.
#
# Objetivo absoluto o relativo
#   Un arbol no extrapola: si el nivel del sistema cambia (2025 es el primer ano
#   en que la generacion baja), un modelo entrenado sobre MW sigue prediciendo el
#   nivel de los anos que vio. El objetivo relativo divide cada hora por el
#   promedio de la ultima semana conocida (hace 48 horas) y el modelo aprende la
#   forma; el nivel lo pone el denominador. Cual de los dos se usa se decide con
#   la validacion de 2024, no con la prueba.
#
# Perdidas comparadas (mismas variables, mismos hiperparametros):
#   L2 (media), L1 (mediana), Huber (robusta a picos), cuantilicas P10/P50/P90
#   y una L2 asimetrica propia que pesa el doble quedarse corto que pasarse,
#   porque subestimar la demanda obliga a activar reserva cara y sobreestimarla
#   solo deja capacidad ociosa. El intervalo P10-P90 se calibra ademas por
#   prediccion conformal (CQR) con los residuos de 2024.
#
#   Rscript R/11_ml_day_ahead.R

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readr)
  library(lightgbm)
  library(forecast)
  library(jsonlite)
})

REAL_HOURLY_PATH <- "data/real/sen_hourly_real.csv.gz"
OUT_TABLES <- "output/tables"
TZ_CL <- "America/Santiago"
UNDER_WEIGHT <- 2          # costo relativo de subestimar vs sobreestimar
SEED <- 42
LAGS <- c(48, 72, 96, 120, 144, 168, 336)
HUBER_MW <- 600            # errores mayores a esto pesan linealmente

load_real <- function(path = REAL_HOURLY_PATH) {
  read_csv(path, show_col_types = FALSE) |>
    arrange(ts_utc) |>
    mutate(local = with_tz(ts_utc, TZ_CL),
           date_local = as.Date(local, tz = TZ_CL),
           year = year(local))
}

#' Variables que existen 48 horas antes de cada hora objetivo.
make_features <- function(d) {
  y <- d$gen_mw
  lagv <- function(k) dplyr::lag(y, k)
  for (k in LAGS) d[[paste0("lag_", k)]] <- lagv(k)
  roll <- function(x, w) as.numeric(stats::filter(x, rep(1 / w, w), sides = 1))
  d$roll24_lag48 <- dplyr::lag(roll(y, 24), 48)
  d$roll168_lag48 <- dplyr::lag(roll(y, 168), 48)
  d$same_hour_mean_4w <- rowMeans(cbind(lagv(168), lagv(336), lagv(504), lagv(672)), na.rm = TRUE)
  hol <- d |> distinct(date_local, holiday)
  hol_prev <- setNames(hol$holiday, as.character(hol$date_local + 1))
  hol_next <- setNames(hol$holiday, as.character(hol$date_local - 1))
  d |>
    mutate(hour = hour(local), dow = wday(local, week_start = 1), doy = yday(local),
           month = month(local), weekend = dow >= 6, holiday = as.integer(holiday),
           after_holiday = as.integer(coalesce(hol_prev[as.character(date_local)], FALSE)),
           before_holiday = as.integer(coalesce(hol_next[as.character(date_local)], FALSE)),
           trend = as.numeric(ts_utc - min(ts_utc), units = "days") / 365.25,
           temp_c = temp_c)
}

LEVEL_FEATURES <- c(paste0("lag_", LAGS), "roll24_lag48", "same_hour_mean_4w")
CALENDAR <- c("hour", "dow", "doy", "month", "weekend", "holiday", "after_holiday", "before_holiday", "temp_c")
FEATURES <- c(LEVEL_FEATURES, "roll168_lag48", CALENDAR, "trend")

usable <- function(d) d |> filter(!is.na(gen_mw), !anomaly, !missing, !is.na(roll168_lag48), !is.na(lag_336))

#' Matriz de entrada y etiqueta. En modo relativo, los niveles pasados se
#' expresan como fraccion del promedio de la ultima semana conocida y la
#' tendencia sale: el nivel lo aporta el denominador, no el arbol.
design <- function(df, target) {
  if (target == "absolute") return(list(X = as.matrix(df[FEATURES]), y = df$gen_mw, scale = rep(1, nrow(df))))
  s <- df$roll168_lag48
  X <- as.matrix(cbind(df[LEVEL_FEATURES] / s, df[CALENDAR]))
  list(X = X, y = df$gen_mw / s, scale = s)
}

metrics <- function(y, p, scale) {
  e <- p - y
  list(mape = 100 * mean(abs(e) / y), rmse = sqrt(mean(e^2)), mae = mean(abs(e)),
       mase = mean(abs(e)) / scale, bias = mean(e),
       cost = mean(UNDER_WEIGHT * pmax(-e, 0) + pmax(e, 0)))
}

pinball <- function(y, q, tau) mean(pmax(tau * (y - q), (tau - 1) * (y - q)))

#' L2 asimetrica: gradiente y hessiano con peso UNDER_WEIGHT cuando la prediccion
#' queda bajo el valor real.
asym_l2 <- function(preds, dtrain) {
  y <- get_field(dtrain, "label")
  w <- ifelse(preds < y, UNDER_WEIGHT, 1)
  list(grad = w * (preds - y), hess = w)
}

base_params <- function(p) c(list(learning_rate = 0.03, num_threads = 4, verbose = -1, seed = SEED,
                                  feature_pre_filter = FALSE), p)

spec_params <- function(params, s, target) {
  p <- base_params(params)
  if (!is.null(s$custom)) {
    p$objective <- s$custom
    p$metric <- "l2"
    return(p)
  }
  p$objective <- s$objective
  if (!is.null(s$alpha)) p$alpha <- s$alpha
  if (s$objective == "huber") p$alpha <- if (target == "absolute") HUBER_MW else HUBER_MW / 9500
  p
}

fit_lgb <- function(train, valid, params, s = list(objective = "regression"), target = "absolute",
                    nrounds = 5000, early = 150) {
  a <- design(train, target)
  b <- design(valid, target)
  dtr <- lgb.Dataset(a$X, label = a$y)
  dva <- lgb.Dataset.create.valid(dtr, b$X, label = b$y)
  lgb.train(params = spec_params(params, s, target), data = dtr, nrounds = nrounds,
            valids = list(valid = dva), early_stopping_rounds = early, verbose = -1)
}

#' Prediccion en MW, deshaciendo la escala del objetivo relativo.
predict_mw <- function(model, df, target, custom = FALSE) {
  a <- design(df, target)
  p <- if (custom) predict(model, a$X, type = "raw") else predict(model, a$X)
  p * a$scale
}

#' Busqueda aleatoria de hiperparametros con L2 y early stopping en 2024.
tune <- function(train, valid, n = 16) {
  set.seed(SEED)
  grid <- data.frame(num_leaves = sample(c(31, 63, 127, 255), n, TRUE),
                     min_data_in_leaf = sample(c(20, 50, 100, 200), n, TRUE),
                     feature_fraction = sample(c(0.6, 0.8, 1.0), n, TRUE),
                     bagging_fraction = sample(c(0.7, 0.9, 1.0), n, TRUE),
                     lambda_l2 = sample(c(0, 1, 10), n, TRUE))
  res <- lapply(seq_len(n), function(i) {
    p <- as.list(grid[i, ])
    p$bagging_freq <- ifelse(p$bagging_fraction < 1, 1, 0)
    m <- fit_lgb(train, valid, p)
    c(grid[i, ], best_iter = m$best_iter,
      valid_mape = 100 * mean(abs(predict_mw(m, valid, "absolute") - valid$gen_mw) / valid$gen_mw))
  })
  res <- bind_rows(lapply(res, as.data.frame))
  res[order(res$valid_mape), ]
}

#' STL+ETS clasico (estacionalidad 24 y 168) sobre las ultimas 8 semanas,
#' reajustado cada dia: pronostica las horas del dia D desde la medianoche de D-1.
stl_ets_day_ahead <- function(d, test_dates) {
  y_filled <- fill_gaps(d$gen_mw)
  out <- rep(NA_real_, nrow(d))
  for (dt in as.character(test_dates)) {
    target <- which(d$date_local == as.Date(dt))
    if (length(target) == 0) next
    end <- target[1] - 25                       # medianoche de D-1: 24 h mas de distancia
    start <- max(1, end - 8 * 168 + 1)
    fit <- stlm(msts(y_filled[start:end], seasonal.periods = c(24, 168)), method = "ets")
    fc <- forecast(fit, h = 24 + length(target))$mean
    out[target] <- as.numeric(fc)[25:(24 + length(target))]
  }
  out
}

fill_gaps <- function(x) {
  i <- which(!is.na(x))
  approx(i, x[i], xout = seq_along(x), rule = 2)$y
}

#' Prediccion conformal cuantilica (Romano et al., 2019): cuanto hay que abrir
#' el intervalo [q_lo, q_hi] para que cubra lo prometido en un set de
#' calibracion que el modelo no vio.
cqr_margin <- function(y, q_lo, q_hi, coverage = 0.8) {
  score <- pmax(q_lo - y, y - q_hi)
  n <- length(score)
  as.numeric(quantile(score, min(1, ceiling((n + 1) * coverage) / n), type = 1))
}

#' Adaptive Conformal Inference (Gibbs y Candes, 2021) con retardo: el nivel del
#' cuantil de la hora i se ajusta con los fallos de las horas que ya se conocian
#' al pronosticarla (i - delay o antes), alpha_i = alpha + gamma * suma(alpha - fallo).
#' Un margen fijo calibrado en 2024 se usa todo 2025; este sigue la estacion.
#' `y`, `q_lo`, `q_hi` en orden temporal, sin NA. gamma fijado antes de mirar resultados.
aci_margins <- function(y, q_lo, q_hi, calib_scores, coverage = 0.8, gamma = 0.005, delay = 48) {
  alpha <- 1 - coverage
  n <- length(y)
  ncal <- length(calib_scores)
  margin <- numeric(n)
  miss <- numeric(n)
  drift <- 0
  for (i in seq_len(n)) {
    if (i > delay) drift <- drift + gamma * (alpha - miss[i - delay])
    a <- alpha + drift
    level <- min(1, max(0, ceiling((ncal + 1) * (1 - a)) / ncal))
    margin[i] <- if (a <= 0) max(calib_scores) else as.numeric(quantile(calib_scores, level, type = 1))
    miss[i] <- as.numeric(y[i] < q_lo[i] - margin[i] | y[i] > q_hi[i] + margin[i])
  }
  margin
}

SPECS <- list(
  l2 = list(objective = "regression"), l1 = list(objective = "regression_l1"),
  huber = list(objective = "huber"), asym_l2 = list(custom = asym_l2),
  q10 = list(objective = "quantile", alpha = 0.1), q50 = list(objective = "quantile", alpha = 0.5),
  q90 = list(objective = "quantile", alpha = 0.9))

main <- function() {
  dir.create(OUT_TABLES, recursive = TRUE, showWarnings = FALSE)
  d <- make_features(load_real())
  all_ok <- usable(d)
  train <- all_ok |> filter(year <= 2023)
  valid <- all_ok |> filter(year == 2024)
  test <- all_ok |> filter(year == 2025)
  full <- bind_rows(train, valid)
  scale <- mean(abs(train$gen_mw - train$lag_168), na.rm = TRUE)   # MASE: naive semanal en entrenamiento
  cat(sprintf("train %d, valid %d, test %d horas\n", nrow(train), nrow(valid), nrow(test)))

  tuned <- tune(train, valid)
  best <- as.list(tuned[1, c("num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "lambda_l2")])
  best$bagging_freq <- ifelse(best$bagging_fraction < 1, 1, 0)
  cat("mejores hiperparametros:", paste(names(best), unlist(best), collapse = ", "), "\n")

  preds <- tibble(ts_utc = test$ts_utc, local = test$local, actual = test$gen_mw,
                  naive_168 = test$lag_168, naive_4w = test$same_hour_mean_4w)
  valid_preds <- tibble(actual = valid$gen_mw)
  iters <- list()
  for (target in c("absolute", "relative")) {
    tag <- if (target == "absolute") "lgb_" else "lgb_rel_"
    for (nm in names(SPECS)) {
      s <- SPECS[[nm]]
      custom <- !is.null(s$custom)
      m <- fit_lgb(train, valid, best, s, target)
      iters[[paste0(tag, nm)]] <- m$best_iter
      valid_preds[[paste0(tag, nm)]] <- predict_mw(m, valid, target, custom)
      preds[[paste0(tag, nm, "_trainonly")]] <- predict_mw(m, test, target, custom)
      a <- design(full, target)
      final <- lgb.train(params = spec_params(best, s, target), data = lgb.Dataset(a$X, label = a$y),
                         nrounds = max(m$best_iter, 50), verbose = -1)
      preds[[paste0(tag, nm)]] <- predict_mw(final, test, target, custom)
      if (nm == "l2") assign(paste0("importance_", target), lgb.importance(final))
      cat(target, nm, "best_iter", m$best_iter, "\n")
    }
  }
  valid_mape <- sapply(c("lgb_l2", "lgb_rel_l2"), function(k)
    100 * mean(abs(valid_preds[[k]] - valid_preds$actual) / valid_preds$actual))
  chosen <- if (valid_mape["lgb_rel_l2"] < valid_mape["lgb_l2"]) "relative" else "absolute"
  cat("validacion 2024, MAPE L2: absoluto", round(valid_mape[1], 3), "relativo", round(valid_mape[2], 3),
      "-> se usa", chosen, "\n")

  preds$stl_ets <- stl_ets_day_ahead(d, sort(unique(test$date_local)))[match(test$ts_utc, d$ts_utc)]

  point <- c("naive_168", "naive_4w", "stl_ets",
             paste0("lgb_", c("l2", "l1", "huber", "asym_l2", "q50")),
             paste0("lgb_rel_", c("l2", "l1", "huber", "asym_l2", "q50")),
             "lgb_l2_trainonly", "lgb_rel_l2_trainonly")
  ok <- Reduce(`&`, lapply(point, function(nm) !is.na(preds[[nm]])))
  tab <- bind_rows(lapply(point, function(nm) c(model = nm, metrics(preds$actual[ok], preds[[nm]][ok], scale))))
  tab <- tab |> mutate(across(-model, as.numeric))

  # Intervalos P10-P90: tal como salen del modelo, y calibrados por CQR con 2024.
  interval_of <- function(lo, hi) {
    y <- preds$actual[ok]; lo <- lo[ok]; hi <- hi[ok]
    list(coverage = mean(y >= lo & y <= hi), below = mean(y < lo), above = mean(y > hi),
         mean_width_mw = mean(hi - lo), pinball_lo = pinball(y, lo, 0.1), pinball_hi = pinball(y, hi, 0.9))
  }
  intervals <- list()
  for (tag in c("lgb_", "lgb_rel_")) {
    margin <- cqr_margin(valid_preds$actual, valid_preds[[paste0(tag, "q10")]], valid_preds[[paste0(tag, "q90")]])
    lo <- preds[[paste0(tag, "q10_trainonly")]] - margin
    hi <- preds[[paste0(tag, "q90_trainonly")]] + margin
    preds[[paste0(tag, "cqr_lo")]] <- lo
    preds[[paste0(tag, "cqr_hi")]] <- hi
    intervals[[paste0(tag, "raw")]] <- interval_of(preds[[paste0(tag, "q10")]], preds[[paste0(tag, "q90")]])
    intervals[[paste0(tag, "cqr")]] <- c(interval_of(lo, hi), list(margin_mw = margin))
    # ACI: mismo modelo y mismos puntajes de calibracion 2024, margen que se ajusta en 2025.
    q_lo <- preds[[paste0(tag, "q10_trainonly")]]; q_hi <- preds[[paste0(tag, "q90_trainonly")]]
    calib <- pmax(valid_preds[[paste0(tag, "q10")]] - valid_preds$actual, valid_preds$actual - valid_preds[[paste0(tag, "q90")]])
    idx <- which(ok)[order(preds$ts_utc[ok])]
    m_aci <- rep(NA_real_, nrow(preds))
    m_aci[idx] <- aci_margins(preds$actual[idx], q_lo[idx], q_hi[idx], calib)
    preds[[paste0(tag, "aci_lo")]] <- q_lo - m_aci
    preds[[paste0(tag, "aci_hi")]] <- q_hi + m_aci
    intervals[[paste0(tag, "aci")]] <- c(interval_of(q_lo - m_aci, q_hi + m_aci),
                                          list(margin_mw_mean = mean(m_aci, na.rm = TRUE)))
    month <- format(preds$local[ok], "%Y-%m")
    for (kind in c("cqr", "aci")) {
      lo_k <- preds[[paste0(tag, kind, "_lo")]][ok]; hi_k <- preds[[paste0(tag, kind, "_hi")]][ok]
      cov_m <- tapply(preds$actual[ok] >= lo_k & preds$actual[ok] <= hi_k, month, mean)
      intervals[[paste0(tag, kind)]]$worst_month_coverage <- min(cov_m)
      intervals[[paste0(tag, kind)]]$best_month_coverage <- max(cov_m)
    }
  }

  # Validacion cruzada por anos: entrenar hasta Y-2, validar Y-1, probar Y.
  cv <- bind_rows(lapply(2021:2025, function(Y) {
    tr <- all_ok |> filter(year <= Y - 2); va <- all_ok |> filter(year == Y - 1); te <- all_ok |> filter(year == Y)
    row <- data.frame(test_year = Y)
    for (target in c("absolute", "relative")) {
      m <- fit_lgb(tr, va, best, target = target)
      row[[paste0(target, "_mape")]] <- 100 * mean(abs(predict_mw(m, te, target) - te$gen_mw) / te$gen_mw)
    }
    row$naive_mape <- 100 * mean(abs(te$lag_168 - te$gen_mw) / te$gen_mw, na.rm = TRUE)
    row
  }))

  profile <- d |> filter(!is.na(gen_mw)) |> group_by(year, hour = hour(local)) |>
    summarise(mean_mw = mean(gen_mw), .groups = "drop")

  write_csv(tab, file.path(OUT_TABLES, "ml_day_ahead_metrics.csv"))
  write_csv(preds, file.path(OUT_TABLES, "ml_day_ahead_predictions_2025.csv.gz"))
  write_csv(tuned, file.path(OUT_TABLES, "ml_tuning_2024.csv"))
  write_csv(cv, file.path(OUT_TABLES, "ml_yearly_cv.csv"))
  write_csv(as.data.frame(get(paste0("importance_", chosen))), file.path(OUT_TABLES, "ml_feature_importance.csv"))
  write_csv(profile, file.path(OUT_TABLES, "real_daily_profile_by_year.csv"))
  write_json(list(split = list(train = nrow(train), valid = nrow(valid), test = nrow(test), test_scored = sum(ok)),
                  best_params = best, best_iter = iters, valid_mape_l2 = as.list(valid_mape), chosen_target = chosen,
                  intervals = intervals, mase_scale_mw = scale,
                  excluded_hours = list(anomaly = sum(d$anomaly), missing = sum(d$missing))),
             file.path(OUT_TABLES, "ml_day_ahead_summary.json"), auto_unbox = TRUE, pretty = TRUE, digits = 6)
  print(tab |> mutate(across(-model, ~ round(.x, 3))), n = 30)
  str(intervals)
  print(cv)
}

if (sys.nframe() == 0) main()
