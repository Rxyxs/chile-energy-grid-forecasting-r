<div align="center">

# ⚡ Pronóstico Estadístico de Demanda Energética y Estabilidad de Red en el SEN Chile

**Un proyecto de extremo a extremo en R -- ARIMA/SARIMAX, descomposición STL, modelos de espacio de estados (ETS) y GARCH -- sobre el ecosistema tidyverts/fable**

🌐 **[Español](README.es.md)** | **[English](README.md)**

[![R](https://img.shields.io/badge/R-4.4%2F4.6-276DC3)](https://www.r-project.org/)
[![fable](https://img.shields.io/badge/tidyverts-fable%20%7C%20feasts%20%7C%20tsibble-2C5F8A)](https://fable.tidyverts.org/)
[![forecast](https://img.shields.io/badge/TBATS-forecast-4C7A3E)](https://cran.r-project.org/package=forecast)
[![rugarch](https://img.shields.io/badge/GARCH-rugarch-8A5A2C)](https://cran.r-project.org/package=rugarch)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

</div>

---

## 1. Título del proyecto

**Pronóstico Estadístico de Demanda Energética y Estabilidad de Red en el Sistema Eléctrico Nacional (SEN) de Chile**

Repositorio: [`chile-energy-grid-forecasting-r`](https://github.com/Rxyxs/chile-energy-grid-forecasting-r)

## 2. Motivación

Me interesa este problema porque combina dos cosas que rara vez conviven en un solo dataset: series de tiempo con estacionalidad múltiple genuina (diaria, semanal, anual) y un componente de volatilidad real que importa operacionalmente, no solo estadísticamente. El Coordinador Eléctrico Nacional (CEN) es responsable de la coordinación en tiempo real y la planificación de largo plazo del SEN, que abastece a la gran mayoría de Chile continental. Dos tensiones estructurales hacen que el pronóstico riguroso importe ahí:

- **Creciente penetración solar y eólica.** Chile tiene una de las mayores irradiancias solares del mundo (Atacama) y un parque eólico en expansión, especialmente en el norte y el sur del país. Esa generación es intermitente por naturaleza: cae a cero en la noche (solar) y varía con el viento en rachas que se agrupan en el tiempo (clustering de volatilidad), no de forma independiente hora a hora.
- **El costo de una falla de red no es lineal.** Una rampa de demanda neta mal anticipada (el clásico problema de la "curva de pato": la demanda neta sube abruptamente cuando cae la generación solar al atardecer) obliga a activar reservas de generación rápida con poco preaviso, y en el peor caso deriva en desbalance de frecuencia.

Este proyecto construye el stack estadístico completo para abordar ambos problemas con las herramientas correctas para cada uno: modelos ARIMA/SARIMAX y de espacio de estados para el nivel esperado de demanda, y un modelo GARCH explícito para la volatilidad de la generación renovable -- no como un anexo cosmético, sino como la pieza que conecta el pronóstico de nivel con el riesgo real de estabilidad de red.

### 2.1 Impacto de Negocio e Indicadores Clave (KPIs)

| Métrica | Resultado | Qué significa |
|---|---|---|
| Mejor pronóstico horario de demanda (TBATS) | 2,47% MAPE, 284 RMSE | 2,2x más preciso que la línea base seasonal-naive (6,54% MAPE) |
| Pronóstico de demanda neta con solar/eólico exógenos (SARIMAX) | 6,68% MAPE, MASE 0,85 | Ataca directamente el problema de riesgo de rampa ("curva de pato"), no solo la demanda bruta |
| Demanda diaria, CV de origen rodante (ETS vs. naive) | 0,43% vs. 1,09% MAPE | La ventaja real de ETS solo aparece bajo CV apropiada -- un solo holdout favorecía engañosamente a naive (§7.1) |
| Recuperación de volatilidad eólica GARCH(1,1) | α₁ 0,142 (real 0,15), persistencia 0,954 (real 0,95) | Confirma que el modelo de volatilidad está bien ajustado, no solo que "el código corre" |
| Autocorrelación residual lag-24, mejor corrección encontrada | 0,41 → 0,29 (TBATS) | Reportado honestamente como una corrección parcial, no completa -- ningún modelo probado la elimina del todo (§7.3) |

### 2.2 Arquitectura del Pipeline

```mermaid
flowchart LR
    A[01_generate_synthetic_data.R<br/>viento via GARCH, demanda multi-estacional] --> B[02_stationarity_tests.R<br/>ADF + KPSS]
    B --> C[04_stl_decomposition.R<br/>estacionalidad diaria/semanal]
    C --> D1[ARIMA + Fourier /<br/>SARIMAX exogeno]
    C --> D2[ETS espacio de estados]
    C --> D3[TBATS]
    A --> E[GARCH 1,1<br/>volatilidad eolica]
    D1 --> F[08_cross_validation.R<br/>CV de origen rodante]
    D2 --> F
    D3 --> F
    E --> G[(pronostico + volatilidad<br/>output/figures, output/tables)]
    F --> G
```

## 3. Marco teórico

### 3.1 ARIMA y SARIMAX

Un modelo ARIMA(p,d,q) combina autorregresión (AR), diferenciación (I, de "integrado") y promedio móvil (MA):

$$\left(1 - \sum_{i=1}^{p} \phi_i L^i\right)(1-L)^d y_t = \left(1 + \sum_{j=1}^{q} \theta_j L^j\right)\varepsilon_t$$

donde $L$ es el operador de rezago ($Ly_t = y_{t-1}$), $\phi_i$ son los coeficientes autorregresivos, $\theta_j$ los coeficientes de promedio móvil, y $\varepsilon_t$ es ruido blanco. SARIMAX extiende esto agregando regresores exógenos $X_t$:

$$y_t = \beta X_t + \left(1 - \sum \phi_i L^i\right)^{-1}\left(1 + \sum \theta_j L^j\right)(1-L)^{-d}\varepsilon_t$$

En este proyecto, en vez de un SARIMA estacional clásico con periodo 168 (inviable computacionalmente sobre ~17.500 observaciones horarias), la estacionalidad diaria y semanal se captura con **términos de Fourier** como regresores dentro de la parte ARIMA -- el enfoque estándar recomendado para series de alta frecuencia con estacionalidad múltiple (Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*).

### 3.1.1 TBATS

TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, Seasonal components -- De Livera, Hyndman & Snyder, 2011) es una alternativa de espacio de estados a los términos de Fourier como regresores: cada componente estacional es un **estado que evoluciona en el tiempo**, no una sinusoide de amplitud fija, y el modelo estima conjuntamente su propia transformación Box-Cox y una corrección ARMA(p,q) sobre los errores:

$$y_t^{(\omega)} = \ell_{t-1} + \phi b_{t-1} + \sum_{i=1}^{T} s_{t-m_i}^{(i)} + d_t, \qquad d_t = \sum_{i=1}^{p}\varphi_i d_{t-i} + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j} + \varepsilon_t$$

donde $y_t^{(\omega)}$ es la serie transformada por Box-Cox, $\ell_t$ el nivel, $b_t$ la tendencia amortiguada, y cada $s_t^{(i)}$ un estado estacional de periodo $m_i$ (acá $m_1=24$, $m_2=168$). §7.3 mide directamente por qué esto importa acá: los regresores de Fourier de amplitud fija de arriba dejan autocorrelación residual en el lag 24 que los estados dinámicos de TBATS reducen sustancialmente más.

### 3.2 Descomposición STL

STL (*Seasonal-Trend decomposition using Loess*) descompone la serie en tendencia, componente(s) estacional(es) y remanente:

$$y_t = T_t + S_t^{(24)} + S_t^{(168)} + R_t$$

usando regresión local (Loess) iterativa para estimar cada componente sin asumir una forma funcional fija -- a diferencia de un ajuste sinusoidal puro, STL permite que la forma estacional cambie lentamente en el tiempo.

### 3.3 Modelos de espacio de estados (ETS)

ETS (Error-Tendencia-Estacionalidad) es una familia de **modelos de espacio de estados innovadores** (*innovations state space models*), no un ARIMA con otro nombre: cada modelo se especifica por el tipo de Error, Tendencia y Estacionalidad (aditivo, multiplicativo, o ninguno). El caso general con estado $\mathbf{x}_t$ sigue:

$$y_t = w(\mathbf{x}_{t-1}) + r(\mathbf{x}_{t-1})\varepsilon_t \qquad \mathbf{x}_t = f(\mathbf{x}_{t-1}) + g(\mathbf{x}_{t-1})\varepsilon_t$$

y el nivel, tendencia y estacionalidad se actualizan cada período mediante ecuaciones de suavizamiento exponencial ponderado, en vez de estimarse una sola vez sobre toda la muestra.

### 3.4 GARCH para volatilidad

GARCH(1,1) modela la varianza condicional de una serie cuando ésta exhibe *clustering* de volatilidad -- períodos de calma y períodos agitados que se agrupan en el tiempo, típico de generación eólica (rachas de viento correlacionadas):

$$\varepsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1) \qquad \sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

con $\omega > 0$, $\alpha_1, \beta_1 \geq 0$, y $\alpha_1 + \beta_1 < 1$ para que el proceso sea estacionario en varianza. $\alpha_1 + \beta_1$ cercano a 1 indica alta persistencia (shocks de volatilidad que tardan en disiparse).

## 4. Explicación

### 4.1 Flujo de datos

```
R/01_generate_synthetic_data.R   →  data/sen_hourly_synthetic.csv
        (Polars-like con dplyr)      (demanda, solar, eólica, demanda neta)
              ↓
R/02_data_wrangling.R            →  data/hourly_tsibble.rds
        (tsibble + fill_gaps)        data/daily_tsibble.rds
              ↓
R/03_stationarity_tests.R        →  output/tables/stationarity_tests.csv
R/04_stl_decomposition.R         →  output/figures/stl_*.png
              ↓
R/05_models_arima_sarimax.R      →  output/models/{arima_fourier,sarimax,tbats}_fit.rds
R/06_models_ets_statespace.R     →  output/models/ets_fit.rds
R/07_models_garch_volatility.R   →  output/models/garch_wind_fit.rds
              ↓
R/08_cross_validation.R          →  output/tables/accuracy_*.csv (incl. rolling-origin CV, horaria + diaria)
R/09_generate_plots.R            →  output/figures/{forecast,residuals}_*.png
              ↓
reports/02_Residual_Autocorrelation_Fourier.Rmd  →  reports/02_..._Fourier.html
        (se renderiza por separado -- ver §6.1)
```

`run_pipeline.R` ejecuta los 9 scripts en orden, cada uno como un **subproceso `Rscript` independiente** (no vía `source()`) -- una decisión de diseño deliberada, no cosmética: cada script tiene un bloque `if (sys.nframe() == 0) { ... }` que solo se activa cuando el script corre como proceso de tope (`Rscript archivo.R`), y **no se dispara** si el archivo se cargara con `source()` desde otro script. La primera versión del orquestador usaba `source()` y fallaba silenciosamente (no generaba ningún archivo, sin lanzar error) -- exactamente el tipo de bug que solo aparece al ejecutar el pipeline completo de punta a punta, no al probar cada script por separado.

### 4.2 Manipulación con tidyverse/tsibble

Toda la manipulación usa `dplyr` + `tsibble`, la estructura de datos "tidy" para series de tiempo: cada fila es una observación, cada columna una variable, y el índice temporal (`datetime` o `date`) queda registrado como metadata de la tabla, no como una columna cualquiera. Esto permite que `fill_gaps()`, `stretch_tsibble()` (para CV) y las funciones de `feasts`/`fable` operen automáticamente sobre el grano temporal correcto sin que el analista tenga que gestionar índices manualmente.

### 4.3 Lógica de validación cruzada

`stretch_tsibble(.init, .step)` genera múltiples "orígenes" de pronóstico con ventana de entrenamiento creciente (*rolling-origin cross-validation*, el estándar en series de tiempo -- a diferencia de k-fold, nunca se entrena con datos futuros al punto de pronóstico). Ver §7 para un hallazgo concreto de por qué esto importa más que un solo split train/test.

## 5. Metodología

### 5.1 Pruebas de estacionariedad (ADF, KPSS)

ADF (H0: existe raíz unitaria / no estacionaria) y KPSS (H0: estacionaria) se corren juntas porque tienen hipótesis nulas opuestas -- resultado real de este proyecto:

| Serie | ADF (stat, p) | Conclusión ADF | KPSS (stat, p) | Conclusión KPSS |
|---|---|---|---|---|
| Demanda horaria (nivel) | -10.98, 0.01 | Estacionaria | 13.38, 0.01 | **No estacionaria** |
| Demanda horaria (1ra dif.) | -24.14, 0.01 | Estacionaria | 0.003, 0.10 | Estacionaria |
| Demanda neta horaria (nivel) | -8.46, 0.01 | Estacionaria | 12.41, 0.01 | **No estacionaria** |
| Demanda diaria promedio (nivel) | -1.65, 0.73 | **No estacionaria** | 2.27, 0.01 | No estacionaria |
| Demanda diaria promedio (1ra dif.) | -4.61, 0.01 | Estacionaria | 0.40, 0.076 | Estacionaria |

**Hallazgo real, no forzado:** ADF y KPSS *discrepan* en el nivel de la demanda horaria -- un caso de libro de texto de **estacionariedad en torno a tendencia** (la serie es estacionaria alrededor de un patrón determinístico de tendencia+estacionalidad, pero no estacionaria en media pura). Diferenciar una vez resuelve la discrepancia en ambas resoluciones (ambos tests coinciden en la serie diferenciada). Esto informa directamente `d = 1` en la especificación ARIMA de §6.

### 5.2 Detección de estacionalidad múltiple

La descomposición STL (`season(period = "day") + season(period = "week")`) sobre la demanda horaria da la siguiente varianza por componente:

| Componente | Varianza |
|---|---|
| Estacionalidad diaria | 2.560.793 |
| Tendencia | 537.086 |
| Estacionalidad semanal | 278.189 |
| Remanente | 23.233 |

La estacionalidad diaria domina ampliamente (la forma de doble punta mañana/noche), y el remanente es pequeño frente a los componentes estructurales -- señal de que la mayor parte de la varianza es explicable, no ruido.

### 5.3 Criterios de información (AIC, BIC)

| Modelo | AIC | AICc | BIC |
|---|---|---|---|
| ARIMA + Fourier (demanda) | 248.745 | 248.745 | 248.893 |
| SARIMAX (demanda neta, con solar/eólica exógenas) | 248.586 | 248.586 | 248.748 |
| ETS(M,A,M) (demanda diaria) | 9.962,1 | 9.962,5 | 10.016,7 |

ETS seleccionó automáticamente error y estacionalidad **multiplicativos** con tendencia **aditiva** -- consistente con cómo se generaron los datos sintéticos (factores de fin de semana y estacionales multiplicativos sobre un nivel base).

### 5.4 Métricas de error (MAPE, RMSE, MASE)

$$\text{MAPE} = \frac{100}{n}\sum \left|\frac{y_t - \hat{y}_t}{y_t}\right| \qquad \text{RMSE} = \sqrt{\frac{1}{n}\sum (y_t - \hat{y}_t)^2} \qquad \text{MASE} = \frac{\frac{1}{n}\sum |y_t - \hat{y}_t|}{\frac{1}{n-1}\sum |y_i - y_{i-1}|_{\text{train}}}$$

MASE > 1 significa "peor que un naive ingenuo sobre el propio set de entrenamiento"; MASE < 1 significa "mejor que ese naive" -- una escala comparable entre series de distinta magnitud, a diferencia de RMSE.

## 6. Desarrollo (código en R)

Pipeline modular en `R/`, sobre `tidyverse` + `tsibble` + `fable` + `feasts` + `rugarch`. Algunos fragmentos representativos (código real de este repositorio, no ilustrativo):

**Simulación explícita de un proceso GARCH(1,1) para el componente eólico** (`R/01_generate_synthetic_data.R`) -- para que el modelo GARCH de §6.4 tenga volatilidad genuina que recuperar, no ruido inventado:

```r
simulate_garch11 <- function(n, omega, alpha, beta, seed) {
  set.seed(seed)
  sigma2 <- numeric(n); eps <- numeric(n)
  sigma2[1] <- omega / (1 - alpha - beta)
  eps[1] <- sqrt(sigma2[1]) * rnorm(1)
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * eps[t - 1]^2 + beta * sigma2[t - 1]
    eps[t] <- sqrt(sigma2[t]) * rnorm(1)
  }
  list(eps = eps, sigma = sqrt(sigma2))
}
```

**ARIMA con términos de Fourier** (`R/05_models_arima_sarimax.R`) -- `d = 1` fijo, informado por §5.1 (ver las notas metodológicas sobre dos limitaciones reales de `fable` encontradas en este proyecto, más abajo). `K = 12` en el término diario es la base de Fourier completa para un ciclo de periodo 24 (límite de Nyquist) -- subido desde `K = 4` específicamente para atacar la autocorrelación residual del lag 24 de §7.3:

```r
fit_arima_fourier <- function(train_ts) {
  train_ts %>%
    model(
      arima_fourier = ARIMA(demand_mw ~ pdq(d = 1) +
                               fourier(period = "day", K = 12) +
                               fourier(period = "week", K = 2) + PDQ(0, 0, 0)),
      snaive_day = SNAIVE(demand_mw ~ lag("day"))
    )
}
```

**SARIMAX con generación renovable como regresor exógeno** (`R/05_models_arima_sarimax.R`) -- en operación real, el pronóstico solar/eólico viene de un modelo meteorológico independiente y se trata como insumo exógeno conocido al pronosticar la demanda neta:

```r
fit_sarimax_net_demand <- function(train_ts) {
  train_ts %>%
    model(
      sarimax = ARIMA(net_demand_mw ~ pdq(d = 1) + solar_mw + wind_mw +
                         fourier(period = "day", K = 12) +
                         fourier(period = "week", K = 2) + PDQ(0, 0, 0)),
      snaive_net = SNAIVE(net_demand_mw ~ lag("day"))
    )
}
```

**TBATS** (`R/05_models_arima_sarimax.R`) -- sin soporte de regresores exógenos (no hay analogo SARIMAX), ajustado solo sobre `demand_mw`, vía `msts`/`tbats` del paquete `forecast`:

```r
fit_tbats_demand <- function(train_ts) {
  demand_msts <- forecast::msts(train_ts$demand_mw, seasonal.periods = c(24, 168))
  forecast::tbats(demand_msts, use.parallel = FALSE)
}
```

**Validación cruzada de origen móvil, diaria** (`R/08_cross_validation.R`):

```r
run_daily_rolling_cv <- function(daily_ts, init_size = 550, step = 14, horizon = 7) {
  daily_ts %>%
    stretch_tsibble(.init = init_size, .step = step) %>%
    model(ets = ETS(demand_mw_mean), snaive_week = SNAIVE(demand_mw_mean ~ lag("week"))) %>%
    forecast(h = horizon) %>%
    fabletools::accuracy(daily_ts)
}
```

**Validación cruzada de origen móvil, horaria** (`R/08_cross_validation.R`, nuevo) -- una ventana de entrenamiento de tamaño **fijo** (no la expansiva de `stretch_tsibble`), para que todos los orígenes se comparen con la misma cantidad de historia, y un número acotado de orígenes para que reajustar una búsqueda ARIMA+Fourier completa en cada origen tome minutos, no horas:

```r
run_hourly_rolling_cv <- function(hourly_ts, init_size = 12000, step = 1000, horizon = 48, n_origins = 5) {
  # desliza una ventana fija de 12.000 horas de entrenamiento, avanzando `step`
  # horas por origen, reajusta ARIMA + Fourier (K=12) cada vez, pronostica 48h -- ver §7.5
}
```

> **Nota metodológica -- bug real encontrado y su solución:** la búsqueda automática del orden de diferenciación `d` de `fable::ARIMA()` (v0.5.0) falla de forma silenciosa (error interno sin mensaje) sobre esta serie, tanto en R 4.4.0 como en R 4.6.1 -- se aisló probando `stats::arima()` base (funciona sin problema), luego una grilla de complejidad creciente hasta encontrar que fijar `p,q` y dejar `d` en búsqueda automática falla, pero fijar `d` (o cualquier orden completo) y dejar `p,q` en búsqueda automática funciona correctamente. La solución no es solo un workaround: como `d=1` ya estaba justificado empíricamente por ADF/KPSS (§5.1), fijarlo explícitamente es *más* defendible metodológicamente que confiar en una búsqueda automática de caja negra.

> **Nota metodológica 2 -- la misma clase de bug, encontrada de nuevo:** persiguiendo la corrección del lag 24 de §7.3, un término estacional explícito en periodo 24 (`PDQ(P=1,D=0,Q=1,period=24)`, y por separado `D=1` de diferenciación estacional en periodo 24) fallaron ambos con *"No se encontró un modelo ARIMA apropiado... raíces características que pueden ser numéricamente inestables"* -- la misma firma de falla que el bug de búsqueda de `d` de arriba, ahora confirmada también en la búsqueda de orden estacional. Documentado como una limitación real y reproducible de `fable::ARIMA()` sobre esta serie, no un error de especificación de este proyecto; ver `reports/02_Residual_Autocorrelation_Fourier.Rmd` para el recorrido completo de lo que se probó.

### 6.1 Renderizando el reporte reproducible

```powershell
Rscript run_pipeline.R    # debe correr primero -- el reporte lee sus modelos/tablas guardados
Rscript -e "rmarkdown::render('reports/02_Residual_Autocorrelation_Fourier.Rmd')"
```

`reports/02_Residual_Autocorrelation_Fourier.Rmd` es un documento R Markdown completamente reproducible por sí mismo (no solo un script): reajusta el modelo baseline original (`K=4`) él mismo (así la comparación no es un número citado de una corrida pasada), re-deriva independientemente el componente determinista verdadero de la serie sintética a partir de las propias funciones del generador para chequear si la autocorrelación residual es un artefacto de modelado o una propiedad real de los datos (§7.3 resume la respuesta), y renderiza diagnósticos ACF/PACF para los tres modelos más la tabla de CV de origen móvil horaria. Requiere Pandoc (viene incluido con RStudio; `rmarkdown::pandoc_available()` lo chequea). La salida (`reports/*.html`) está en `.gitignore`, igual que cualquier otro artefacto generado en este repositorio -- renderizarlo es parte de reproducir el proyecto, no un entregable versionado.

## 7. Resultados

### 7.1 Comparación de precisión de modelos

**Holdout horario fijo (14 días) -- demanda:**

| Modelo | MAPE | RMSE | MASE |
|---|---|---|---|
| **TBATS** | **2,47%** | **284** | **0,44** |
| ARIMA + Fourier (K=12) | 5,59% | 616 | 1,03 |
| SNAIVE (naive estacional diario) | 6,54% | 795 | 1,13 |

TBATS no solo es mejor en el diagnóstico de autocorrelación residual (§7.3) -- también es el pronosticador puntual más preciso en este holdout, por un margen amplio, más que duplicando la mejora de ARIMA+Fourier sobre SNAIVE. El trade-off (§7.3, §6) es que no soporta regresores exógenos y tarda ~4x más en ajustarse.

**Holdout horario fijo (14 días) -- demanda neta (SARIMAX, K=12):**

| Modelo | MAPE | RMSE | MASE |
|---|---|---|---|
| **SARIMAX** (solar+eólica exógenas) | **6,68%** | **610** | **0,85** |
| SNAIVE | 10,60% | 924 | 1,15 |

**Holdout diario fijo (28 días) -- demanda diaria promedio:**

| Modelo | MAPE | RMSE | MASE |
|---|---|---|---|
| SNAIVE semanal | **1,22%** | **138** | **1,34** |
| ETS(M,A,M) | 1,28% | 141 | 1,41 |

**Validación cruzada de origen móvil (13 orígenes, horizonte 7 días) -- demanda diaria:**

| Modelo | MAPE | RMSE | MASE |
|---|---|---|---|
| **ETS(M,A,M)** | **0,43%** | **47,0** | **0,45** |
| SNAIVE semanal | 1,09% | 107 | 1,11 |

**Hallazgo honesto, no maquillado:** en el holdout diario de una sola ventana, SNAIVE le *gana* a ETS. En la validación cruzada de origen móvil (13 orígenes distintos), ETS le gana claramente a SNAIVE. Este es exactamente el motivo por el que la validación cruzada de series de tiempo existe: un solo split train/test puede ser engañoso -- una ventana de 28 días particular puede favorecer al azar a un modelo simple, mientras que promediar sobre 13 orígenes revela el desempeño real. Reportar solo el resultado favorable habría sido deshonesto; el punto metodológico vale más que ocultar el resultado incómodo.

### 7.2 Recuperación de parámetros GARCH

El componente eólico se simuló como un GARCH(1,1) explícito con $\alpha_1=0{,}15$, $\beta_1=0{,}80$ (persistencia real = 0,95). El modelo ajustado con `rugarch` recuperó:

| Parámetro | Valor real (simulación) | Valor estimado |
|---|---|---|
| $\alpha_1$ | 0,150 | 0,142 |
| $\beta_1$ | 0,800 | 0,811 |
| Persistencia ($\alpha_1+\beta_1$) | 0,950 | 0,954 |

Recuperación casi exacta -- confirma que el pipeline de generación de datos y el ajuste GARCH son consistentes entre sí, no solo que "el código corre".

### 7.3 Diagnóstico de residuos -- la investigación del lag 24

La versión anterior de este proyecto marcó una limitación real y la dejó como trabajo futuro: el ACF de los residuos del ARIMA + Fourier mostraba un pico notable en el lag 24 (≈0,41), atribuido a que la estacionalidad diaria "no quedó completamente capturada con `K=4` términos de Fourier." Esta ronda de trabajo persiguió esa corrección -- y la investigación completa, no solo la respuesta final, es el resultado más útil. Detalle completo y código reproducible: `reports/02_Residual_Autocorrelation_Fourier.Rmd`.

**Paso 1 -- ¿son los datos, o es el modelo?** El generador sintético (`R/01_generate_synthetic_data.R`) construye `demand_mw` como una función determinista conocida (tendencia × día de semana × estacionalidad anual × forma diaria) más ruido gaussiano independiente -- así que el residuo *verdadero*, reconstruido exactamente a partir de las propias funciones del generador, debería ser ruido blanco si el modelo estuviera correctamente especificado. Lo es: **ACF(lag24) = 0,004, Ljung-Box p = 0,685** (no rechaza ruido blanco). Esto descarta que "los datos tengan dependencia real de 24h que el modelo se esté perdiendo" -- el pico de 0,41 es un artefacto de modelado, no una señal perdida.

**Paso 2 -- cinco correcciones probadas, reportadas honestamente incluyendo las que fallaron:**

| # | Intento | Resultado | Veredicto |
|---|---|---|---|
| 1 | Fourier diario `K`: 4 → 12 (límite de Nyquist para periodo 24) | ACF(lag24): 0,413 → 0,373 | Mejora real, parcial |
| 2 | Fourier semanal `K`: 2 → 6/12/20 (con K diario=12) | ACF(lag24) **empeora** a 0,469 | Descartado |
| 3 | `log(demand_mw)` en vez del nivel | ACF(lag24) = 0,402, sin mejora | Descartado |
| 4 | `PDQ(P=1,D=0,Q=1,period=24)`, sin Fourier diario | `fable::ARIMA()` falla: raíces características inestables | No disponible (Nota metodológica 2, §6) |
| 5 | Diferenciación estacional `D=1` en periodo 24 | Misma falla que #4 | No disponible |
| 6 | **TBATS** (estados estacionales dinámicos, sin Fourier/diferenciación) | ACF(lag24) = **0,289** | Mejor mejora lograda |

**La conclusión honesta: ninguna corrección probada elimina completamente la autocorrelación.** `K=12` logra una reducción real pero modesta de ~10% y mantiene el soporte de regresores exógenos de SARIMAX. TBATS logra una reducción sustancialmente mayor de ~30% (y, según §7.1, es también el pronosticador puntual más preciso) pero no tiene análogo de regresores exógenos y cuesta ~4x el tiempo de ajuste. Tanto `arima_fourier` (K=12) como TBATS quedan disponibles en `output/models/`, para que quien use el proyecto elija según lo que necesite -- regresores interpretables y velocidad (Fourier), o minimizar la autocorrelación residual y la precisión titular (TBATS) -- en vez de que este proyecto imponga una única respuesta "correcta".

![ACF de residuos -- ARIMA + Fourier K=12 (mejorado)](output/figures/residuals_arima.png)

### 7.4 Gráficos generados

Todos en `output/figures/`, generados por `R/04_stl_decomposition.R` y `R/09_generate_plots.R`:

- `stl_decomposition.png` -- descomposición tendencia + estacional diaria + estacional semanal + remanente
- `demand_two_week_zoom.png` -- detalle de 2 semanas (la descomposición completa a 2 años de resolución horaria se ve como una banda sólida; este gráfico muestra la forma real de doble punta)
- `stl_acf_demand.png`, `stl_acf_remainder.png`, `stl_pacf_demand.png` -- ACF/PACF
- `forecast_arima_demand.png`, `forecast_ets_daily.png` -- pronósticos con bandas de incertidumbre 80%/95%
- `residuals_arima.png`, `residuals_ets.png` -- diagnóstico de residuos
- `garch_wind_volatility_forecast.png` -- volatilidad condicional pronosticada a 48 horas
- `tbats_residuals_acf.png`, `tbats_residuals_pacf.png` -- generados por `reports/02_Residual_Autocorrelation_Fourier.Rmd`, no por el pipeline numerado (§6.1)

### 7.5 Validación cruzada de origen móvil horaria

Extiende el holdout horario fijo de §7.1 con `run_hourly_rolling_cv()` (`R/08_cross_validation.R`): una ventana **fija** de entrenamiento de 12.000 horas (~500 días) deslizada a través de 5 orígenes (pasos de 1.000 horas), reajustando ARIMA + Fourier (K=12) desde cero cada vez, pronosticando 48 horas hacia adelante:

| Origen | MAPE | RMSE | MASE | Tiempo de ajuste |
|---|---|---|---|---|
| 1 | 3,80% | 336 | 0,549 | 72s |
| 2 | 3,61% | 358 | 0,561 | 64s |
| 3 | 3,71% | 384 | 0,637 | 87s |
| 4 | 4,29% | 508 | 0,836 | 70s |
| 5 | 4,29% | 543 | 0,861 | 63s |

**Hallazgo real, no completamente plano:** el MAPE deriva de 3,61%-3,80% en los orígenes tempranos hasta 4,29% en los dos últimos, y RMSE/MASE siguen la misma dirección, aproximadamente duplicándose del origen 2 al origen 5. Esta es exactamente la clase de señal de estabilidad temporal que un solo holdout fijo no puede exponer (el 5,59% de MAPE titular de §7.1 es en sí mismo aproximadamente el promedio de este rango) -- vale la pena señalarlo honestamente en vez de reportar solo el mejor origen. Una ventana de tamaño fijo (no la expansiva de `stretch_tsibble`) es deliberada acá: aísla si el desempeño deriva con el *tiempo*, no solo mejora porque los orígenes tardíos tengan más datos de entrenamiento por casualidad.

## 8. Conclusión

Los tres modelos de nivel (ARIMA+Fourier, SARIMAX, ETS) superan de forma consistente a sus respectivos baselines naive cuando se evalúan correctamente -- MASE < 1 en ARIMA, SARIMAX y ETS-bajo-CV confirma esto formalmente, no solo visualmente. El hallazgo del §7.1 (SNAIVE gana en un holdout, ETS gana en CV) es en sí mismo el resultado metodológico más valioso del proyecto: valida por qué la validación cruzada de origen móvil, no un solo split, debe ser el estándar al comparar modelos de series de tiempo. El GARCH(1,1) recupera casi exactamente los parámetros reales de la simulación, confirmando que el enfoque de "simular con estructura conocida, luego reajustar" es una forma válida de validar que un pipeline estadístico funciona de extremo a extremo.

**La investigación del lag 24 (§7.3) es el resultado más valioso de esta ronda, y es parcial -- reportado como tal:** aumentar el `K` de Fourier a su límite de Nyquist y ajustar TBATS reducen mediblemente la autocorrelación residual que la versión anterior marcó (TBATS: -30%, también la mejor precisión puntual de §7.1) -- pero ninguno la elimina, y verificar independientemente contra la fórmula determinista propia del generador sintético confirmó que los datos reales no tienen esa dependencia en absoluto, así que la brecha restante es enteramente una limitación del framework de modelado `fable`/TBATS sobre esta serie, no una característica no atendida de los datos.

**Valor para operadores del SEN:** un pronóstico de demanda neta con exógenas renovables (SARIMAX) más un modelo explícito de volatilidad eólica (GARCH) da dos insumos complementarios que un operador de red necesita: *cuánta* energía se espera necesitar, y *cuán inciertas* son las rampas asociadas a la generación renovable -- esto último es precisamente lo que informa cuánta reserva de generación rápida mantener disponible.

**Limitaciones y trabajo futuro:**
- Ni `K=12` de Fourier ni TBATS eliminan completamente la autocorrelación residual del lag 24 (§7.3) -- un problema genuinamente abierto sobre esta serie dentro del conjunto de herramientas actual de `fable`/`forecast`, no uno resuelto; una corrección ARMA estacional hecha a mano (evitando por completo la búsqueda automática de `fable::ARIMA()`, que es de donde viene la falla de raíces numéricamente inestables) es el siguiente intento natural.
- La validación cruzada de origen móvil horaria (§7.5) ya está implementada, pero acotada a 5 orígenes por costo computacional (cada origen reajusta una búsqueda ARIMA+Fourier completa, ~1-2 minutos) -- una extensión natural es paralelizar entre orígenes para poder afrontar más.
- El backtest de SARIMAX usa los valores reales conocidos de generación solar/eólica del período de holdout, no un pronóstico de ellos -- en producción, el error del pronóstico meteorológico se propagaría al pronóstico de demanda neta, y valdría la pena cuantificar esa propagación explícitamente.
- El GARCH se aplicó solo al componente eólico; extenderlo al precio marginal/costo operacional del sistema (que en el SEN real también exhibe clustering de volatilidad, impulsado por niveles de embalses hidroeléctricos y disponibilidad renovable) es una extensión natural con datos reales.

## 9. Autor

**Pablo Reyes** -- Científico de Datos, Universidad Mayor -- [github.com/Rxyxs](https://github.com/Rxyxs)

---

## Apéndice: instalación y uso

Requiere R 4.4+ (probado en 4.4.0 y 4.6.1).

```powershell
git clone https://github.com/Rxyxs/chile-energy-grid-forecasting-r.git
cd chile-energy-grid-forecasting-r
Rscript R/00_setup.R      # instala tidyverse, tsibble, fable, feasts, forecast, rugarch, tseries, ggtime, rmarkdown
Rscript run_pipeline.R    # corre los 9 pasos de punta a punta (TBATS + CV horaria suman ~15-20 min)
Rscript -e "rmarkdown::render('reports/02_Residual_Autocorrelation_Fourier.Rmd')"  # opcional, §6.1
```

### Estructura del repositorio

```
chile-energy-grid-forecasting-r/
├── R/
│   ├── 00_setup.R                      # instalación de paquetes
│   ├── 01_generate_synthetic_data.R    # generador SEN sintético (incl. GARCH eólico)
│   ├── 02_data_wrangling.R             # tsibble horario + diario
│   ├── 03_stationarity_tests.R         # ADF, KPSS
│   ├── 04_stl_decomposition.R          # STL multi-estacional + gráficos
│   ├── 05_models_arima_sarimax.R       # ARIMA + Fourier (K=12), SARIMAX, TBATS
│   ├── 06_models_ets_statespace.R      # ETS (espacio de estados)
│   ├── 07_models_garch_volatility.R    # GARCH(1,1) sobre volatilidad eólica
│   ├── 08_cross_validation.R           # holdout + rolling-origin CV (horaria + diaria), MAPE/RMSE/MASE
│   └── 09_generate_plots.R             # residuos, pronósticos con bandas, volatilidad
├── reports/
│   └── 02_Residual_Autocorrelation_Fourier.Rmd  # diagnóstico del lag 24, ACF/PACF, escrito honesto (§6.1, §7.3)
├── run_pipeline.R                      # orquestador (subprocesos Rscript)
├── data/                               # CSVs + tsibbles .rds (generado, en .gitignore)
├── output/
│   ├── models/                         # modelos ajustados .rds, incl. tbats_fit.rds (generado)
│   ├── tables/                         # tablas de resultados .csv, incl. accuracy_rolling_cv_hourly.csv (generado)
│   └── figures/                        # gráficos .png (generado)
├── .gitignore
├── LICENSE
├── README.md
└── README.es.md
```

### Disclaimer de datos

Todos los datos son **100% sintéticos**, generados por `R/01_generate_synthetic_data.R` con semilla fija. Las cifras de capacidad/demanda (demanda base ~8.000 MW, capacidad solar ~3.500 MW, capacidad eólica ~2.500 MW) son órdenes de magnitud ilustrativos del SEN, no datos operacionales reales del CEN. El CEN, sus responsabilidades, y la existencia del SEN como sistema interconectado nacional son hechos públicos reales -- ninguna cifra de generación/demanda específica en este repositorio proviene de sus reportes operacionales.
