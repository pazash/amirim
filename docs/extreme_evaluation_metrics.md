# Extreme Weather Evaluation Metrics — Scientific Specification

> **Scope**: Evaluation of Aurora 1.5 (and HRES baseline) on **heat waves** and **heavy precipitation** events using ExtremeWeatherBench (EWB).
>
> **Verification target**: ERA5 reanalysis (0.25° × 0.25°, 6-hourly).
>
> **Document version**: 1.0 — August 2025

---

## 1. Background and Motivation

Standard global-aggregate metrics (RMSE, ACC) systematically mask model deficiencies
during extreme events because such events occupy a tiny fraction of the
spatio-temporal domain (Pasche _et al._, 2025; Camps-Valls _et al._, 2025).
AI weather models — including GraphCast, Pangu-Weather, FourCastNet and Aurora —
exhibit a well-documented **amplitude bias** (a.k.a. "regression-to-the-mean")
in which predicted surface intensities collapse toward climatological values
at medium-range lead times, even when the large-scale circulation pattern remains
discernible (Aurora diagnostic study, 2026; Pasche _et al._, 2025).

This document specifies a set of **amplitude metrics** and **spatial-extent
metrics**, evaluated across multiple lead times, designed to quantify precisely
where and how Aurora 1.5 diverges from HRES during high-impact events.

---

## 2. Event Definitions

### 2.1 Heat Waves

| Attribute | Specification | Justification |
|---|---|---|
| **Variable** | 2-metre temperature (`2t`, K) | Standard proxy for surface heat stress; available in Aurora outputs and ERA5 |
| **Threshold** | Grid-point-local 90th percentile of daily maximum `2t` from the 1991–2020 ERA5 climatology | WMO guidance; used by Copernicus C3S and the ETCCDI `TX90p` index (Alexander _et al._, 2006) |
| **Duration** | ≥ 3 consecutive days above threshold | WMO/WHO common definition; balances sensitivity and specificity |
| **Region** | Event bounding box supplied by EWB case YAML + 5° buffer on each edge | Captures the full spatial footprint and allows evaluation of spatial displacement errors |

**Note**: EWB provides curated case metadata (start date, end date, bounding box)
for each catalogued heat wave. We rely on these metadata and do _not_ re-detect events.

### 2.2 Heavy Precipitation

| Attribute | Specification | Justification |
|---|---|---|
| **Variable** | Total precipitation (`tp`, m) accumulated over 24 h | Standard WMO reporting interval; avoids sub-daily timing noise |
| **Threshold** | Grid-point-local 95th percentile of _wet-day_ (≥ 1 mm/day) precipitation from the 1991–2020 ERA5 climatology | Follows ETCCDI `R95p` convention (Zhang _et al._, 2011); wet-day conditioning avoids dilution by dry grid points |
| **Duration** | Single 24-h window or longer as defined by the EWB case | Heavy precipitation events range from flash (hours) to multi-day; EWB case metadata dictates the window |
| **Region** | Event bounding box from EWB case YAML + 5° buffer | Same justification as for heat waves |

---

## 3. Lead Time Convention

### 3.1 Definition

Lead time is defined as:

```
τ = t_valid − t_init
```

where `t_init` is the forecast initialization time and `t_valid` is the valid
time of the prediction being evaluated.

### 3.2 Convention for Multi-Day Events

For events spanning multiple days (e.g., a heat wave from July 5–10), we
adopt the **"lead time to event onset"** convention:

- The model is initialized at `t_init`, which is `τ_onset` hours _before_ the
  event start date.
- All subsequent forecast time-steps within the event window inherit
  **incrementing** lead times relative to `t_init`.

| Concept | Formula |
|---|---|
| Lead time to onset | `τ_onset = t_event_start − t_init` |
| Lead time at step _k_ | `τ_k = τ_onset + k × Δt` (where Δt = 6 h for Aurora) |

**Rationale**: This is the standard convention in operational early-warning
evaluation (NWS SEHOS; Copernicus C3S; EWB default pipeline). It measures
how much advance notice a model can give before the hazard begins.

### 3.3 Target Lead-Time Buckets

We evaluate at the following initialization offsets _to event onset_:

| Bucket label | τ_onset (hours) | Meteorological range |
|---|---|---|
| **Day 1** | 24 | Short-range |
| **Day 3** | 72 | Short-range |
| **Day 5** | 120 | Medium-range |

At each bucket, the model is rolled out through the entire event duration,
producing a set of lead-time-specific forecasts within the event window.
All metrics below are then computed **per lead time step** and reported as
functions of `τ_k`.

---

## 4. Metrics

We organise metrics into two families: **amplitude metrics** (do the forecasts
capture the intensity of the extreme?) and **spatial-extent metrics** (do the
forecasts capture where the extreme occurs?).

### 4.1 Amplitude Metrics

#### 4.1.1 Peak Amplitude Error (PAE)

**Purpose**: Quantify the model's bias in predicting the most extreme value
observed during the event, which is the most operationally critical quantity.

**Definition** (per lead time `τ`, per event case _c_):

```
PAE(τ, c) = max_s∈S [ f(s, τ) ] − max_s∈S [ o(s, τ) ]
```

where:
- `f(s, τ)` = forecast field at grid point _s_ and lead time `τ`
- `o(s, τ)` = ERA5 (truth) field at the same point and valid time
- _S_ = set of grid points within the event bounding box

**Interpretation**: PAE < 0 indicates underestimation of the peak (the
well-known amplitude-bias). PAE > 0 indicates overestimation.

**Literature basis**: Pasche _et al._ (2025) report peak-intensity errors
as the primary diagnostic distinguishing AI models from HRES during the
2021 Pacific Northwest heatwave.

---

#### 4.1.2 Conditional Bias of Extremes (CBE)

**Purpose**: Measure the _mean_ forecast bias computed **only over grid points
where the truth exceeds the extreme threshold** — i.e., how well does the
model capture the intensity _where_ the extreme is actually occurring?

**Definition** (per lead time `τ`):

```
CBE(τ) = (1 / |E|) × Σ_{s∈E} [ f(s, τ) − o(s, τ) ]
```

where:
- _E_ = { s ∈ S : o(s, τ) ≥ Q_p } (the set of "extreme" grid points)
- Q_p = the p-th percentile threshold (p = 90 for heat waves, p = 95 for
  precipitation, computed from the event-specific ERA5 field at that valid
  time, not the climatology)

**Interpretation**: Negative CBE quantifies systematic underestimation at
extreme grid points. Unlike global RMSE, CBE is not diluted by the vast
majority of non-extreme grid points.

**Literature basis**: Conditional biases restricted to extreme quantiles are
recommended by Taggart _et al._ (2022, _Monthly Weather Review_) for
extreme-event verification.

---

#### 4.1.3 Root Mean Squared Error (RMSE)

**Purpose**: Provide a standard reference metric for global (domain-level)
model accuracy. This is the conventional baseline used by WeatherBench 2
(Rasp _et al._, 2024) and allows direct comparison with published results.

**Definition** (per lead time `τ`, over domain _S_):

```
RMSE(τ) = sqrt( (1 / |S|) × Σ_{s∈S} [ f(s, τ) − o(s, τ) ]² )
```

**Note**: We compute RMSE over the event bounding box (+ buffer), not
globally, to maintain focus on the event region while still providing an
aggregate accuracy measure.

---

### 4.2 Spatial-Extent Metrics

#### 4.2.1 Intersection over Union (IoU) / Critical Success Index (CSI)

**Purpose**: Measure the spatial overlap between the binary footprint of
predicted and observed extremes. IoU is equivalent to the Critical Success
Index (CSI), a standard in forecast verification.

**Definition**: Given binary fields derived by thresholding both forecast
and observation at the same threshold Q_p:

```
F_binary(s) = 1 if f(s, τ) ≥ Q_p, else 0
O_binary(s) = 1 if o(s, τ) ≥ Q_p, else 0

IoU(τ) = TP / (TP + FP + FN)
```

where:
- TP = Σ_s F_binary(s) × O_binary(s)  (true positives)
- FP = Σ_s F_binary(s) × (1 − O_binary(s))  (false positives)
- FN = Σ_s (1 − F_binary(s)) × O_binary(s)  (false negatives)

**Threshold**:
- Heat waves: Q_p = 90th percentile of the ERA5 `2t` field at the valid time
  within the event bounding box.
- Precipitation: Q_p = 95th percentile of the ERA5 24-h `tp` field (wet days
  only) at the valid time within the event bounding box.

**Interpretation**: IoU = 1 indicates perfect spatial overlap. IoU = 0
indicates no overlap. The metric is harsh on both displacement and area errors.

**Literature basis**: CSI/IoU is standard for binary event verification
(Wilks, 2011, _Statistical Methods in the Atmospheric Sciences_; Roberts and
Lean, 2008).

---

#### 4.2.2 Fractions Skill Score (FSS)

**Purpose**: Assess the spatial scale at which the forecast becomes
"useful" by comparing the fractional coverage of extreme events within
neighbourhoods of increasing size. FSS avoids the "double-penalty" problem
inherent in point-wise metrics: a small spatial displacement in an intense
forecast scores poorly in RMSE but well in FSS at appropriate scales.

**Definition**: For a given neighbourhood radius _r_ (in grid points):

```
FSS(τ, r) = 1 − [ Σ_s (O_r(s) − F_r(s))² ] / [ Σ_s O_r(s)² + Σ_s F_r(s)² ]
```

where:
- `F_r(s)` = fraction of grid points within a (2r+1)×(2r+1) box centred on
  _s_ where the forecast exceeds Q_p
- `O_r(s)` = same, for the observation

**Neighbourhood radii**: We evaluate at r ∈ {1, 3, 5, 10, 20} grid points,
corresponding to approximately {25 km, 75 km, 125 km, 250 km, 500 km}
at 0.25° resolution.

**Uniform skill threshold** (Roberts and Lean, 2008):

```
FSS_uniform = 0.5 + f_o / 2
```

where `f_o` is the observed fraction of extreme grid points in the domain.
A model is considered "useful" at scale _r_ when FSS(r) ≥ FSS_uniform.

**Threshold**: Same Q_p as for IoU (90th percentile for heat, 95th for precip).

**Literature basis**: FSS is the standard spatial verification score for
high-resolution precipitation (Roberts and Lean, 2008, _Monthly Weather
Review_; Mittermaier _et al._, 2013). It is increasingly applied to
temperature extremes (Ebert, 2008).

---

#### 4.2.3 Extreme Area Ratio (EAR)

**Purpose**: Quantify whether the model predicts the correct _total area_
of the extreme event, independent of exact location.

**Definition**:

```
EAR(τ) = A_f(τ) / A_o(τ)
```

where:
- `A_f(τ)` = number of grid points where f(s, τ) ≥ Q_p
- `A_o(τ)` = number of grid points where o(s, τ) ≥ Q_p

**Interpretation**:
- EAR = 1 → correct total extreme area
- EAR < 1 → model under-predicts the spatial extent (too few extreme grid points)
- EAR > 1 → model over-predicts the spatial extent

**Literature basis**: Area-based diagnostics are a component of the SAL
(Structure-Amplitude-Location) framework (Wernli _et al._, 2008) and
are recommended for extreme precipitation evaluation.

---

## 5. Summary Table

| # | Metric | Family | Variable(s) | What it measures | EWB base class |
|---|---|---|---|---|---|
| 1 | Peak Amplitude Error (PAE) | Amplitude | 2t / tp | Bias at domain max | `BaseMetric` |
| 2 | Conditional Bias of Extremes (CBE) | Amplitude | 2t / tp | Mean bias at extreme grid points | `BaseMetric` |
| 3 | RMSE | Amplitude (baseline) | 2t / tp | Global domain error | `BaseMetric` (built-in) |
| 4 | Intersection over Union (IoU/CSI) | Spatial | 2t / tp | Binary footprint overlap | `ThresholdMetric` |
| 5 | Fractions Skill Score (FSS) | Spatial | 2t / tp | Scale-dependent spatial skill | `BaseMetric` |
| 6 | Extreme Area Ratio (EAR) | Spatial | 2t / tp | Total extreme area bias | `BaseMetric` |

---

## 6. Lead-Time-Dependent Analysis Protocol

For each event case _c_ in the EWB case list:

1. For each τ_onset ∈ {24, 72, 120} hours:
   a. Initialize Aurora at `t_init = t_event_start − τ_onset`.
   b. Roll out Aurora with 6-h steps through the event end.
   c. Also extract the HRES forecast initialised at the same `t_init`,
      slicing the lead times that fall within the event window.
   d. Compute all six metrics at every 6-h step within the event window.
2. Aggregate results:
   - **Per-case**: Plot each metric vs. lead time `τ_k` for Aurora and HRES.
   - **Cross-case**: Average each metric across all cases of the same event
     type at matched `τ_k` values.
   - **Summary**: Report mean ± standard deviation across cases for each
     τ_onset bucket.

---

## 7. Key References

1. **Pasche, O.C., Wider, J., Zhang, Z., Camps-Valls, G., and Runge, J.** (2025).
   "Validating Deep Learning Weather Forecast Models on Recent High-Impact
   Extreme Events." _Artificial Intelligence for the Earth Systems_, 4(1).

2. **Camps-Valls, G. _et al._** (2025). "Artificial intelligence for modeling
   and understanding extreme weather and climate events." _Nature Communications_.

3. **Roberts, N.M. and Lean, H.W.** (2008). "Scale-Selective Verification of
   Rainfall Accumulations from High-Resolution Forecasts of Convective Events."
   _Monthly Weather Review_, 136, 78–97.

4. **Wernli, H., Paulat, M., Hagen, M., and Frei, C.** (2008). "SAL—A Novel
   Quality Measure for the Verification of Quantitative Precipitation
   Forecasts." _Monthly Weather Review_, 136(11), 4470–4487.

5. **Wilks, D.S.** (2011). _Statistical Methods in the Atmospheric Sciences_.
   3rd ed. Academic Press.

6. **Alexander, L.V. _et al._** (2006). "Global observed changes in daily
   climate extremes of temperature and precipitation." _Journal of Geophysical
   Research_, 111(D5).

7. **Rasp, S., Hoyer, S., Merose, A., _et al._** (2024). "WeatherBench 2: A
   Benchmark for the Next Generation of Data-Driven Global Weather Models."
   _Journal of Advances in Modeling Earth Systems_, 16(6).

8. **Taggart, R.J., Leuenberger, M., and Germann, U.** (2022). "Conditional
   verification for extreme events." _Monthly Weather Review_, 150(6).

---

## 8. Glossary

| Term | Definition |
|---|---|
| **τ** | Lead time (hours from initialization to valid time) |
| **τ_onset** | Lead time from initialization to event start |
| **Q_p** | The p-th percentile of the reference field (ERA5) |
| **EWB** | ExtremeWeatherBench |
| **HRES** | ECMWF High-Resolution Deterministic Forecast (IFS) |
| **ERA5** | ECMWF Reanalysis v5, used as verification truth |
| **2t** | 2-metre temperature |
| **tp** | Total precipitation |
| **FSS** | Fractions Skill Score |
| **IoU** | Intersection over Union |
| **CSI** | Critical Success Index (= IoU for binary events) |
| **PAE** | Peak Amplitude Error |
| **CBE** | Conditional Bias of Extremes |
| **EAR** | Extreme Area Ratio |
| **SAL** | Structure-Amplitude-Location verification framework |
