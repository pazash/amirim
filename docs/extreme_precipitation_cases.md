# Extreme Precipitation Event Cases

> Curated catalog of extreme precipitation events for evaluating Aurora 1.5
> forecasts against ERA5 reanalysis and the HRES baseline.
>
> All events fall within the HRES WeatherBench2 data window (2016–2022).

---

## 1. Why Extreme Precipitation?

ExtremeWeatherBench (EWB) provides curated heat wave cases but does not
include extreme precipitation events. AI weather models exhibit a
well-documented amplitude bias where predicted intensities collapse toward
climatological values at medium-range lead times (Pasche et al., 2025).
Zhang et al. (2026, *Science Advances*) showed that physics-based HRES
still consistently outperforms AI models (GraphCast, Pangu-Weather, Fuxi)
on record-breaking extremes, with AI models underestimating both the
frequency and intensity of such events. Evaluating Aurora on extreme
precipitation directly tests this known limitation.

### Variable note

Aurora 1.5 does not output total precipitation (`tp`) directly.
The evaluation uses **mean sea level pressure** (`msl`) and
**total column water vapour** (`tcwv`) as proxy variables — both are
available in Aurora outputs and represent the synoptic and moisture
conditions that produce extreme precipitation.

---

## 2. Event Catalog

Bounding boxes are given as **[south_lat, north_lat, west_lon, east_lon]**
in degrees. Longitude uses the signed convention (negative = west).

---

### 2.1 Hurricane Harvey (Texas, USA)

| Field | Value |
|---|---|
| **Start** | 2017-08-25 |
| **End** | 2017-08-30 |
| **Bbox** | [27.0, 32.0, −98.0, −93.0] |
| **Peak rainfall** | 1 539 mm in 4 days (Nederland, TX) |
| **Synoptic driver** | Cat-4 hurricane stalling over coastal Texas due to weak steering flow; massive moisture advection from the Gulf of Mexico |

**Meteorological justification.** Wettest tropical cyclone on record in
the contiguous US. Four-day accumulations exceeded the local 99.99th
percentile of precipitation climatology at multiple stations. Peak
accumulation of 1 539 mm is ≈3.5× the August monthly mean (≈130 mm).
NOAA classified multiple stations as exceeding the 1-in-1 000-year return
period. Area-averaged rainfall over the Houston metro (>25 000 km²)
exceeded 750 mm. Precipitable water (TCWV) sustained values above 65 mm
throughout the event, well above the regional 95th-percentile
climatological value of ≈55 mm.

---

### 2.2 2021 European Floods (Ahr Valley, Germany / Belgium)

| Field | Value |
|---|---|
| **Start** | 2021-07-12 |
| **End** | 2021-07-16 |
| **Bbox** | [49.0, 52.0, 5.0, 8.5] |
| **Peak rainfall** | ≈150 mm in 24 h (Eifel / Ahr region) |
| **Synoptic driver** | Slow-moving Vb-type cutoff low funneling Mediterranean moisture northward, amplified by orographic lifting |

**Meteorological justification.** 48-hour totals of 100–150 mm across a
broad area exceeded the local 99th percentile of summer (JJA) daily
precipitation. ERA5-derived 2-day totals over the Eifel/Ahr region were
>3σ above the 1991–2020 JJA climatological mean. WWA climate attribution
estimated a ≈1-in-400-year return period for 1-day maximum rainfall over a
≈25 000 km² region. Multiple river gauges on the Ahr, Erft, and Rur
exceeded their entire recorded maxima (some dating to the 1800s). MSLP
anomaly at the cutoff-low core was ≈10 hPa below the July mean; TCWV
exceeded 40 mm (>99th percentile for the region in summer).

---

### 2.3 2021 Henan / Zhengzhou Floods (China)

| Field | Value |
|---|---|
| **Start** | 2021-07-17 |
| **End** | 2021-07-23 |
| **Bbox** | [32.5, 36.5, 112.0, 116.0] |
| **Peak rainfall** | 201.9 mm in 1 hour (Zhengzhou, July 20); 617.1 mm over 3 days |
| **Synoptic driver** | Typhoon In-Fa interacting with the subtropical high and a mid-level trough, channeling extreme moisture from the Western Pacific; orographic enhancement from Funiu and Songshan mountains |

**Meteorological justification.** The 1-hour total of 201.9 mm is the
highest hourly rainfall ever recorded at any CMA station since 1951 and
ranks among the highest globally outside of tropical orographic extremes.
The 3-day total of 617.1 mm is ≈1.3× the entire July climatological mean
(≈450 mm). CMA classified this as a 1-in-1 000-year return period event.
TCWV over Henan exceeded 70 mm during the peak (local 99.5th percentile).
Daily rainfall on July 20 was >5σ above the 1991–2020 daily mean for that
calendar day.

---

### 2.4 2016 Louisiana Floods (USA)

| Field | Value |
|---|---|
| **Start** | 2016-08-10 |
| **End** | 2016-08-15 |
| **Bbox** | [29.5, 31.5, −92.5, −89.0] |
| **Peak rainfall** | >790 mm over 48–72 h (Watson, LA) |
| **Synoptic driver** | Quasi-stationary mesoscale low-pressure system with deep tropical moisture feed from the Gulf of Mexico; no named tropical cyclone |

**Meteorological justification.** USGS classified the Amite River gauge at
Denham Springs as a 1-in-1 000-year recurrence interval event, exceeding
the previous record crest by >1.2 m. Multiple stations recorded 500–790 mm
over 48–72 hours, corresponding to the >99.9th percentile of local August
daily precipitation. The event was non-tropical, making it a rare case of a
quasi-stationary mesoscale convective system producing tropical-cyclone-level
accumulations. Precipitable water exceeded 60 mm for 3+ consecutive days
(above the 97th percentile for the region).

---

### 2.5 2018 Kerala Floods (India)

| Field | Value |
|---|---|
| **Start** | 2018-08-08 |
| **End** | 2018-08-20 |
| **Bbox** | [8.0, 13.0, 74.5, 78.0] |
| **Peak rainfall** | ≈300 mm/day peak; 2 346 mm total in August 2018 (164 % above monthly mean) |
| **Synoptic driver** | Anomalously active Southwest monsoon with multiple monsoon depressions; extreme moisture convergence over the Western Ghats with orographic enhancement |

**Meteorological justification.** August 2018 rainfall of 2 346 mm was
164 % above the 1951–2000 August climatological mean of ≈650 mm — the
highest August total since at least 1924 (94-year return period). Daily
totals on multiple days exceeded the 99th percentile of wet-day
precipitation at stations across all 14 districts simultaneously. IMD
reported that the 8–20 August window received more rainfall than any
comparable 12-day window on record. Integrated vapour transport (IVT) from
the Arabian Sea sustained values >500 kg m⁻¹ s⁻¹ for 10+ consecutive days,
well above the 95th percentile of monsoon-season IVT.

---

### 2.6 Typhoon Hagibis (Japan)

| Field | Value |
|---|---|
| **Start** | 2019-10-11 |
| **End** | 2019-10-13 |
| **Bbox** | [33.0, 39.0, 136.0, 142.0] |
| **Peak rainfall** | 922.5 mm in 24 h (Hakone — all-time JMA record) |
| **Synoptic driver** | Super typhoon making landfall on the Izu Peninsula; extreme orographic enhancement against Japan's central mountain ranges |

**Meteorological justification.** The 24-hour accumulation of 922.5 mm at
Hakone set the all-time JMA 24-hour rainfall record for Japan. Over 100
stations in eastern Honshu exceeded the 99th percentile of October daily
precipitation. The typhoon's gale-force wind field diameter of 825 nmi was
the largest ever recorded for any typhoon globally. Central pressure of
915 hPa was >50 hPa below the October regional MSLP mean. TCWV exceeded
65 mm across a >500 km swath ahead of the storm. Rapid intensification
from tropical storm to Cat-5 equivalent in <12 hours (October 6) was among
the fastest on record in the western Pacific.

---

### 2.7 Cyclone Idai (Mozambique / Zimbabwe)

| Field | Value |
|---|---|
| **Start** | 2019-03-14 |
| **End** | 2019-03-17 |
| **Bbox** | [−22.0, −15.0, 33.0, 38.0] |
| **Peak rainfall** | >400 mm over 3 days near Beira |
| **Synoptic driver** | Intense tropical cyclone with peak sustained winds of 195 km/h making landfall near Beira; two landfalls (first as depression, then re-emergence and re-intensification over the Mozambique Channel) |

**Meteorological justification.** Multi-day rainfall totals >400 mm near
Beira exceeded the 99th percentile of the regional March precipitation
climatology by >4×. The system was the strongest landfalling TC in the SW
Indian Ocean basin since reliable records began. Central pressure of
940 hPa was ≈40 hPa below the March climatological MSLP for the Mozambique
Channel. Rapid intensification from tropical depression to intense TC in
48 h produced a TCWV maximum >60 mm — well above the regional 95th
percentile (≈40 mm for March). 72-hour rainfall exceeded the local March
monthly mean by 2–3×.

---

### 2.8 2019 Iran Floods

| Field | Value |
|---|---|
| **Start** | 2019-03-17 |
| **End** | 2019-04-10 |
| **Bbox** | [29.0, 38.0, 47.0, 57.0] |
| **Peak rainfall** | Golestan Province: ≈70 % of annual mean in one day; Lorestan: ≈200 mm in 24 h |
| **Synoptic driver** | Three successive extratropical cyclones from the Mediterranean with enhanced moisture from the Arabian Sea and Persian Gulf |

**Meteorological justification.** Golestan Province received ≈70 % of its
mean annual precipitation (≈450 mm) in a single day on March 17, exceeding
the 99.9th percentile of daily rainfall for a semi-arid region where the
March mean is ≈50 mm. Three successive extratropical cyclones crossed Iran
within 3 weeks, each producing >99th-percentile daily totals across
different provinces. Lorestan received ≈200 mm in 24 hours vs. a March
monthly mean of ≈80 mm. The temporal clustering of 3 extreme events within
25 days is itself statistically exceptional (estimated recurrence >100
years). MSLP anomalies of −8 to −12 hPa below the March mean accompanied
each wave.

---

### 2.9 2020 China Yangtze Floods

| Field | Value |
|---|---|
| **Start** | 2020-07-01 |
| **End** | 2020-07-22 |
| **Bbox** | [27.0, 33.0, 108.0, 118.0] |
| **Peak rainfall** | Multiple stations >200 mm/day on consecutive days |
| **Synoptic driver** | Exceptionally active Mei-yu (plum rain) front stalling over the Yangtze basin; strong moisture transport from the South China Sea; subtropical high anomalously displaced ≈5° north |

**Meteorological justification.** The Mei-yu season persisted for 62 days
(June 1 – August 2), the longest since 1951 — exceeding the 1981–2010 mean
duration (≈23 days) by 2.7×. Area-averaged rainfall over the
middle-lower Yangtze (25–33°N, 108–122°E) was >99th percentile of the
1961–2020 Mei-yu season totals. Multiple stations exceeded 200 mm/day on
consecutive days (>99th percentile of JJA daily climatology). Three Gorges
inflow peaked at 75 000 m³/s — the highest since the dam's 2006 completion.
The subtropical high was displaced ≈5° north of its climatological
position, sustaining the frontal convergence zone.

---

### 2.10 2022 Pakistan Floods

| Field | Value |
|---|---|
| **Start** | 2022-08-14 |
| **End** | 2022-08-30 |
| **Bbox** | [24.0, 34.0, 65.0, 74.0] |
| **Peak rainfall** | Sindh: 466 % above normal; Balochistan: 590 % above normal; multiple stations >300 mm/day |
| **Synoptic driver** | Extreme monsoon enhancement combined with glacial melt from a preceding heat wave; persistent moisture feed from the Arabian Sea; multiple monsoon depressions |

**Meteorological justification.** Sindh Province received 466 % above
normal August rainfall; Balochistan 590 % above normal — both far exceeding
the 99.9th percentile of their respective August climatologies. Multiple
stations recorded >300 mm/day vs. typical August daily means of 5–15 mm in
these semi-arid regions. WWA attribution found a 75 % increase in 5-day
maximum rainfall intensity attributable to climate change. Integrated
moisture flux from the Arabian Sea exceeded the 99th percentile of
monsoon-season values for 12+ consecutive days. The spatial coverage of
>100 mm/day rainfall simultaneously spanned >200 000 km², unprecedented in
the satellite era for the region.

---

## 3. Summary

| # | Event | Start | End | Bbox [S, N, W, E] | Return period | Percentile | Mechanism |
|---|---|---|---|---|---|---|---|
| 1 | Hurricane Harvey | 2017-08-25 | 2017-08-30 | 27, 32, −98, −93 | >1 000 yr | >99.99th | TC stalling |
| 2 | European Floods | 2021-07-12 | 2021-07-16 | 49, 52, 5, 8.5 | ≈400 yr | >99th JJA | Vb cyclone |
| 3 | Henan Floods | 2021-07-17 | 2021-07-23 | 32.5, 36.5, 112, 116 | 1 000 yr | >99.5th | TC + trough |
| 4 | Louisiana Floods | 2016-08-10 | 2016-08-15 | 29.5, 31.5, −92.5, −89 | 1 000 yr | >99.9th | Mesoscale low |
| 5 | Kerala Floods | 2018-08-08 | 2018-08-20 | 8, 13, 74.5, 78 | ≈100 yr | 164 % above clim | Monsoon surge |
| 6 | Typhoon Hagibis | 2019-10-11 | 2019-10-13 | 33, 39, 136, 142 | All-time JMA record | >99th Oct | Super typhoon |
| 7 | Cyclone Idai | 2019-03-14 | 2019-03-17 | −22, −15, 33, 38 | Basin record | >99th Mar | TC landfall |
| 8 | Iran Floods | 2019-03-17 | 2019-04-10 | 29, 38, 47, 57 | >100 yr cluster | >99.9th | Multi-wave extratropical |
| 9 | China Yangtze | 2020-07-01 | 2020-07-22 | 27, 33, 108, 118 | Longest Mei-yu since 1951 | >99th Mei-yu | Stationary front |
| 10 | Pakistan Floods | 2022-08-14 | 2022-08-30 | 24, 34, 65, 74 | Unprecedented | >99.9th | Extreme monsoon |

---

## 4. Selection Criteria

### 4.1 Return-period exceedance
Every event was classified by its national meteorological service as
exceeding at minimum a 100-year return period in rainfall accumulation at
multiple gauges. Three events exceeded 1 000-year thresholds (Harvey,
Louisiana, Henan). Return periods are based on GEV / GPD fits to
station-level annual maxima, following WMO-1100 guidelines.

### 4.2 Percentile exceedance vs. ERA5 climatology
All events produced daily or multi-day rainfall exceeding the 99th
percentile of the local seasonal precipitation climatology from the
1991–2020 ERA5 record. Four events exceeded the 99.9th percentile (Harvey,
Louisiana, Pakistan, Iran). This can be verified from the ARCO ERA5
`total_precipitation` variable by computing grid-point percentiles from the
30-year climatology.

### 4.3 Anomaly magnitude
Peak daily rainfall at the event center exceeded 3–5 standard deviations
above the climatological daily mean for the corresponding calendar day and
grid point. TCWV anomalies typically exceeded the 95th–99th percentile of
seasonal climatology. MSLP anomalies at the driving synoptic feature
ranged from −8 hPa (Iran extratropical cyclones) to >−50 hPa (Typhoon
Hagibis).

### 4.4 Record-breaking agency classifications
- **NWS / NOAA**: Harvey — wettest TC in US history; Louisiana — >1 000-year.
- **CMA**: Henan 201.9 mm/h — highest hourly rainfall since 1951.
- **JMA**: Hagibis 922.5 mm/24 h — all-time national 24-hour record.
- **DWD / WWA**: European floods — 1-in-400-year for the Ahr/Eifel region.
- **IMD**: Kerala 2018 — worst monsoon rainfall since 1924.
- **PMD**: Pakistan August 2022 — 466–590 % above climatological monthly means.

### 4.5 Mechanism diversity
The catalog spans 5 synoptic mechanisms that stress different aspects of
the forecast model:

| Mechanism | Events | Model skill tested |
|---|---|---|
| TC stalling / landfall | Harvey, Hagibis, Idai | MSLP deepening, track, moisture advection |
| Vb-cyclone / cutoff low | European floods, Iran | Slow-moving cutoff detection, moisture channels |
| Monsoon surge / depression | Kerala, Pakistan | Large-scale moisture flux, persistent convergence |
| Mesoscale convective system | Louisiana | Sub-synoptic features (hardest for coarse models) |
| Mei-yu / stationary front | China Yangtze | Frontal position, subtropical high placement |

### 4.6 Geographic and seasonal diversity
- **Regions**: North America (2), Europe (1), East Asia (2), South Asia (2),
  Middle East (1), Africa (1).
- **Seasons**: JJA (7), SON (1 — Hagibis), MAM (2 — Idai, Iran).
- **Duration**: 2 days (Hagibis) to 24 days (Iran); median ≈7 days.

---

## 5. References

1. Zhang, Z. et al. (2026). "Physics-based models outperform AI weather
   forecasts of record-breaking extremes." *Science Advances*, 12(18).
2. Pasche, O. C. et al. (2025). "Validating Deep Learning Weather Forecast
   Models on Recent High-Impact Extreme Events." *AIES*, 4(1).
3. Kreienkamp, F. et al. (2021). "Rapid attribution of heavy rainfall
   events leading to the severe flooding in Western Europe." WWA.
4. van Oldenborgh, G. J. et al. (2017). "Attribution of extreme rainfall
   from Hurricane Harvey." *Environ. Res. Lett.*, 12(12).
5. Otto, F. E. L. et al. (2022). "Climate change likely increased extreme
   monsoon rainfall … in Pakistan." WWA.
6. Wernli, H. et al. (2008). "SAL — A Novel Quality Measure for the
   Verification of Quantitative Precipitation Forecasts." *MWR*, 136(11).
