# create_dataset.py  (NEW - REALISTIC VERSION)
# =====================================================================
# The old version generated everything with random.choice, so the AI had no
# patterns to learn from. This version injects REALISTIC patterns:
#   - More robbery/assault at night, more theft in markets in the evening
#   - More crime on weekends and at month-end (salary days)
#   - Ramadan / Eid / Ashura effects
#   - 2 fixed hotspots inside every area (crimes cluster around them)
#   - A slight upward trend every year
#   - Crime waves (an area's level stays high/low for a few weeks)
#   - 2 "spike episodes" (for the anomaly-detection demo)
# NOTE: This is still SYNTHETIC data. Write "simulated data" in your report.
# =====================================================================

import os
import shutil

import numpy as np
import pandas as pd

from ai_engine import calendar_features

N_RECORDS = 8000                 # approx total records
SEED = 42
OUT_PATH = 'data/larkana_crime_dataset_5000.csv'   # same file name kept so older scripts do not break

STATIONS = {
    'Civil Line Police Station': ['Civil Line', 'Civil Line Area A', 'Civil Line Area B'],
    'Waleed Line Police Station': ['Waleed Area', 'Waleed Extension', 'Waleed North'],
    'Market Line Police Station': ['Market Area', 'Market Commercial', 'Market Residential'],
}
BASE_COORDS = {
    'Civil Line': (27.560, 68.205), 'Civil Line Area A': (27.555, 68.200), 'Civil Line Area B': (27.565, 68.210),
    'Waleed Area': (27.580, 68.220), 'Waleed Extension': (27.575, 68.225), 'Waleed North': (27.585, 68.215),
    'Market Area': (27.570, 68.210), 'Market Commercial': (27.575, 68.205), 'Market Residential': (27.565, 68.215),
}
AREA_INTENSITY = {   # relative crime level of each area
    'Civil Line': 1.0, 'Civil Line Area A': 0.7, 'Civil Line Area B': 0.5,
    'Waleed Area': 1.1, 'Waleed Extension': 0.8, 'Waleed North': 0.6,
    'Market Area': 1.4, 'Market Commercial': 1.9, 'Market Residential': 0.8,
}
MARKET_AREAS = {'Market Area', 'Market Commercial'}
SHIFT_MULT = {'Morning': 0.6, 'Evening': 1.0, 'Night': 1.3}
DOW_MULT = [0.95, 0.95, 0.95, 1.0, 1.1, 1.25, 1.2]            # Mon..Sun
SHIFT_HOURS = {'Morning': list(range(6, 14)), 'Evening': list(range(14, 22)),
               'Night': [22, 23, 0, 1, 2, 3, 4, 5]}
TYPES = ['Theft', 'Robbery', 'Assault', 'Murder']
TYPE_BASE = np.array([0.55, 0.25, 0.17, 0.03])
TYPE_SHIFT = {'Morning': [0.9, 0.6, 0.7, 0.5], 'Evening': [1.3, 1.0, 1.0, 0.8], 'Night': [0.8, 1.6, 1.4, 1.5]}

# Spike episodes: (area, start, end, shifts, boosted crime type, intensity multiplier)
SPIKES = [
    ('Waleed North', '2024-08-05', '2024-08-25', ['Evening', 'Night'], 'Robbery', 3.5),
    ('Market Commercial', '2025-12-08', '2025-12-21', ['Evening'], 'Theft', 3.0),
]


def in_spike(a, s, d):
    for sa, st, en, shifts, ctype, mult in SPIKES:
        if a == sa and s in shifts and pd.Timestamp(st) <= d <= pd.Timestamp(en):
            return ctype, mult
    return None, 1.0


def main():
    rng = np.random.default_rng(SEED)
    areas = [a for v in STATIONS.values() for a in v]
    station_of = {a: s for s, v in STATIONS.items() for a in v}
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='D')
    cal = calendar_features(dates)

    hotspots = {}                                     # 2 fixed hotspots in each area
    for a in areas:
        lat, lon = BASE_COORDS[a]
        hotspots[a] = [(lat + rng.uniform(-0.004, 0.004), lon + rng.uniform(-0.004, 0.004)) for _ in range(2)]

    # ---- "crime waves": each area's slowly changing level (AR(1) process) ----
    # This makes crimes cluster over time (some weeks high, some low) - real data behaves the same way
    wave = {}
    for a in areas:
        x, series = 0.0, []
        for _ in range(len(dates)):
            x = 0.97 * x + rng.normal(0, 0.08)
            series.append(np.exp(x))
        wave[a] = series

    # ---- intensity (lambda) of every (day, area, shift) ----
    keys, lam = [], []
    for di, d in enumerate(dates):
        c = cal.iloc[di]
        trend = 1 + 0.08 * (di / 365.0)
        for a in areas:
            for s in ['Morning', 'Evening', 'Night']:
                v = AREA_INTENSITY[a] * SHIFT_MULT[s] * DOW_MULT[int(c['DayOfWeek'])] * trend * wave[a][di]
                if c['IsMonthEnd']:
                    v *= 1.2
                if c['IsRamadan']:
                    v *= {'Morning': 0.7, 'Evening': 1.25 if a in MARKET_AREAS else 1.0, 'Night': 0.9}[s]
                if c['IsEidWeek']:
                    v *= 1.4 if a in MARKET_AREAS else 1.1
                if c['IsAshura']:
                    v *= 0.7
                v *= in_spike(a, s, d)[1]
                keys.append((d, a, s))
                lam.append(v)
    lam = np.array(lam)
    lam = lam * (N_RECORDS / lam.sum())
    counts = rng.poisson(lam)

    # ---- details of every crime ----
    rows = []
    for (d, a, s), n in zip(keys, counts):
        for _ in range(int(n)):
            w = TYPE_BASE * np.array(TYPE_SHIFT[s])
            if a in MARKET_AREAS and s == 'Evening':
                w[0] *= 1.3                                    # evening theft in markets
            boosted, _ = in_spike(a, s, d)
            if boosted:
                w[TYPES.index(boosted)] *= 5
            ctype = TYPES[rng.choice(4, p=w / w.sum())]
            hour = int(rng.choice(SHIFT_HOURS[s]))
            minute = int(rng.integers(0, 60))
            if rng.random() < 0.7:                             # 70% of crimes happen around a hotspot
                hl, ho = hotspots[a][rng.choice(2, p=[0.6, 0.4])]
                lat, lon = hl + rng.normal(0, 0.0012), ho + rng.normal(0, 0.0012)
            else:                                              # 30% are spread across the area
                bl, bo = BASE_COORDS[a]
                lat, lon = bl + rng.uniform(-0.008, 0.008), bo + rng.uniform(-0.008, 0.008)
            rows.append((station_of[a], a, ctype, d.strftime('%Y-%m-%d'), f"{hour:02d}:{minute:02d}",
                         round(lat, 6), round(lon, 6)))

    df = pd.DataFrame(rows, columns=['Police_Station', 'Area', 'Crime_Type', 'Date', 'Time', 'Latitude', 'Longitude'])
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)
    df.insert(0, 'Crime_ID', df.index + 1)

    os.makedirs('data', exist_ok=True)
    backup = 'data/larkana_crime_dataset_ORIGINAL_random.csv'
    if os.path.exists(OUT_PATH) and not os.path.exists(backup):
        shutil.copy(OUT_PATH, backup)
        print(f"Purana dataset backup ho gaya: {backup}")
    df.to_csv(OUT_PATH, index=False)
    print(f"Dataset ban gaya: {OUT_PATH}  ({len(df)} records)")
    print(df['Crime_Type'].value_counts().to_string())


if __name__ == '__main__':
    main()
