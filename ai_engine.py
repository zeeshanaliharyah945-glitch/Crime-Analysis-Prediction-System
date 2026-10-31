# ai_engine.py
# =====================================================================
#  CRIME AI ENGINE  -  the core of all AI features lives in this file
# =====================================================================
#  Features implemented here:
#   1. Honest risk model   (area x day x shift, time-based split, no leakage)
#   2. Explainable AI      (a "why?" explanation for every prediction)
#   3. Patrol recommendation
#   4. Crime forecasting   (next 30 days)
#   5. Hotspot detection   (DBSCAN clustering)
#   6. Grid risk prediction (500 m cells, risk for next week)
#   7. Anomaly detection   (Isolation Forest + z-score)
#   8. Fairness / bias audit
#   9. PDF report builder  (with charts)
#
#  Do not run this file directly. It is imported by 06_train_ai_models.py and
#  ai_features_gui.py.
# =====================================================================

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              IsolationForest, RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss, roc_auc_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# PATHS / CONSTANTS
# ---------------------------------------------------------------------
DATA_PATH = 'data/larkana_crime_dataset_5000.csv'
OUT_DIR = 'outputs/ai'
MODEL_PATH = 'outputs/models/ai_risk_bundle.pkl'
MAP_DIR = 'outputs/heatmaps'

SEVERITY = {'Theft': 2, 'Robbery': 4, 'Assault': 3, 'Murder': 5}
CRIME_TYPES = ['Theft', 'Robbery', 'Assault', 'Murder']
SHIFTS = ['Morning', 'Evening', 'Night']
SHIFT_CODE = {'Morning': 0, 'Evening': 1, 'Night': 2}
SHIFT_HOURS = {'Morning': '06:00-13:59', 'Evening': '14:00-21:59', 'Night': '22:00-05:59'}
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']

# Pakistan calendar (approximate dates) - Ramadan / Eid / Ashura
RAMADAN = [('2023-03-23', '2023-04-20'), ('2024-03-11', '2024-04-09'), ('2025-03-01', '2025-03-30')]
EID_DATES = pd.to_datetime(['2023-04-22', '2024-04-10', '2025-03-31',      # Eid-ul-Fitr
                            '2023-06-29', '2024-06-17', '2025-06-07'])     # Eid-ul-Adha
ASHURA = pd.to_datetime(['2023-07-29', '2024-07-17', '2025-07-06'])

FEATURES = ['AreaCode', 'ShiftCode', 'DayOfWeek', 'Month', 'IsWeekend', 'IsMonthEnd',
            'IsRamadan', 'IsEidWeek', 'IsAshura', 'Lag1', 'Lag7', 'Lag30', 'AreaLag7']


def hour_to_shift(h):
    if 6 <= h < 14:
        return 'Morning'
    if 14 <= h < 22:
        return 'Evening'
    return 'Night'


def calendar_features(dates):
    """Converts a list of dates into calendar features (weekend, Ramadan, Eid...)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    ram = np.zeros(len(d), dtype=bool)
    for a, b in RAMADAN:
        ram |= (d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))
    eid = np.zeros(len(d), dtype=bool)
    for e in EID_DATES:
        eid |= np.abs((d - e).days.values) <= 3
    ash = np.zeros(len(d), dtype=bool)
    for e in ASHURA:
        ash |= np.abs((d - e).days.values) <= 1
    return pd.DataFrame({
        'DayOfWeek': d.dayofweek.values,
        'Month': d.month.values,
        'IsWeekend': (d.dayofweek.values >= 5).astype(int),
        'IsMonthEnd': ((d.day.values >= 28) | (d.day.values <= 2)).astype(int),
        'IsRamadan': ram.astype(int),
        'IsEidWeek': eid.astype(int),
        'IsAshura': ash.astype(int),
    })


def load_crimes(path=DATA_PATH):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Time'].astype(str).str[:2].astype(int)
    df['Shift'] = df['Hour'].apply(hour_to_shift)
    df['Severity'] = df['Crime_Type'].map(SEVERITY).fillna(2)
    return df


def _metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return {'AUC': roc_auc_score(y, p), 'Accuracy': accuracy_score(y, pred),
            'Precision': pr, 'Recall': rc, 'F1': f1, 'Brier': brier_score_loss(y, p)}


def _best_f1_threshold(y, p):
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.8, 76):
        pred = (p >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        if f1 > best_f:
            best_f, best_t = f1, t
    return best_t


def _nb_sum(m):
    """Sum of the 8 neighbouring cells of a 2D grid (spatial smoothing)."""
    p = np.pad(m, 1)
    out = np.zeros_like(m, dtype=float)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out += p[1 + dr:1 + dr + m.shape[0], 1 + dc:1 + dc + m.shape[1]]
    return out


def _level_color(v, vmax):
    t = min(max(v / vmax, 0.0), 1.0) if vmax > 0 else 0
    return '#%02x%02x%02x' % (255, int(235 * (1 - t)), int(60 * (1 - t)))


def _ascii(s):
    """PDF (fpdf 1.7) only supports latin-1 - remove emoji / special characters."""
    return str(s).encode('latin-1', 'replace').decode('latin-1')


# =====================================================================
#  MAIN CLASS
# =====================================================================
class CrimeAI:
    def __init__(self, data_path=DATA_PATH, model_path=MODEL_PATH, out_dir=OUT_DIR):
        self.data_path, self.model_path, self.out_dir = data_path, model_path, out_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(MAP_DIR, exist_ok=True)

        self.df = load_crimes(data_path)
        self.areas = sorted(self.df['Area'].unique())
        self.area_code = {a: i for i, a in enumerate(self.areas)}
        self.station_of = self.df.drop_duplicates('Area').set_index('Area')['Police_Station'].to_dict()
        self.first_date = self.df['Date'].min()
        self.last_date = self.df['Date'].max()
        self.avg_sev = self.df.groupby('Area')['Severity'].mean().to_dict()
        self._build_counts()

        self.bundle = None
        self.forecast_df = None
        self.forecast_eval = None
        self.hotspots = None
        self.grid_pred = None
        self.grid_eval = None
        self.anomalies = None
        self.load_model()

    # -----------------------------------------------------------------
    # Basic counts (date x area x shift)
    # -----------------------------------------------------------------
    def _build_counts(self):
        dates = pd.date_range(self.first_date, self.last_date, freq='D')
        c = self.df.groupby(['Date', 'Area', 'Shift']).size()
        idx = pd.MultiIndex.from_product([dates, self.areas, SHIFTS], names=['Date', 'Area', 'Shift'])
        self.panel_counts = c.reindex(idx, fill_value=0).rename('Count').reset_index()
        self.wide = self.panel_counts.pivot_table(index='Date', columns=['Area', 'Shift'],
                                                  values='Count', aggfunc='sum')
        self.area_daily = (self.panel_counts.groupby(['Date', 'Area'])['Count'].sum()
                           .unstack('Area').reindex(columns=self.areas))

    # =================================================================
    # FEATURE 1: HONEST RISK MODEL
    # =================================================================
    def build_panel(self):
        """One row per (day, area, shift). Lag features only look at PREVIOUS days => no leakage."""
        p = self.panel_counts.sort_values(['Area', 'Shift', 'Date']).reset_index(drop=True)
        g = p.groupby(['Area', 'Shift'])['Count']
        p['Lag1'] = g.shift(1)
        p['Lag7'] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
        p['Lag30'] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum())

        ad = self.area_daily.shift(1).rolling(7, min_periods=1).sum()
        ad_long = ad.reset_index().melt(id_vars='Date', var_name='Area', value_name='AreaLag7')
        p = p.merge(ad_long, on=['Date', 'Area'], how='left')

        p = pd.concat([p, calendar_features(p['Date'])], axis=1)
        p['AreaCode'] = p['Area'].map(self.area_code)
        p['ShiftCode'] = p['Shift'].map(SHIFT_CODE)
        p['Target'] = (p['Count'] >= 1).astype(int)
        p[['Lag1', 'Lag7', 'Lag30', 'AreaLag7']] = p[['Lag1', 'Lag7', 'Lag30', 'AreaLag7']].fillna(0)
        p = p[p['Date'] >= self.first_date + pd.Timedelta(days=30)].reset_index(drop=True)
        return p

    def features_for(self, area, date, shift):
        """Feature row for any (area, date, shift) (future dates work too)."""
        date = pd.Timestamp(date).normalize()
        anchor = min(date, self.last_date + pd.Timedelta(days=1))
        one = pd.Timedelta(days=1)
        s = self.wide[(area, shift)]
        cal = calendar_features([date]).iloc[0]
        row = {
            'AreaCode': self.area_code[area], 'ShiftCode': SHIFT_CODE[shift],
            'DayOfWeek': cal['DayOfWeek'], 'Month': cal['Month'], 'IsWeekend': cal['IsWeekend'],
            'IsMonthEnd': cal['IsMonthEnd'], 'IsRamadan': cal['IsRamadan'],
            'IsEidWeek': cal['IsEidWeek'], 'IsAshura': cal['IsAshura'],
            'Lag1': float(s.get(anchor - one, 0)),
            'Lag7': float(s.loc[anchor - 7 * one: anchor - one].sum()),
            'Lag30': float(s.loc[anchor - 30 * one: anchor - one].sum()),
            'AreaLag7': float(self.area_daily[area].loc[anchor - 7 * one: anchor - one].sum()),
        }
        return pd.DataFrame([row])[FEATURES]

    def train_risk_model(self, verbose=True):
        panel = self.build_panel()
        cutoff = self.last_date - pd.Timedelta(days=364)          # the last 365 days = TEST
        val_start = cutoff - pd.Timedelta(days=180)               # the 180 days before that = VALIDATION
        train = panel[panel['Date'] < cutoff]
        test = panel[panel['Date'] >= cutoff]
        fit = train[train['Date'] < val_start]
        val = train[train['Date'] >= val_start]

        def make(name):
            if name == 'RandomForest':
                return RandomForestClassifier(n_estimators=300, min_samples_leaf=25,
                                              random_state=42, n_jobs=-1)
            return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=4,
                                                  min_samples_leaf=40, l2_regularization=1.0,
                                                  early_stopping=False, random_state=42)

        def baseline_probs(src, dst):
            rate = src.groupby(['AreaCode', 'ShiftCode', 'DayOfWeek'])['Target'].mean().rename('p')
            m = dst[['AreaCode', 'ShiftCode', 'DayOfWeek']].merge(rate.reset_index(), how='left',
                                                                  on=['AreaCode', 'ShiftCode', 'DayOfWeek'])
            return m['p'].fillna(src['Target'].mean()).values

        if verbose:
            print(f"   Train: {train['Date'].min().date()} -> {train['Date'].max().date()} ({len(train)} rows)")
            print(f"   Test : {test['Date'].min().date()} -> {test['Date'].max().date()} ({len(test)} rows)")
            print(f"   Crime base rate (test): {test['Target'].mean():.3f}")

        rows, val_auc, thr_map, final_models = [], {}, {}, {}
        # Baseline (simple historical average)
        pb_val = baseline_probs(fit, val)
        thr_b = _best_f1_threshold(val['Target'], pb_val)
        pb_test = baseline_probs(train, test)
        m = _metrics(test['Target'], pb_test, thr_b)
        rows.append({'Model': 'Baseline (historical average)', 'Val_AUC': roc_auc_score(val['Target'], pb_val), **m})

        for name in ['RandomForest', 'GradientBoosting']:
            mdl = make(name).fit(fit[FEATURES], fit['Target'])
            pv = mdl.predict_proba(val[FEATURES])[:, 1]
            val_auc[name] = roc_auc_score(val['Target'], pv)
            thr_map[name] = (_best_f1_threshold(val['Target'], pv), pv)
            fin = make(name).fit(train[FEATURES], train['Target'])
            final_models[name] = fin
            pt = fin.predict_proba(test[FEATURES])[:, 1]
            rows.append({'Model': name, 'Val_AUC': val_auc[name], **_metrics(test['Target'], pt, thr_map[name][0])})

        best = max(val_auc, key=val_auc.get)                       # selection uses VALIDATION, not the test set
        final = final_models[best]
        thr, pv_best = thr_map[best]
        cut_low, cut_high = np.quantile(pv_best, [0.33, 0.67])
        test = test.copy()
        test['Prob'] = final.predict_proba(test[FEATURES])[:, 1]

        imp = permutation_importance(final, test[FEATURES], test['Target'], scoring='roc_auc',
                                     n_repeats=5, random_state=42, n_jobs=-1)
        importance = (pd.DataFrame({'Feature': FEATURES, 'Importance': imp.importances_mean,
                                    'Std': imp.importances_std})
                      .sort_values('Importance', ascending=False).reset_index(drop=True))

        comparison = pd.DataFrame(rows)
        comparison['Selected'] = comparison['Model'] == best
        self.bundle = {
            'model': final, 'model_name': best, 'features': FEATURES, 'threshold': thr,
            'cut_low': float(cut_low), 'cut_high': float(cut_high),
            'background': train[FEATURES].sample(150, random_state=42).reset_index(drop=True),
            'comparison': comparison, 'importance': importance,
            'test_pred': test[['Date', 'Area', 'Shift', 'Count', 'Target', 'Prob']].reset_index(drop=True),
            'train_end': cutoff, 'base_rate': float(test['Target'].mean()),
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.bundle, f)
        comparison.to_csv(os.path.join(self.out_dir, 'risk_model_comparison.csv'), index=False)
        importance.to_csv(os.path.join(self.out_dir, 'feature_importance.csv'), index=False)
        if verbose:
            print(f"   Selected model (by validation AUC): {best}")
            print(comparison.round(3).to_string(index=False))
        return comparison

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.bundle = pickle.load(f)
            except Exception:
                self.bundle = None
        return self.bundle

    def level(self, p):
        b = self.bundle
        if p >= b['cut_high']:
            return 'High'
        if p >= b['cut_low']:
            return 'Medium'
        return 'Low'

    def risk_for(self, area, date, shift):
        X = self.features_for(area, date, shift)
        p = float(self.bundle['model'].predict_proba(X)[0, 1])
        return p, self.level(p), X

    # =================================================================
    # FEATURE 2: EXPLAINABLE AI  (marginal occlusion - SHAP-like idea, no extra library)
    # =================================================================
    def describe_feature(self, f, v):
        v = float(v)
        if f == 'AreaCode':
            return f"Area: {self.areas[int(v)]}"
        if f == 'ShiftCode':
            return f"{SHIFTS[int(v)]} shift ({SHIFT_HOURS[SHIFTS[int(v)]]})"
        if f == 'DayOfWeek':
            return f"Day: {DAY_NAMES[int(v)]}"
        if f == 'Month':
            return f"Month: {MONTH_NAMES[int(v) - 1]}"
        if f == 'IsWeekend':
            return 'Weekend' if v else 'Weekday'
        if f == 'IsMonthEnd':
            return 'Month-end / salary days' if v else 'Mid-month days'
        if f == 'IsRamadan':
            return 'Ramadan' if v else 'Not Ramadan'
        if f == 'IsEidWeek':
            return 'Eid week' if v else 'Not Eid week'
        if f == 'IsAshura':
            return 'Ashura days (tight security)' if v else 'Normal security'
        if f == 'Lag1':
            return f"{int(v)} crime(s) yesterday in this area-shift"
        if f == 'Lag7':
            return f"{int(v)} crime(s) in last 7 days (this area-shift)"
        if f == 'Lag30':
            return f"{int(v)} crime(s) in last 30 days (this area-shift)"
        if f == 'AreaLag7':
            return f"{int(v)} crime(s) in last 7 days (whole area)"
        return f

    def explain(self, X, top=5):
        """Replace each feature with its 'average situation' value and measure how much the probability changes."""
        model, bg = self.bundle['model'], self.bundle['background']
        p0 = float(model.predict_proba(X[FEATURES])[0, 1])
        blocks = []
        for f in FEATURES:
            Z = pd.concat([X[FEATURES]] * len(bg), ignore_index=True)
            Z[f] = bg[f].values
            blocks.append(Z)
        probs = model.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1].reshape(len(FEATURES), len(bg))
        out = []
        for i, f in enumerate(FEATURES):
            contrib = p0 - probs[i].mean()
            out.append({'Feature': f, 'Text': self.describe_feature(f, X[f].values[0]),
                        'Contribution': float(contrib)})
        out.sort(key=lambda r: abs(r['Contribution']), reverse=True)
        return out[:top]

    @staticmethod
    def explain_text(exp):
        lines = []
        for e in exp:
            arrow = 'UP  ' if e['Contribution'] > 0 else 'DOWN'
            lines.append(f"  {arrow} {e['Contribution'] * 100:+.1f} pts  <-  {e['Text']}")
        return '\n'.join(lines)

    # =================================================================
    # FEATURE 3: PATROL RECOMMENDATION
    # =================================================================
    def risk_table(self, date, shift, with_reasons=True):
        rows = []
        for a in self.areas:
            p, lvl, X = self.risk_for(a, date, shift)
            reason = ''
            if with_reasons:
                ups = [e for e in self.explain(X, top=6) if e['Contribution'] > 0][:2]
                reason = '; '.join(e['Text'] for e in ups)
            rows.append({'Area': a, 'Station': self.station_of[a], 'Risk_Prob': p, 'Level': lvl,
                         'Avg_Severity': self.avg_sev[a], 'Reason': reason})
        return pd.DataFrame(rows).sort_values('Risk_Prob', ascending=False).reset_index(drop=True)

    def patrol_plan(self, date, shift, officers=12):
        t = self.risk_table(date, shift)
        w = (t['Risk_Prob'] ** 2) * t['Avg_Severity']
        quota = officers * w / w.sum()
        alloc = np.floor(quota).astype(int)
        rem = int(officers - alloc.sum())
        for idx in (quota - alloc).sort_values(ascending=False).index[:rem]:
            alloc.loc[idx] += 1
        t['Officers'] = alloc
        station = (t.groupby('Station')['Officers'].sum().reset_index()
                   .sort_values('Officers', ascending=False))
        return t, station

    # =================================================================
    # FEATURE 4: FORECASTING (next 30 days) - direct multi-horizon model
    # =================================================================
    def _fc_rows(self, origins, H, cal_ext, with_target=True):
        A = self.area_daily.values.astype(float)
        T, n = A.shape
        cs = np.vstack([np.zeros((1, n)), np.cumsum(A, axis=0)])

        def roll_mean(idx, w):
            lo = np.maximum(idx - w + 1, 0)
            return (cs[idx + 1] - cs[lo]) / (idx - lo + 1)[:, None]

        L7, L28, L90 = roll_mean(origins, 7), roll_mean(origins, 28), roll_mean(origins, 90)
        chunks = []
        for h in range(1, H + 1):
            t = origins + h
            ok = t < T if with_target else np.ones(len(origins), dtype=bool)
            if not ok.any():
                continue
            o, tt, m = origins[ok], t[ok], int(ok.sum())
            d = {
                'AreaCode': np.tile(np.arange(n), m), 'H': np.full(m * n, h),
                'DayOfWeek': np.repeat(cal_ext['DayOfWeek'].values[tt], n),
                'Month': np.repeat(cal_ext['Month'].values[tt], n),
                'IsWeekend': np.repeat(cal_ext['IsWeekend'].values[tt], n),
                'IsMonthEnd': np.repeat(cal_ext['IsMonthEnd'].values[tt], n),
                'IsRamadan': np.repeat(cal_ext['IsRamadan'].values[tt], n),
                'IsEidWeek': np.repeat(cal_ext['IsEidWeek'].values[tt], n),
                'IsAshura': np.repeat(cal_ext['IsAshura'].values[tt], n),
                'L7': L7[ok].ravel(), 'L28': L28[ok].ravel(), 'L90': L90[ok].ravel(),
                'Origin': np.repeat(o, n), 'TargetIdx': np.repeat(tt, n),
            }
            if with_target:
                d['y'] = A[tt].ravel()
            chunks.append(pd.DataFrame(d))
        return pd.concat(chunks, ignore_index=True)

    # The model uses the "level" (L28) as an offset: y = L28 * ratio, and the AI only learns the ratio
    # (weekend/Eid/Ramadan effect + recent trend L7/L28, L90/L28). This keeps it stable even on small data.
    FC_FEATS = ['AreaCode', 'H', 'DayOfWeek', 'IsWeekend', 'IsMonthEnd', 'IsRamadan',
                'IsEidWeek', 'IsAshura', 'R7', 'R90']

    def _fc_fit(self, d):
        d = d.assign(R7=d['L7'] / (d['L28'] + 0.1), R90=d['L90'] / (d['L28'] + 0.1))
        lvl = d['L28'] + 0.1
        m = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.04, max_iter=120,
                                          max_depth=3, min_samples_leaf=200, l2_regularization=5.0,
                                          early_stopping=False, random_state=42)
        m.fit(d[self.FC_FEATS], d['y'] / lvl, sample_weight=lvl)
        return m

    def _fc_predict(self, m, d):
        d = d.assign(R7=d['L7'] / (d['L28'] + 0.1), R90=d['L90'] / (d['L28'] + 0.1))
        return np.clip(m.predict(d[self.FC_FEATS]) * (d['L28'] + 0.1), 0, None).values

    def train_forecaster(self, horizon=30, verbose=True):
        T = len(self.area_daily)
        dates_ext = pd.date_range(self.first_date, periods=T + horizon, freq='D')
        cal_ext = calendar_features(dates_ext)
        eval_start = T - 120

        # ---- Evaluation: train only on data before eval_start, test on later forecast origins ----
        origins = np.arange(27, T - 1, 2)
        rows = self._fc_rows(origins, horizon, cal_ext)
        tr = rows[rows['TargetIdx'] < eval_start]
        te = rows[(rows['Origin'] >= eval_start) & (rows['Origin'] % 7 == eval_start % 7)].copy()
        m = self._fc_fit(tr)
        te['Model'] = self._fc_predict(m, te)
        te['Naive'] = te['L28']
        area_mean = tr.groupby('AreaCode')['y'].mean()
        te['AreaMean'] = te['AreaCode'].map(area_mean)
        res = []
        for col, label in [('Model', 'AI model (Poisson boosting)'),
                           ('Naive', 'Baseline: last 28-day average'),
                           ('AreaMean', 'Baseline: area historical mean')]:
            daily_mae = float(np.abs(te[col] - te['y']).mean())
            w = te[te['H'] <= 28].copy()
            w['Wk'] = (w['H'] - 1) // 7
            g = w.groupby(['Origin', 'AreaCode', 'Wk']).agg(pred=(col, 'sum'), act=('y', 'sum'))
            res.append({'Model': label, 'Daily_MAE': daily_mae,
                        'Weekly_MAE': float(np.abs(g['pred'] - g['act']).mean())})
        self.forecast_eval = pd.DataFrame(res)

        # ---- Final: train on all data, forecast the next `horizon` days from T-1 ----
        full = rows
        mf = self._fc_fit(full)
        fut = self._fc_rows(np.array([T - 1]), horizon, cal_ext, with_target=False)
        fut['Forecast'] = self._fc_predict(mf, fut)
        fut['Date'] = dates_ext[fut['TargetIdx'].values]
        fut['Area'] = [self.areas[i] for i in fut['AreaCode']]
        self.forecast_df = fut[['Date', 'Area', 'Forecast']].reset_index(drop=True)
        self.forecast_df.to_csv(os.path.join(self.out_dir, 'forecast_next30.csv'), index=False)
        self.forecast_eval.to_csv(os.path.join(self.out_dir, 'forecast_eval.csv'), index=False)
        if verbose:
            print(self.forecast_eval.round(3).to_string(index=False))
        return self.forecast_df

    def forecast_summary(self):
        """Area-wise forecast for the next 4 weeks (weekly totals)."""
        f = self.forecast_df.copy()
        start = f['Date'].min()
        f['Week'] = ((f['Date'] - start).dt.days // 7 + 1).clip(upper=4)
        piv = f.pivot_table(index='Area', columns='Week', values='Forecast', aggfunc='sum')
        piv.columns = [f'Week {c}' for c in piv.columns]
        piv['Total (30d)'] = f.groupby('Area')['Forecast'].sum()
        return piv.round(1).reset_index().sort_values('Total (30d)', ascending=False)

    # =================================================================
    # FEATURE 5: HOTSPOT DETECTION (DBSCAN)
    # =================================================================
    def find_hotspots(self, eps_m=60, min_samples=30):
        X = np.radians(self.df[['Latitude', 'Longitude']].values)
        db = DBSCAN(eps=eps_m / 6371000.0, min_samples=min_samples,
                    metric='haversine', algorithm='ball_tree').fit(X)
        d = self.df.assign(Cluster=db.labels_)
        d = d[d['Cluster'] >= 0]
        recent_start = self.last_date - pd.Timedelta(days=90)
        rows = []
        for cid, g in d.groupby('Cluster'):
            lat, lon = g['Latitude'].mean(), g['Longitude'].mean()
            dy = (g['Latitude'] - lat) * 111320.0
            dx = (g['Longitude'] - lon) * 111320.0 * np.cos(np.radians(lat))
            radius = float(np.percentile(np.sqrt(dx ** 2 + dy ** 2), 90))
            top = g['Crime_Type'].value_counts()
            rows.append({
                'Latitude': round(lat, 6), 'Longitude': round(lon, 6), 'Radius_m': round(radius),
                'Crimes': len(g), 'Severity_Score': int(g['Severity'].sum()),
                'Area': g['Area'].mode().iloc[0],
                'Dominant_Crime': top.index[0], 'Dominant_Share_%': round(100 * top.iloc[0] / len(g)),
                'Peak_Shift': g['Shift'].mode().iloc[0],
                'Last_90_Days': int((g['Date'] >= recent_start).sum()),
            })
        hs = pd.DataFrame(rows)
        if len(hs):
            hs = hs.sort_values('Severity_Score', ascending=False).reset_index(drop=True)
            hs.insert(0, 'Rank', hs.index + 1)
        self.hotspots = hs
        hs.to_csv(os.path.join(self.out_dir, 'hotspots.csv'), index=False)
        return hs

    def save_hotspot_map(self, path=os.path.join(MAP_DIR, 'ai_hotspots_map.html')):
        import folium
        from folium.plugins import HeatMap
        hs = self.hotspots
        from map_utils import create_map
        m = create_map([self.df['Latitude'].mean(), self.df['Longitude'].mean()], zoom_start=14, theme='light')
        HeatMap(self.df[['Latitude', 'Longitude']].values.tolist(), radius=9, blur=12, min_opacity=0.25).add_to(m)
        for _, r in hs.iterrows():
            color = 'red' if r['Rank'] <= 3 else ('orange' if r['Rank'] <= 6 else 'blue')
            html = (f"<b>Hotspot #{r['Rank']}</b><br>Area: {r['Area']}<br>Crimes: {r['Crimes']}<br>"
                    f"Main crime: {r['Dominant_Crime']} ({r['Dominant_Share_%']}%)<br>"
                    f"Peak shift: {r['Peak_Shift']}<br>Last 90 days: {r['Last_90_Days']}")
            folium.Circle([r['Latitude'], r['Longitude']], radius=max(r['Radius_m'], 80), color=color,
                          fill=True, fill_opacity=0.25, popup=folium.Popup(html, max_width=250),
                          tooltip=f"Hotspot #{r['Rank']}").add_to(m)
            folium.Marker([r['Latitude'], r['Longitude']],
                          icon=folium.DivIcon(html=f"<div style='font-size:14px;font-weight:bold;color:{color}'>#{r['Rank']}</div>")
                          ).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(path)
        return path

    # =================================================================
    # FEATURE 6: GRID RISK PREDICTION (500 m cells, next week)
    # =================================================================
    def _grid_cube(self, cell_m=500):
        df = self.df
        self._lat0, self._lon0 = df['Latitude'].min(), df['Longitude'].min()
        self._dlat = cell_m / 111320.0
        self._dlon = cell_m / (111320.0 * np.cos(np.radians(df['Latitude'].mean())))
        r = np.floor((df['Latitude'] - self._lat0) / self._dlat).astype(int).values
        c = np.floor((df['Longitude'] - self._lon0) / self._dlon).astype(int).values
        nr, nc = r.max() + 1, c.max() + 1
        ws = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='D')
        valid = ((ws >= self.first_date) & (ws + pd.Timedelta(days=6) <= self.last_date)).values
        weeks = pd.date_range(ws[valid].min(), ws[valid].max(), freq='7D')
        cube = np.zeros((len(weeks), nr, nc))
        widx = ((ws[valid] - weeks[0]).dt.days // 7).values
        np.add.at(cube, (widx, r[valid], c[valid]), 1)
        return cube, weeks, nr, nc

    @staticmethod
    def _grid_feats(cube, i, nr, nc):
        l1, l2 = cube[i - 1], cube[i - 2]
        l4 = cube[i - 4:i].sum(axis=0)
        l12 = cube[max(i - 12, 0):i].mean(axis=0)
        nb4 = _nb_sum(l4)
        rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        return np.column_stack([l1.ravel(), l2.ravel(), l4.ravel(), l12.ravel(), nb4.ravel(),
                                rr.ravel(), cc.ravel()])

    def train_grid_model(self, cell_m=500, topk=10, verbose=True):
        cube, weeks, nr, nc = self._grid_cube(cell_m)
        W = len(weeks)
        X, y, wk = [], [], []
        for i in range(12, W):
            X.append(self._grid_feats(cube, i, nr, nc))
            y.append(cube[i].ravel())
            wk.append(np.full(nr * nc, i))
        X, y, wk = np.vstack(X), np.concatenate(y), np.concatenate(wk)
        test_weeks = 26
        tr, te = wk < W - test_weeks, wk >= W - test_weeks

        def mk():
            return HistGradientBoostingRegressor(loss='poisson', learning_rate=0.05, max_iter=150,
                                                 max_depth=3, min_samples_leaf=30, l2_regularization=1.0,
                                                 early_stopping=False, random_state=42)
        m = mk().fit(X[tr], y[tr])
        pred = np.clip(m.predict(X[te]), 0, None)
        base = X[te][:, 2] / 4.0                                    # last-4-week average
        mae_m, mae_b = float(np.abs(pred - y[te]).mean()), float(np.abs(base - y[te]).mean())

        hit_m, hit_b, hit_r = [], [], []
        P, B, Y, Wk = pred.reshape(-1, nr * nc), base.reshape(-1, nr * nc), y[te].reshape(-1, nr * nc), wk[te][::nr * nc]
        for k in range(len(P)):
            tot = Y[k].sum()
            if tot == 0:
                continue
            hit_m.append(Y[k][np.argsort(-P[k])[:topk]].sum() / tot)
            hit_b.append(Y[k][np.argsort(-B[k])[:topk]].sum() / tot)
            hit_r.append(topk / (nr * nc))
        self.grid_eval = pd.DataFrame([
            {'Method': 'AI model (HistGB Poisson)', 'MAE_per_cell': mae_m, f'Top{topk}_Cells_Capture_%': 100 * np.mean(hit_m)},
            {'Method': 'Baseline: last 4-week average', 'MAE_per_cell': mae_b, f'Top{topk}_Cells_Capture_%': 100 * np.mean(hit_b)},
            {'Method': 'Random cells', 'MAE_per_cell': np.nan, f'Top{topk}_Cells_Capture_%': 100 * np.mean(hit_r)},
        ])
        self.grid_eval.to_csv(os.path.join(self.out_dir, 'grid_eval.csv'), index=False)

        # Final model: train on all data, predict next week
        mf = mk().fit(X, y)
        cube_ext = np.concatenate([cube, np.zeros((1, nr, nc))])
        Xn = self._grid_feats(cube_ext, W, nr, nc)
        pn = np.clip(mf.predict(Xn), 0, None)
        rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        g = pd.DataFrame({'Row': rr.ravel(), 'Col': cc.ravel(), 'Pred_Next_Week': pn,
                          'Last_4wk_Avg': Xn[:, 2] / 4.0})
        g['Lat_Center'] = self._lat0 + (g['Row'] + 0.5) * self._dlat
        g['Lon_Center'] = self._lon0 + (g['Col'] + 0.5) * self._dlon
        r_idx = np.floor((self.df['Latitude'] - self._lat0) / self._dlat).astype(int)
        c_idx = np.floor((self.df['Longitude'] - self._lon0) / self._dlon).astype(int)
        top = self.df.assign(Row=r_idx, Col=c_idx).groupby(['Row', 'Col'])['Crime_Type'].agg(lambda s: s.mode().iloc[0])
        g = g.merge(top.rename('Main_Crime').reset_index(), how='left', on=['Row', 'Col'])
        g['Main_Crime'] = g['Main_Crime'].fillna('-')
        g = g.sort_values('Pred_Next_Week', ascending=False).reset_index(drop=True)
        g['Rank'] = g.index + 1
        g['Risk_Level'] = np.where(g['Rank'] <= topk, 'High', np.where(g['Rank'] <= 2 * topk + 5, 'Medium', 'Low'))
        g['Cell_ID'] = 'R' + g['Row'].astype(str) + 'C' + g['Col'].astype(str)
        g['Week_Start'] = (weeks[-1] + pd.Timedelta(days=7)).date()
        self.grid_pred = g[['Rank', 'Cell_ID', 'Lat_Center', 'Lon_Center', 'Pred_Next_Week',
                            'Last_4wk_Avg', 'Main_Crime', 'Risk_Level', 'Week_Start', 'Row', 'Col']]
        self.grid_pred.to_csv(os.path.join(self.out_dir, 'grid_risk_next_week.csv'), index=False)
        self._grid_shape = (nr, nc)
        if verbose:
            print(self.grid_eval.round(3).to_string(index=False))
            print(f"   Grid: {nr} x {nc} cells ({cell_m} m), weeks: {W}")
        return self.grid_pred

    def save_grid_map(self, path=os.path.join(MAP_DIR, 'ai_grid_risk_map.html')):
        import folium
        g = self.grid_pred
        vmax = g['Pred_Next_Week'].max()
        from map_utils import create_map
        m = create_map([self.df['Latitude'].mean(), self.df['Longitude'].mean()], zoom_start=14, theme='light')
        for _, r in g[g['Pred_Next_Week'] > 0.05].iterrows():
            b = [[self._lat0 + r['Row'] * self._dlat, self._lon0 + r['Col'] * self._dlon],
                 [self._lat0 + (r['Row'] + 1) * self._dlat, self._lon0 + (r['Col'] + 1) * self._dlon]]
            folium.Rectangle(bounds=b, color='#444', weight=1, fill=True,
                             fill_color=_level_color(r['Pred_Next_Week'], vmax), fill_opacity=0.55,
                             tooltip=f"{r['Cell_ID']} | {r['Risk_Level']} | expected {r['Pred_Next_Week']:.1f} crimes | main: {r['Main_Crime']}").add_to(m)
        folium.LayerControl().add_to(m)
        m.save(path)
        return path

    # =================================================================
    # FEATURE 7: ANOMALY DETECTION
    # =================================================================
    def detect_anomalies(self):
        df = self.df
        ws = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='D')
        valid = ((ws >= self.first_date) & (ws + pd.Timedelta(days=6) <= self.last_date)).values
        weeks = pd.date_range(ws[valid].min(), ws[valid].max(), freq='7D')
        wk = (df[valid].assign(Week=ws[valid]).groupby(['Week', 'Area', 'Crime_Type']).size()
              .unstack('Crime_Type', fill_value=0).reindex(columns=CRIME_TYPES, fill_value=0))
        full = pd.MultiIndex.from_product([weeks, self.areas], names=['Week', 'Area'])
        wk = wk.reindex(full, fill_value=0).reset_index()
        wk['Total'] = wk[CRIME_TYPES].sum(axis=1)

        g = wk.groupby('Area')['Total']
        wk['Expected'] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
        sd = g.transform(lambda s: s.shift(1).rolling(12, min_periods=6).std())
        eff = np.maximum(sd.fillna(1.0), np.sqrt(np.maximum(wk['Expected'].fillna(1.0), 1.0)))
        wk['Z_Score'] = (wk['Total'] - wk['Expected']) / eff

        for t in CRIME_TYPES:                                   # each crime type has its own expected level
            wk[f'E_{t}'] = wk.groupby('Area')[t].transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
        wk = wk.dropna(subset=['Expected']).reset_index(drop=True)
        sev = sum(wk[t] * SEVERITY[t] for t in CRIME_TYPES)
        feat = np.column_stack([wk['Total'], wk['Z_Score'], sev, wk['Robbery'], wk['Assault'] + wk['Murder']])
        iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
        iso.fit(StandardScaler().fit_transform(feat))
        wk['IF_Anomaly'] = iso.predict(StandardScaler().fit_transform(feat)) == -1

        wk['Level'] = np.where((wk['Z_Score'] >= 3.0) | ((wk['Z_Score'] >= 2.5) & wk['IF_Anomaly']), 'High',
                               np.where(wk['IF_Anomaly'] & (wk['Z_Score'] >= 2.0), 'Medium', ''))
        a = wk[(wk['Level'] != '') & (wk['Total'] > wk['Expected'])].copy()
        excess = pd.DataFrame({t: a[t] - a[f'E_{t}'] for t in CRIME_TYPES})
        a['Top_Crime'] = excess.idxmax(axis=1)
        a['Change_%'] = (100 * (a['Total'] - a['Expected']) / a['Expected'].clip(lower=0.5)).round(0)
        a['Message'] = [f"{r.Area}: {int(r.Total)} crimes vs ~{r.Expected:.1f} expected (+{r['Change_%']:.0f}%), mostly {r.Top_Crime}"
                        for _, r in a.iterrows()]
        a = a.rename(columns={'Week': 'Week_Start', 'Total': 'Count'})
        a['Expected'] = a['Expected'].round(1)
        a['Z_Score'] = a['Z_Score'].round(2)
        a = a[['Week_Start', 'Area', 'Count', 'Expected', 'Change_%', 'Z_Score', 'Top_Crime', 'Level', 'Message']]
        self.anomalies = a.sort_values('Week_Start', ascending=False).reset_index(drop=True)
        self.anomalies.to_csv(os.path.join(self.out_dir, 'anomalies.csv'), index=False)
        self._weekly_area = wk
        return self.anomalies

    # =================================================================
    # FEATURE 8: FAIRNESS / BIAS AUDIT
    # =================================================================
    def fairness_audit(self):
        t = self.bundle['test_pred'].copy()
        t['Station'] = t['Area'].map(self.station_of)
        t['Flag'] = (t['Prob'] >= self.bundle['cut_high']).astype(int)
        overall_fpr = t.loc[t['Target'] == 0, 'Flag'].mean()
        total_flags, total_crimes = t['Flag'].sum(), t['Count'].sum()

        def group_stats(col):
            rows = []
            for name, g in t.groupby(col):
                neg, pos = g[g['Target'] == 0], g[g['Target'] == 1]
                rows.append({
                    col: name, 'Cells': len(g), 'Actual_Rate': g['Target'].mean(),
                    'Avg_Pred_Prob': g['Prob'].mean(), 'Calibration_Gap': g['Prob'].mean() - g['Target'].mean(),
                    'High_Flag_Rate': g['Flag'].mean(),
                    'FPR': neg['Flag'].mean() if len(neg) else np.nan,
                    'TPR': pos['Flag'].mean() if len(pos) else np.nan,
                    'Flag_Share_%': 100 * g['Flag'].sum() / max(total_flags, 1),
                    'Crime_Share_%': 100 * g['Count'].sum() / max(total_crimes, 1),
                })
            r = pd.DataFrame(rows)
            r['FPR_vs_Overall'] = r['FPR'] / overall_fpr if overall_fpr else np.nan
            return r.round(3)

        by_area, by_station = group_stats('Area'), group_stats('Station')
        by_area.to_csv(os.path.join(self.out_dir, 'fairness_by_area.csv'), index=False)
        by_station.to_csv(os.path.join(self.out_dir, 'fairness_by_station.csv'), index=False)

        findings = []
        overall_rate = t['Target'].mean()
        for _, r in by_area.iterrows():
            if abs(r['Calibration_Gap']) > 0.05:
                kind = 'OVER-estimates' if r['Calibration_Gap'] > 0 else 'UNDER-estimates'
                findings.append(f"- [WARNING] {r['Area']}: model {kind} risk (calibration gap {r['Calibration_Gap']:+.3f}).")
            if r['FPR_vs_Overall'] > 1.5:
                explained = r['Actual_Rate'] > 1.15 * overall_rate
                note = (f"expected: this area's real crime rate ({r['Actual_Rate']:.2f}) is much higher than overall ({overall_rate:.2f})"
                        if explained else "NOT explained by a higher crime rate - investigate")
                findings.append(f"- [{'INFO' if explained else 'WARNING'}] {r['Area']}: false-alarm rate is "
                                f"{r['FPR_vs_Overall']:.2f}x overall ({note}).")
            if r['Crime_Share_%'] > 0 and r['Flag_Share_%'] > 1.5 * r['Crime_Share_%']:
                findings.append(f"- [WARNING] {r['Area']}: gets {r['Flag_Share_%']:.1f}% of High-risk flags but only "
                                f"{r['Crime_Share_%']:.1f}% of crimes (over-policing risk).")
        if not any('[WARNING]' in f for f in findings):
            findings.append('- No serious bias warning found (checks: calibration gap > 0.05, unexplained FPR > 1.5x, flag-share > 1.5x crime-share).')

        txt = f"""FAIRNESS / BIAS AUDIT REPORT
Generated: {datetime.now():%Y-%m-%d %H:%M}
Model: {self.bundle['model_name']} | Test period: {t['Date'].min().date()} to {t['Date'].max().date()}
Overall false-alarm rate (High flag when no crime): {overall_fpr:.3f}

WHAT WAS CHECKED
1. Calibration gap  = average predicted probability - actual crime rate (per area / station)
2. False-positive rate parity (FPR) = how often an area is flagged High when no crime happened
3. Flag share vs crime share = does an area get more police attention than its crimes justify

FINDINGS
{chr(10).join(findings)}

IMPORTANT LIMITATIONS (mention these in the viva)
- The model only learns from RECORDED crimes. If an area has more policing, more crime gets recorded
  there -> a feedback loop. Model output should therefore be treated as a "suggestion", not an order.
- Different areas can genuinely have different base crime rates; a difference in flag rate is not always bias.
- The data is synthetic/simulated: a new audit on real FIR data is required before any real deployment.
"""
        with open(os.path.join(self.out_dir, 'fairness_report.txt'), 'w', encoding='utf-8') as f:
            f.write(txt)
        return by_area, by_station, txt

    # =================================================================
    # LOAD SAVED RESULTS (for the GUI)
    # =================================================================
    def load_artifacts(self):
        def rd(name, **kw):
            p = os.path.join(self.out_dir, name)
            return pd.read_csv(p, **kw) if os.path.exists(p) else None
        self.forecast_df = rd('forecast_next30.csv', parse_dates=['Date'])
        self.forecast_eval = rd('forecast_eval.csv')
        self.hotspots = rd('hotspots.csv')
        self.grid_pred = rd('grid_risk_next_week.csv')
        self.grid_eval = rd('grid_eval.csv')
        self.anomalies = rd('anomalies.csv', parse_dates=['Week_Start'])
        return self

    # =================================================================
    # FEATURE 9: CHARTS + PDF REPORT
    # =================================================================
    def make_charts(self, folder=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        folder = folder or os.path.join(self.out_dir, 'charts')
        os.makedirs(folder, exist_ok=True)
        paths = {}

        # 1. Monthly trend
        mo = self.df.groupby(self.df['Date'].dt.to_period('M')).size()
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(mo.index.astype(str), mo.values, marker='o', color='#c0392b')
        ax.set_title('Monthly crime count'); ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.grid(alpha=.3); fig.tight_layout()
        paths['trend'] = os.path.join(folder, 'trend.png'); fig.savefig(paths['trend'], dpi=130); plt.close(fig)

        # 2. Forecast
        if self.forecast_df is not None:
            hist = self.area_daily.sum(axis=1).iloc[-90:]
            fc = self.forecast_df.groupby('Date')['Forecast'].sum()
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(hist.index, hist.values, label='Actual (last 90 days)', color='#2c3e50')
            ax.plot(fc.index, fc.values, label='Forecast (next 30 days)', color='#e67e22', linestyle='--')
            ax.set_title('City-wide daily crimes: actual vs forecast'); ax.legend(); ax.grid(alpha=.3)
            fig.tight_layout()
            paths['forecast'] = os.path.join(folder, 'forecast.png'); fig.savefig(paths['forecast'], dpi=130); plt.close(fig)

        # 3. Feature importance
        if self.bundle is not None:
            imp = self.bundle['importance'].head(8).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.barh(imp['Feature'], imp['Importance'], color='#2980b9')
            ax.set_title('What drives the risk model? (permutation importance, AUC drop)'); fig.tight_layout()
            paths['importance'] = os.path.join(folder, 'importance.png'); fig.savefig(paths['importance'], dpi=130); plt.close(fig)

        # 4. Hotspot scatter
        if self.hotspots is not None and len(self.hotspots):
            fig, ax = plt.subplots(figsize=(6, 4.6))
            ax.scatter(self.df['Longitude'], self.df['Latitude'], s=3, alpha=.25, color='gray')
            for _, r in self.hotspots.head(10).iterrows():
                ax.scatter(r['Longitude'], r['Latitude'], s=r['Crimes'] * 1.2, alpha=.45, color='red')
                ax.annotate(f"#{r['Rank']}", (r['Longitude'], r['Latitude']), fontsize=8, ha='center')
            ax.set_title('Top hotspots (DBSCAN)'); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            fig.tight_layout()
            paths['hotspots'] = os.path.join(folder, 'hotspots.png'); fig.savefig(paths['hotspots'], dpi=130); plt.close(fig)

        # 5. Grid risk heat
        if self.grid_pred is not None:
            nr, nc = int(self.grid_pred['Row'].max()) + 1, int(self.grid_pred['Col'].max()) + 1
            M = np.zeros((nr, nc))
            M[self.grid_pred['Row'].values, self.grid_pred['Col'].values] = self.grid_pred['Pred_Next_Week'].values
            fig, ax = plt.subplots(figsize=(6, 4.6))
            im = ax.imshow(M, origin='lower', cmap='YlOrRd'); fig.colorbar(im, label='Expected crimes next week')
            ax.set_title('Grid risk prediction (500 m cells)'); fig.tight_layout()
            paths['grid'] = os.path.join(folder, 'grid.png'); fig.savefig(paths['grid'], dpi=130); plt.close(fig)

        # 6. Anomaly chart (area with the biggest alert)
        if self.anomalies is not None and len(self.anomalies) and hasattr(self, '_weekly_area'):
            area = self.anomalies.sort_values('Z_Score', ascending=False).iloc[0]['Area']
            w = self._weekly_area[self._weekly_area['Area'] == area]
            al = w[(w['Z_Score'] >= 2.5)]
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(w['Week'], w['Total'], color='#34495e', label='Weekly crimes')
            ax.plot(w['Week'], w['Expected'], color='#27ae60', linestyle='--', label='Expected (12-wk avg)')
            ax.scatter(al['Week'], al['Total'], color='red', zorder=5, label='Anomaly')
            ax.set_title(f'Anomaly detection - {area}'); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
            paths['anomaly'] = os.path.join(folder, 'anomaly.png'); fig.savefig(paths['anomaly'], dpi=130); plt.close(fig)
        return paths

    def build_pdf_report(self, path=None):
        from fpdf import FPDF
        path = path or os.path.join('outputs', 'reports', f"ai_crime_report_{datetime.now():%Y%m%d_%H%M%S}.pdf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        charts = self.make_charts()
        by_area, by_station, fair_txt = self.fairness_audit()

        class PDF(FPDF):
            def header(s):
                s.set_font('Arial', 'B', 9); s.set_text_color(120, 120, 120)
                s.cell(0, 6, 'AI Crime Analysis & Prediction System - Larkana', 0, 1, 'R'); s.ln(2)

            def footer(s):
                s.set_y(-12); s.set_font('Arial', 'I', 8); s.set_text_color(120, 120, 120)
                s.cell(0, 8, f'Page {s.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.set_auto_page_break(True, 15)

        def title(t):
            pdf.set_font('Arial', 'B', 14); pdf.set_text_color(192, 57, 43)
            pdf.cell(0, 9, _ascii(t), 0, 1); pdf.set_text_color(0, 0, 0)

        def para(t, size=10):
            pdf.set_font('Arial', '', size); pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, _ascii(t)); pdf.ln(1)

        def table(df, widths, size=8):
            pdf.set_font('Arial', 'B', size); pdf.set_fill_color(230, 230, 230)
            for c, w in zip(df.columns, widths):
                pdf.cell(w, 6, _ascii(c)[:int(w / 1.9)], 1, 0, 'C', True)
            pdf.ln(); pdf.set_font('Arial', '', size)
            for _, r in df.iterrows():
                for v, w in zip(r.values, widths):
                    s = f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
                    pdf.cell(w, 5.5, _ascii(s)[:int(w / 1.7)], 1)
                pdf.ln()
            pdf.ln(2)

        def img(key, w=170):
            if key in charts:
                pdf.image(charts[key], w=w); pdf.ln(2)

        # ---- Page 1: summary ----
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20); pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 12, 'AI Crime Analysis Report', 0, 1)
        pdf.set_font('Arial', '', 10); pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 6, _ascii(f"Generated {datetime.now():%Y-%m-%d %H:%M} | Data: {self.first_date.date()} to {self.last_date.date()} | {len(self.df)} records"), 0, 1)
        pdf.set_text_color(0, 0, 0); pdf.ln(3)
        title('1. Executive summary')
        top_area = self.df['Area'].value_counts().idxmax()
        night = (self.df['Shift'] == 'Night').mean() * 100
        lines = [f"Total records: {len(self.df)} across {len(self.areas)} areas. Most crimes: {top_area}. Night shift share: {night:.0f}%."]
        if self.hotspots is not None and len(self.hotspots):
            h = self.hotspots.iloc[0]
            lines.append(f"Top hotspot: #1 near {h['Area']} ({h['Crimes']} crimes, mostly {h['Dominant_Crime']}, peak: {h['Peak_Shift']}).")
        if self.forecast_df is not None:
            lines.append(f"Forecast: about {self.forecast_df['Forecast'].sum():.0f} crimes expected city-wide in the next 30 days.")
        if self.grid_pred is not None:
            g = self.grid_pred.iloc[0]
            lines.append(f"Highest-risk grid cell next week: {g['Cell_ID']} (expected {g['Pred_Next_Week']:.1f} crimes, main: {g['Main_Crime']}).")
        if self.anomalies is not None and len(self.anomalies):
            lines.append(f"Latest anomaly alert: {self.anomalies.iloc[0]['Message']} (week of {pd.to_datetime(self.anomalies.iloc[0]['Week_Start']).date()}).")
        for l in lines:
            para('- ' + l)
        img('trend')

        # ---- Page 2: model ----
        title('2. Risk model (honest evaluation)')
        para("Target: will at least one crime occur in this area during this shift on this day? "
             "Features use only PAST information (lags), and the model is tested on the final 365 days "
             "(time-based split), so there is no data leakage.")
        cmp = self.bundle['comparison'][['Model', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1']].round(3)
        table(cmp, [62, 22, 22, 22, 22, 22])
        img('importance')

        # ---- Hotspots + grid ----
        pdf.add_page()
        title('3. Hotspots (DBSCAN clustering)')
        if self.hotspots is not None and len(self.hotspots):
            ht = self.hotspots.head(6)[['Rank', 'Area', 'Crimes', 'Dominant_Crime', 'Peak_Shift', 'Last_90_Days']]
            table(ht, [14, 50, 22, 40, 30, 30])
        img('hotspots', 120)
        title('4. Grid risk prediction (next week)')
        if self.grid_pred is not None:
            gt = self.grid_pred.head(6)[['Rank', 'Cell_ID', 'Pred_Next_Week', 'Last_4wk_Avg', 'Main_Crime', 'Risk_Level']].round(2)
            table(gt, [14, 30, 36, 36, 34, 30])
        img('grid', 110)

        # ---- Forecast + anomalies ----
        pdf.add_page()
        title('5. Forecast (next 30 days)')
        img('forecast')
        if self.forecast_eval is not None:
            table(self.forecast_eval.round(3), [90, 40, 40])
        title('6. Anomaly alerts')
        img('anomaly')
        if self.anomalies is not None and len(self.anomalies):
            at = self.anomalies.head(6).copy()
            at['Week_Start'] = pd.to_datetime(at['Week_Start']).dt.date.astype(str)
            table(at[['Week_Start', 'Area', 'Count', 'Expected', 'Z_Score', 'Level']], [32, 55, 22, 26, 26, 24])

        # ---- Fairness ----
        pdf.add_page()
        title('7. Fairness / bias audit')
        para(fair_txt.split('FINDINGS')[1].split('IMPORTANT')[0].strip())
        ba = by_area[['Area', 'Actual_Rate', 'Avg_Pred_Prob', 'Calibration_Gap', 'High_Flag_Rate', 'FPR']].round(3)
        table(ba, [50, 26, 28, 30, 30, 22])
        para('Limitation: ' + fair_txt.split('IMPORTANT LIMITATIONS')[1].strip().split('\n')[1].strip('- '))
        pdf.output(path)
        return path
