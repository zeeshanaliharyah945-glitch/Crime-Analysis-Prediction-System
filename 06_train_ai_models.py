# 06_train_ai_models.py
# =====================================================================
#  This script trains / runs ALL AI features in one go and saves the
#  results to outputs/ai/ (the GUI reads these files).
#
#  How to run:   python 06_train_ai_models.py
#  Time: roughly 1-2 minutes.
# =====================================================================

import os
import sys
import time
import traceback

from ai_engine import DATA_PATH, CrimeAI


def step(n, total, title):
    print("\n" + "=" * 62)
    print(f"STEP {n}/{total}: {title}")
    print("=" * 62)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found: {DATA_PATH}\nRun this first:  python create_dataset.py")
        sys.exit(1)

    t0 = time.time()
    ai = CrimeAI()
    print(f"Data loaded: {len(ai.df)} records | {ai.first_date.date()} -> {ai.last_date.date()} | {len(ai.areas)} areas")
    failed = []
    total = 8

    def run(n, title, fn):
        step(n, total, title)
        try:
            fn()
        except Exception:
            failed.append(title)
            print(f"[ERROR] {title} fail ho gaya:")
            traceback.print_exc()

    run(1, "Honest Risk Model (time-based split, no leakage)", lambda: ai.train_risk_model())

    def explain_demo():
        p, lvl, X = ai.risk_for('Market Commercial', ai.last_date + __import__('pandas').Timedelta(days=1), 'Night')
        print(f"Example prediction -> Market Commercial, Night, next day: {p * 100:.1f}% ({lvl})")
        print("Why? (Explainable AI):")
        print(ai.explain_text(ai.explain(X)))
    run(2, "Explainable AI demo", explain_demo)

    run(3, "Crime Forecasting (next 30 days)", lambda: ai.train_forecaster())

    def hot():
        hs = ai.find_hotspots()
        print(hs.head(5)[['Rank', 'Area', 'Crimes', 'Dominant_Crime', 'Peak_Shift', 'Radius_m']].to_string(index=False))
        print("Map saved:", ai.save_hotspot_map())
    run(4, "Hotspot Detection (DBSCAN)", hot)

    def grid():
        g = ai.train_grid_model()
        print(g.head(5)[['Rank', 'Cell_ID', 'Pred_Next_Week', 'Last_4wk_Avg', 'Main_Crime', 'Risk_Level']].round(2).to_string(index=False))
        print("Map saved:", ai.save_grid_map())
    run(5, "Grid Risk Prediction (500 m cells)", grid)

    def anom():
        a = ai.detect_anomalies()
        print(f"{len(a)} anomalies mile. Taaza alerts:")
        for m in a['Message'].head(5):
            print("  !", m)
    run(6, "Anomaly Detection", anom)

    def fair():
        _, _, txt = ai.fairness_audit()
        print(txt.split('FINDINGS')[1].split('IMPORTANT')[0].strip())
    run(7, "Fairness / Bias Audit", fair)

    def pdf():
        print("PDF saved:", ai.build_pdf_report())
    run(8, "AI PDF Report (with charts)", pdf)

    print("\n" + "=" * 62)
    if failed:
        print("KUCH STEPS FAIL HUE:", ', '.join(failed))
    else:
        print("ALL AI FEATURES ARE READY!")
    print(f"Total time: {time.time() - t0:.0f} sec")
    print("Results: outputs/ai/ | Maps: outputs/heatmaps/ | PDF: outputs/reports/")
    print("Now start the GUI:  python crime_analysis_gui.py   ->  '🧠 AI FEATURES' button")
    print("=" * 62)


if __name__ == '__main__':
    main()
