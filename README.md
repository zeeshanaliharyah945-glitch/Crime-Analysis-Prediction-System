# Crime Analysis & Prediction System (Larkana)

AI-powered crime hotspot detection and predictive policing system: machine-learning
risk models, forecasting, DBSCAN hotspots, grid risk prediction, anomaly alerts,
fairness audit, interactive maps, a desktop GUI and a chatbot.

> The dataset is **synthetic / simulated**. Audit the models on real FIR data before any real deployment.

## Project structure

```
crime_project/
├── data/                       datasets (raw + processed)
├── outputs/
│   ├── ai/                     forecasts, hotspots, grid risk, alerts, fairness report, charts
│   ├── charts/                 EDA and model-evaluation charts
│   ├── heatmaps/               interactive HTML maps
│   ├── models/                 trained models (.pkl)
│   ├── powerbi/                CSV exports for Power BI
│   └── reports/                text / PDF reports (old GUI PDFs in reports/archive)
├── create_dataset.py           generate the synthetic dataset
├── 01_data_preprocessing.py    cleaning + feature engineering
├── 02_eda_analysis.py          exploratory analysis and charts
├── 03_machine_learning.py      Random Forest / Logistic Regression
├── 04_hotspot_mapping.py       hotspot + prediction maps
├── 05_export_for_powerbi.py    Power BI export
├── 06_train_ai_models.py       all AI features (risk, forecast, hotspots, grid, anomaly, fairness, PDF)
├── ai_engine.py                AI engine (used by 06 and the GUI)
├── ai_chatbot.py               crime chatbot (English / Roman Urdu input)
├── ai_features_gui.py          AI features window
├── crime_analysis_gui.py       main desktop GUI
├── animated_hotspot_map.py     animated hotspot map (standalone)
├── map_utils.py                shared map helper (key-free basemaps)
├── run_all.py                  run the whole pipeline
└── requirements.txt
```

## Setup

```
python -m venv venv
venv\Scripts\activate          (Windows)     |     source venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
```

## Run

```
python run_all.py                  # full pipeline (dataset -> models -> maps -> AI features)
python crime_analysis_gui.py       # desktop GUI (run run_all.py or 06_train_ai_models.py first)
python animated_hotspot_map.py     # animated map only
```

Always run commands from the project root folder.

## Maps and the "API KEY REQUIRED" message

CARTO began requiring an API key for its basemaps in late August 2026, which made the old
CartoDB tiles show an "API KEY REQUIRED" watermark. All maps now go through `map_utils.create_map()`,
which uses key-free Esri and OpenStreetMap basemaps. Use the layer control (top-right of the map)
to switch between Light, Dark, Streets, Satellite and OpenStreetMap.

Optional: if you have a free CARTO key, set `CARTO_API_KEY` and two extra CARTO basemaps appear.
The GUI opens maps through a small local web server so that OpenStreetMap tiles also load
(they are blocked on pages opened as `file://`).

## Chatbot

Works offline by default. If `ANTHROPIC_API_KEY` is set, Claude is used to parse the question
into a safe JSON spec (the model never executes code). Questions can be typed in English or Roman Urdu;
answers are in English.
