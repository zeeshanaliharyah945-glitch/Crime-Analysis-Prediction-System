# 04_hotspot_mapping.py (FIXED VERSION)
# Purpose: Create interactive crime hotspot maps
import os
import pandas as pd
import numpy as np
import pickle
import folium
from folium.plugins import HeatMap, MarkerCluster
from map_utils import create_map, open_map
import webbrowser
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 4: CRIME HOTSPOT MAPPING")
print("="*60)

# ==================== LOAD DATA ====================
print("\n📂 Loading data...")
df = pd.read_csv('data/processed_crime_data.csv')
print(f"✅ Loaded {len(df)} records")

# ==================== LOAD MODEL ====================
print("\n🔮 Loading prediction model...")
with open('outputs/models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('outputs/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✅ Model loaded")

# ==================== PREPARE FEATURES FOR PREDICTION ====================
print("\n🔧 Preparing features for risk prediction...")

feature_columns = [
    'Year', 'Month', 'Day', 'DayOfWeek', 'Hour',
    'Severity', 'Crime_Type_Encoded', 'Police_Station_Encoded',
    'Area_Encoded', 'TimePeriod_Encoded', 'Station_Crime_Count'
]

# Aggregate by area
area_data = df.groupby('Area').agg({
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Crime_Count': 'first',
    'Station_Crime_Count': 'first',
    'Police_Station_Encoded': 'first',
    'Area_Encoded': 'first'
}).reset_index()

# Add time features (using typical values)
area_data['Year'] = 2024
area_data['Month'] = 12
area_data['Day'] = 25
area_data['DayOfWeek'] = 2
area_data['Hour'] = 20
area_data['Severity'] = 3
area_data['TimePeriod_Encoded'] = 3
area_data['Crime_Type_Encoded'] = 1

# Make predictions
X_pred = area_data[feature_columns]
X_pred_scaled = scaler.transform(X_pred)
predictions = model.predict(X_pred_scaled)
probabilities = model.predict_proba(X_pred_scaled)[:, 1]

# Add predictions to dataframe
area_data['High_Risk_Prediction'] = predictions
area_data['Risk_Probability'] = probabilities
area_data['Risk_Level'] = area_data['Risk_Probability'].apply(
    lambda x: 'Critical' if x > 0.8 else 'High' if x > 0.6 else 'Medium' if x > 0.4 else 'Low'
)

print(f"   Predictions for {len(area_data)} areas")
print(f"\n   Risk Level Distribution:")
for level in ['Critical', 'High', 'Medium', 'Low']:
    count = (area_data['Risk_Level'] == level).sum()
    print(f"   - {level}: {count} areas ({count/len(area_data)*100:.1f}%)")

# ==================== CREATE MAP ====================
print("\n🗺️ Creating interactive map...")

center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

crime_map = create_map(
    location=[center_lat, center_lon],
    zoom_start=13,
    theme='light'
)

# Add title
title_html = '''
<div style="position: fixed; top: 10px; left: 50px; z-index: 1000; background-color: white; 
            padding: 10px; border-radius: 5px; border: 2px solid red; font-family: Arial;">
    <h3>Crime Hotspot Map - High Risk Areas</h3>
</div>
'''
crime_map.get_root().html.add_child(folium.Element(title_html))

# Add Heatmap
print("   🔥 Adding heatmap layer...")
heat_data = [[row['Latitude'], row['Longitude'], row['Crime_Count']] 
             for _, row in area_data.iterrows()]
HeatMap(heat_data, min_opacity=0.4, radius=15, blur=10,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
).add_to(crime_map)

# Color mapping
risk_colors = {'Critical': 'darkred', 'High': 'red', 'Medium': 'orange', 'Low': 'green'}

# Add markers
print("   📍 Adding area markers...")
marker_cluster = MarkerCluster().add_to(crime_map)

for _, area in area_data.iterrows():
    popup_html = f"""
    <div style="font-family: Arial; width: 250px;">
        <h4 style="color: {risk_colors[area['Risk_Level']]};">{area['Area']}</h4>
        <hr>
        <b>Risk Level:</b> <span style="color: {risk_colors[area['Risk_Level']]};">{area['Risk_Level']}</span><br>
        <b>Risk Probability:</b> {area['Risk_Probability']:.1%}<br>
        <b>Crime Count:</b> {int(area['Crime_Count'])}<br>
        <b>Prediction:</b> {"⚠️ HIGH RISK" if area['High_Risk_Prediction'] == 1 else "✅ LOW RISK"}<br>
        <hr>
        <i>Recommended: {'Increase patrols' if area['High_Risk_Prediction'] == 1 else 'Regular monitoring'}</i>
    </div>
    """
    
    folium.Marker(
        location=[area['Latitude'], area['Longitude']],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color=risk_colors[area['Risk_Level']], icon='info-sign', prefix='glyphicon'),
        tooltip=f"{area['Area']} - {area['Risk_Level']} Risk"
    ).add_to(marker_cluster)

# Add circle markers (size based on crime count)
print("   ⚪ Adding circle markers...")
for _, area in area_data.iterrows():
    radius = np.sqrt(area['Crime_Count']) * 2
    folium.CircleMarker(
        location=[area['Latitude'], area['Longitude']],
        radius=radius,
        popup=f"{area['Area']}<br>Crimes: {int(area['Crime_Count'])}",
        color=risk_colors[area['Risk_Level']],
        fill=True,
        fill_opacity=0.3,
        weight=2
    ).add_to(crime_map)

# Add layer control
folium.LayerControl().add_to(crime_map)

# Save map
crime_map.save('outputs/heatmaps/crime_hotspot_map.html')
print(f"   ✅ Map saved: outputs/heatmaps/crime_hotspot_map.html")

# ==================== CREATE PREDICTION SUMMARY MAP ====================
print("\n🗺️ Creating prediction summary map...")

pred_map = create_map(location=[center_lat, center_lon], zoom_start=13, theme='light')

# Add prediction heatmap
pred_heat_data = [[row['Latitude'], row['Longitude'], row['Risk_Probability']] 
                  for _, row in area_data.iterrows()]
HeatMap(pred_heat_data, min_opacity=0.3, radius=20, blur=15,
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
).add_to(pred_map)

# Add high risk markers only
high_risk_areas = area_data[area_data['Risk_Level'].isin(['High', 'Critical'])]
for _, area in high_risk_areas.iterrows():
    folium.Marker(
        location=[area['Latitude'], area['Longitude']],
        popup=f"<b>{area['Area']}</b><br>Risk: {area['Risk_Probability']:.1%}<br>⚠️ HIGH RISK AREA",
        icon=folium.Icon(color='red', icon='warning-sign', prefix='glyphicon')
    ).add_to(pred_map)

folium.LayerControl().add_to(pred_map)
pred_map.save('outputs/heatmaps/prediction_risk_map.html')
print(f"   ✅ Map saved: outputs/heatmaps/prediction_risk_map.html")

# ==================== GENERATE RISK REPORT (FIXED - NO SPECIAL CHARACTERS) ====================
print("\n📄 Generating Risk Assessment Report...")

top_risks = area_data.nlargest(10, 'Risk_Probability')[['Area', 'Risk_Probability', 'Crime_Count', 'Risk_Level']]

risk_report = f"""
================================================================================
                    CRIME RISK ASSESSMENT REPORT
                        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

OVERVIEW
--------------------------------------------------------------------------------
Total Areas Analyzed: {len(area_data)}
Critical Risk Areas: {(area_data['Risk_Level'] == 'Critical').sum()}
High Risk Areas: {(area_data['Risk_Level'] == 'High').sum()}
Medium Risk Areas: {(area_data['Risk_Level'] == 'Medium').sum()}
Low Risk Areas: {(area_data['Risk_Level'] == 'Low').sum()}

TOP 10 HIGHEST RISK AREAS
--------------------------------------------------------------------------------
"""
for idx, (_, area) in enumerate(top_risks.iterrows(), 1):
    risk_report += f"{idx}. {area['Area']}\n"
    risk_report += f"   - Risk Level: {area['Risk_Level']}\n"
    risk_report += f"   - Risk Probability: {area['Risk_Probability']:.1%}\n"
    risk_report += f"   - Historical Crime Count: {int(area['Crime_Count'])}\n\n"

risk_report += """
RECOMMENDATIONS
--------------------------------------------------------------------------------
CRITICAL & HIGH RISK AREAS:
   [1] Increase police patrol frequency
   [2] Install additional CCTV cameras
   [3] Conduct community awareness programs
   [4] Improve street lighting
   [5] Deploy rapid response teams

MEDIUM RISK AREAS:
   [1] Monitor crime patterns closely
   [2] Implement community watch programs
   [3] Regular security audits

LOW RISK AREAS:
   [1] Maintain current security measures
   [2] Preventive awareness campaigns

TIME-BASED RECOMMENDATIONS
--------------------------------------------------------------------------------
   [1] Peak crime hours: 20:00 - 04:00 (Night shift)
   [2] Increase patrols on weekends
   [3] Focus on high-risk areas during December

================================================================================
"""

# Save report with utf-8 encoding to avoid Unicode errors
with open('outputs/reports/risk_assessment_report.txt', 'w', encoding='utf-8') as f:
    f.write(risk_report)
print("   ✅ Saved: outputs/reports/risk_assessment_report.txt")

# Also print to console
print(risk_report)

# ==================== OPEN MAP IN BROWSER ====================
print("\n🌐 Opening map in browser...")
open_map('outputs/heatmaps/crime_hotspot_map.html', serve=False)

print("\n" + "="*60)
print("✅ CRIME HOTSPOT MAPPING COMPLETE!")
print("="*60)
print("\nFiles generated:")
print("   - outputs/heatmaps/crime_hotspot_map.html (main map)")
print("   - outputs/heatmaps/prediction_risk_map.html (prediction map)")
print("   - outputs/reports/risk_assessment_report.txt")