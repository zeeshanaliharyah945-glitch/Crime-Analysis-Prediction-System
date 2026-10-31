# animated_hotspot_map.py
# Creates interactive hotspot map with animations for risk areas

import pandas as pd
import numpy as np
import pickle
import folium
from folium.plugins import HeatMap, MarkerCluster, BeautifyIcon
from folium import features
from map_utils import create_map, open_map
import webbrowser
import os
from datetime import datetime
import json
import random

print("="*60)
print("🗺️ GENERATING ANIMATED HOTSPOT MAP WITH RISK ANIMATIONS")
print("="*60)

# ==================== LOAD DATA ====================
print("\n📂 Loading data...")
df = pd.read_csv('data/processed_crime_data.csv')

# Load model
with open('outputs/models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('outputs/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('outputs/models/label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

print(f"✅ Loaded {len(df)} records")

# ==================== PREPARE RISK PREDICTIONS ====================
print("\n🔮 Calculating risk predictions for each area...")

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

# Add time features
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

area_data['Risk_Probability'] = probabilities
area_data['Risk_Level'] = area_data['Risk_Probability'].apply(
    lambda x: 'Critical' if x > 0.8 else 'High' if x > 0.6 else 'Medium' if x > 0.4 else 'Low'
)

print(f"✅ Predictions complete for {len(area_data)} areas")

# ==================== CREATE ANIMATED MAP ====================
print("\n🗺️ Creating animated hotspot map...")

center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

# Create base map with custom style
crime_map = create_map(
    location=[center_lat, center_lon],
    zoom_start=13,
    theme='dark',  # Dark theme for better visibility
    control_scale=True
)

# ==================== ADD CUSTOM HTML/CSS FOR ANIMATIONS ====================
animation_css = """
<style>
@keyframes pulse-red {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7);
        opacity: 0.7;
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 20px rgba(255, 0, 0, 0);
        opacity: 1;
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(255, 0, 0, 0);
        opacity: 0.7;
    }
}

@keyframes pulse-orange {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.7);
        opacity: 0.7;
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 15px rgba(255, 165, 0, 0);
        opacity: 1;
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(255, 165, 0, 0);
        opacity: 0.7;
    }
}

@keyframes pulse-green {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(0, 200, 81, 0.7);
        opacity: 0.7;
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 10px rgba(0, 200, 81, 0);
        opacity: 1;
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(0, 200, 81, 0);
        opacity: 0.7;
    }
}

@keyframes glow {
    0% {
        filter: drop-shadow(0 0 2px rgba(255, 0, 0, 0.5));
    }
    50% {
        filter: drop-shadow(0 0 15px rgba(255, 0, 0, 0.8));
    }
    100% {
        filter: drop-shadow(0 0 2px rgba(255, 0, 0, 0.5));
    }
}

@keyframes float {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-10px);
    }
    100% {
        transform: translateY(0px);
    }
}

@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}

.pulse-red {
    animation: pulse-red 1.5s infinite;
    border-radius: 50%;
}

.pulse-orange {
    animation: pulse-orange 2s infinite;
    border-radius: 50%;
}

.pulse-green {
    animation: pulse-green 2.5s infinite;
    border-radius: 50%;
}

.glow-effect {
    animation: glow 1s infinite;
}

.float-effect {
    animation: float 3s ease-in-out infinite;
}

.rotate-effect {
    animation: rotate 4s linear infinite;
}

.risk-badge {
    font-family: Arial, sans-serif;
    font-size: 12px;
    font-weight: bold;
    padding: 3px 8px;
    border-radius: 20px;
    display: inline-block;
    margin: 2px;
}

.risk-critical {
    background-color: #FF0000;
    color: white;
    box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
}

.risk-high {
    background-color: #FF6600;
    color: white;
}

.risk-medium {
    background-color: #FFAA00;
    color: black;
}

.risk-low {
    background-color: #00C851;
    color: white;
}

.glass-card {
    background: rgba(30, 58, 95, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.risk-meter {
    width: 100%;
    height: 8px;
    background-color: #333;
    border-radius: 4px;
    overflow: hidden;
    margin: 10px 0;
}

.risk-meter-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1s ease-in-out;
}

.warning-icon {
    font-size: 24px;
    display: inline-block;
}

.info-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #00B4D8;
}
</style>
"""

# Add CSS to map
crime_map.get_root().header.add_child(folium.Element(animation_css))

# ==================== ADD HEATMAP LAYER ====================
print("🔥 Adding heatmap layer...")
heat_data = [[row['Latitude'], row['Longitude'], row['Crime_Count']] 
             for _, row in area_data.iterrows()]

HeatMap(heat_data, 
        min_opacity=0.4,
        max_zoom=15,
        radius=20,
        blur=15,
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
).add_to(crime_map)

# ==================== ADD ANIMATED MARKERS FOR EACH AREA ====================
print("📍 Adding animated markers with risk animations...")

# Create marker clusters with animation
high_risk_cluster = MarkerCluster(name="⚠️ HIGH RISK AREAS", overlay=True).add_to(crime_map)
medium_risk_cluster = MarkerCluster(name="🟡 MEDIUM RISK AREAS", overlay=True).add_to(crime_map)
low_risk_cluster = MarkerCluster(name="✅ LOW RISK AREAS", overlay=True).add_to(crime_map)

for _, area in area_data.iterrows():
    risk_prob = area['Risk_Probability']
    risk_level = area['Risk_Level']
    crime_count = int(area['Crime_Count'])
    
    # Determine risk color and animation class
    if risk_level == 'Critical' or risk_level == 'High':
        marker_color = 'red'
        animation_class = 'pulse-red glow-effect float-effect'
        risk_text = "🔴 CRITICAL RISK"
        risk_badge_class = "risk-critical"
        warning_text = "⚠️ IMMEDIATE ACTION REQUIRED ⚠️"
    elif risk_level == 'Medium':
        marker_color = 'orange'
        animation_class = 'pulse-orange'
        risk_text = "🟠 HIGH RISK"
        risk_badge_class = "risk-high"
        warning_text = "⚠️ INCREASE PATROLS NEEDED"
    else:
        marker_color = 'green'
        animation_class = 'pulse-green'
        risk_text = "🟢 LOW RISK"
        risk_badge_class = "risk-low"
        warning_text = "✅ REGULAR MONITORING"
    
    # Create animated icon
    icon_html = f"""
    <div class="{animation_class}" style="
        width: 30px;
        height: 30px;
        background-color: {marker_color};
        border-radius: 50%;
        border: 3px solid white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    ">
        <span style="color: white; font-size: 16px;">⚠️</span>
    </div>
    """
    
    # Create custom icon
    icon = folium.DivIcon(
        html=icon_html,
        icon_size=(30, 30),
        icon_anchor=(15, 15)
    )
    
    # Create glass effect popup with risk meter
    risk_percent = int(risk_prob * 100)
    
    # Determine meter color
    if risk_percent >= 70:
        meter_color = '#FF0000'
    elif risk_percent >= 40:
        meter_color = '#FFA500'
    else:
        meter_color = '#00C851'
    
    popup_html = f"""
    <div class="glass-card" style="width: 320px;">
        <div class="info-title">
            🚔 <span class="float-effect">📍 {area['Area']}</span>
        </div>
        
        <div class="risk-badge {risk_badge_class}">
            {risk_text}
        </div>
        
        <div style="margin: 15px 0;">
            <div class="warning-icon float-effect">{'🔴' if risk_level in ['Critical', 'High'] else '🟡' if risk_level == 'Medium' else '🟢'}</div>
            <span style="font-weight: bold; font-size: 18px;">Risk: {risk_percent}%</span>
        </div>
        
        <div class="risk-meter">
            <div class="risk-meter-fill" style="width: {risk_percent}%; background-color: {meter_color};"></div>
        </div>
        
        <div style="margin: 10px 0;">
            <span style="font-weight: bold;">📊 Statistics:</span><br>
            • Total Crimes: <b>{crime_count}</b><br>
            • Crime Density: <b>{crime_count / 10:.1f} per km²</b><br>
            • Risk Level: <b style="color: {meter_color};">{risk_level.upper()}</b>
        </div>
        
        <div style="margin: 10px 0; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 10px;">
            <span style="font-weight: bold;">💡 Recommendations:</span><br>
            {warning_text}<br>
            • Increase police patrols in this area<br>
            • Install additional CCTV cameras<br>
            • Conduct community awareness programs
        </div>
        
        <div style="font-size: 11px; color: #888; text-align: center; margin-top: 10px;">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    """
    
    # Create marker with custom icon
    marker = folium.Marker(
        location=[area['Latitude'], area['Longitude']],
        popup=folium.Popup(popup_html, max_width=350, min_width=300),
        tooltip=f"⚠️ {area['Area']} - {risk_text} ({risk_percent}%)",
        icon=folium.Icon(color=marker_color, icon='info-sign', prefix='glyphicon')
    )
    
    # Add to appropriate cluster
    if risk_level in ['Critical', 'High']:
        marker.add_to(high_risk_cluster)
    elif risk_level == 'Medium':
        marker.add_to(medium_risk_cluster)
    else:
        marker.add_to(low_risk_cluster)
    
    # Add animated circle marker (size based on crime count)
    radius = np.sqrt(crime_count) * 3
    
    circle_html = f"""
    <div class="{animation_class}" style="
        width: {radius * 2}px;
        height: {radius * 2}px;
        background-color: {marker_color};
        border-radius: 50%;
        opacity: 0.4;
        border: 2px solid {marker_color};
    "></div>
    """
    
    folium.CircleMarker(
        location=[area['Latitude'], area['Longitude']],
        radius=radius,
        popup=f"{area['Area']}<br>Crimes: {crime_count}<br>Risk: {risk_percent}%",
        color=marker_color,
        fill=True,
        fill_opacity=0.3,
        weight=2,
        className=animation_class
    ).add_to(crime_map)

# ==================== ADD RISK LEGEND WITH ANIMATION ====================
print("📊 Adding animated risk legend...")

legend_html = f'''
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000; background: rgba(30, 58, 95, 0.95); 
            backdrop-filter: blur(10px); border-radius: 15px; padding: 15px; 
            border: 1px solid rgba(255,255,255,0.2); font-family: Arial; min-width: 180px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
    <div style="font-weight: bold; margin-bottom: 10px; color: #00B4D8; text-align: center;">
        📊 RISK LEVEL LEGEND
    </div>
    <div>
        <div style="margin: 8px 0;">
            <span class="pulse-red" style="display: inline-block; width: 15px; height: 15px; 
                  background-color: #FF0000; border-radius: 50%; margin-right: 8px;"></span>
            <span style="color: white;">Critical Risk (80-100%)</span>
        </div>
        <div style="margin: 8px 0;">
            <span class="pulse-orange" style="display: inline-block; width: 15px; height: 15px; 
                  background-color: #FF6600; border-radius: 50%; margin-right: 8px;"></span>
            <span style="color: white;">High Risk (60-79%)</span>
        </div>
        <div style="margin: 8px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                  background-color: #FFAA00; border-radius: 50%; margin-right: 8px;"></span>
            <span style="color: white;">Medium Risk (40-59%)</span>
        </div>
        <div style="margin: 8px 0;">
            <span class="pulse-green" style="display: inline-block; width: 15px; height: 15px; 
                  background-color: #00C851; border-radius: 50%; margin-right: 8px;"></span>
            <span style="color: white;">Low Risk (0-39%)</span>
        </div>
    </div>
    <hr style="border-color: #444; margin: 10px 0;">
    <div style="font-size: 11px; text-align: center; color: #888;">
        🔴 Pulsing markers = Active high risk areas<br>
        💡 Click markers for detailed info
    </div>
</div>
'''

crime_map.get_root().html.add_child(folium.Element(legend_html))

# ==================== ADD TITLE HEADER ====================
title_html = f'''
<div style="position: fixed; top: 10px; left: 50px; z-index: 1000; background: rgba(0,0,0,0.8); 
            backdrop-filter: blur(10px); border-radius: 15px; padding: 12px 25px; 
            border-left: 5px solid #FF0000; font-family: Arial;">
    <div style="font-size: 18px; font-weight: bold; color: white;">
        🚔 CRIME HOTSPOT MAP - ANIMATED RISK INDICATORS
    </div>
    <div style="font-size: 12px; color: #ccc;">
        Real-time risk assessment | Pulsing markers = Active high-risk zones
    </div>
</div>
'''
crime_map.get_root().html.add_child(folium.Element(title_html))

# ==================== ADD FLOATING STATS PANEL ====================
stats_html = f'''
<div style="position: fixed; top: 80px; right: 20px; z-index: 1000; background: rgba(0,0,0,0.85); 
            backdrop-filter: blur(10px); border-radius: 15px; padding: 12px; 
            border: 1px solid rgba(255,255,255,0.2); font-family: Arial; min-width: 180px;">
    <div style="font-weight: bold; color: #00B4D8; text-align: center;">📊 LIVE STATS</div>
    <hr style="border-color: #444; margin: 8px 0;">
    <div style="font-size: 12px;">
        🔴 Critical: {(area_data['Risk_Level'] == 'Critical').sum()}<br>
        🟠 High: {(area_data['Risk_Level'] == 'High').sum()}<br>
        🟡 Medium: {(area_data['Risk_Level'] == 'Medium').sum()}<br>
        🟢 Low: {(area_data['Risk_Level'] == 'Low').sum()}
    </div>
    <hr style="border-color: #444; margin: 8px 0;">
    <div style="font-size: 10px; color: #888; text-align: center;">
        Updated: {datetime.now().strftime('%H:%M:%S')}
    </div>
</div>
'''
crime_map.get_root().html.add_child(folium.Element(stats_html))

# ==================== ADD LAYER CONTROL ====================
folium.LayerControl().add_to(crime_map)

# ==================== SAVE MAP ====================
print("\n💾 Saving animated map...")
output_path = 'outputs/heatmaps/animated_crime_hotspot.html'
crime_map.save(output_path)
print(f"✅ Animated map saved: {output_path}")

# ==================== OPEN MAP ====================
print("\n🌐 Opening animated map in browser...")
open_map(output_path, serve=False)

print("\n" + "="*60)
print("✅ ANIMATED HOTSPOT MAP GENERATED!")
print("="*60)
print("\n🎬 ANIMATION FEATURES:")
print("   • 🔴 Pulsing red markers = Critical/High risk areas")
print("   • 🟠 Pulsing orange markers = Medium risk areas")
print("   • 🟢 Pulsing green markers = Low risk areas")
print("   • 📊 Risk meter shows exact percentage")
print("   • 💫 Floating/glowing effects on active zones")
print("   • 🔄 Animated risk legend")
print("\n💡 Click on any marker to see:")
print("   • Exact risk percentage")
print("   • Crime statistics")
print("   • Safety recommendations")
print("   • Real-time updates")
print("="*60)