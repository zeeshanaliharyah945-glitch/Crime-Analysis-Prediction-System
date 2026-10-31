# 05_export_for_powerbi.py
# Purpose: Export data for Power BI dashboard

import pandas as pd
import numpy as np
import pickle
import os

print("="*60)
print("STEP 5: EXPORTING DATA FOR POWER BI")
print("="*60)

# Create output directory
os.makedirs('outputs/powerbi', exist_ok=True)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/processed_crime_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"✅ Loaded {len(df)} records")

# ==================== 1. FACT CRIME TABLE ====================
print("\n📊 Creating fact_crime table...")

fact_crime = df[['Crime_ID', 'Date', 'Time', 'Hour', 'Latitude', 'Longitude',
                 'Crime_Type', 'Police_Station', 'Area', 'Severity', 'Risk_Score']].copy()
fact_crime.to_csv('outputs/powerbi/fact_crime.csv', index=False)
print(f"   ✅ Saved: {len(fact_crime)} records")

# ==================== 2. DIM DATE TABLE ====================
print("\n📅 Creating dim_date table...")

date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
dim_date = pd.DataFrame({
    'Date': date_range,
    'Year': date_range.year,
    'Month': date_range.month,
    'Month_Name': date_range.strftime('%B'),
    'Quarter': date_range.quarter,
    'Day': date_range.day,
    'DayOfWeek': date_range.dayofweek,
    'Weekday_Name': date_range.strftime('%A'),
    'IsWeekend': (date_range.dayofweek >= 5).astype(int)
})
dim_date.to_csv('outputs/powerbi/dim_date.csv', index=False)
print(f"   ✅ Saved: {len(dim_date)} records")

# ==================== 3. DIM CRIME TYPE ====================
print("\n🏷️ Creating dim_crime_type table...")

severity_map = {'Murder': 5, 'Robbery': 4, 'Assault': 3, 'Theft': 2}
dim_crime = pd.DataFrame({
    'Crime_Type': df['Crime_Type'].unique(),
    'Severity_Level': [severity_map[c] for c in df['Crime_Type'].unique()]
})
dim_crime.to_csv('outputs/powerbi/dim_crime_type.csv', index=False)
print(f"   ✅ Saved: {len(dim_crime)} records")

# ==================== 4. DIM AREA ====================
print("\n🗺️ Creating dim_area table...")

dim_area = df.groupby('Area').agg({
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Crime_Count': 'first',
    'Police_Station': 'first'
}).reset_index()
dim_area.to_csv('outputs/powerbi/dim_area.csv', index=False)
print(f"   ✅ Saved: {len(dim_area)} records")

# ==================== 5. DAILY STATS ====================
print("\n📈 Creating daily_stats table...")

daily_stats = df.groupby(df['Date'].dt.date).agg({
    'Crime_ID': 'count',
    'Risk_Score': 'mean'
}).reset_index()
daily_stats.columns = ['Date', 'Daily_Crimes', 'Avg_Risk_Score']
daily_stats.to_csv('outputs/powerbi/daily_stats.csv', index=False)
print(f"   ✅ Saved: {len(daily_stats)} records")

# ==================== 6. HOURLY STATS ====================
print("\n⏰ Creating hourly_stats table...")

hourly_stats = df.groupby('Hour').agg({
    'Crime_ID': 'count',
    'Risk_Score': 'mean'
}).reset_index()
hourly_stats.columns = ['Hour', 'Crime_Count', 'Avg_Risk_Score']
hourly_stats.to_csv('outputs/powerbi/hourly_stats.csv', index=False)
print(f"   ✅ Saved: {len(hourly_stats)} records")

# ==================== 7. PREDICTIONS ====================
print("\n🔮 Creating predictions table...")

with open('outputs/models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('outputs/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_columns = ['Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Severity',
                   'Crime_Type_Encoded', 'Police_Station_Encoded', 'Area_Encoded',
                   'TimePeriod_Encoded', 'Station_Crime_Count']

area_pred = df.groupby('Area').agg({
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Crime_Count': 'first',
    'Station_Crime_Count': 'first',
    'Police_Station_Encoded': 'first',
    'Area_Encoded': 'first'
}).reset_index()

area_pred['Year'] = 2024
area_pred['Month'] = 12
area_pred['Day'] = 25
area_pred['DayOfWeek'] = 2
area_pred['Hour'] = 20
area_pred['Severity'] = 3
area_pred['TimePeriod_Encoded'] = 3
area_pred['Crime_Type_Encoded'] = 1

X_pred = area_pred[feature_columns]
X_pred_scaled = scaler.transform(X_pred)
area_pred['Risk_Probability'] = model.predict_proba(X_pred_scaled)[:, 1]
area_pred['Risk_Level'] = pd.cut(area_pred['Risk_Probability'], 
                                  bins=[0, 0.4, 0.6, 0.8, 1],
                                  labels=['Low', 'Medium', 'High', 'Critical'])

area_pred[['Area', 'Latitude', 'Longitude', 'Risk_Probability', 'Risk_Level']].to_csv(
    'outputs/powerbi/predictions.csv', index=False)
print(f"   ✅ Saved: {len(area_pred)} predictions")

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("✅ POWER BI EXPORT COMPLETE!")
print("="*60)
print("\nFiles saved in 'outputs/powerbi/' directory:")
print("   - fact_crime.csv")
print("   - dim_date.csv")
print("   - dim_crime_type.csv")
print("   - dim_area.csv")
print("   - daily_stats.csv")
print("   - hourly_stats.csv")
print("   - predictions.csv")
print("\n📊 TO CREATE POWER BI DASHBOARD:")
print("   1. Open Power BI Desktop")
print("   2. Click 'Get Data' → 'Text/CSV'")
print("   3. Load all CSV files from 'outputs/powerbi/'")
print("   4. Create relationships between tables")
print("   5. Build your visualizations!")