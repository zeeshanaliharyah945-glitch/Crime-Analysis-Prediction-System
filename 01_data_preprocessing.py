# 01_data_preprocessing.py
# Purpose: Clean and prepare crime dataset for analysis

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 1: DATA PREPROCESSING")
print("="*60)

# ==================== LOAD DATA ====================
print("\n📂 Loading dataset...")
df = pd.read_csv('data/larkana_crime_dataset_5000.csv')
print(f"✅ Loaded {len(df)} records")
print(f"   Columns: {df.columns.tolist()}")

# ==================== DATA CLEANING ====================
print("\n🧹 Cleaning data...")

# Remove any duplicate rows
initial_rows = len(df)
df = df.drop_duplicates()
print(f"   Removed {initial_rows - len(df)} duplicate rows")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ==================== FEATURE ENGINEERING ====================
print("\n🔧 Creating new features...")

# Time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Weekday'] = df['Date'].dt.day_name()
df['Quarter'] = df['Date'].dt.quarter

# Hour extraction
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour

# Time period categorization
def get_time_period(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['TimePeriod'] = df['Hour'].apply(get_time_period)

# Severity mapping
severity_map = {'Murder': 5, 'Robbery': 4, 'Assault': 3, 'Theft': 2}
df['Severity'] = df['Crime_Type'].map(severity_map)

# Risk score
df['Risk_Score'] = df['Severity'] * (1 + (df['Hour'] >= 20).astype(int) * 0.5)

print(f"   Added features: Year, Month, Day, DayOfWeek, Weekday, Quarter, Hour, TimePeriod, Severity, Risk_Score")

# ==================== ENCODE CATEGORICAL VARIABLES ====================
print("\n🏷️ Encoding categorical variables...")

from sklearn.preprocessing import LabelEncoder

encoders = {}
categorical_cols = ['Crime_Type', 'Police_Station', 'Area', 'TimePeriod']

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"   Encoded {col}: {len(le.classes_)} unique values")

# ==================== CREATE AGGREGATED FEATURES ====================
print("\n📊 Creating aggregated features...")

# Crime count by area
area_counts = df.groupby('Area').size().reset_index(name='Crime_Count')
df = df.merge(area_counts, on='Area', how='left')

# Crime count by police station
station_counts = df.groupby('Police_Station').size().reset_index(name='Station_Crime_Count')
df = df.merge(station_counts, on='Police_Station', how='left')

print(f"   Added Crime_Count (by area) and Station_Crime_Count")

# ==================== CREATE TARGET VARIABLE ====================
print("\n🎯 Creating target variable for prediction...")

median_crime_count = df['Crime_Count'].median()
df['High_Risk_Area'] = (df['Crime_Count'] > median_crime_count).astype(int)
print(f"   Median crime count: {median_crime_count}")
print(f"   High risk areas (1): {(df['High_Risk_Area'] == 1).sum()}")
print(f"   Low risk areas (0): {(df['High_Risk_Area'] == 0).sum()}")

# ==================== SAVE PROCESSED DATA ====================
print("\n💾 Saving processed data...")

df.to_csv('data/processed_crime_data.csv', index=False)
print(f"✅ Saved to 'data/processed_crime_data.csv'")

# Save encoders
with open('outputs/models/label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print(f"✅ Saved encoders to 'outputs/models/label_encoders.pkl'")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("✅ DATA PREPROCESSING COMPLETE!")
print("="*60)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print("\nSample of processed data:")
print(df[['Crime_ID', 'Crime_Type', 'Area', 'Hour', 'Severity', 'Risk_Score', 'High_Risk_Area']].head())