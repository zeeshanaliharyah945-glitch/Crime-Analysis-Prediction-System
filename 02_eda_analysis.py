# 02_eda_analysis.py (FIXED VERSION - No special characters in reports)
# Purpose: Visualize crime patterns and trends

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking charts
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Load processed data
print("\n📂 Loading processed data...")
df = pd.read_csv('data/processed_crime_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"✅ Loaded {len(df)} records")

# Create output directory if not exists
import os
os.makedirs('outputs/charts', exist_ok=True)

# ==================== 1. CRIME TYPE DISTRIBUTION ====================
print("\n📊 Creating Crime Type Distribution Chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

crime_counts = df['Crime_Type'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Bar chart
axes[0].bar(crime_counts.index, crime_counts.values, color=colors, edgecolor='black')
axes[0].set_title('Crime Type Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Crime Type')
axes[0].set_ylabel('Number of Incidents')
for bar, count in zip(axes[0].patches, crime_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 str(count), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(crime_counts.values, labels=crime_counts.index, autopct='%1.1f%%', 
            colors=colors, explode=[0.05, 0.05, 0.05, 0.05])
axes[1].set_title('Crime Type Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/charts/01_crime_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/charts/01_crime_distribution.png")

# ==================== 2. TIME-BASED ANALYSIS ====================
print("\n⏰ Creating Time-Based Analysis Charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Hourly distribution
hourly_crimes = df.groupby('Hour').size()
axes[0, 0].bar(hourly_crimes.index, hourly_crimes.values, color='#45B7D1', edgecolor='black')
axes[0, 0].set_title('Crimes by Hour of Day', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Hour (0-23)')
axes[0, 0].set_ylabel('Number of Crimes')
axes[0, 0].set_xticks(range(0, 24, 2))

# Time period distribution
period_counts = df['TimePeriod'].value_counts()
period_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
axes[0, 1].pie(period_counts.values, labels=period_counts.index, autopct='%1.1f%%', 
               colors=period_colors, explode=[0.05]*len(period_counts))
axes[0, 1].set_title('Crimes by Time Period', fontsize=14, fontweight='bold')

# Day of week distribution
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['Weekday'] = pd.Categorical(df['Weekday'], categories=day_order, ordered=True)
weekday_counts = df.groupby('Weekday', observed=True).size()
axes[1, 0].bar(weekday_counts.index, weekday_counts.values, color='#96CEB4', edgecolor='black')
axes[1, 0].set_title('Crimes by Day of Week', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Number of Crimes')
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

# Monthly trend
monthly_crimes = df.groupby(df['Date'].dt.to_period('M')).size()
axes[1, 1].plot(monthly_crimes.index.astype(str), monthly_crimes.values, 
                marker='o', color='#FF6B6B', linewidth=2, markersize=6)
axes[1, 1].set_title('Monthly Crime Trends', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Number of Crimes')
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/02_time_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/charts/02_time_analysis.png")

# ==================== 3. GEOGRAPHICAL ANALYSIS ====================
print("\n🗺️ Creating Geographical Analysis Charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Area distribution
area_counts = df['Area'].value_counts().head(10)
axes[0].barh(range(len(area_counts)), area_counts.values, color='#4ECDC4', edgecolor='black')
axes[0].set_yticks(range(len(area_counts)))
axes[0].set_yticklabels(area_counts.index)
axes[0].set_title('Top 10 Areas by Crime Count', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Crimes')

# Police station distribution
station_counts = df['Police_Station'].value_counts()
axes[1].bar(station_counts.index, station_counts.values, color='#FF6B6B', edgecolor='black')
axes[1].set_title('Crimes by Police Station', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Police Station')
axes[1].set_ylabel('Number of Crimes')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('outputs/charts/03_geographical_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/charts/03_geographical_distribution.png")

# ==================== 4. SEVERITY ANALYSIS ====================
print("\n⚠️ Creating Severity Analysis Charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Severity distribution
severity_counts = df['Severity'].value_counts().sort_index()
axes[0].bar([f"Level {s}" for s in severity_counts.index], severity_counts.values, 
            color='#45B7D1', edgecolor='black')
axes[0].set_title('Crime Severity Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Severity Level')
axes[0].set_ylabel('Number of Crimes')
for bar, count in zip(axes[0].patches, severity_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 str(count), ha='center', fontweight='bold')

# Risk score by crime type
risk_by_crime = df.groupby('Crime_Type')['Risk_Score'].mean().sort_values()
axes[1].barh(risk_by_crime.index, risk_by_crime.values, color='#96CEB4', edgecolor='black')
axes[1].set_title('Average Risk Score by Crime Type', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Risk Score')

plt.tight_layout()
plt.savefig('outputs/charts/04_severity_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/charts/04_severity_analysis.png")

# ==================== 5. CORRELATION HEATMAP ====================
print("\n📊 Creating Correlation Heatmap...")

numeric_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Severity', 'Risk_Score']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/charts/05_correlation_heatmap.png")

# ==================== GENERATE SUMMARY REPORT (FIXED - NO SPECIAL CHARACTERS) ====================
print("\n📄 Generating Summary Report...")

# Get top crime hour
peak_hour = df.groupby('Hour').size().idxmax()
peak_hour_count = df.groupby('Hour').size().max()

# Get peak crime day
peak_day = df.groupby('Weekday', observed=True).size().idxmax()

# Get most active time period
active_period = df['TimePeriod'].mode()[0]

# Get highest crime area
highest_crime_area = df['Area'].value_counts().index[0]
highest_crime_count = df['Area'].value_counts().iloc[0]

# Get highest crime police station
highest_crime_station = df['Police_Station'].value_counts().index[0]

report = f"""
================================================================================
                    CRIME ANALYSIS SUMMARY REPORT
================================================================================

DATASET OVERVIEW
--------------------------------------------------------------------------------
Total Records: {len(df):,}
Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}
Number of Areas: {df['Area'].nunique()}
Number of Police Stations: {df['Police_Station'].nunique()}

CRIME STATISTICS
--------------------------------------------------------------------------------
Total Theft Cases: {len(df[df['Crime_Type'] == 'Theft']):,}
Total Robbery Cases: {len(df[df['Crime_Type'] == 'Robbery']):,}
Total Assault Cases: {len(df[df['Crime_Type'] == 'Assault']):,}
Total Murder Cases: {len(df[df['Crime_Type'] == 'Murder']):,}

Most Common Crime: {df['Crime_Type'].mode()[0]}
Least Common Crime: {df['Crime_Type'].value_counts().index[-1]}

TIME PATTERNS
--------------------------------------------------------------------------------
Peak Crime Hour: {peak_hour}:00 ({peak_hour_count} crimes)
Peak Crime Day: {peak_day}
Most Active Time Period: {active_period}

GEOGRAPHICAL INSIGHTS
--------------------------------------------------------------------------------
Highest Crime Area: {highest_crime_area} ({highest_crime_count} crimes)
Highest Crime Police Station: {highest_crime_station}

RISK ANALYSIS
--------------------------------------------------------------------------------
Average Risk Score: {df['Risk_Score'].mean():.2f}
Highest Risk Crime Type: {df.groupby('Crime_Type')['Risk_Score'].mean().idxmax()}

================================================================================
"""

print(report)

# Save with utf-8 encoding
with open('outputs/reports/eda_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("   ✅ Saved: outputs/reports/eda_summary_report.txt")

print("\n" + "="*60)
print("✅ EXPLORATORY DATA ANALYSIS COMPLETE!")
print("="*60)
print("\nCharts saved in: outputs/charts/")
print("Report saved in: outputs/reports/")