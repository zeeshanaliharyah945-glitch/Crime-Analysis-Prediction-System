# run_all.py
# Purpose: Run all project components sequentially

import subprocess
import sys
import os

def run_script(script_name, description):
    print("\n" + "="*60)
    print(f"RUNNING: {description}")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], capture_output=False, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}")
        print(f"Error code: {e.returncode}")
        return False

def main():
    print("="*60)
    print("CRIME ANALYSIS & PREDICTION SYSTEM")
    print("Running Complete Pipeline")
    print("="*60)
    
    # Create directories
    os.makedirs('outputs/charts', exist_ok=True)
    os.makedirs('outputs/heatmaps', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/powerbi', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists('data/larkana_crime_dataset_5000.csv'):
        print("\n⚠️ Dataset not found! Running create_dataset.py first...")
        if not run_script('create_dataset.py', 'Dataset Creation'):
            print("Failed to create dataset")
            return
    
    # Run scripts in order
    scripts = [
        ('01_data_preprocessing.py', 'Data Preprocessing'),
        ('02_eda_analysis.py', 'Exploratory Data Analysis'),
        ('03_machine_learning.py', 'Machine Learning Training'),
        ('04_hotspot_mapping.py', 'Hotspot Mapping'),
        ('05_export_for_powerbi.py', 'Power BI Export'),
        ('06_train_ai_models.py', 'AI Features (risk, forecast, hotspots, grid, anomaly, fairness, PDF)')
    ]
    
    for script, description in scripts:
        if not run_script(script, description):
            print(f"\n❌ Pipeline stopped at {description}")
            return
    
    print("\n" + "="*60)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📁 OUTPUT FILES GENERATED:")
    print("   📊 outputs/charts/ - All visualization charts")
    print("   🗺️ outputs/heatmaps/ - Interactive HTML maps")
    print("   🤖 outputs/models/ - Trained ML models")
    print("   📄 outputs/reports/ - Analysis reports")
    print("   📈 outputs/powerbi/ - Data for Power BI dashboard")
    print("   🧠 outputs/ai/ - AI features results (forecast, hotspots, grid, alerts, fairness)")
    print("\n🚀 NEXT STEPS:")
    print("   1. Open 'outputs/heatmaps/crime_hotspot_map.html' in browser")
    print("   2. Check 'outputs/reports/' for analysis reports")
    print("   3. Import CSV files from 'outputs/powerbi/' into Power BI")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()