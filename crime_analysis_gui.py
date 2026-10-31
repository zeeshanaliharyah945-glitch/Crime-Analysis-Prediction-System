# crime_analysis_gui.py - COMPLETE WITH ANIMATED MAP INTEGRATION
# Features: Animated hotspot map inside GUI | No separate command needed

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import webbrowser
import os
from datetime import datetime
import warnings
from fpdf import FPDF
import tempfile
import time
import threading
import json
import folium
from folium.plugins import HeatMap, MarkerCluster
from map_utils import create_map, open_map
import random

warnings.filterwarnings('ignore')

# Set appearance mode and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AnimatedButton(ctk.CTkButton):
    """Custom button with hover animation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def on_enter(self, e):
        self.configure(fg_color=self.hover_color)
    
    def on_leave(self, e):
        self.configure(fg_color=self.fg_color)

class CrimeAnalysisApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Crime Analysis & Prediction System")
        self.root.geometry("1500x850")
        self.root.minsize(1300, 750)
        
        # Initialize data variables
        self.df = None
        self.model = None
        self.scaler = None
        self.encoders = None
        self.data_loaded = False
        
        # INCREASED FONT SIZES
        self.fonts = {
            'title': ctk.CTkFont(size=22, weight="bold"),
            'header': ctk.CTkFont(size=26, weight="bold"),
            'subheader': ctk.CTkFont(size=18, weight="bold"),
            'body': ctk.CTkFont(size=14),
            'body_bold': ctk.CTkFont(size=14, weight="bold"),
            'small': ctk.CTkFont(size=12),
            'kpi_value': ctk.CTkFont(size=32, weight="bold"),
            'kpi_label': ctk.CTkFont(size=14),
            'button': ctk.CTkFont(size=15, weight="bold")
        }
        
        # UPGRADED COLORS
        self.colors = {
            'primary': '#2B5B84',
            'secondary': '#1E3A5F',
            'accent': '#00B4D8',
            'danger': '#E63946',
            'success': '#2A9D8F',
            'warning': '#F4A261',
            'dark': '#1a1a2e',
            'light': '#f0f0f0',
            'theft': '#4ECDC4',
            'robbery': '#FF6B6B',
            'assault': '#F4A261',
            'murder': '#E63946',
            'high_risk': '#FF4444',
            'medium_risk': '#FFA500',
            'low_risk': '#00C851'
        }
        
        # Create main container
        self.create_widgets()
        
        # Load data
        self.load_initial_data()
        
        self.root.mainloop()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # ==================== SIDEBAR ====================
        self.sidebar = ctk.CTkFrame(self.root, width=260, corner_radius=0, fg_color=self.colors['secondary'])
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        
        # Logo/Title
        title_label = ctk.CTkLabel(self.sidebar, text="🚔 CRIME ANALYSIS\n& PREDICTION SYSTEM", 
                                   font=self.fonts['title'], text_color="white")
        title_label.pack(pady=35)
        
        # Navigation Buttons
        nav_buttons = [
            ("🏠 DASHBOARD", self.show_dashboard),
            ("📊 CRIME STATISTICS", self.show_statistics),
            ("🗺️ ANIMATED CRIME MAP", self.show_animated_map),
            ("🤖 PREDICT RISK", self.show_prediction),
            ("📈 ANALYTICS", self.show_analytics),
            ("🧠 AI FEATURES", self.open_ai_features),
            ("📄 PDF REPORTS", self.show_reports),
            ("❓ HELP", self.show_help)
        ]
        
        for text, command in nav_buttons:
            btn = AnimatedButton(self.sidebar, text=text, command=command, 
                                font=self.fonts['button'], height=48,
                                fg_color="transparent", hover_color=self.colors['primary'],
                                corner_radius=10)
            btn.pack(pady=8, padx=20, fill="x")
        
        # Status bar
        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Loading...", 
                                         font=self.fonts['small'], text_color="gray")
        self.status_label.pack(side="bottom", pady=20)
        
        # ==================== MAIN CONTENT AREA ====================
        self.main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['dark'])
        self.main_frame.pack(side="right", fill="both", expand=True)
        
        # Header
        self.header_frame = ctk.CTkFrame(self.main_frame, height=70, fg_color=self.colors['primary'])
        self.header_frame.pack(fill="x", padx=0, pady=0)
        
        self.header_label = ctk.CTkLabel(self.header_frame, text="DASHBOARD", 
                                         font=self.fonts['header'], text_color="white")
        self.header_label.pack(side="left", padx=35, pady=18)
        
        # Content area
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color=self.colors['dark'])
        self.content_frame.pack(fill="both", expand=True, padx=25, pady=25)
        
        # Loading message
        self.loading_label = ctk.CTkLabel(self.content_frame, text="📊 LOADING DATA AND MODELS...\nPlease wait", 
                                          font=self.fonts['subheader'], text_color="white")
        self.loading_label.pack(expand=True)
    
    def clear_content(self):
        """Clear the content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def show_glass_message(self, title, message, color):
        """Show glass effect message dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Message")
        dialog.geometry("400x200")
        dialog.attributes('-topmost', True)
        dialog.configure(fg_color=self.colors['secondary'])
        
        ctk.CTkLabel(dialog, text=title, font=self.fonts['subheader'], text_color=color).pack(pady=20)
        ctk.CTkLabel(dialog, text=message, font=self.fonts['body'], wraplength=350).pack(pady=10)
        
        def close():
            for i in range(10, 0, -1):
                dialog.attributes('-alpha', i/10)
                time.sleep(0.03)
            dialog.destroy()
        
        btn = AnimatedButton(dialog, text="OK", command=close, font=self.fonts['button'],
                            fg_color=color, hover_color=self.colors['primary'])
        btn.pack(pady=20)
        
        dialog.attributes('-alpha', 0)
        for i in range(1, 11):
            dialog.attributes('-alpha', i/10)
            time.sleep(0.02)
    
    def load_initial_data(self):
        """Load data and models"""
        try:
            self.status_label.configure(text="Status: Loading data...")
            
            if os.path.exists('data/processed_crime_data.csv'):
                self.df = pd.read_csv('data/processed_crime_data.csv')
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                print(f"Loaded {len(self.df)} records")
            else:
                self.status_label.configure(text="Status: Error - Run preprocessing first")
                return
            
            if os.path.exists('outputs/models/random_forest_model.pkl'):
                with open('outputs/models/random_forest_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                print("Model loaded")
            
            if os.path.exists('outputs/models/scaler.pkl'):
                with open('outputs/models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if os.path.exists('outputs/models/label_encoders.pkl'):
                with open('outputs/models/label_encoders.pkl', 'rb') as f:
                    self.encoders = pickle.load(f)
            
            self.data_loaded = True
            self.status_label.configure(text="Status: Ready")
            
            if hasattr(self, 'loading_label'):
                self.loading_label.destroy()
            
            self.show_dashboard()
            
        except Exception as e:
            print(f"Error: {e}")
            self.status_label.configure(text=f"Status: Error")
            if hasattr(self, 'loading_label'):
                self.loading_label.destroy()
            self.show_error_page(str(e))
    
    def show_error_page(self, error_msg):
        """Show error page"""
        error_frame = ctk.CTkFrame(self.content_frame, fg_color=self.colors['secondary'], corner_radius=15)
        error_frame.pack(expand=True, padx=50, pady=50)
        
        ctk.CTkLabel(error_frame, text="⚠️ ERROR", font=self.fonts['header'], 
                    text_color=self.colors['danger']).pack(pady=20)
        ctk.CTkLabel(error_frame, text=f"Data not loaded properly!\n\n{error_msg}\n\nPlease run: python run_all.py", 
                    font=self.fonts['body']).pack(pady=20)
    
    # ==================== 1. DASHBOARD ====================
    def show_dashboard(self):
        if not self.data_loaded:
            messagebox.showwarning("Not Ready", "Please wait for data to load...")
            return
        
        self.clear_content()
        self.header_label.configure(text="📊 DASHBOARD")
        
        # KPI Cards Frame
        cards_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        cards_frame.pack(fill="x", pady=15)
        
        total_crimes = len(self.df)
        avg_risk = self.df['Risk_Score'].mean()
        high_risk_areas = self.df[self.df['High_Risk_Area'] == 1]['Area'].nunique()
        most_common_crime = self.df['Crime_Type'].mode()[0]
        
        kpis = [
            ("TOTAL CRIMES", f"{total_crimes:,}", self.colors['accent']),
            ("AVG RISK SCORE", f"{avg_risk:.1f}", self.colors['warning']),
            ("HIGH RISK AREAS", str(high_risk_areas), self.colors['danger']),
            ("MOST COMMON", most_common_crime, self.colors['success'])
        ]
        
        for title, value, color in kpis:
            card = ctk.CTkFrame(cards_frame, fg_color=self.colors['secondary'], corner_radius=15)
            card.pack(side="left", padx=12, pady=12, fill="both", expand=True)
            
            title_label = ctk.CTkLabel(card, text=title, font=self.fonts['kpi_label'], text_color="gray")
            title_label.pack(pady=(18, 8))
            
            value_label = ctk.CTkLabel(card, text=value, font=self.fonts['kpi_value'], text_color=color)
            value_label.pack(pady=(0, 18))
        
        # Charts Frame
        charts_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        charts_frame.pack(fill="both", expand=True, pady=20)
        
        # Crime Type Distribution Chart
        chart1_frame = ctk.CTkFrame(charts_frame, fg_color=self.colors['secondary'], corner_radius=15)
        chart1_frame.pack(side="left", padx=12, fill="both", expand=True)
        
        crime_counts = self.df['Crime_Type'].value_counts()
        
        fig1 = Figure(figsize=(5.5, 4.5), dpi=100, facecolor=self.colors['secondary'])
        ax1 = fig1.add_subplot(111)
        colors = [self.colors['theft'], self.colors['robbery'], self.colors['assault'], self.colors['murder']]
        ax1.pie(crime_counts.values, labels=crime_counts.index, autopct='%1.1f%%', colors=colors,
                textprops={'fontsize': 12, 'color': 'white'})
        ax1.set_title('Crime Type Distribution', color='white', fontsize=14, fontweight='bold')
        ax1.set_facecolor(self.colors['secondary'])
        
        canvas1 = FigureCanvasTkAgg(fig1, master=chart1_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(pady=15, padx=15)
        
        # Time Series Chart
        chart2_frame = ctk.CTkFrame(charts_frame, fg_color=self.colors['secondary'], corner_radius=15)
        chart2_frame.pack(side="right", padx=12, fill="both", expand=True)
        
        monthly_crimes = self.df.groupby(self.df['Date'].dt.to_period('M')).size()
        
        fig2 = Figure(figsize=(5.5, 4.5), dpi=100, facecolor=self.colors['secondary'])
        ax2 = fig2.add_subplot(111)
        ax2.plot(monthly_crimes.index.astype(str), monthly_crimes.values, marker='o', 
                color=self.colors['accent'], linewidth=2.5, markersize=7)
        ax2.set_title('Monthly Crime Trends', color='white', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month', color='white', fontsize=11)
        ax2.set_ylabel('Number of Crimes', color='white', fontsize=11)
        ax2.tick_params(colors='white', labelsize=10)
        ax2.set_facecolor(self.colors['secondary'])
        ax2.grid(True, alpha=0.2)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=chart2_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(pady=15, padx=15)
    
    # ==================== 2. CRIME STATISTICS ====================
    def show_statistics(self):
        if not self.data_loaded:
            messagebox.showwarning("Not Ready", "Please wait...")
            return
        
        self.clear_content()
        self.header_label.configure(text="📊 CRIME STATISTICS")
        
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill="both", expand=True)
        
        # Tab 1 - Summary Statistics
        summary_frame = ctk.CTkFrame(notebook, fg_color=self.colors['dark'])
        notebook.add(summary_frame, text="📋 SUMMARY STATISTICS")
        
        columns = ("Metric", "Value")
        tree = ttk.Treeview(summary_frame, columns=columns, show="headings", height=18)
        tree.heading("Metric", text="METRIC")
        tree.heading("Value", text="VALUE")
        tree.pack(padx=20, pady=20, fill="both", expand=True)
        
        total_records = len(self.df)
        theft_count = len(self.df[self.df['Crime_Type'] == 'Theft'])
        robbery_count = len(self.df[self.df['Crime_Type'] == 'Robbery'])
        assault_count = len(self.df[self.df['Crime_Type'] == 'Assault'])
        murder_count = len(self.df[self.df['Crime_Type'] == 'Murder'])
        peak_hour = self.df.groupby('Hour').size().idxmax()
        highest_area = self.df['Area'].value_counts().index[0]
        
        stats = [
            ("📊 DATASET OVERVIEW", ""),
            ("Total Records", f"{total_records:,}"),
            ("Number of Areas", str(self.df['Area'].nunique())),
            ("Number of Police Stations", str(self.df['Police_Station'].nunique())),
            ("", ""),
            ("🔴 CRIME TOTALS", ""),
            (f"Theft", f"{theft_count:,} ({(theft_count/total_records)*100:.1f}%)"),
            (f"Robbery", f"{robbery_count:,} ({(robbery_count/total_records)*100:.1f}%)"),
            (f"Assault", f"{assault_count:,} ({(assault_count/total_records)*100:.1f}%)"),
            (f"Murder", f"{murder_count:,} ({(murder_count/total_records)*100:.1f}%)"),
            ("", ""),
            ("⏰ TIME PATTERNS", ""),
            (f"Peak Crime Hour", f"{peak_hour}:00"),
            ("Peak Crime Day", str(self.df.groupby('Weekday', observed=True).size().idxmax())),
            ("", ""),
            ("🗺️ GEOGRAPHICAL", ""),
            (f"Highest Crime Area", f"{highest_area} ({self.df['Area'].value_counts().iloc[0]} crimes)"),
        ]
        
        for metric, value in stats:
            tree.insert("", "end", values=(metric, value))
        
        # Tab 2 - Area-wise Stats
        area_frame = ctk.CTkFrame(notebook, fg_color=self.colors['dark'])
        notebook.add(area_frame, text="📍 AREA-WISE STATISTICS")
        
        area_stats = self.df.groupby('Area').size().sort_values(ascending=False)
        max_count = area_stats.max()
        
        stats_canvas = tk.Canvas(area_frame, bg=self.colors['dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(area_frame, orient="vertical", command=stats_canvas.yview)
        scrollable_frame = ctk.CTkFrame(stats_canvas, fg_color="transparent")
        
        scrollable_frame.bind("<Configure>", lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all")))
        stats_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        stats_canvas.configure(yscrollcommand=scrollbar.set)
        
        stats_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        for idx, (area, count) in enumerate(area_stats.items()):
            bar_width = (count / max_count) * 300
            area_row = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            area_row.pack(fill="x", pady=8)
            
            area_name = ctk.CTkLabel(area_row, text=f"{idx+1}. {area}", font=self.fonts['body'], width=200, anchor="w")
            area_name.pack(side="left", padx=10)
            
            if count > max_count * 0.7:
                bar_color = self.colors['danger']
            elif count > max_count * 0.4:
                bar_color = self.colors['warning']
            else:
                bar_color = self.colors['success']
            
            bar_canvas = tk.Canvas(area_row, width=bar_width, height=25, bg=bar_color, highlightthickness=0)
            bar_canvas.pack(side="left", padx=10)
            
            count_label = ctk.CTkLabel(area_row, text=str(count), font=self.fonts['body_bold'], width=50)
            count_label.pack(side="left")
    
    # ==================== 3. ANIMATED MAP (INTEGRATED - NO SEPARATE COMMAND) ====================
    def generate_animated_map(self):
        """Generate animated hotspot map with risk animations"""
        
        # Show generation dialog
        gen_dialog = ctk.CTkToplevel(self.root)
        gen_dialog.title("Generating Animated Map")
        gen_dialog.geometry("450x250")
        gen_dialog.attributes('-topmost', True)
        gen_dialog.configure(fg_color=self.colors['secondary'])
        
        ctk.CTkLabel(gen_dialog, text="🗺️ GENERATING ANIMATED CRIME MAP", 
                    font=self.fonts['subheader'], text_color=self.colors['accent']).pack(pady=20)
        
        progress = ctk.CTkProgressBar(gen_dialog, width=350, height=12)
        progress.pack(pady=15)
        
        status_label = ctk.CTkLabel(gen_dialog, text="Calculating risk predictions...", font=self.fonts['body'])
        status_label.pack(pady=10)
        
        # Step 1: Calculate risk predictions
        status_label.configure(text="Step 1/4: Calculating risk predictions...")
        progress.set(0.25)
        gen_dialog.update()
        time.sleep(0.5)
        
        # Prepare area data
        area_data = self.df.groupby('Area').agg({
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Crime_Count': 'first',
            'Station_Crime_Count': 'first',
            'Police_Station_Encoded': 'first',
            'Area_Encoded': 'first'
        }).reset_index()
        
        area_data['Year'] = 2024
        area_data['Month'] = 12
        area_data['Day'] = 25
        area_data['DayOfWeek'] = 2
        area_data['Hour'] = 20
        area_data['Severity'] = 3
        area_data['TimePeriod_Encoded'] = 3
        area_data['Crime_Type_Encoded'] = 1
        
        feature_columns = ['Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Severity',
                           'Crime_Type_Encoded', 'Police_Station_Encoded', 'Area_Encoded',
                           'TimePeriod_Encoded', 'Station_Crime_Count']
        
        X_pred = area_data[feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred)
        probabilities = self.model.predict_proba(X_pred_scaled)[:, 1]
        
        area_data['Risk_Probability'] = probabilities
        area_data['Risk_Level'] = area_data['Risk_Probability'].apply(
            lambda x: 'Critical' if x > 0.8 else 'High' if x > 0.6 else 'Medium' if x > 0.4 else 'Low'
        )
        
        status_label.configure(text="Step 2/4: Creating map with animations...")
        progress.set(0.5)
        gen_dialog.update()
        time.sleep(0.5)
        
        # Create map
        center_lat = self.df['Latitude'].mean()
        center_lon = self.df['Longitude'].mean()
        
        crime_map = create_map(
            location=[center_lat, center_lon],
            zoom_start=13,
            theme='dark',
            control_scale=True
        )
        
        # Animation CSS
        animation_css = """
        <style>
        @keyframes pulse-red {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,0,0,0.7); opacity: 0.7; }
            70% { transform: scale(1); box-shadow: 0 0 0 20px rgba(255,0,0,0); opacity: 1; }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,0,0,0); opacity: 0.7; }
        }
        @keyframes pulse-orange {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,165,0,0.7); opacity: 0.7; }
            70% { transform: scale(1); box-shadow: 0 0 0 15px rgba(255,165,0,0); opacity: 1; }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,165,0,0); opacity: 0.7; }
        }
        @keyframes pulse-green {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0,200,81,0.7); opacity: 0.7; }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0,200,81,0); opacity: 1; }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0,200,81,0); opacity: 0.7; }
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        .pulse-red { animation: pulse-red 1.5s infinite; border-radius: 50%; }
        .pulse-orange { animation: pulse-orange 2s infinite; border-radius: 50%; }
        .pulse-green { animation: pulse-green 2.5s infinite; border-radius: 50%; }
        .float-effect { animation: float 3s ease-in-out infinite; }
        .glass-card { background: rgba(30,58,95,0.95); backdrop-filter: blur(10px); border-radius: 15px; padding: 15px; border: 1px solid rgba(255,255,255,0.2); }
        .risk-critical { background-color: #FF0000; color: white; padding: 3px 8px; border-radius: 20px; display: inline-block; }
        .risk-high { background-color: #FF6600; color: white; padding: 3px 8px; border-radius: 20px; display: inline-block; }
        .risk-low { background-color: #00C851; color: white; padding: 3px 8px; border-radius: 20px; display: inline-block; }
        </style>
        """
        crime_map.get_root().header.add_child(folium.Element(animation_css))
        
        # Add heatmap
        heat_data = [[row['Latitude'], row['Longitude'], row['Crime_Count']] for _, row in area_data.iterrows()]
        HeatMap(heat_data, min_opacity=0.4, radius=20, blur=15,
                gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}).add_to(crime_map)
        
        status_label.configure(text="Step 3/4: Adding animated markers...")
        progress.set(0.75)
        gen_dialog.update()
        time.sleep(0.5)
        
        # Add markers
        high_cluster = MarkerCluster(name="🔴 HIGH RISK AREAS").add_to(crime_map)
        low_cluster = MarkerCluster(name="🟢 LOW RISK AREAS").add_to(crime_map)
        
        for _, area in area_data.iterrows():
            risk_prob = area['Risk_Probability']
            risk_level = area['Risk_Level']
            crime_count = int(area['Crime_Count'])
            risk_percent = int(risk_prob * 100)
            
            if risk_level in ['Critical', 'High']:
                marker_color = 'red'
                animation_class = 'pulse-red'
                risk_text = "🔴 CRITICAL/HIGH RISK"
            elif risk_level == 'Medium':
                marker_color = 'orange'
                animation_class = 'pulse-orange'
                risk_text = "🟠 MEDIUM RISK"
            else:
                marker_color = 'green'
                animation_class = 'pulse-green'
                risk_text = "🟢 LOW RISK"
            
            popup_html = f"""
            <div class="glass-card" style="width: 300px;">
                <div style="font-size: 16px; font-weight: bold; color: #00B4D8;">📍 {area['Area']}</div>
                <div class="{animation_class.replace('pulse', 'risk') if 'risk' in risk_text.lower() else 'risk-' + risk_level.lower()} float-effect" style="margin: 10px 0;">
                    {risk_text} - {risk_percent}%
                </div>
                <div class="risk-meter" style="background:#333; height:8px; border-radius:4px; margin:10px 0;">
                    <div style="width:{risk_percent}%; height:100%; background:{marker_color}; border-radius:4px;"></div>
                </div>
                <div>📊 Crimes: <b>{crime_count}</b></div>
                <div>⚠️ Risk Probability: <b>{risk_percent}%</b></div>
                <hr style="border-color:#444;">
                <div style="font-size:11px;">💡 Increase patrols in this area</div>
            </div>
            """
            
            folium.Marker(
                location=[area['Latitude'], area['Longitude']],
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{area['Area']} - {risk_text} ({risk_percent}%)",
                icon=folium.Icon(color=marker_color, icon='warning-sign', prefix='glyphicon')
            ).add_to(high_cluster if risk_level in ['Critical', 'High'] else low_cluster)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000; background: rgba(30,58,95,0.95); 
                    backdrop-filter: blur(10px); border-radius: 15px; padding: 15px; border: 1px solid rgba(255,255,255,0.2);">
            <div style="font-weight: bold; color: #00B4D8;">📊 RISK LEGEND</div>
            <div><span class="pulse-red" style="display:inline-block; width:12px; height:12px; background:red; border-radius:50%;"></span> Critical/High (60-100%)</div>
            <div><span style="display:inline-block; width:12px; height:12px; background:orange; border-radius:50%;"></span> Medium (40-59%)</div>
            <div><span class="pulse-green" style="display:inline-block; width:12px; height:12px; background:green; border-radius:50%;"></span> Low (0-39%)</div>
        </div>
        '''
        crime_map.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(crime_map)
        
        status_label.configure(text="Step 4/4: Saving map...")
        progress.set(1.0)
        gen_dialog.update()
        time.sleep(0.5)
        
        # Save map
        output_path = 'outputs/heatmaps/animated_crime_hotspot.html'
        crime_map.save(output_path)
        
        gen_dialog.destroy()
        
        # Open map
        open_map(output_path)
        self.show_glass_message("✅ ANIMATED MAP READY!", 
                               "Map has been generated and opened in your browser.\n\nRed pulsing markers = High risk areas\nClick on any marker for details!", 
                               self.colors['success'])
    
    def show_animated_map(self):
        """Show animated map option in GUI"""
        if not self.data_loaded:
            messagebox.showwarning("Not Ready", "Please wait for data to load...")
            return
        
        self.clear_content()
        self.header_label.configure(text="🗺️ ANIMATED CRIME HOTSPOT MAP")
        
        # Main frame
        main_container = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        main_container.pack(fill="both", expand=True)
        
        # Glass effect info card
        info_card = ctk.CTkFrame(main_container, fg_color=self.colors['secondary'], corner_radius=20)
        info_card.pack(fill="x", pady=20, padx=30)
        
        ctk.CTkLabel(info_card, text="✨ ANIMATED CRIME HOTSPOT MAP WITH RISK INDICATORS", 
                    font=self.fonts['subheader'], text_color=self.colors['accent']).pack(pady=15)
        
        # Animation preview description
        preview_frame = ctk.CTkFrame(info_card, fg_color="transparent")
        preview_frame.pack(pady=10)
        
        animations_text = """
        🎬 ANIMATION FEATURES:
        
        🔴 RED PULSING MARKERS  →  Critical/High Risk Areas (60-100% risk)
        🟠 ORANGE MARKERS      →  Medium Risk Areas (40-59% risk)
        🟢 GREEN PULSING       →  Low Risk Areas (0-39% risk)
        
        ✨ GLASS EFFECT POPUPS →  Click any marker for detailed information
        📊 RISK METER          →  Shows exact percentage
        🔥 HEATMAP OVERLAY     →  Crime density visualization
        💫 FLOATING EFFECTS    →  Active risk zones have float animation
        """
        
        preview_label = ctk.CTkLabel(preview_frame, text=animations_text, 
                                     font=self.fonts['body'], justify="left")
        preview_label.pack(pady=10)
        
        # Risk Legend
        legend_frame = ctk.CTkFrame(info_card, fg_color="transparent")
        legend_frame.pack(pady=15)
        
        ctk.CTkLabel(legend_frame, text="📊 RISK LEVEL LEGEND", 
                    font=self.fonts['body_bold']).pack(pady=5)
        
        legend_items_frame = ctk.CTkFrame(legend_frame, fg_color="transparent")
        legend_items_frame.pack()
        
        legends = [
            ("🔴", "Critical Risk", "80-100%", self.colors['danger']),
            ("🟠", "High Risk", "60-79%", self.colors['warning']),
            ("🟡", "Medium Risk", "40-59%", "#FFD700"),
            ("🟢", "Low Risk", "0-39%", self.colors['success'])
        ]
        
        for icon, label, range_val, color in legends:
            item = ctk.CTkFrame(legend_items_frame, fg_color="transparent")
            item.pack(side="left", padx=15)
            ctk.CTkLabel(item, text=icon, font=ctk.CTkFont(size=20)).pack()
            ctk.CTkLabel(item, text=label, font=self.fonts['small'], text_color=color).pack()
            ctk.CTkLabel(item, text=range_val, font=self.fonts['small'], text_color="gray").pack()
        
        # Generate Map Button
        btn_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        btn_frame.pack(pady=30)
        
        def generate_and_open():
            self.generate_animated_map()
        
        generate_btn = AnimatedButton(btn_frame, text="🗺️ GENERATE & OPEN ANIMATED MAP", 
                                     command=generate_and_open, font=self.fonts['button'],
                                     height=60, width=400, fg_color=self.colors['accent'],
                                     hover_color=self.colors['primary'], corner_radius=15)
        generate_btn.pack(pady=10)
        
        # Info about map
        info_text = """
        💡 TIPS:
        • Map generation takes 5-10 seconds
        • Map will open automatically in your browser
        • Click on any pulsing marker to see risk details
        • Red pulsing markers = Immediate attention needed
        • Use zoom to see area details
        """
        
        info_frame = ctk.CTkFrame(main_container, fg_color=self.colors['secondary'], corner_radius=15)
        info_frame.pack(pady=20, padx=30, fill="x")
        
        ctk.CTkLabel(info_frame, text=info_text, font=self.fonts['small'], 
                    justify="left").pack(pady=15, padx=20)
    
    # ==================== 4. PREDICT RISK ====================
    def show_prediction(self):
        if not self.data_loaded:
            messagebox.showwarning("Not Ready", "Please wait...")
            return
        
        self.clear_content()
        self.header_label.configure(text="🤖 PREDICT CRIME RISK")
        
        pred_container = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        pred_container.pack(fill="both", expand=True)
        
        # Input Form
        input_frame = ctk.CTkFrame(pred_container, fg_color=self.colors['secondary'], corner_radius=20)
        input_frame.pack(side="left", padx=15, fill="both", expand=True)
        
        ctk.CTkLabel(input_frame, text="🔮 ENTER CRIME DETAILS", 
                    font=self.fonts['subheader']).pack(pady=25)
        
        entries = {}
        
        ctk.CTkLabel(input_frame, text="📍 Select Area:", font=self.fonts['body_bold']).pack(pady=(15, 5))
        areas = list(self.df['Area'].unique())
        area_var = ctk.StringVar(value=areas[0])
        area_menu = ctk.CTkComboBox(input_frame, values=areas, variable=area_var, width=280, font=self.fonts['body'])
        area_menu.pack()
        entries['area'] = area_var
        
        ctk.CTkLabel(input_frame, text="🔫 Crime Type:", font=self.fonts['body_bold']).pack(pady=(15, 5))
        crime_types = list(self.df['Crime_Type'].unique())
        crime_var = ctk.StringVar(value=crime_types[0])
        crime_menu = ctk.CTkComboBox(input_frame, values=crime_types, variable=crime_var, width=280, font=self.fonts['body'])
        crime_menu.pack()
        entries['crime'] = crime_var
        
        ctk.CTkLabel(input_frame, text="⏰ Hour (0-23):", font=self.fonts['body_bold']).pack(pady=(15, 5))
        hour_var = ctk.StringVar(value="20")
        hour_entry = ctk.CTkEntry(input_frame, textvariable=hour_var, width=280, font=self.fonts['body'])
        hour_entry.pack()
        entries['hour'] = hour_var
        
        ctk.CTkLabel(input_frame, text="📅 Day of Week (0=Mon,6=Sun):", font=self.fonts['body_bold']).pack(pady=(15, 5))
        day_var = ctk.StringVar(value="2")
        day_entry = ctk.CTkEntry(input_frame, textvariable=day_var, width=280, font=self.fonts['body'])
        day_entry.pack()
        entries['day'] = day_var
        
        def make_prediction():
            try:
                area = entries['area'].get()
                crime_type = entries['crime'].get()
                hour = int(entries['hour'].get())
                day_of_week = int(entries['day'].get())
                
                if hour < 0 or hour > 23:
                    raise ValueError("Hour must be 0-23")
                if day_of_week < 0 or day_of_week > 6:
                    raise ValueError("Day must be 0-6")
                
                area_encoded = self.encoders['Area'].transform([area])[0]
                crime_encoded = self.encoders['Crime_Type'].transform([crime_type])[0]
                police_station = self.df[self.df['Area'] == area]['Police_Station'].iloc[0]
                station_encoded = self.encoders['Police_Station'].transform([police_station])[0]
                station_count = self.df[self.df['Area'] == area]['Station_Crime_Count'].iloc[0]
                
                severity_map = {'Theft': 2, 'Robbery': 4, 'Assault': 3, 'Murder': 5}
                severity = severity_map[crime_type]
                
                if 5 <= hour < 12:
                    period = 'Morning'
                elif 12 <= hour < 17:
                    period = 'Afternoon'
                elif 17 <= hour < 21:
                    period = 'Evening'
                else:
                    period = 'Night'
                period_encoded = self.encoders['TimePeriod'].transform([period])[0]
                
                features = np.array([[
                    2024, 12, 25, day_of_week, hour, severity,
                    crime_encoded, station_encoded, area_encoded,
                    period_encoded, station_count
                ]])
                
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
                probability = self.model.predict_proba(features_scaled)[0][1]
                
                self.show_prediction_result(area, crime_type, hour, day_of_week, prediction, probability)
                
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        
        predict_btn = AnimatedButton(input_frame, text="🔮 PREDICT RISK", 
                                    command=make_prediction, font=self.fonts['button'],
                                    height=50, width=280, fg_color=self.colors['accent'],
                                    hover_color=self.colors['success'], corner_radius=12)
        predict_btn.pack(pady=30)
        
        # Info Panel
        info_frame = ctk.CTkFrame(pred_container, fg_color=self.colors['secondary'], corner_radius=20)
        info_frame.pack(side="right", padx=15, fill="both", expand=True)
        
        ctk.CTkLabel(info_frame, text="📋 HOW TO USE", font=self.fonts['subheader']).pack(pady=20)
        
        help_text = """
        1️⃣ Select Area
        2️⃣ Select Crime Type  
        3️⃣ Enter Hour (0-23)
           • Night hours = Higher risk
        4️⃣ Enter Day (0-6)
           • Weekends = More crimes
        5️⃣ Click PREDICT
        
        ⚡ AI will analyze patterns
        📊 Shows risk percentage
        💡 Gives recommendations
        """
        
        ctk.CTkLabel(info_frame, text=help_text, font=self.fonts['body'], 
                    justify="left").pack(pady=20, padx=20)
    
    def show_prediction_result(self, area, crime_type, hour, day, prediction, probability):
        """Show prediction result dialog"""
        result_dialog = ctk.CTkToplevel(self.root)
        result_dialog.title("Prediction Result")
        result_dialog.geometry="500x500"
        result_dialog.attributes('-topmost', True)
        result_dialog.configure(fg_color=self.colors['secondary'])
        
        result_dialog.attributes('-alpha', 0)
        for i in range(1, 11):
            result_dialog.attributes('-alpha', i/10)
            result_dialog.update()
            time.sleep(0.02)
        
        if prediction == 1:
            risk_level = "HIGH RISK"
            risk_color = self.colors['danger']
            icon = "⚠️"
            bg_color = "#4a0000"
        else:
            risk_level = "LOW RISK"
            risk_color = self.colors['success']
            icon = "✅"
            bg_color = "#004d00"
        
        header_frame = ctk.CTkFrame(result_dialog, fg_color=bg_color, corner_radius=15)
        header_frame.pack(fill="x", padx=20, pady=15)
        
        ctk.CTkLabel(header_frame, text=f"{icon} {risk_level} {icon}", 
                    font=ctk.CTkFont(size=24, weight="bold"), text_color=risk_color).pack(pady=15)
        
        meter_frame = ctk.CTkFrame(result_dialog, fg_color=self.colors['dark'], corner_radius=15)
        meter_frame.pack(padx=20, pady=15, fill="x")
        
        ctk.CTkLabel(meter_frame, text=f"RISK PROBABILITY: {probability:.1%}", 
                    font=self.fonts['subheader']).pack(pady=10)
        
        progress_bar = ctk.CTkProgressBar(meter_frame, width=400, height=20, corner_radius=10)
        progress_bar.pack(pady=10)
        progress_bar.set(probability)
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        details_frame = ctk.CTkFrame(result_dialog, fg_color=self.colors['dark'], corner_radius=15)
        details_frame.pack(padx=20, pady=15, fill="x")
        
        details = [
            f"📍 Area: {area}",
            f"🔫 Crime Type: {crime_type}",
            f"⏰ Time: {hour}:00",
            f"📅 Day: {days[day]}"
        ]
        
        for detail in details:
            ctk.CTkLabel(details_frame, text=detail, font=self.fonts['body']).pack(pady=5)
        
        suggestions_frame = ctk.CTkFrame(result_dialog, fg_color=self.colors['dark'], corner_radius=15)
        suggestions_frame.pack(padx=20, pady=15, fill="x")
        
        ctk.CTkLabel(suggestions_frame, text="💡 RECOMMENDATIONS", 
                    font=self.fonts['body_bold'], text_color=self.colors['accent']).pack(pady=10)
        
        if prediction == 1:
            suggestions = [
                "🚨 IMMEDIATE ACTION REQUIRED",
                "→ Increase police patrols",
                "→ Install CCTV cameras",
                "→ Improve street lighting",
                "→ Deploy rapid response team"
            ]
        else:
            suggestions = [
                "✅ MAINTAIN CURRENT SECURITY",
                "→ Regular monitoring",
                "→ Community watch programs",
                "→ Preventive awareness"
            ]
        
        for suggestion in suggestions:
            color = self.colors['warning'] if "→" in suggestion else self.colors['accent']
            ctk.CTkLabel(suggestions_frame, text=suggestion, font=self.fonts['small'], 
                        text_color=color).pack(pady=3)
        
        def close_dialog():
            for i in range(10, 0, -1):
                result_dialog.attributes('-alpha', i/10)
                result_dialog.update()
                time.sleep(0.02)
            result_dialog.destroy()
        
        close_btn = AnimatedButton(result_dialog, text="CLOSE", command=close_dialog,
                                  font=self.fonts['button'], fg_color=self.colors['primary'],
                                  hover_color=risk_color, corner_radius=10)
        close_btn.pack(pady=15)
    
    # ==================== 5. ANALYTICS ====================
    def show_analytics(self):
        if not self.data_loaded:
            messagebox.showwarning("Not Ready", "Please wait...")
            return
        
        self.clear_content()
        self.header_label.configure(text="📈 ANALYTICS DASHBOARD")
        
        canvas = tk.Canvas(self.content_frame, bg=self.colors['dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Hourly Chart
        chart1_frame = ctk.CTkFrame(scrollable_frame, fg_color=self.colors['secondary'], corner_radius=15)
        chart1_frame.pack(pady=15, padx=20, fill="x")
        
        ctk.CTkLabel(chart1_frame, text="⏰ CRIMES BY HOUR", font=self.fonts['subheader']).pack(pady=12)
        
        hourly_data = self.df.groupby('Hour').size()
        fig1 = Figure(figsize=(11, 5), dpi=100, facecolor=self.colors['secondary'])
        ax1 = fig1.add_subplot(111)
        bars = ax1.bar(hourly_data.index, hourly_data.values, color=self.colors['accent'], edgecolor='white')
        
        max_val = hourly_data.max()
        for bar, val in zip(bars, hourly_data.values):
            if val > max_val * 0.8:
                bar.set_color(self.colors['danger'])
            elif val > max_val * 0.5:
                bar.set_color(self.colors['warning'])
        
        ax1.set_title('Crime Distribution by Hour', color='white', fontsize=13)
        ax1.set_xlabel('Hour', color='white', fontsize=11)
        ax1.set_ylabel('Crimes', color='white', fontsize=11)
        ax1.tick_params(colors='white')
        ax1.set_facecolor(self.colors['secondary'])
        
        canvas1 = FigureCanvasTkAgg(fig1, master=chart1_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(pady=12, padx=20)
        
        # Day of Week Chart
        chart2_frame = ctk.CTkFrame(scrollable_frame, fg_color=self.colors['secondary'], corner_radius=15)
        chart2_frame.pack(pady=15, padx=20, fill="x")
        
        ctk.CTkLabel(chart2_frame, text="📅 CRIMES BY DAY", font=self.fonts['subheader']).pack(pady=12)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_data = self.df.groupby('Weekday', observed=True).size().reindex(day_order)
        
        fig2 = Figure(figsize=(11, 5), dpi=100, facecolor=self.colors['secondary'])
        ax2 = fig2.add_subplot(111)
        bars2 = ax2.bar(weekday_data.index, weekday_data.values, color=self.colors['warning'], edgecolor='white')
        
        for day, bar in zip(weekday_data.index, bars2):
            if day in ['Saturday', 'Sunday']:
                bar.set_color(self.colors['danger'])
        
        ax2.set_title('Crime Distribution by Day', color='white', fontsize=13)
        ax2.set_xlabel('Day', color='white', fontsize=11)
        ax2.set_ylabel('Crimes', color='white', fontsize=11)
        ax2.tick_params(colors='white')
        ax2.set_facecolor(self.colors['secondary'])
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=chart2_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(pady=12, padx=20)
        
        # Top Areas Chart
        chart3_frame = ctk.CTkFrame(scrollable_frame, fg_color=self.colors['secondary'], corner_radius=15)
        chart3_frame.pack(pady=15, padx=20, fill="x")
        
        ctk.CTkLabel(chart3_frame, text="📍 TOP HIGH-RISK AREAS", font=self.fonts['subheader']).pack(pady=12)
        
        area_data = self.df['Area'].value_counts().head(10)
        
        fig3 = Figure(figsize=(11, 5), dpi=100, facecolor=self.colors['secondary'])
        ax3 = fig3.add_subplot(111)
        ax3.barh(range(len(area_data)), area_data.values, color=self.colors['danger'], edgecolor='white')
        ax3.set_yticks(range(len(area_data)))
        ax3.set_yticklabels(area_data.index)
        ax3.set_title('Top 10 Areas by Crime Count', color='white', fontsize=13)
        ax3.set_xlabel('Crimes', color='white', fontsize=11)
        ax3.tick_params(colors='white')
        ax3.set_facecolor(self.colors['secondary'])
        
        canvas3 = FigureCanvasTkAgg(fig3, master=chart3_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(pady=12, padx=20)
    
    # ==================== 6. PDF REPORTS ====================
    def show_reports(self):
        if not self.data_loaded:
            messagebox.showwarning("Not Ready", "Please wait...")
            return
        
        self.clear_content()
        self.header_label.configure(text="📄 PDF REPORTS")
        
        reports_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        reports_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(reports_frame, text="📄 GENERATE PDF REPORT", 
                    font=self.fonts['subheader']).pack(pady=20)
        
        date_frame = ctk.CTkFrame(reports_frame, fg_color=self.colors['secondary'], corner_radius=15)
        date_frame.pack(pady=20, padx=50, fill="x")
        
        ctk.CTkLabel(date_frame, text="📅 SELECT DATE RANGE", font=self.fonts['body_bold']).pack(pady=10)
        
        start_frame = ctk.CTkFrame(date_frame, fg_color="transparent")
        start_frame.pack(pady=8)
        ctk.CTkLabel(start_frame, text="Start Date:", font=self.fonts['body']).pack(side="left", padx=10)
        start_entry = ctk.CTkEntry(start_frame, width=180, font=self.fonts['body'])
        start_entry.pack(side="left", padx=10)
        start_entry.insert(0, "2023-01-01")
        
        end_frame = ctk.CTkFrame(date_frame, fg_color="transparent")
        end_frame.pack(pady=8)
        ctk.CTkLabel(end_frame, text="End Date:", font=self.fonts['body']).pack(side="left", padx=10)
        end_entry = ctk.CTkEntry(end_frame, width=180, font=self.fonts['body'])
        end_entry.pack(side="left", padx=10)
        end_entry.insert(0, "2024-12-31")
        
        def generate_pdf():
            try:
                start_date = pd.to_datetime(start_entry.get())
                end_date = pd.to_datetime(end_entry.get())
                filtered_df = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)]
                
                if len(filtered_df) == 0:
                    messagebox.showwarning("No Data", "No data found for selected range")
                    return
                
                gen_dialog = ctk.CTkToplevel(self.root)
                gen_dialog.title("Generating PDF")
                gen_dialog.geometry("400x200")
                gen_dialog.attributes('-topmost', True)
                gen_dialog.configure(fg_color=self.colors['secondary'])
                
                ctk.CTkLabel(gen_dialog, text="📄 GENERATING PDF REPORT", 
                            font=self.fonts['subheader'], text_color=self.colors['accent']).pack(pady=20)
                
                progress = ctk.CTkProgressBar(gen_dialog, width=300)
                progress.pack(pady=15)
                
                for i in range(101):
                    progress.set(i/100)
                    gen_dialog.update()
                    time.sleep(0.01)
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 20)
                pdf.set_text_color(0, 51, 102)
                pdf.cell(0, 15, "CRIME ANALYSIS REPORT", ln=1, align="C")
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
                pdf.cell(0, 8, f"Date Range: {start_date.date()} to {end_date.date()}", ln=1, align="C")
                pdf.ln(10)
                
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "SUMMARY STATISTICS", ln=1)
                pdf.set_font("Arial", "", 11)
                pdf.cell(0, 8, f"Total Crimes: {len(filtered_df)}", ln=1)
                pdf.cell(0, 8, f"Average Daily Crimes: {len(filtered_df) / max(1, (end_date - start_date).days):.1f}", ln=1)
                pdf.ln(5)
                
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "CRIME BREAKDOWN", ln=1)
                pdf.set_font("Arial", "", 11)
                for crime in filtered_df['Crime_Type'].unique():
                    count = len(filtered_df[filtered_df['Crime_Type'] == crime])
                    pdf.cell(0, 8, f"{crime}: {count} ({count/len(filtered_df)*100:.1f}%)", ln=1)
                pdf.ln(5)
                
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "TOP 5 HIGH CRIME AREAS", ln=1)
                pdf.set_font("Arial", "", 11)
                for area, count in filtered_df['Area'].value_counts().head(5).items():
                    pdf.cell(0, 8, f"{area}: {count} crimes", ln=1)
                
                filename = f"crime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                report_dir = os.path.join('outputs', 'reports')
                os.makedirs(report_dir, exist_ok=True)
                filepath = os.path.abspath(os.path.join(report_dir, filename))
                pdf.output(filepath)
                
                gen_dialog.destroy()
                self.show_glass_message("✅ PDF GENERATED", f"Report saved as:\n{filename}", self.colors['success'])
                os.startfile(filepath)
                
            except Exception as e:
                if 'gen_dialog' in locals():
                    gen_dialog.destroy()
                messagebox.showerror("Error", f"Failed: {str(e)}")
        
        gen_btn = AnimatedButton(reports_frame, text="📄 GENERATE PDF REPORT", 
                                command=generate_pdf, font=self.fonts['button'],
                                height=55, width=350, fg_color=self.colors['success'],
                                hover_color=self.colors['primary'], corner_radius=15)
        gen_btn.pack(pady=30)
    
    # ==================== 7. HELP ====================
    def show_help(self):
        self.clear_content()
        self.header_label.configure(text="❓ HELP")
        
        help_frame = ctk.CTkFrame(self.content_frame, fg_color=self.colors['secondary'], corner_radius=20)
        help_frame.pack(fill="both", expand=True, padx=50, pady=30)
        
        help_text = """
        🚔 CRIME ANALYSIS & PREDICTION SYSTEM
        
        ================================================
        
        📊 DASHBOARD - Key metrics and charts
        📈 STATISTICS - Detailed crime breakdown
        🗺️ ANIMATED MAP - Interactive risk map with animations
        🤖 PREDICT - AI-based risk prediction
        📈 ANALYTICS - Advanced visualizations
        📄 PDF REPORTS - Generate professional reports
        
        ================================================
        
        💡 ANIMATED MAP FEATURES:
        • Red pulsing markers = High risk areas
        • Click markers for risk percentage
        • Heatmap shows crime density
        • Automatic generation in browser
        
        🎯 PREDICTION TIPS:
        • Night hours (20-04) = Higher risk
        • Weekends = More crimes
        • December = Peak crime month
        """
        
        ctk.CTkLabel(help_frame, text=help_text, font=self.fonts['body'], 
                    justify="left").pack(pady=30, padx=30)


    def open_ai_features(self):
        """Opens the AI features window (ai_features_gui.py)"""
        try:
            from ai_features_gui import AIFeaturesWindow
            if getattr(self, 'ai_window', None) is not None and self.ai_window.winfo_exists():
                self.ai_window.focus()
                return
            self.ai_window = AIFeaturesWindow(self.root)
        except Exception as e:
            messagebox.showerror("AI Features", f"Could not open the AI window:\n{e}\n\nRun this first: python 06_train_ai_models.py")


# ==================== RUN ====================
if __name__ == "__main__":
    print("="*50)
    print("🚀 CRIME ANALYSIS & PREDICTION SYSTEM")
    print("WITH INTEGRATED ANIMATED MAP")
    print("="*50)
    
    if not os.path.exists('data/processed_crime_data.csv'):
        print("❌ Processed data not found!")
        print("Please run: python run_all.py")
    else:
        print("✅ Launching application...")
        app = CrimeAnalysisApp()
         ###.\venv\Scripts\Activate.ps1
         ###python crime_analysis_gui.py