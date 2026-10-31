# ai_features_gui.py
# =====================================================================
#  AI FEATURES WINDOW  -  opens from the "🧠 AI FEATURES" button of the main GUI
#  Tabs: Risk & Why | Patrol | Forecast | Hotspots | Grid Risk | Alerts |
#        Fairness | Chatbot | Report
#  You must run `python 06_train_ai_models.py` first.
# =====================================================================

import os
import subprocess
import sys
import tkinter as tk
import webbrowser
from tkinter import ttk, messagebox

import customtkinter as ctk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ai_chatbot import CrimeChatbot
from map_utils import open_map
from ai_engine import MAP_DIR, SHIFT_HOURS, SHIFTS, CrimeAI

BG = '#2b2b2b'
FG = '#e8e8e8'
ACCENT = '#1f538d'
LEVEL_COLORS = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}


def open_file(path):
    path = os.path.abspath(path)
    if sys.platform.startswith('win'):
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])


class AIFeaturesWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("🧠 AI Features - Crime Analysis & Prediction")
        self.geometry("1250x800")
        self.minsize(1050, 680)

        self._style()
        try:
            self.ai = CrimeAI().load_artifacts()
        except Exception as e:
            messagebox.showerror("Data error", f"Failed to load dataset:\n{e}", parent=self)
            self.destroy()
            return
        self.bot = CrimeChatbot(self.ai)
        self.canvases = {}

        head = ctk.CTkFrame(self, fg_color=ACCENT, corner_radius=0)
        head.pack(fill='x')
        ctk.CTkLabel(head, text="🧠 AI FEATURES", font=("Segoe UI", 22, "bold"), text_color="white").pack(side='left', padx=20, pady=10)
        ready = self.ai.bundle is not None
        msg = "✅ Models ready" if ready else "⚠️ Models not found - run `python 06_train_ai_models.py` first"
        ctk.CTkLabel(head, text=msg, font=("Segoe UI", 13), text_color="white").pack(side='right', padx=20)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=10)
        for name in ["🎯 Risk & Why", "👮 Patrol", "📈 Forecast", "🔥 Hotspots", "🧩 Grid Risk",
                     "🚨 Alerts", "⚖️ Fairness", "💬 Chatbot", "📄 Report"]:
            self.tabs.add(name)
        self._tab_risk(self.tabs.tab("🎯 Risk & Why"))
        self._tab_patrol(self.tabs.tab("👮 Patrol"))
        self._tab_forecast(self.tabs.tab("📈 Forecast"))
        self._tab_hotspots(self.tabs.tab("🔥 Hotspots"))
        self._tab_grid(self.tabs.tab("🧩 Grid Risk"))
        self._tab_alerts(self.tabs.tab("🚨 Alerts"))
        self._tab_fairness(self.tabs.tab("⚖️ Fairness"))
        self._tab_chat(self.tabs.tab("💬 Chatbot"))
        self._tab_report(self.tabs.tab("📄 Report"))
        self.after(250, self.lift)

    # ------------------------------------------------------------- helpers
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use('clam')
        except Exception:
            pass
        st.configure('AI.Treeview', background=BG, fieldbackground=BG, foreground=FG, rowheight=26, borderwidth=0)
        st.configure('AI.Treeview.Heading', background=ACCENT, foreground='white', font=('Segoe UI', 10, 'bold'))
        st.map('AI.Treeview', background=[('selected', ACCENT)])

    def _need_model(self):
        if self.ai.bundle is None:
            messagebox.showwarning("Model missing", "Please run `python 06_train_ai_models.py` first.", parent=self)
            return False
        return True

    def _table(self, parent, df, height=10, level_col=None):
        """Shows a DataFrame in a Treeview. Any previous table is replaced."""
        frame = ctk.CTkFrame(parent, fg_color='transparent')
        if df is None or len(df) == 0:
            ctk.CTkLabel(frame, text="No data (run 06_train_ai_models.py first).", text_color='gray').pack(pady=10)
            return frame
        cols = list(df.columns)
        tv = ttk.Treeview(frame, columns=cols, show='headings', height=height, style='AI.Treeview')
        sb = ttk.Scrollbar(frame, orient='vertical', command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=140, anchor='w', stretch=True)
        if 'Reason' in cols:
            tv.column('Reason', width=380)
        if 'Message' in cols:
            tv.column('Message', width=420)
        for lv, col in LEVEL_COLORS.items():
            tv.tag_configure(lv, foreground=col)
        for _, r in df.iterrows():
            vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values]
            tag = (str(r[level_col]),) if level_col and str(r[level_col]) in LEVEL_COLORS else ()
            tv.insert('', 'end', values=vals, tags=tag)
        tv.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        return frame

    def _replace(self, key, parent, widget, **pack):
        old = self.canvases.get(key)
        if old is not None:
            try:
                old.destroy()
            except Exception:
                pass
        widget.pack(**pack)
        self.canvases[key] = widget

    def _figure(self, key, parent, fig, **pack):
        cv = FigureCanvasTkAgg(fig, master=parent)
        w = cv.get_tk_widget()
        self._replace(key, parent, w, **(pack or dict(fill='both', expand=True)))
        cv.draw()

    @staticmethod
    def _dark_fig(w=6, h=3.2):
        fig = Figure(figsize=(w, h), dpi=100, facecolor=BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        for s in ax.spines.values():
            s.set_color('#666')
        ax.title.set_color(FG)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        return fig, ax

    def _parse_date(self, text):
        try:
            return pd.Timestamp(text.strip())
        except Exception:
            messagebox.showerror("Date error", "Date format: YYYY-MM-DD (e.g. 2026-01-05)", parent=self)
            return None

    @staticmethod
    def _shift_of(text):
        return text.split()[0]

    # ------------------------------------------------------------- TAB 1: risk + why
    def _tab_risk(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(fill='x', padx=10, pady=8)
        shift_vals = [f"{s} ({SHIFT_HOURS[s]})" for s in SHIFTS]
        self.r_area = ctk.CTkComboBox(top, values=self.ai.areas, width=200)
        self.r_area.set(self.ai.areas[0])
        self.r_date = ctk.CTkEntry(top, width=130)
        self.r_date.insert(0, str((self.ai.last_date + pd.Timedelta(days=1)).date()))
        self.r_shift = ctk.CTkComboBox(top, values=shift_vals, width=220)
        self.r_shift.set(shift_vals[2])
        for lbl, w in [("Area", self.r_area), ("Date", self.r_date), ("Shift", self.r_shift)]:
            ctk.CTkLabel(top, text=lbl).pack(side='left', padx=(12, 4))
            w.pack(side='left')
        ctk.CTkButton(top, text="🔮 Predict & Explain", command=self._do_risk, width=170).pack(side='left', padx=16)

        self.r_result = ctk.CTkLabel(tab, text="Select an area, date and shift, then click Predict.",
                                     font=("Segoe UI", 22, "bold"))
        self.r_result.pack(pady=(10, 0))
        self.r_text = ctk.CTkLabel(tab, text="", font=("Consolas", 13), justify='left')
        self.r_text.pack(pady=4)
        self.r_chart = ctk.CTkFrame(tab, fg_color='transparent')
        self.r_chart.pack(fill='both', expand=True, padx=10, pady=6)
        ctk.CTkLabel(tab, text="Note: the data ends on 31-Dec-2025; for future dates the lag features are built from the last available data.",
                     text_color='gray', font=("Segoe UI", 11)).pack(pady=(0, 4))

    def _do_risk(self):
        if not self._need_model():
            return
        d = self._parse_date(self.r_date.get())
        if d is None:
            return
        area, shift = self.r_area.get(), self._shift_of(self.r_shift.get())
        if area not in self.ai.areas:
            messagebox.showerror("Area", "Please choose an area from the dropdown.", parent=self)
            return
        p, lvl, X = self.ai.risk_for(area, d, shift)
        base = self.ai.bundle['base_rate']
        self.r_result.configure(text=f"Probability of a crime: {p * 100:.1f}%   →   {lvl.upper()} RISK",
                                text_color=LEVEL_COLORS[lvl])
        exp = self.ai.explain(X, top=8)
        self.r_text.configure(text=f"(Average base rate for this area-shift ≈ {base * 100:.0f}%)   Why?\n" +
                              self.ai.explain_text(exp[:5]))
        fig, ax = self._dark_fig(7.5, 3.2)
        names = [e['Text'][:42] for e in exp][::-1]
        vals = [e['Contribution'] * 100 for e in exp][::-1]
        ax.barh(names, vals, color=['#e74c3c' if v > 0 else '#2ecc71' for v in vals])
        ax.axvline(0, color='#aaa', lw=0.8)
        ax.set_title("What is raising / lowering the risk (percentage points)")
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()
        self._figure('risk', self.r_chart, fig)

    # ------------------------------------------------------------- TAB 2: patrol
    def _tab_patrol(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(fill='x', padx=10, pady=8)
        shift_vals = [f"{s} ({SHIFT_HOURS[s]})" for s in SHIFTS]
        self.p_date = ctk.CTkEntry(top, width=130)
        self.p_date.insert(0, str((self.ai.last_date + pd.Timedelta(days=1)).date()))
        self.p_shift = ctk.CTkComboBox(top, values=shift_vals, width=220)
        self.p_shift.set(shift_vals[2])
        self.p_off = ctk.CTkEntry(top, width=70)
        self.p_off.insert(0, "12")
        for lbl, w in [("Date", self.p_date), ("Shift", self.p_shift), ("Officers", self.p_off)]:
            ctk.CTkLabel(top, text=lbl).pack(side='left', padx=(12, 4))
            w.pack(side='left')
        ctk.CTkButton(top, text="👮 Generate Patrol Plan", command=self._do_patrol, width=190).pack(side='left', padx=16)
        self.p_summary = ctk.CTkLabel(tab, text="", font=("Segoe UI", 15, "bold"), justify='left')
        self.p_summary.pack(pady=6)
        self.p_table = ctk.CTkFrame(tab, fg_color='transparent')
        self.p_table.pack(fill='both', expand=True, padx=10, pady=6)

    def _do_patrol(self):
        if not self._need_model():
            return
        d = self._parse_date(self.p_date.get())
        if d is None:
            return
        try:
            n = max(1, int(self.p_off.get()))
        except ValueError:
            messagebox.showerror("Officers", "Enter the number of officers (e.g. 12).", parent=self)
            return
        t, station = self.ai.patrol_plan(d, self._shift_of(self.p_shift.get()), n)
        top3 = ', '.join(t['Area'].head(3))
        st = '   |   '.join(f"{r.Station.replace(' Police Station', '')}: {r.Officers}" for r in station.itertuples())
        self.p_summary.configure(text=f"Top 3 risky areas: {top3}\nStation-wise officers → {st}")
        show = t[['Area', 'Station', 'Risk_Prob', 'Level', 'Officers', 'Reason']].copy()
        show['Risk_Prob'] = (show['Risk_Prob'] * 100).round(1).astype(str) + '%'
        self._replace('patrol', self.p_table, self._table(self.p_table, show, 12, 'Level'), fill='both', expand=True)

    # ------------------------------------------------------------- TAB 3: forecast
    def _tab_forecast(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(fill='x', padx=10, pady=8)
        self.f_area = ctk.CTkComboBox(top, values=['All areas'] + self.ai.areas, width=220, command=lambda _: self._draw_forecast())
        self.f_area.set('All areas')
        ctk.CTkLabel(top, text="Area").pack(side='left', padx=(12, 4))
        self.f_area.pack(side='left')
        ev = ""
        if self.ai.forecast_eval is not None:
            r = self.ai.forecast_eval.iloc[0]
            b = self.ai.forecast_eval.iloc[1]
            ev = f"   Weekly MAE: AI {r['Weekly_MAE']:.2f} vs baseline {b['Weekly_MAE']:.2f}"
        ctk.CTkLabel(top, text=ev, text_color='gray').pack(side='left', padx=12)
        self.f_chart = ctk.CTkFrame(tab, fg_color='transparent')
        self.f_chart.pack(fill='both', expand=True, padx=10)
        self.f_table = ctk.CTkFrame(tab, fg_color='transparent')
        self.f_table.pack(fill='x', padx=10, pady=6)
        self._draw_forecast()

    def _draw_forecast(self):
        f = self.ai.forecast_df
        if f is None:
            lbl = ctk.CTkLabel(self.f_chart, text="Forecast not found. Please run `python 06_train_ai_models.py` first.")
            self._replace('fc', self.f_chart, lbl, pady=30)
            return
        sel = self.f_area.get()
        if sel == 'All areas':
            hist = self.ai.area_daily.sum(axis=1).iloc[-60:]
            fc = f.groupby('Date')['Forecast'].sum()
        else:
            hist = self.ai.area_daily[sel].iloc[-60:]
            fc = f[f['Area'] == sel].set_index('Date')['Forecast']
        fig, ax = self._dark_fig(9, 3.4)
        ax.plot(hist.index, hist.values, color='#5dade2', label='Actual (last 60 days)')
        ax.plot(fc.index, fc.values, color='#f39c12', ls='--', marker='o', ms=3, label='Forecast (next 30 days)')
        ax.set_title(f"Daily crimes - {sel}")
        ax.legend(facecolor=BG, labelcolor=FG)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self._figure('fc', self.f_chart, fig)
        summ = self.ai.forecast_summary()
        self._replace('fc_t', self.f_table, self._table(self.f_table, summ, 5), fill='x')

    # ------------------------------------------------------------- TAB 4: hotspots
    def _tab_hotspots(self, tab):
        ctk.CTkLabel(tab, text="Crime hotspots found by DBSCAN clustering (ranked by severity score)",
                     font=("Segoe UI", 14, "bold")).pack(pady=8)
        row = ctk.CTkFrame(tab, fg_color='transparent')
        row.pack()
        ctk.CTkButton(row, text="🗺️ Open Hotspot Map in Browser", width=240,
                      command=lambda: self._open_map('ai_hotspots_map.html')).pack(side='left', padx=6)
        t = ctk.CTkFrame(tab, fg_color='transparent')
        t.pack(fill='both', expand=True, padx=10, pady=10)
        self._table(t, self.ai.hotspots, 14).pack(fill='both', expand=True)

    # ------------------------------------------------------------- TAB 5: grid
    def _tab_grid(self, tab):
        ctk.CTkLabel(tab, text="City divided into 500 m cells - expected crime count for next week",
                     font=("Segoe UI", 14, "bold")).pack(pady=8)
        ev = ""
        if self.ai.grid_eval is not None and len(self.ai.grid_eval) >= 3:
            c = [x for x in self.ai.grid_eval.columns if 'Capture' in x][0]
            ev = (f"Test: the top-10 cells capture {self.ai.grid_eval.iloc[0][c]:.0f}% of actual crimes "
                  f"(baseline {self.ai.grid_eval.iloc[1][c]:.0f}%, random {self.ai.grid_eval.iloc[2][c]:.0f}%)")
        ctk.CTkLabel(tab, text=ev, text_color='gray').pack()
        ctk.CTkButton(tab, text="🗺️ Open Grid Risk Map in Browser", width=250,
                      command=lambda: self._open_map('ai_grid_risk_map.html')).pack(pady=8)
        t = ctk.CTkFrame(tab, fg_color='transparent')
        t.pack(fill='both', expand=True, padx=10, pady=6)
        g = self.ai.grid_pred
        show = None if g is None else g.head(20)[['Rank', 'Cell_ID', 'Lat_Center', 'Lon_Center', 'Pred_Next_Week',
                                                   'Last_4wk_Avg', 'Main_Crime', 'Risk_Level']].round(3)
        self._table(t, show, 14, 'Risk_Level').pack(fill='both', expand=True)

    # ------------------------------------------------------------- TAB 6: alerts
    def _tab_alerts(self, tab):
        ctk.CTkLabel(tab, text="Anomaly alerts (Isolation Forest + z-score) - newest first",
                     font=("Segoe UI", 14, "bold")).pack(pady=8)
        a = self.ai.anomalies
        if a is not None and len(a):
            latest = a.head(3)
            for r in latest.itertuples():
                color = LEVEL_COLORS.get(r.Level, 'white')
                ctk.CTkLabel(tab, text=f"🚨 [{r.Level}] week of {pd.to_datetime(r.Week_Start).date()} - {r.Message}",
                             text_color=color, font=("Segoe UI", 13)).pack(anchor='w', padx=20)
        t = ctk.CTkFrame(tab, fg_color='transparent')
        t.pack(fill='both', expand=True, padx=10, pady=10)
        show = None
        if a is not None:
            show = a.head(30).copy()
            show['Week_Start'] = pd.to_datetime(show['Week_Start']).dt.date.astype(str)
        self._table(t, show, 14, 'Level').pack(fill='both', expand=True)

    # ------------------------------------------------------------- TAB 7: fairness
    def _tab_fairness(self, tab):
        p = os.path.join(self.ai.out_dir, 'fairness_report.txt')
        box = ctk.CTkTextbox(tab, height=330, font=("Consolas", 12))
        box.pack(fill='both', expand=True, padx=10, pady=10)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                box.insert('1.0', f.read())
        else:
            box.insert('1.0', "Fairness report not found. Please run `python 06_train_ai_models.py` first.")
        box.configure(state='disabled')
        pa = os.path.join(self.ai.out_dir, 'fairness_by_area.csv')
        if os.path.exists(pa):
            df = pd.read_csv(pa)[['Area', 'Actual_Rate', 'Avg_Pred_Prob', 'Calibration_Gap', 'High_Flag_Rate', 'FPR', 'Flag_Share_%', 'Crime_Share_%']]
            f = ctk.CTkFrame(tab, fg_color='transparent')
            f.pack(fill='x', padx=10, pady=6)
            self._table(f, df, 9).pack(fill='x')

    # ------------------------------------------------------------- TAB 8: chatbot
    def _tab_chat(self, tab):
        mode = "Claude API (ANTHROPIC_API_KEY found)" if os.environ.get('ANTHROPIC_API_KEY') else "Offline mode (rule-based)"
        ctk.CTkLabel(tab, text=f"💬 Crime Chatbot - {mode}", font=("Segoe UI", 14, "bold")).pack(pady=6)
        self.chat = ctk.CTkTextbox(tab, font=("Consolas", 13), wrap='word')
        self.chat.pack(fill='both', expand=True, padx=10, pady=4)
        try:
            self.chat.tag_config('user', foreground='#5dade2')
            self.chat.tag_config('bot', foreground='#58d68d')
        except Exception:
            pass
        chips = ctk.CTkFrame(tab, fg_color='transparent')
        chips.pack(fill='x', padx=10)
        for q in ["which area is the most dangerous?", "how many robberies happened at night last month?",
                  "which area will have the most crime next week?", "where should we patrol tonight?"]:
            ctk.CTkButton(chips, text=q, height=26, fg_color='#34495e', font=("Segoe UI", 11),
                          command=lambda x=q: self._chat_send(x)).pack(side='left', padx=3, pady=4)
        row = ctk.CTkFrame(tab, fg_color='transparent')
        row.pack(fill='x', padx=10, pady=8)
        self.chat_in = ctk.CTkEntry(row, placeholder_text="Type your question (English or Roman Urdu)...", height=38)
        self.chat_in.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.chat_in.bind('<Return>', lambda e: self._chat_send())
        ctk.CTkButton(row, text="Send", width=90, height=38, command=self._chat_send).pack(side='left')
        self._chat_write("Bot", "Hello! Type 'help' to see example questions.\n", 'bot')

    def _chat_write(self, who, text, tag):
        self.chat.configure(state='normal')
        self.chat.insert('end', f"{who}: ", tag)
        self.chat.insert('end', text + "\n")
        self.chat.configure(state='disabled')
        self.chat.see('end')

    def _chat_send(self, text=None):
        q = (text or self.chat_in.get()).strip()
        if not q:
            return
        self.chat_in.delete(0, 'end')
        self._chat_write("You", q, 'user')
        self._chat_write("Bot", "thinking...", 'bot')
        self.update_idletasks()
        try:
            out = self.bot.ask(q)
            ans = out['answer']
            if out.get('table') is not None:
                ans += "\n" + out['table'].head(8).to_string(index=False)
            ans += f"\n[mode: {out.get('mode')}]"
        except Exception as e:
            ans = f"Error: {e}"
        # remove the "thinking..." line
        self.chat.configure(state='normal')
        idx = self.chat.search("Bot: thinking...", '1.0', backwards=True)
        if idx:
            self.chat.delete(idx, f"{idx} lineend+1c")
        self.chat.configure(state='disabled')
        self._chat_write("Bot", ans + "\n", 'bot')

    # ------------------------------------------------------------- TAB 9: report
    def _tab_report(self, tab):
        ctk.CTkLabel(tab, text="📄 AI PDF Report", font=("Segoe UI", 18, "bold")).pack(pady=(30, 6))
        ctk.CTkLabel(tab, text="Charts, hotspots, grid risk, forecast, anomaly alerts and fairness audit in one PDF.",
                     text_color='gray').pack()
        self.rep_status = ctk.CTkLabel(tab, text="", font=("Segoe UI", 13))
        ctk.CTkButton(tab, text="Generate AI Report (PDF)", width=260, height=44, command=self._do_report).pack(pady=20)
        self.rep_status.pack()
        ctk.CTkButton(tab, text="📂 Open reports folder", width=200,
                      command=lambda: open_file(os.path.join('outputs', 'reports'))).pack(pady=10)

    def _do_report(self):
        if not self._need_model():
            return
        self.rep_status.configure(text="Building PDF... (10-20 sec)")
        self.update_idletasks()
        try:
            path = self.ai.build_pdf_report()
            self.rep_status.configure(text=f"✅ Saved: {path}")
            open_file(path)
        except Exception as e:
            self.rep_status.configure(text=f"❌ Error: {e}")

    # ------------------------------------------------------------- misc
    def _open_map(self, name):
        p = os.path.join(MAP_DIR, name)
        if not os.path.exists(p):
            messagebox.showwarning("Map missing", f"{p} not found. Please run `python 06_train_ai_models.py` first.", parent=self)
            return
        open_map(p)


if __name__ == '__main__':
    root = ctk.CTk()
    root.withdraw()
    win = AIFeaturesWindow(root)
    win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
