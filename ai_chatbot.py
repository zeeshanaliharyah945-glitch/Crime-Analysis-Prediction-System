# ai_chatbot.py
# =====================================================================
#  CRIME CHATBOT  -  ask questions in natural language
# =====================================================================
#  Two modes:
#   1) OFFLINE (default)  : rule-based parser, needs no internet and no API key.
#   2) CLAUDE MODE        : if ANTHROPIC_API_KEY is set, Claude converts the question
#                           into a small JSON "spec" (area, crime, shift, dates...).
#
#  SAFETY: the LLM never runs code. It only returns JSON with fixed fields, and
#  our own pandas code executes the query. If the JSON is invalid, the chatbot
#  falls back to the rule-based parser.
#
#  Example questions that work:
#    "how many robberies happened at Market Commercial at night last month?"
#    "which area has the most theft in 2025?"
#    "which crime type is the most frequent?"
#    "show the trend for Waleed North"
#    "which area will have the most crime next week?"
#
#  Note: the word lists below intentionally include common Roman Urdu spellings
#  (e.g. "chori", "raat", "pichle mahine") as accepted INPUT vocabulary, so users
#  can type questions in either English or Roman Urdu. All answers are in English.
# =====================================================================

import json
import os
import re

import pandas as pd

from ai_engine import CRIME_TYPES, DAY_NAMES, MONTH_NAMES, SHIFTS

LLM_MODEL = os.environ.get('CRIME_LLM_MODEL', 'claude-sonnet-5-5')

CRIME_WORDS = {
    'Theft': ['theft', 'thefts', 'chori', 'choriyan', 'churi', 'snatching', 'stolen'],
    'Robbery': ['robbery', 'robberies', 'dakaiti', 'dakaity', 'dacoity', 'loot', 'lootmar', 'lut'],
    'Assault': ['assault', 'assaults', 'hamla', 'hamle', 'jhagra', 'jhagray', 'marpeet', 'fight', 'fights'],
    'Murder': ['murder', 'murders', 'qatal', 'qatl', 'khoon', 'katal'],
}
SHIFT_WORDS = {
    'Night': ['night', 'raat', 'raatko', 'late'],
    'Morning': ['morning', 'subah', 'subha', 'sawere', 'savere'],
    'Evening': ['evening', 'shaam', 'sham'],
}
MONTH_ALIASES = {m.lower(): i + 1 for i, m in enumerate(MONTH_NAMES)}
MONTH_ALIASES.update({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
                      'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12})

RANK_WORDS = ['most', 'highest', 'top', 'worst', 'dangerous', 'khatarnak', 'zyada', 'ziada', 'sab se', 'sabse',
              'maximum', 'max', 'ranking', 'rank', 'busiest']
TREND_WORDS = ['trend', 'barh', 'badh', 'increase', 'increasing', 'decrease', 'decreasing', 'kam', 'growth',
               'compare', 'comparison', 'change']
AVG_WORDS = ['average', 'avg', 'ausat', 'rozana', 'daily']
FORECAST_WORDS = ['forecast', 'predict', 'prediction', 'expected', 'agle', 'agla', 'next', 'aane', 'future']
RISK_WORDS = ['risk', 'patrol', 'khatra', 'safe', 'unsafe']

VALID_METRICS = ['count', 'ranking', 'trend', 'average', 'forecast', 'risk']
VALID_GROUPS = [None, 'Area', 'Crime_Type', 'Shift', 'Weekday', 'Month']

HELP = ("I can answer questions like these (English or Roman Urdu input):\n"
        " - how many robberies happened at Market Commercial at night last month?\n"
        " - which area has the most theft in 2025?\n"
        " - which crime type is the most frequent?\n"
        " - which day of the week is the most dangerous?\n"
        " - show the trend for Waleed North\n"
        " - which area will have the most crime next week?\n"
        " - where should we patrol tonight?\n"
        "Note: 'today / last / this month' are measured from the last date in the dataset.")


def _tokens(q):
    return re.findall(r"[a-z0-9]+", q.lower())


class CrimeChatbot:
    def __init__(self, ai):
        self.ai = ai
        self.df = ai.df
        self.anchor = ai.last_date

    # -------------------------------------------------------------
    # 1) Question -> spec (rules)
    # -------------------------------------------------------------
    def _find_area(self, q):
        ql = ' ' + ' '.join(_tokens(q)) + ' '
        best = None
        for a in self.ai.areas:
            if f' {a.lower()} ' in ql and (best is None or len(a) > len(best)):
                best = a
        if best:
            ext = [a for a in self.ai.areas if a != best and a.startswith(best + ' ')]
            if ext:                                   # e.g. "Civil Line" -> Civil Line + Area A + Area B
                return [best] + ext, f"{best} (all sub-areas)"
            return [best], best
        for pre in sorted({a.split()[0] for a in self.ai.areas}):
            if f' {pre.lower()} ' in ql:
                grp = [a for a in self.ai.areas if a.startswith(pre)]
                return grp, f"{pre} (all areas)"
        return None, None

    def _find_period(self, q):
        ql = q.lower()
        a = self.anchor
        m = re.search(r"(?:last|past|pichle|pichly|pichhle|pichlay)\s+(\d+)\s+(day|days|din|dinon|week|weeks|hafte|haftay|month|months|mahine|mahinon)", ql)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            days = n * (7 if unit.startswith(('week', 'haft')) else 30 if unit.startswith(('month', 'mah')) else 1)
            return a - pd.Timedelta(days=days - 1), a, f"last {n} {unit}"
        years = re.findall(r"\b(20[2-3]\d)\b", ql)
        toks = set(_tokens(ql))
        mon = next((MONTH_ALIASES[t] for t in _tokens(ql) if t in MONTH_ALIASES and t != 'may' or (t == 'may' and years)), None)
        if mon:
            y = int(years[0]) if years else (a.year if mon <= a.month else a.year - 1)
            s = pd.Timestamp(year=y, month=mon, day=1)
            return s, s + pd.offsets.MonthEnd(0), f"{MONTH_NAMES[mon - 1]} {y}"
        if years:
            y = int(years[0])
            return pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31), str(y)
        if re.search(r"last month|past month|pichle mahine|pichla mahina|pichle maheene|pichla mahine", ql):
            return a - pd.Timedelta(days=29), a, "last 30 days"
        if re.search(r"last week|past week|pichle hafte|pichla hafta|pichle haftay", ql):
            return a - pd.Timedelta(days=6), a, "last 7 days"
        if re.search(r"last year|pichle saal|pichla saal|pichle sal", ql):
            return a - pd.Timedelta(days=364), a, "last 365 days"
        if re.search(r"this month|is mahine|is month", ql):
            return a.replace(day=1), a, "this month"
        if re.search(r"this year|is saal|is sal", ql):
            return pd.Timestamp(a.year, 1, 1), a, "this year"
        if toks & {'aaj', 'today'}:
            return a, a, f"{a.date()} (latest day)"
        return self.ai.first_date, a, "entire dataset (2023-2025)"

    def parse_rules(self, question):
        q = question.lower()
        toks = _tokens(q)
        tset = set(toks)
        spec = {'metric': 'count', 'group_by': None, 'areas': None, 'area_label': None,
                'crime_type': None, 'shift': None, 'top_n': 5, 'horizon': 7}
        spec['areas'], spec['area_label'] = self._find_area(q)
        for c, words in CRIME_WORDS.items():
            if tset & set(words):
                spec['crime_type'] = c
        for s, words in SHIFT_WORDS.items():
            if tset & set(words):
                spec['shift'] = s
        spec['start'], spec['end'], spec['period_label'] = self._find_period(q)

        def any_phrase(ps):
            return any(p in q for p in ps)

        if any_phrase(['which area', 'kaunsa area', 'konsa area', 'kaunse area', 'konse area', 'kis area', 'kis jagah', 'kahan', 'kaun si jagah']):
            spec['group_by'] = 'Area'
        elif any_phrase(['which crime', 'kaunsa crime', 'konsa crime', 'kaunsi crime', 'kis qisam', 'crime type', 'konsa jurm', 'kaunsa jurm']):
            spec['group_by'] = 'Crime_Type'
        elif any_phrase(['which shift', 'kaunsi shift', 'konsi shift', 'kis waqt', 'kaunse waqt', 'konsa waqt', 'which time']):
            spec['group_by'] = 'Shift'
        elif any_phrase(['which day', 'kaunsa din', 'konsa din', 'kis din']):
            spec['group_by'] = 'Weekday'
        elif any_phrase(['which month', 'kaunsa mahina', 'konsa mahina', 'kaunse mahine', 'konse mahine']):
            spec['group_by'] = 'Month'

        ranking = any(w in q for w in RANK_WORDS)
        if tset & set(FORECAST_WORDS) and (tset & {'week', 'hafte', 'hafta', 'month', 'mahine', 'mahina', 'forecast', 'predict', 'expected', 'future'}):
            spec['metric'] = 'forecast'
            spec['horizon'] = 30 if tset & {'month', 'mahine', 'mahina'} else 7
            if not spec['areas'] and (ranking or spec['group_by'] == 'Area'):
                spec['group_by'] = 'Area'
        elif tset & set(RISK_WORDS):
            spec['metric'] = 'risk'
            spec['shift'] = spec['shift'] or 'Night'
        elif tset & set(TREND_WORDS):
            spec['metric'] = 'trend'
        elif tset & set(AVG_WORDS):
            spec['metric'] = 'average'
        elif spec['group_by'] or ranking:
            spec['metric'] = 'ranking'
            spec['group_by'] = spec['group_by'] or 'Area'
        return spec

    # -------------------------------------------------------------
    # 2) Optional: Claude -> spec
    # -------------------------------------------------------------
    def llm_spec(self, question):
        key = os.environ.get('ANTHROPIC_API_KEY')
        if not key:
            return None
        try:
            import requests
            system = (
                "You convert a crime-statistics question (English or Roman Urdu) into a JSON object. "
                "Reply with JSON ONLY, no prose. Fields: "
                '{"metric": one of ["count","ranking","trend","average","forecast","risk"], '
                '"group_by": one of [null,"Area","Crime_Type","Shift","Weekday","Month"], '
                f'"area": one of {self.ai.areas} or a prefix like "Civil Line" or null, '
                f'"crime_type": one of {CRIME_TYPES} or null, "shift": one of {SHIFTS} or null, '
                '"start": "YYYY-MM-DD" or null, "end": "YYYY-MM-DD" or null, "horizon": 7 or 30}. '
                f"Dataset covers {self.ai.first_date.date()} to {self.anchor.date()}; treat {self.anchor.date()} as 'today'. "
                "'last month' means the last 30 days. Use metric ranking+group_by for 'which/most' questions.")
            r = requests.post('https://api.anthropic.com/v1/messages',
                              headers={'x-api-key': key, 'anthropic-version': '2023-06-01',
                                       'content-type': 'application/json'},
                              json={'model': LLM_MODEL, 'max_tokens': 300, 'system': system,
                                    'messages': [{'role': 'user', 'content': question}]}, timeout=25)
            r.raise_for_status()
            text = r.json()['content'][0]['text']
            raw = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            return self._validate_llm(raw)
        except Exception:
            return None

    def _validate_llm(self, raw):
        metric = raw.get('metric') if raw.get('metric') in VALID_METRICS else 'count'
        group = raw.get('group_by') if raw.get('group_by') in VALID_GROUPS else None
        area = raw.get('area')
        areas, label = None, None
        if isinstance(area, str) and area:
            if area in self.ai.areas:
                ext = [a for a in self.ai.areas if a.startswith(area + ' ')]
                areas, label = [area] + ext, area
            else:
                grp = [a for a in self.ai.areas if a.lower().startswith(area.lower())]
                areas, label = (grp or None), (area if grp else None)
        ct = raw.get('crime_type') if raw.get('crime_type') in CRIME_TYPES else None
        sh = raw.get('shift') if raw.get('shift') in SHIFTS else None

        def d(x, default):
            try:
                return pd.Timestamp(x)
            except Exception:
                return default
        start = max(d(raw.get('start'), self.ai.first_date), self.ai.first_date)
        end = min(d(raw.get('end'), self.anchor), self.anchor)
        if start > end:
            start, end = self.ai.first_date, self.anchor
        hz = 30 if raw.get('horizon') == 30 else 7
        return {'metric': metric, 'group_by': group, 'areas': areas, 'area_label': label, 'crime_type': ct,
                'shift': sh, 'top_n': 5, 'horizon': hz, 'start': start, 'end': end,
                'period_label': f"{start.date()} to {end.date()}"}

    # -------------------------------------------------------------
    # 3) Spec -> answer (our own safe pandas code)
    # -------------------------------------------------------------
    def _subset(self, spec):
        d = self.df
        d = d[(d['Date'] >= spec['start']) & (d['Date'] <= spec['end'])]
        if spec['areas']:
            d = d[d['Area'].isin(spec['areas'])]
        if spec['crime_type']:
            d = d[d['Crime_Type'] == spec['crime_type']]
        if spec['shift']:
            d = d[d['Shift'] == spec['shift']]
        return d

    @staticmethod
    def _what(spec):
        parts = []
        if spec['crime_type']:
            parts.append(spec['crime_type'])
        if spec['shift']:
            parts.append(f"{spec['shift']} shift")
        return ' / '.join(parts) if parts else 'all crimes'

    def execute(self, spec):
        where = spec['area_label'] or 'Larkana (all areas)'
        what = self._what(spec)
        m = spec['metric']

        if m == 'forecast':
            f = self.ai.forecast_df
            if f is None:
                return {'answer': "Forecast is not available yet. Please run `python 06_train_ai_models.py` first.", 'table': None}
            f = f[f['Date'] <= f['Date'].min() + pd.Timedelta(days=spec['horizon'] - 1)]
            if spec['areas']:
                f = f[f['Area'].isin(spec['areas'])]
            tot = f['Forecast'].sum()
            label = 'the next 7 days' if spec['horizon'] == 7 else 'the next 30 days'
            if spec['group_by'] == 'Area' or not spec['areas']:
                t = f.groupby('Area')['Forecast'].sum().round(1).sort_values(ascending=False).reset_index()
                ans = (f"Over {label}, about {tot:.0f} crimes are expected in {where}. "
                       f"Highest: {t.iloc[0]['Area']} ({t.iloc[0]['Forecast']:.0f}).")
                return {'answer': ans, 'table': t}
            return {'answer': f"Over {label}, about {tot:.0f} crimes are expected in {where} (all crime types combined).", 'table': None}

        if m == 'risk':
            if self.ai.bundle is None:
                return {'answer': "The risk model has not been trained yet. Please run `python 06_train_ai_models.py` first.", 'table': None}
            date = self.anchor + pd.Timedelta(days=1)
            t = self.ai.risk_table(date, spec['shift'], with_reasons=False)
            t = t[['Area', 'Risk_Prob', 'Level']].copy()
            t['Risk_Prob'] = (t['Risk_Prob'] * 100).round(1).astype(str) + '%'
            top = t.head(3)['Area'].tolist()
            return {'answer': f"Highest risk for the {spec['shift']} shift on {date.date()}: {', '.join(top)}. "
                              f"(Prioritise these areas for patrols.)", 'table': t}

        d = self._subset(spec)
        days = max((spec['end'] - spec['start']).days + 1, 1)

        if m == 'count':
            n = len(d)
            br = d['Crime_Type'].value_counts()
            ans = f"{n} incidents of {what} were recorded in {where} ({spec['period_label']})."
            if n and not spec['crime_type']:
                ans += " (" + ', '.join(f"{k}: {v}" for k, v in br.items()) + ")"
            return {'answer': ans, 'table': br.rename_axis('Crime_Type').reset_index(name='Count') if n else None}

        if m == 'average':
            n = len(d)
            return {'answer': f"Average for {what} in {where} ({spec['period_label']}): {n / days:.2f} incidents/day "
                              f"({n / days * 7:.1f} per week). Total: {n}.", 'table': None}

        if m == 'ranking':
            g = spec['group_by'] or 'Area'
            if g == 'Weekday':
                s = d['Date'].dt.dayofweek.map(lambda i: DAY_NAMES[i])
            elif g == 'Month':
                s = d['Date'].dt.month.map(lambda i: MONTH_NAMES[i - 1])
            else:
                s = d[g]
            t = s.value_counts().head(spec['top_n']).rename_axis(g).reset_index(name='Count')
            if t.empty:
                return {'answer': "No data found for this filter.", 'table': None}
            return {'answer': f"Top {g} for {what} ({spec['period_label']}): "
                              f"{t.iloc[0][g]} ({t.iloc[0]['Count']} incidents).", 'table': t}

        if m == 'trend':
            span = (spec['end'] - spec['start']).days + 1
            freq = 'W' if span <= 90 else 'ME'
            series = d.set_index('Date').resample(freq).size()
            t = series.rename('Count').reset_index()
            t['Date'] = t['Date'].dt.date.astype(str)
            half = len(series) // 2
            if half >= 1:
                a1, a2 = series.iloc[:half].mean(), series.iloc[half:].mean()
                ch = 100 * (a2 - a1) / a1 if a1 else 0
                if abs(ch) < 2:
                    ans = (f"The trend for {what} in {where} ({spec['period_label']}) is roughly stable "
                           f"(no notable difference between the first and second half, {ch:+.1f}%).")
                else:
                    word = 'increase' if ch > 0 else 'decrease'
                    ans = (f"Trend for {what} in {where} ({spec['period_label']}): a {abs(ch):.0f}% {word} "
                           f"in the second half compared with the first half.")
            else:
                ans = f"Not enough data to show a trend for {what} in {where}."
            return {'answer': ans, 'table': t}
        return {'answer': HELP, 'table': None}

    # -------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------
    def ask(self, question):
        q = question.strip()
        if not q:
            return {'answer': HELP, 'table': None, 'mode': 'rules', 'spec': None}
        if q.lower() in ('hi', 'hello', 'salam', 'assalam o alaikum', 'assalamualaikum', 'help', 'madad', '?'):
            return {'answer': "Hello! I am the Crime Assistant.\n" + HELP, 'table': None, 'mode': 'rules', 'spec': None}
        spec, mode = self.llm_spec(q), 'claude'
        if spec is None:
            spec, mode = self.parse_rules(q), 'rules'
            nothing = (spec['metric'] == 'count' and not spec['areas'] and not spec['crime_type'] and not spec['shift']
                       and spec['period_label'].startswith('entire dataset'))
            count_words = {'kitni', 'kitne', 'kitna', 'many', 'total', 'count', 'number', 'crimes', 'crime', 'incidents'}
            if nothing and not (set(_tokens(q)) & count_words):
                return {'answer': "Sorry, I could not understand that question.\n" + HELP, 'table': None, 'mode': mode, 'spec': spec}
        out = self.execute(spec)
        out['mode'], out['spec'] = mode, spec
        return out
