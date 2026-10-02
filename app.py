"""
app.py : Walmart Sales Forecasting dashboard (Streamlit)

Needs backend.py in the same folder (unchanged).
Run:  streamlit run app.py
"""

import io
import sys
import html
import asyncio
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backend as be

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="Walmart Sales Forecasting", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================================= #
# THEME
# ============================================================================= #

DARK = dict(
    bg="#0A1020", surface="#111A2E", surface2="#17233D", border="#25344F",
    text="#E8EEF9", muted="#9AABC7", primary="#3B82F6", primary2="#2563EB",
    accent="#FFC220", success="#22C55E", warn="#F59E0B", danger="#F87171",
    success_bg="rgba(34,197,94,.13)", warn_bg="rgba(245,158,11,.13)",
    danger_bg="rgba(248,113,113,.13)", info_bg="rgba(59,130,246,.13)",
    shadow="0 2px 10px rgba(0,0,0,.35)", shadow_hover="0 12px 30px rgba(0,0,0,.5)",
    grid="rgba(148,163,184,.16)", ring="rgba(59,130,246,.35)", glow="rgba(59,130,246,.35)",
)
LIGHT = dict(
    bg="#F3F6FB", surface="#FFFFFF", surface2="#F1F5FB", border="#DCE3EF",
    text="#0F172A", muted="#5A6B85", primary="#0071CE", primary2="#005BA8",
    accent="#F5A800", success="#16A34A", warn="#D97706", danger="#DC2626",
    success_bg="rgba(22,163,74,.10)", warn_bg="rgba(217,119,6,.10)",
    danger_bg="rgba(220,38,38,.09)", info_bg="rgba(0,113,206,.09)",
    shadow="0 2px 10px rgba(15,23,42,.07)", shadow_hover="0 12px 28px rgba(15,23,42,.14)",
    grid="rgba(100,116,139,.20)", ring="rgba(0,113,206,.25)", glow="rgba(0,113,206,.30)",
)

dark_mode = st.session_state.get("dark_mode", True)
P = DARK if dark_mode else LIGHT

STATIC_CSS = """
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--text)}
.stApp{font-family:'Inter','Segoe UI',system-ui,sans-serif}
h1,h2,h3,h4,h5,h6,p,label,button,input,textarea,li{font-family:'Inter','Segoe UI',system-ui,sans-serif}
header[data-testid="stHeader"]{background:transparent!important}
[data-testid="stAppDeployButton"],[data-testid="stDeployButton"],[data-testid="stMainMenu"],[data-testid="stToolbarActions"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important}
.block-container,[data-testid="stMainBlockContainer"]{padding:3.6rem 2.4rem 3rem!important;max-width:1500px}
[data-testid="stHorizontalBlock"]{gap:1.25rem}
[data-testid="stVerticalBlock"]{gap:1rem}
hr{border-color:var(--border)!important}

/* text colours that stay readable in both themes */
.stApp p,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp label,.stApp label *{color:var(--text)!important}
[data-testid="stCaptionContainer"] *{color:var(--muted)!important}
[data-testid="stSpinner"] *{color:var(--text)!important}

/* sidebar */
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{background:var(--surface)!important}
section[data-testid="stSidebar"]{border-right:1px solid var(--border)}
[data-testid="stSidebarCollapseButton"] button,[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapsedControl"] button{color:var(--text)!important;border-radius:10px;transition:background .15s}
[data-testid="stExpandSidebarButton"]{background:var(--surface)!important;border:1px solid var(--border)!important;box-shadow:var(--shadow);margin:.4rem 0 0 .4rem}
[data-testid="stSidebarCollapseButton"] button:hover,[data-testid="stExpandSidebarButton"]:hover,[data-testid="stSidebarCollapsedControl"] button:hover{background:var(--surface2)!important}
[data-testid="stSidebarCollapseButton"] *,[data-testid="stExpandSidebarButton"] *,[data-testid="stSidebarCollapsedControl"] *{color:var(--text)!important;fill:var(--text)!important}
.brand{display:flex;align-items:center;gap:.8rem;margin:.2rem 0 .4rem}
.brand-name{font-weight:700;font-size:1.1rem;color:var(--text)}
.brand-sub{font-size:.78rem;color:var(--muted)}
.nav-label{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:700;margin:1.2rem 0 .45rem}

/* top bar */
.topbar{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--accent);border-radius:16px;padding:1rem 1.4rem;box-shadow:var(--shadow);margin-bottom:1.1rem}
.tb-left{display:flex;align-items:center;gap:.9rem}
.tb-title{font-size:1.3rem;font-weight:700;color:var(--text);line-height:1.2}
.tb-sub{font-size:.85rem;color:var(--muted)}
.tb-right{display:flex;gap:.5rem;flex-wrap:wrap}
.brand-mark{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,var(--primary),var(--primary2));color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.25rem;flex-shrink:0}
.chip{display:inline-flex;align-items:center;gap:.45rem;padding:.3rem .8rem;border-radius:999px;font-size:.78rem;font-weight:600;border:1px solid var(--border);background:var(--surface2);color:var(--text);margin:0 .3rem .3rem 0}
.chip .dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.chip.ok .dot{background:var(--success)}
.chip.bad .dot{background:var(--danger)}

/* nav tabs (top navbar) */
[data-testid="stTabs"] [role="tablist"]{gap:.4rem;background:var(--surface);border:1px solid var(--border)!important;border-radius:14px;padding:.35rem;box-shadow:var(--shadow)}
[role="tab"]{height:auto;padding:.65rem 1.4rem;border-radius:10px;background:transparent;border:none!important;box-shadow:none!important;transition:background .15s,transform .15s}
[role="tab"]::after,[role="tab"]::before,[role="tablist"]::after,[role="tablist"]::before{display:none!important}
[role="tab"] p{font-weight:600;font-size:.95rem;color:var(--muted)!important;transition:color .15s}
[role="tab"]:hover{background:var(--surface2);transform:translateY(-1px)}
[role="tab"]:hover p{color:var(--text)!important}
[role="tab"][aria-selected="true"]{background:var(--primary)}
[role="tab"][aria-selected="true"] p{color:#fff!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],[data-testid="stTabs"] [data-baseweb="tab-border"]{display:none!important}
[data-testid="stTabPanel"],div[role="tabpanel"]{padding-top:1.3rem}

/* section headings */
.section-head{display:flex;align-items:center;gap:.7rem;margin:.1rem 0 .2rem}
.section-head .idx{background:var(--primary);color:#fff;font-weight:700;font-size:.78rem;padding:.2rem .55rem;border-radius:8px}
.section-head .stitle{font-size:1.3rem;font-weight:700;color:var(--text)}
.note{color:var(--muted);font-size:.9rem;margin:0 0 .9rem;line-height:1.5}
.panel-title{font-weight:700;font-size:1.05rem;color:var(--text);margin-bottom:.3rem}
.group-label{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--primary);font-weight:700;margin:.9rem 0 .1rem}

/* cards */
.cards{display:grid;grid-template-columns:repeat(var(--n,4),minmax(0,1fr));gap:1rem;margin:.1rem 0 1.1rem}
@media(max-width:1000px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1rem 1.15rem 1rem 1.3rem;box-shadow:var(--shadow);position:relative;overflow:hidden;transition:transform .18s,box-shadow .18s,border-color .18s}
.card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--primary)}
.card.t-accent:before{background:var(--accent)}
.card.t-success:before{background:var(--success)}
.card.t-warn:before{background:var(--warn)}
.card.t-danger:before{background:var(--danger)}
.card:hover{transform:translateY(-3px);border-color:var(--primary);box-shadow:var(--shadow-hover)}
.c-label{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:700}
.c-value{font-size:clamp(1.1rem,1.55vw,1.6rem);font-weight:700;color:var(--text);margin-top:.35rem;line-height:1.2;overflow-wrap:anywhere}
.c-sub{font-size:.78rem;color:var(--muted);margin-top:.3rem}

/* callouts */
.callout{border-radius:12px;padding:.9rem 1.1rem;border:1px solid var(--border);border-left-width:5px;background:var(--surface);margin:.2rem 0 1rem}
.callout .ct{font-weight:700;color:var(--text);font-size:.95rem;margin-bottom:.25rem}
.callout .cb{color:var(--text);opacity:.9;font-size:.9rem;line-height:1.5}
.callout .ca{margin-top:.5rem;font-weight:600;font-size:.88rem;color:var(--text)}
.callout.info{border-left-color:var(--primary);background:var(--info-bg)}
.callout.success{border-left-color:var(--success);background:var(--success-bg)}
.callout.warn{border-left-color:var(--warn);background:var(--warn-bg)}
.callout.danger{border-left-color:var(--danger);background:var(--danger-bg)}

/* forms, inputs */
[data-testid="stForm"]{background:var(--surface);border:1px solid var(--border)!important;border-radius:16px;padding:1.3rem 1.4rem;box-shadow:var(--shadow)}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--border)!important;border-radius:16px}
[data-testid="stNumberInputContainer"],[data-testid="stDateInputField"],[data-testid="stTextInputRootElement"],[data-testid="stSelectbox"] [role="group"],[data-testid="stMultiSelect"] [role="group"],div[data-baseweb="input"],div[data-baseweb="select"]>div{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:10px!important;box-shadow:none!important;transition:border-color .15s,box-shadow .15s}
[data-testid="stNumberInputContainer"]:hover,[data-testid="stDateInputField"]:hover,[data-testid="stTextInputRootElement"]:hover,[data-testid="stSelectbox"] [role="group"]:hover,[data-testid="stMultiSelect"] [role="group"]:hover{border-color:var(--primary)!important}
[data-testid="stNumberInputContainer"]:focus-within,[data-testid="stDateInputField"]:focus-within,[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stSelectbox"] [role="group"]:focus-within,[data-testid="stMultiSelect"] [role="group"]:focus-within{border-color:var(--primary)!important;box-shadow:0 0 0 3px var(--ring)!important}
input,textarea{background:transparent!important;color:var(--text)!important;-webkit-text-fill-color:var(--text)!important;caret-color:var(--text)}
input::placeholder{color:var(--muted)!important;-webkit-text-fill-color:var(--muted)!important}
[data-testid="stSelectbox"] *{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important}
[data-testid="stSelectbox"] svg,[data-testid="stMultiSelect"] svg:not([title]){fill:var(--muted)!important}
[data-testid="stNumberInput"] button{background:var(--surface)!important;color:var(--text)!important;border:none!important;border-left:1px solid var(--border)!important;transition:background .15s}
[data-testid="stNumberInput"] button *{color:var(--text)!important;fill:var(--text)!important}
[data-testid="stNumberInput"] button:hover{background:var(--primary)!important}
[data-testid="stNumberInput"] button:hover *{color:#fff!important;fill:#fff!important}
[data-testid="stSelectboxVirtualDropdown"],[data-testid="stSelectboxVirtualDropdown"] *,[data-testid="stDateInputCalendar"],[data-testid="stDateInputCalendar"] *{background-color:var(--surface)!important;color:var(--text)!important;-webkit-text-fill-color:var(--text)!important}
[data-testid="stSelectboxVirtualDropdown"],[data-testid="stDateInputCalendar"]{border:1px solid var(--border)!important;border-radius:12px!important;box-shadow:var(--shadow-hover)!important}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover *,[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] *{background-color:var(--surface2)!important}
[data-testid="stDateInputCalendar"] [aria-selected="true"],[data-testid="stDateInputCalendar"] [aria-selected="true"] *{background-color:var(--primary)!important;color:#fff!important;-webkit-text-fill-color:#fff!important}
[data-testid="stDateInputCalendar"] [aria-disabled="true"],[data-testid="stDateInputCalendar"] [aria-disabled="true"] *{color:var(--muted)!important;-webkit-text-fill-color:var(--muted)!important}
[data-testid="stDateInputCalendar"] button:hover,[data-testid="stDateInputCalendar"] [role="gridcell"]:hover,[data-testid="stDateInputCalendar"] [role="gridcell"]:hover *{background-color:var(--surface2)!important}
[data-testid="stDateInputField"] *{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important}
span[data-baseweb="tag"]{background:var(--primary)!important;border-radius:8px}
span[data-baseweb="tag"] *{color:#fff!important}
[data-testid="stSliderThumbValue"]{color:var(--primary)!important;font-weight:700}
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:var(--muted)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--surface2)!important;border:1.5px dashed var(--border)!important;border-radius:12px;transition:border-color .15s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--primary)!important}
[data-testid="stFileUploaderDropzone"] *,[data-testid="stFileUploaderFile"] *{color:var(--text)!important}
[data-testid="stFileUploaderDropzone"] small{color:var(--muted)!important}

/* buttons */
button[data-testid^="stBaseButton-primary"]{background:linear-gradient(135deg,var(--primary),var(--primary2))!important;border:none!important;border-radius:10px!important;padding:.6rem 1.5rem!important;box-shadow:0 4px 14px var(--glow);transition:transform .15s,box-shadow .15s,filter .15s}
button[data-testid^="stBaseButton-primary"] *{color:#fff!important;font-weight:600}
button[data-testid^="stBaseButton-primary"]:hover{transform:translateY(-2px);box-shadow:0 10px 24px var(--glow);filter:brightness(1.1)}
button[data-testid^="stBaseButton-primary"]:active{transform:translateY(0)}
button[data-testid^="stBaseButton-secondary"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:.55rem 1.3rem!important;transition:transform .15s,border-color .15s,box-shadow .15s}
button[data-testid^="stBaseButton-secondary"] *{color:var(--text)!important;font-weight:600}
button[data-testid^="stBaseButton-secondary"]:hover{border-color:var(--primary)!important;transform:translateY(-2px);box-shadow:var(--shadow-hover)}
button[data-testid^="stBaseButton-secondary"]:hover *{color:var(--primary)!important}

/* charts */
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]){background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:.5rem;box-shadow:var(--shadow);overflow:hidden;box-sizing:border-box;transition:border-color .2s,box-shadow .2s}
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]):hover{border-color:var(--primary);box-shadow:var(--shadow-hover)}
[data-testid="stPlotlyChart"],[data-testid="stPlotlyChart"]>div,[data-testid="stPlotlyChart"] .js-plotly-plot,[data-testid="stPlotlyChart"] .plot-container{overflow:hidden!important;max-width:100%;box-sizing:border-box}
"""

CSS = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"
       ":root{" + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in P.items()) + "}" + STATIC_CSS)
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ============================================================================= #
# SMALL UI HELPERS
# ============================================================================= #

def h(x):
    """Escape text for HTML, and neutralise $ so Markdown never treats it as maths."""
    return html.escape(str(x)).replace("$", "&#36;")


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def fmt_money(x, d=0):
    return f"${x:,.{d}f}"


def section(idx, title, text):
    md(f'<div class="section-head"><span class="idx">{idx}</span><span class="stitle">{h(title)}</span></div>'
       f'<div class="note">{h(text)}</div>')


def note(text):
    md(f'<div class="note">{h(text)}</div>')


def cards(items):
    """items: list of (label, value, sub, tone). tone in '', accent, success, warn, danger."""
    out = f'<div class="cards" style="--n:{len(items)}">'
    for label, value, sub, tone in items:
        cls = f" t-{tone}" if tone else ""
        out += (f'<div class="card{cls}"><div class="c-label">{h(label)}</div>'
                f'<div class="c-value">{h(value)}</div>'
                + (f'<div class="c-sub">{h(sub)}</div>' if sub else "") + "</div>")
    md(out + "</div>")


def callout(level, title, body, action=None):
    out = f'<div class="callout {level}"><div class="ct">{h(title)}</div><div class="cb">{h(body)}</div>'
    if action:
        out += f'<div class="ca">{h(action)}</div>'
    md(out + "</div>")


def chip(text, ok=None):
    cls = "" if ok is None else (" ok" if ok else " bad")
    return f'<span class="chip{cls}"><span class="dot"></span>{h(text)}</span>'


def style_fig(fig, title, height=340, legend=False, xtitle=None, ytitle=None):
    """Apply the active theme to a Plotly figure so it is readable in light and dark mode."""
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color=P["text"])),
        height=height, margin=dict(t=56, b=18, l=14, r=14),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=P["text"], size=12),
        hoverlabel=dict(bgcolor=P["surface2"], bordercolor=P["border"], font=dict(color=P["text"])),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=P["text"]), bgcolor="rgba(0,0,0,0)"),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=P["muted"], activecolor=P["primary"]),
    )
    axis = dict(gridcolor=P["grid"], linecolor=P["border"], zerolinecolor=P["grid"],
                tickfont=dict(color=P["muted"]), title_font=dict(color=P["muted"]))
    fig.update_xaxes(title_text=xtitle, automargin=True, **axis)
    fig.update_yaxes(title_text=ytitle, automargin=True, **axis)
    return fig


def show(fig, key):
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


# ============================================================================= #
# CACHED DATA HELPERS
# ============================================================================= #

@st.cache_data(show_spinner=False)
def get_kpis():
    return be.dataset_kpis()


@st.cache_data(show_spinner=False)
def get_analytics(codes):
    df, err = be.load_master_data()
    if err or df is None:
        return None, err
    view = df[df["Type"].isin(codes)] if codes else df
    if len(view) == 0:
        view = df
    v = view.copy()

    trend = v.groupby("Date", observed=True)["Weekly_Sales"].mean().reset_index().sort_values("Date")

    v["TypeName"] = v["Type"].map(be.CODE_TO_TYPE)
    v["Holiday"] = np.where(v["IsHoliday"] == 1, "Holiday", "Regular")
    grp = v.groupby(["TypeName", "Holiday"], observed=True)["Weekly_Sales"].mean().reset_index()
    hol = {"types": [], "holiday": [], "regular": []}
    for t in ["A", "B", "C"]:
        sub = grp[grp["TypeName"] == t]
        if len(sub) == 0:
            continue
        hv = sub[sub["Holiday"] == "Holiday"]["Weekly_Sales"]
        rv = sub[sub["Holiday"] == "Regular"]["Weekly_Sales"]
        hol["types"].append(f"Type {t}")
        hol["holiday"].append(float(hv.iloc[0]) if len(hv) else 0.0)
        hol["regular"].append(float(rv.iloc[0]) if len(rv) else 0.0)
    hm = float(v.loc[v["IsHoliday"] == 1, "Weekly_Sales"].mean())
    rm = float(v.loc[v["IsHoliday"] == 0, "Weekly_Sales"].mean())
    hol["uplift"] = (hm / rm - 1.0) * 100.0 if rm and np.isfinite(hm) else 0.0

    factors = {"Store Size": "Size", "Temperature": "Temperature", "Fuel Price": "Fuel_Price",
               "CPI (cost of living)": "CPI", "Unemployment": "Unemployment"}
    drivers = []
    for label, col in factors.items():
        c = float(v["Weekly_Sales"].corr(v[col]))
        drivers.append((label, c if np.isfinite(c) else 0.0))
    drivers.sort(key=lambda x: x[1])
    return {"n": len(v), "trend": trend, "holiday": hol, "drivers": drivers}, None


@st.cache_data(show_spinner=False)
def get_accuracy(codes):
    try:
        fp = pd.read_csv(be.resolve_data_path("final_sales_predictions.csv"))
    except FileNotFoundError:
        return None
    if codes and "Type" in fp.columns:
        fp = fp[fp["Type"].isin(codes)]
    fp = fp.dropna(subset=["Actual_Sales", "Predicted_Sales"])
    if len(fp) == 0:
        return None
    a = fp["Actual_Sales"].to_numpy(float)
    p = fp["Predicted_Sales"].to_numpy(float)
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(np.sum((a - p) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    s = fp.sample(n=min(2500, len(fp)), random_state=42)
    return {"n": len(fp), "mae": float(np.mean(np.abs(a - p))),
            "rmse": float(np.sqrt(np.mean((a - p) ** 2))), "r2": r2,
            "actual": s["Actual_Sales"].to_numpy(float), "pred": s["Predicted_Sales"].to_numpy(float)}


DEPT_NAMES = {
    1: "Candy & Tobacco", 2: "Health & Beauty", 3: "Stationery & Crafts",
    5: "Electronics & Tech", 38: "Pharmacy & OTC", 92: "Grocery & Food", 95: "DSD Grocery",
}


def dept_label(i):
    return f"Dept {i} - {DEPT_NAMES.get(i, 'General Category')}"


TYPE_FILTER_LABELS = {"A": "Type A : Supercenter", "B": "Type B : Discount Store",
                      "C": "Type C : Neighborhood Market"}

# ============================================================================= #
# SIDEBAR (collapsible with the arrow at its top)
# ============================================================================= #

model, model_err = be.load_model()
kpi = get_kpis()

with st.sidebar:
    md('<div class="brand"><div class="brand-mark">W</div><div><div class="brand-name">Walmart</div>'
       '<div class="brand-sub">Sales Forecasting</div></div></div>')

    md('<div class="nav-label">Appearance</div>')
    st.toggle("Dark mode", value=True, key="dark_mode")

    md('<div class="nav-label">Global filters</div>')
    picked = st.multiselect("Store types", ["A", "B", "C"], default=["A", "B", "C"], key="types",
                            format_func=lambda t: TYPE_FILTER_LABELS[t])
    note("Applies to the Analytics charts. Leave empty to include every type.")

    md('<div class="nav-label">System status</div>')
    md(chip("Model is Ready" if model is not None else "Model is Missing", model is not None)
       + chip("Data is Loaded" if kpi else "Data is Missing", bool(kpi)))

    md('<div class="nav-label">About</div>')
    note("Random Forest regressor on a 17 feature contract. Forecasts, inventory guidance and "
         "staffing plans are all generated from the same model.")

# ============================================================================= #
# TOP BAR + KPI STRIP
# ============================================================================= #

md('<div class="topbar"><div class="tb-left"><div class="brand-mark">W</div><div>'
   '<div class="tb-title">Walmart Sales Forecasting</div>'
   '<div class="tb-sub">Forecast demand, monitor operations, and understand sales drivers.</div></div></div>'
   '<div class="tb-right">'
   + chip("Model ready" if model is not None else "Model missing", model is not None)
   + chip("Data loaded" if kpi else "Data missing", bool(kpi))
   + chip(dt.date.today().strftime("%d %b %Y"))
   + "</div></div>")

if kpi:
    cards([
        ("Records", f"{kpi['records']:,}", "Weekly store/dept rows", ""),
        ("Stores / Depts", f"{kpi['stores']} / {kpi['depts']}", "In the reference data", "accent"),
        ("Avg weekly sales", fmt_money(kpi["avg_weekly"]), "Per store and department", "success"),
        ("Total sales", f"${kpi['total_sales'] / 1e9:,.2f}B", "Across the whole period", ""),
        ("Period", f"{kpi['date_min']:%Y} - {kpi['date_max']:%Y}",
         f"{kpi['date_min']:%b %Y} to {kpi['date_max']:%b %Y}", "accent"),
    ])
else:
    callout("info", "Reference data not found",
            "Put train.csv, features.csv and stores.csv next to app.py to enable KPIs, analytics and the "
            "bundled reference dataset. Forecasting still works without them.")

tab_fc, tab_an, tab_ops = st.tabs(["Forecast", "Analytics", "Model Operations"])

# ============================================================================= #
# TAB 1 : FORECAST
# ============================================================================= #

with tab_fc:
    section("01", "Forecast and Operations",
            "Build a scenario for any upcoming week and review expected sales, stock guidance and staffing.")
    left, right = st.columns([1, 1.55], gap="large")

    with left:
        with st.form("forecast_form"):
            md('<div class="panel-title">Scenario inputs</div>')

            md('<div class="group-label">Store profile</div>')
            a, b = st.columns(2)
            store = a.number_input("Store ID (1 - 45)", 1, 45, 1, 1)
            dept = b.selectbox("Department", list(range(1, 100)), index=4, format_func=dept_label)
            store_type = a.selectbox("Store type", list(be.TYPE_LABELS.keys()),
                                     format_func=lambda c: be.TYPE_LABELS[c])
            size = b.number_input("Store size (sq ft)", 1000, 250000, 150000, 5000)

            md('<div class="group-label">Forecast timing</div>')
            a, b = st.columns(2)
            date = a.date_input("Forecast week", dt.date.today())
            is_holiday = b.selectbox("Holiday week", ["No", "Yes"]) == "Yes"

            md('<div class="group-label">Economic conditions</div>')
            a, b = st.columns(2)
            temp = a.number_input("Temperature (°F)", -10.0, 110.0, 60.0, 0.5, format="%.1f")
            fuel = b.number_input("Fuel price ($/gal)", 1.0, 6.0, 3.5, 0.01, format="%.2f")
            cpi = a.number_input("CPI index", 100.0, 260.0, 211.0, 0.1, format="%.1f")
            unemp = b.number_input("Unemployment (%)", 1.0, 20.0, 8.0, 0.1, format="%.1f")

            md('<div class="group-label">Inventory position</div>')
            stock = st.number_input("Current stock value ($)", 0, value=20000, step=1000)

            run = st.form_submit_button("Run forecast", type="primary")
        note("Promotion inputs (MarkDowns) default to zero, matching the model baseline.")

    if run:
        if model is None:
            st.session_state["fc"] = {"error": model_err or "Model unavailable."}
        else:
            hol = 1 if is_holiday else 0
            wk = int(date.isocalendar()[1])
            row = be.build_feature_row(store, dept, store_type, size, hol, temp, fuel, cpi, unemp,
                                       date.year, date.month, wk)
            pred = float(be.model_predict(model, row)[0])

            # 12 week outlook with the same conditions (holiday flag applies to the first week only)
            dates = [date + dt.timedelta(days=7 * i) for i in range(12)]
            frames = [be.build_feature_row(store, dept, store_type, size, hol if i == 0 else 0,
                                           temp, fuel, cpi, unemp, d.year, d.month, int(d.isocalendar()[1]))
                      for i, d in enumerate(dates)]
            outlook = be.model_predict(model, pd.concat(frames, ignore_index=True))

            ctx = None
            df, err = be.load_master_data()
            if not err and df is not None:
                hist = df[(df["Store"] == store) & (df["Dept"] == dept)]
                if len(hist) > 0:
                    ha = float(hist["Weekly_Sales"].mean())
                    ctx = {"avg": ha, "weeks": len(hist), "delta": (pred / ha - 1) * 100 if ha else 0.0}

            st.session_state["fc"] = {
                "pred": pred, "year": date.year, "month": date.month, "week": wk,
                "inv": be.inventory_status(stock, pred), "stf": be.staffing_plan(pred, hol),
                "stock": float(stock), "ctx": ctx, "dates": dates, "outlook": [float(x) for x in outlook],
                "type_label": be.TYPE_LABELS[store_type], "store": store, "dept": dept,
            }

    with right:
        md('<div class="panel-title">Operational outputs</div>')
        fc = st.session_state.get("fc")
        if not fc:
            callout("info", "Ready for a scenario",
                    "Fill in the store, timing and conditions, then press Run forecast to see expected sales, "
                    "stock guidance, staffing and a 12 week outlook.")
        elif "error" in fc:
            callout("danger", "Forecast unavailable", fc["error"])
        else:
            ctx, inv, stf = fc["ctx"], fc["inv"], fc["stf"]
            first = ("Predicted weekly sales", fmt_money(fc["pred"], 2),
                     f"Store {fc['store']} · Dept {fc['dept']}", "")
            if ctx:
                second = ("Vs store/dept average", f"{ctx['delta']:+.1f}%",
                          f"Average {fmt_money(ctx['avg'])} over {ctx['weeks']} weeks",
                          "success" if ctx["delta"] >= 0 else "warn")
            else:
                second = ("Store type", fc["type_label"].split(":")[0], fc["type_label"], "accent")
            cards([first, second,
                   ("Forecast period", f"Week {fc['week']} · {fc['year']}", f"Month {fc['month']}", "accent")])

            level = {"alert": "danger", "warning": "warn", "success": "success"}[inv["level"]]
            callout(level, inv["status"], inv["detail"], "Recommended action: " + inv["action"])

            c1, c2 = st.columns(2)
            with c1:
                vals = [fc["stock"], fc["pred"], inv["buffer_high"]]
                fig = go.Figure(go.Bar(
                    x=["Stock", "Forecast", "Buffer +20%"], y=vals,
                    marker_color=[P["primary"], P["accent"], P["success"]],
                    text=[fmt_money(v) for v in vals], textposition="outside",
                    textfont=dict(color=P["text"], size=11), cliponaxis=False))
                style_fig(fig, "Stock vs forecast demand", ytitle="Dollars")
                fig.update_yaxes(range=[0, max(vals) * 1.22])
                show(fig, "fc_stock")
            with c2:
                fig = go.Figure(go.Pie(
                    labels=["Cashiers", "Floor staff"], values=[stf["cashiers"], stf["floor"]], hole=0.58,
                    marker=dict(colors=[P["primary"], P["accent"]], line=dict(color=P["surface"], width=3)),
                    textinfo="value", textposition="inside", insidetextfont=dict(color="#FFFFFF", size=15)))
                style_fig(fig, f"Staffing plan : {stf['total']} workers", legend=True)
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5,
                                              font=dict(color=P["text"]), bgcolor="rgba(0,0,0,0)"))
                show(fig, "fc_staff")

            cards([("Total staff", stf["total"], "1 worker per $5,000 forecast", ""),
                   ("Cashiers", stf["cashiers"], "About 35% of the team", "accent"),
                   ("Floor staff", stf["floor"], "Remaining 65%", "success")])
            callout("warn" if stf["surge"] else "info",
                    "Surge staffing recommended" if stf["surge"] else "Standard coverage", stf["reason"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fc["dates"], y=fc["outlook"], mode="lines+markers", name="Forecast",
                                     line=dict(color=P["primary"], width=3),
                                     marker=dict(size=7, color=P["primary"]),
                                     fill="tozeroy", fillcolor=P["info_bg"]))
            if ctx:
                fig.add_hline(y=ctx["avg"], line_dash="dash", line_color=P["accent"],
                              annotation_text="Historical average", annotation_font_color=P["text"])
            style_fig(fig, "12 week outlook (same conditions)", height=360, ytitle="Predicted weekly sales ($)")
            show(fig, "fc_outlook")

# ============================================================================= #
# TAB 2 : ANALYTICS
# ============================================================================= #

with tab_an:
    section("02", "Sales Analytics",
            "Understand how sales moved over time, how holidays lift demand, and which factors matter most.")
    codes = tuple(sorted(be.TYPE_TO_CODE[t] for t in picked))
    data, err = get_analytics(codes)

    if data is None:
        callout("warn", "Analytics unavailable", str(err or "Reference data could not be loaded."))
    else:
        shown = ", ".join(f"Type {t}" for t in picked) if picked else "all store types"
        note(f"Showing {shown} · {data['n']:,} records. Change the filter from the sidebar.")

        t = data["trend"]
        fig = go.Figure(go.Scatter(x=t["Date"], y=t["Weekly_Sales"], mode="lines",
                                   line=dict(color=P["primary"], width=2.5),
                                   fill="tozeroy", fillcolor=P["info_bg"], name="Avg weekly sales"))
        style_fig(fig, "Average weekly sales over time", height=380, ytitle="Avg weekly sales ($)")
        show(fig, "an_trend")

        c1, c2 = st.columns(2)
        with c1:
            hd = data["holiday"]
            fig = go.Figure()
            fig.add_bar(name="Regular", x=hd["types"], y=hd["regular"], marker_color=P["primary"])
            fig.add_bar(name="Holiday", x=hd["types"], y=hd["holiday"], marker_color=P["accent"])
            fig.update_layout(barmode="group")
            style_fig(fig, "Holiday vs regular weeks", height=380, legend=True, ytitle="Avg weekly sales ($)")
            show(fig, "an_holiday")
            note(f"Holiday weeks average {hd['uplift']:+.1f}% compared with regular weeks.")
        with c2:
            labels = [d[0] for d in data["drivers"]]
            vals = [d[1] for d in data["drivers"]]
            fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                                   marker_color=[P["danger"] if v < 0 else P["success"] for v in vals],
                                   text=[f"{v:+.2f}" for v in vals], textposition="outside",
                                   textfont=dict(color=P["text"]), cliponaxis=False))
            style_fig(fig, "What moves sales the most", height=380, xtitle="Correlation with weekly sales")
            fig.update_xaxes(range=[min(-0.1, min(vals) * 1.4), max(0.1, max(vals) * 1.4)])
            show(fig, "an_drivers")
            note("Bars to the right mean the factor rises with sales. Bars to the left mean it falls as sales rise.")

    md('<div class="group-label">Model accuracy</div>')
    acc = get_accuracy(codes)
    if acc is None:
        callout("info", "Accuracy data not found",
                "Place final_sales_predictions.csv next to app.py to see predicted vs actual.")
    else:
        cards([("R² score", f"{acc['r2']:.3f}", "Closer to 1 is better", "success"),
               ("MAE", fmt_money(acc["mae"]), "Average miss per week", "accent"),
               ("RMSE", fmt_money(acc["rmse"]), "Penalises large misses", ""),
               ("Weeks evaluated", f"{acc['n']:,}", "Rows in the results file", "accent")])
        lim = float(max(acc["actual"].max(), acc["pred"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=acc["actual"], y=acc["pred"], mode="markers", name="Weeks",
                                   marker=dict(size=5, opacity=0.5, color=P["primary"])))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect prediction",
                                 line=dict(color=P["accent"], dash="dash", width=2)))
        style_fig(fig, "Model accuracy: predicted vs actual", height=460, legend=True,
                  xtitle="Actual sales ($)", ytitle="Predicted sales ($)")
        show(fig, "an_accuracy")
        note("Each point is one past week. The closer points sit to the dashed line, the more accurate the model was.")

# ============================================================================= #
# TAB 3 : MODEL OPERATIONS
# ============================================================================= #

def read_training_source(source, upload):
    if source.startswith("Upload"):
        if upload is None:
            return None, None, "Please upload a .csv or .xlsx file first."
        try:
            name = upload.name.lower()
            raw = io.BytesIO(upload.getvalue())
            if name.endswith(".csv"):
                return pd.read_csv(raw), upload.name, None
            if name.endswith((".xlsx", ".xls")):
                return pd.read_excel(raw), upload.name, None
            return None, None, "Unsupported file type. Use .csv or .xlsx."
        except Exception as exc:
            return None, None, f"Could not read the file: {exc}"
    ref, err = be.load_master_data()
    if err or ref is None:
        return None, None, f"Reference dataset unavailable. {err or ''}"
    cols = [c for c in (be.FEATURE_ORDER + [be.TARGET_COL]) if c in ref.columns]
    return ref[cols].copy(), "Bundled reference dataset", None


with tab_ops:
    section("03", "Model Operations",
            "Validate a clean training dataset, retrain the Random Forest on new data, and review performance.")

    with st.container(border=True):
        md('<div class="panel-title">Training data source</div>')
        source = st.radio("Source", ["Upload a file (.csv or .xlsx)", "Use the bundled reference dataset"],
                          horizontal=True, label_visibility="collapsed")
        upload = None
        if source.startswith("Upload"):
            upload = st.file_uploader("Clean data file", type=["csv", "xlsx", "xls"])
            note("Required columns: " + ", ".join(be.FEATURE_ORDER + [be.TARGET_COL])
                 + ". Type may be A/B/C. MarkDowns are set to 0 when missing, and Year/Month/Week are derived "
                   "from a Date column when missing.")
        validate = st.button("Validate data")

        if validate:
            raw_df, src, lerr = read_training_source(source, upload)
            if lerr:
                callout("danger", "Cannot read data", lerr)
            else:
                cards([("Rows", f"{raw_df.shape[0]:,}", "", ""),
                       ("Columns", raw_df.shape[1], "", "accent"),
                       ("Missing cells", f"{int(raw_df.isnull().sum().sum()):,}", "", "warn"),
                       ("Duplicate rows", f"{int(raw_df.duplicated().sum()):,}", "", "")])
                try:
                    _, _, rep = be.prepare_training_frame(raw_df)
                    extra = f" Auto-filled: {', '.join(rep['derived'])}." if rep["derived"] else ""
                    callout("success", "Data is valid",
                            f"{rep['rows_used']:,} rows usable, {rep['rows_dropped']:,} dropped.{extra}")
                except Exception as exc:
                    callout("danger", "Schema invalid", str(exc))

    with st.container(border=True):
        md('<div class="panel-title">Retraining controls</div>')
        r1, r2, r3 = st.columns(3)
        n_est = r1.slider("Trees", 50, 300, 120, 10)
        depth = r2.slider("Max depth (0 = unlimited)", 0, 40, 0, 1)
        test_size = r3.slider("Holdout test size", 0.10, 0.40, 0.20, 0.05)
        replace = st.checkbox("Replace the saved model with the newly trained one", value=True)
        retrain = st.button("Retrain model", type="primary")
        note("Retraining runs where the app runs and can take a while on the full dataset.")

        if retrain:
            raw_df, src, lerr = read_training_source(source, upload)
            if lerr:
                callout("danger", "Cannot read data", lerr)
            else:
                X = None
                try:
                    X, y, rep = be.prepare_training_frame(raw_df)
                except Exception as exc:
                    callout("danger", "Schema invalid", str(exc))
                if X is not None:
                    with st.spinner("Training the model..."):
                        try:
                            import os
                            import joblib
                            from sklearn.ensemble import RandomForestRegressor
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

                            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
                            nm = RandomForestRegressor(n_estimators=n_est,
                                                       max_depth=None if depth == 0 else depth,
                                                       random_state=42, n_jobs=-1)
                            nm.fit(Xtr, ytr)
                            pr = nm.predict(Xte)
                            st.session_state["train"] = {
                                "src": src, "r2": float(r2_score(yte, pr)),
                                "mae": float(mean_absolute_error(yte, pr)),
                                "rmse": float(np.sqrt(mean_squared_error(yte, pr))),
                                "ntr": len(Xtr), "nte": len(Xte), "saved": replace,
                                "imp": sorted(zip(be.FEATURE_ORDER, nm.feature_importances_), key=lambda r: r[1]),
                                "actual": yte.to_numpy()[:2500], "pred": pr[:2500],
                            }
                            if replace:
                                os.makedirs(os.path.dirname(be.MODEL_PATH), exist_ok=True)
                                joblib.dump(nm, be.MODEL_PATH)
                                be.load_model.cache_clear()
                        except Exception as exc:
                            callout("danger", "Retraining failed", str(exc))

    tr = st.session_state.get("train")
    if tr:
        callout("success", "Training complete",
                f"Trained on {tr['src']} ({tr['ntr']:,} train rows, {tr['nte']:,} test rows). "
                + ("The saved model was replaced and new forecasts use it." if tr["saved"]
                   else "The model was not saved because the checkbox was off."))
        cards([("R² score", f"{tr['r2']:.3f}", "On the holdout set", "success"),
               ("MAE", fmt_money(tr["mae"]), "Average miss per week", "accent"),
               ("RMSE", fmt_money(tr["rmse"]), "Penalises large misses", "")])
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Bar(x=[v for _, v in tr["imp"]], y=[f for f, _ in tr["imp"]],
                                   orientation="h", marker_color=P["primary"]))
            style_fig(fig, "Feature importance", height=500, xtitle="Importance")
            show(fig, "ops_importance")
        with g2:
            lim = float(max(tr["actual"].max(), tr["pred"].max()))
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=tr["actual"], y=tr["pred"], mode="markers", name="Test rows",
                                       marker=dict(size=5, opacity=0.5, color=P["primary"])))
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect prediction",
                                     line=dict(color=P["accent"], dash="dash", width=2)))
            style_fig(fig, "Holdout: predicted vs actual", height=500, legend=True,
                      xtitle="Actual ($)", ytitle="Predicted ($)")
            show(fig, "ops_holdout")

note("Walmart Sales Forecasting · Random Forest inference · Data Analytics · Model Operations")