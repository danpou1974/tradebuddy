"""
themes.py
=========
Definición de temas visuales para el dashboard.
Oscuro (predeterminado) y Claro Celeste.
Incluye CSS responsive para dispositivos móviles.
"""

THEMES = {
    "🌙 Oscuro": {
        "base": "dark",
        "primaryColor":       "#1D9E75",
        "backgroundColor":    "#0e1117",
        "secondaryBackgroundColor": "#1a1d27",
        "textColor":          "#e0e0e0",
        # Plotly
        "plot_bg":            "rgba(14,17,23,1)",
        "paper_bg":           "rgba(0,0,0,0)",
        "grid_color":         "rgba(255,255,255,0.05)",
        "tick_color":         "rgba(255,255,255,0.4)",
        "candle_up":          "#26a69a",
        "candle_down":        "#ef5350",
        # Cards / UI
        "card_bg":            "#1a1d27",
        "card_border":        "rgba(255,255,255,0.08)",
        "metric_bg":          "#22263a",
        "sidebar_bg":         "#13151f",
        "text_primary":       "#e8eaf0",
        "text_secondary":     "#8b92a8",
        "text_tertiary":      "#4a5068",
        "divider":            "rgba(255,255,255,0.07)",
        # Accents
        "buy_bg":             "rgba(0,200,83,0.12)",
        "sell_bg":            "rgba(213,0,0,0.12)",
        "neutral_bg":         "rgba(158,158,158,0.10)",
    },
    "☀️ Claro Celeste": {
        "base": "light",
        "primaryColor":       "#0077b6",
        "backgroundColor":    "#f0f7ff",
        "secondaryBackgroundColor": "#ffffff",
        "textColor":          "#1a2744",
        # Plotly
        "plot_bg":            "rgba(240,247,255,1)",
        "paper_bg":           "rgba(255,255,255,0)",
        "grid_color":         "rgba(0,119,182,0.08)",
        "tick_color":         "rgba(26,39,68,0.5)",
        "candle_up":          "#0096c7",
        "candle_down":        "#e63946",
        # Cards / UI
        "card_bg":            "#ffffff",
        "card_border":        "rgba(0,119,182,0.15)",
        "metric_bg":          "#e8f4fd",
        "sidebar_bg":         "#dbeeff",
        "text_primary":       "#1a2744",
        "text_secondary":     "#3d5a80",
        "text_tertiary":      "#7a9cbf",
        "divider":            "rgba(0,119,182,0.12)",
        # Accents
        "buy_bg":             "rgba(0,119,182,0.10)",
        "sell_bg":            "rgba(230,57,70,0.10)",
        "neutral_bg":         "rgba(100,120,150,0.08)",
    },
}


def get_streamlit_config(theme: dict) -> dict:
    return {
        "theme.base":                    theme["base"],
        "theme.primaryColor":            theme["primaryColor"],
        "theme.backgroundColor":         theme["backgroundColor"],
        "theme.secondaryBackgroundColor":theme["secondaryBackgroundColor"],
        "theme.textColor":               theme["textColor"],
    }


def apply_css(theme: dict) -> str:
    tb  = theme["text_primary"]
    ts  = theme["text_secondary"]
    tt  = theme["text_tertiary"]
    cb  = theme["card_bg"]
    cbr = theme["card_border"]
    mb  = theme["metric_bg"]
    sb  = theme["sidebar_bg"]
    div = theme["divider"]
    pr  = theme["primaryColor"]

    return f"""
    <style>
    /* ─── VIEWPORT / BASE ─── */
    .stApp {{
        background-color: {theme['backgroundColor']};
    }}

    /* ─── SIDEBAR ─── */
    [data-testid="stSidebar"] {{
        background-color: {sb} !important;
        border-right: 1px solid {cbr};
    }}

    /* ─── CARDS ─── */
    .hmm-card {{
        background: {cb};
        border: 1px solid {cbr};
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 4px;
    }}
    .hmm-card-title {{
        font-size: 10px;
        color: {ts};
        text-transform: uppercase;
        letter-spacing: .07em;
        font-weight: 600;
        margin-bottom: 12px;
    }}

    /* ─── MÉTRICA ─── */
    .hmm-metric {{
        background: {mb};
        border-radius: 8px;
        padding: 10px 13px;
    }}
    .hmm-metric-label {{
        font-size: 10px;
        color: {ts};
        margin-bottom: 3px;
    }}
    .hmm-metric-value {{
        font-size: 18px;
        font-weight: 600;
        color: {tb};
    }}
    .hmm-metric-sub {{
        font-size: 10px;
        margin-top: 3px;
        color: {tt};
    }}

    /* ─── SEÑAL IA ─── */
    .ai-signal-box {{
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
    }}
    .ai-action {{
        font-size: 22px;
        font-weight: 700;
    }}
    .ai-strength {{
        font-size: 12px;
        margin-top: 4px;
        opacity: .85;
    }}
    .ai-conf {{
        font-size: 11px;
        margin-top: 3px;
        opacity: .7;
    }}

    /* ─── INDICADORES ─── */
    .sig-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 3px 0;
        font-size: 10px;
        border-left: 2px solid transparent;
    }}

    /* ─── PROBABILIDADES ─── */
    .prob-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        border-radius: 5px;
        margin: 4px 0;
        border-left: 3px solid transparent;
        font-size: 11px;
    }}

    /* ─── TF PILL ─── */
    .tf-pill {{
        border-radius: 7px;
        padding: 8px 8px;
        text-align: center;
        border-width: 1.5px;
        border-style: solid;
        margin-bottom: 5px;
    }}

    /* ─── RAZONES ─── */
    .reason-item {{
        padding: 6px 10px;
        border-radius: 5px;
        margin: 4px 0;
        font-size: 10px;
        border-left: 3px solid transparent;
        line-height: 1.5;
    }}

    /* ─── DIVIDER ─── */
    .hmm-divider {{
        height: 1px;
        background: {div};
        margin: 8px 0;
    }}

    /* ─── SCORE BAR ─── */
    .score-track {{
        background: {mb};
        border-radius: 6px;
        height: 12px;
        overflow: hidden;
        border: 1px solid {cbr};
    }}

    /* ─── OCULTAR ELEMENTOS INNECESARIOS ─── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ─── BOTONES MÁS GRANDES (TOUCH) ─── */
    .stButton > button {{
        min-height: 42px;
        font-size: 13px;
        border-radius: 8px;
        padding: 6px 12px;
        width: 100%;
        touch-action: manipulation;
    }}

    /* ─── SELECTBOX Y SLIDERS ─── */
    .stSelectbox label, .stMultiSelect label,
    .stSlider label, .stRadio label,
    .stCheckbox label, .stToggle label {{
        font-size: 13px !important;
    }}

    /* ─── SCROLLBAR THIN ─── */
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {cbr}; border-radius: 4px; }}

    /* ═══════════════════════════════════════
       RESPONSIVE — TABLET (≤ 900px)
    ═══════════════════════════════════════ */
    @media screen and (max-width: 900px) {{
        .main .block-container {{
            padding: 0.75rem 0.6rem 2rem !important;
            max-width: 100% !important;
        }}
        .hmm-card {{
            padding: 12px !important;
        }}
        .hmm-metric-value {{
            font-size: 16px !important;
        }}
        .ai-action {{
            font-size: 20px !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 18px !important;
        }}
    }}

    /* ═══════════════════════════════════════
       RESPONSIVE — MÓVIL (≤ 768px)
    ═══════════════════════════════════════ */
    @media screen and (max-width: 768px) {{
        /* Contenedor principal */
        .main .block-container {{
            padding: 0.5rem 0.4rem 3rem !important;
            max-width: 100vw !important;
        }}

        /* Columnas — mínimo razonable para no aplastar */
        [data-testid="column"] {{
            min-width: 130px !important;
        }}

        /* Botones táctiles */
        .stButton > button {{
            min-height: 48px !important;
            font-size: 14px !important;
            border-radius: 10px !important;
        }}

        /* Cards */
        .hmm-card {{
            padding: 10px !important;
            border-radius: 8px !important;
            margin-bottom: 5px !important;
        }}
        .hmm-card-title {{
            font-size: 9px !important;
            margin-bottom: 8px !important;
        }}

        /* Métricas */
        .hmm-metric {{
            padding: 8px 10px !important;
        }}
        .hmm-metric-value {{
            font-size: 15px !important;
        }}
        .hmm-metric-label {{
            font-size: 9px !important;
        }}

        /* Métricas nativas Streamlit */
        [data-testid="stMetric"] {{
            padding: 6px 4px !important;
        }}
        [data-testid="stMetricLabel"] p {{
            font-size: 11px !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 16px !important;
        }}
        [data-testid="stMetricDelta"] {{
            font-size: 11px !important;
        }}

        /* Señal IA */
        .ai-signal-box {{
            padding: 12px 10px !important;
            border-radius: 8px !important;
        }}
        .ai-action {{
            font-size: 18px !important;
        }}
        .ai-strength, .ai-conf {{
            font-size: 11px !important;
        }}

        /* Filas indicadores */
        .sig-row {{
            font-size: 11px !important;
            padding: 6px 8px !important;
        }}

        /* Filas probabilidades */
        .prob-row {{
            font-size: 12px !important;
            padding: 7px 8px !important;
        }}

        /* TF Pills */
        .tf-pill {{
            padding: 6px !important;
        }}

        /* Razones */
        .reason-item {{
            font-size: 11px !important;
            padding: 5px 8px !important;
        }}

        /* Score bar */
        .score-track {{
            height: 10px !important;
        }}

        /* Headers */
        h2 {{ font-size: 1.1rem !important; }}
        h3 {{ font-size: 1rem !important; }}

        /* Gráficos */
        .js-plotly-plot .plotly {{
            max-width: 100% !important;
        }}
        .js-plotly-plot {{
            overflow-x: auto !important;
        }}

        /* DataFrames scrollables */
        .stDataFrame {{
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }}

        /* Toast */
        [data-testid="stToast"] {{
            max-width: 92vw !important;
            font-size: 12px !important;
        }}

        /* Sidebar toggle */
        [data-testid="collapsedControl"] {{
            width: 48px !important;
            height: 48px !important;
        }}

        /* Dividers */
        hr {{ margin: 0.4rem 0 !important; }}

        /* Expanders */
        [data-testid="stExpander"] summary {{
            font-size: 13px !important;
            padding: 10px !important;
        }}

        /* Plotly modebar (toolbar del gráfico) */
        .modebar {{
            right: 2px !important;
            top: 2px !important;
        }}
        .modebar-btn svg {{
            width: 18px !important;
            height: 18px !important;
        }}
    }}

    /* ═══════════════════════════════════════
       RESPONSIVE — MÓVIL PEQUEÑO (≤ 480px)
    ═══════════════════════════════════════ */
    @media screen and (max-width: 480px) {{
        .main .block-container {{
            padding: 0.4rem 0.3rem 3rem !important;
        }}
        .hmm-metric-value {{
            font-size: 14px !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 14px !important;
        }}
        [data-testid="stMetricLabel"] p {{
            font-size: 10px !important;
        }}
        .ai-action {{
            font-size: 16px !important;
        }}
        .sig-row span, .prob-row span {{
            font-size: 10px !important;
        }}
        h2 {{ font-size: 1rem !important; }}
        /* Subheader más compacto */
        .stSubheader {{ font-size: 0.95rem !important; }}
    }}
    </style>
    """
