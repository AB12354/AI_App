import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from huggingface_hub import hf_hub_download
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ── MUST BE FIRST STREAMLIT COMMAND ──
st.set_page_config(
    page_title="VeritasAI - Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

nltk.download('stopwords', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 300
STOP_WORDS  = set(stopwords.words('english'))
stemmer     = PorterStemmer()

DATASETS = ['ISOT', 'GossipCop', 'WELFake', 'LIAR', 'GonzaloA', 'ErfanM', 'Ultimate', 'Combined']

DATASET_INFO = {
    'ISOT':      {'period':'2016-17','rows':'44,898', 'source':'Reuters + Flagged sites',    'tag':'Old'},
    'GossipCop': {'period':'2015-16','rows':'~6,300', 'source':'Entertainment fact-check',   'tag':'Old'},
    'WELFake':   {'period':'2016-20','rows':'72,134', 'source':'IEEE 4 merged sources',      'tag':'Mid'},
    'LIAR':      {'period':'2007-17','rows':'12,800', 'source':'PolitiFact Human labeled',   'tag':'Mid'},
    'GonzaloA':  {'period':'2017-18','rows':'45,000', 'source':'HuggingFace Reuters',        'tag':'New'},
    'ErfanM':    {'period':'2020-21','rows':'44,267', 'source':'HuggingFace Multi-source',   'tag':'New'},
    'Ultimate':  {'period':'2024-25','rows':'50,000', 'source':'HuggingFace Aggregated',     'tag':'New'},
    'Combined':  {'period':'All',    'rows':'~275K',  'source':'All 7 merged',               'tag':'Full'},
}

MODEL_META = {
    'CNN-Text': {'type':'Deep Learning', 'icon':'🔬','color':'#FF2D78','key':'cnn','feat':'seq','desc':'Convolutional text patterns'},
}

TEAM = [
    {'name':'Abdullah Bin Umar','role':'Group Lead · Data Pipeline & EDA',         'emoji':'👨‍💻','id':'F2024-0922'},
    {'name':'Ahmed Ali Qaiser', 'role':'Feature Engineering · TF-IDF & Sequences', 'emoji':'⚙️', 'id':'F2024-1009'},
    {'name':'Abdul Rehman',     'role':'Model Training · 32 Training Runs',         'emoji':'🤖', 'id':'F2024-1237'},
    {'name':'Muhammad Awais',   'role':'Evaluation · Charts & Metrics',             'emoji':'📊', 'id':'F2024-0783'},
]

MEMBER_COLORS = ['#00FF88','#7C3AED','#FFB800','#FF2D78','#00D4FF']

# ── Helper ─────────────────────────────────────────────────────────────────────
def lbl(text):
    return ("<div style='font-size:0.65rem; color:#2A3D5A; text-transform:uppercase; "
            "letter-spacing:0.12em; margin-bottom:0.35rem; font-family:DM Mono,monospace;'>"
            + text + "</div>")

def card(content, border_color=None, extra_style=""):
    left = "border-left:2.5px solid " + border_color + ";" if border_color else ""
    return ("<div style='background:#0A1020; border:1px solid #1A2840; border-radius:11px; "
            "padding:1rem 1.15rem; margin-bottom:0.55rem; " + left + extra_style + "'>"
            + content + "</div>")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

# ── Model Loading from HuggingFace ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    repo = "abjhbjhe/Fake_news"
    loaded = {}
    errors = []

    try:
        cnn_model = tf.keras.models.load_model(hf_hub_download(repo, "cnn_combined.keras"))
    except Exception as e:
        errors.append(f"CNN model: {e}")
        return {ds: {} for ds in DATASETS}, errors

    try:
        tok_path = hf_hub_download(repo, "tokenizer_combined.json")
        with open(tok_path, "r") as f:
            tok_model = tokenizer_from_json(json.load(f))
    except Exception as e:
        errors.append(f"Tokenizer: {e}")
        tok_model = None

    combined = {'cnn': cnn_model, 'tok': tok_model}

    for ds in DATASETS:
        loaded[ds] = combined

    return loaded, errors

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_dataset(text, ds, models):
    m = models.get(ds, {})
    cleaned = clean_text(text)
    preds = {}

    seq_input = None
    if m.get('tok'):
        try:
            seq_input = pad_sequences(
                m['tok'].texts_to_sequences([cleaned]),
                maxlen=MAX_SEQ_LEN, padding='post', truncating='post'
            )
        except: pass

    if m.get('cnn') and seq_input is not None:
        try:
            preds['CNN-Text'] = float(m['cnn'].predict(seq_input, verbose=0).flatten()[0])
        except:
            preds['CNN-Text'] = None
    else:
        preds['CNN-Text'] = None

    return preds, cleaned

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #04080F; --surface: #0A1020; --surface2: #0F1830; --surface3: #152040;
    --border: #1A2840; --border2: #243560; --neon: #00FF88; --neon2: #FF2D78;
    --gold: #FFB800; --purple: #7C3AED; --sky: #00D4FF;
    --text: #F0F6FF; --text2: #6B82A8; --text3: #2A3D5A;
    --font-d: 'Bebas Neue', sans-serif; --font-b: 'Space Grotesk', sans-serif; --font-m: 'DM Mono', monospace;
}
html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--font-b) !important; }
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 55% 45% at 8% 12%, rgba(124,58,237,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 45% 35% at 92% 8%, rgba(0,255,136,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 95%, rgba(0,212,255,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 35% 30% at 5% 85%, rgba(255,45,120,0.06) 0%, transparent 50%) !important;
}
.block-container { padding: 0rem 1.8rem !important; max-width: 1600px !important; }
#MainMenu, footer { visibility: hidden !important; }
header { visibility: visible !important; background: transparent !important; }
[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; color: var(--neon) !important; background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #060C1A 0%, #080F20 100%) !important; border-right: 1px solid var(--border) !important; box-shadow: 6px 0 40px rgba(0,0,0,0.7) !important; padding-top: 0 !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--neon); }
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 4px !important; gap: 2px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text2) !important; border-radius: 9px !important; font-family: var(--font-b) !important; font-weight: 500 !important; font-size: 0.86rem !important; padding: 0.45rem 1.1rem !important; border: none !important; }
.stTabs [aria-selected="true"] { background: rgba(0,255,136,0.1) !important; color: var(--neon) !important; border: 1px solid rgba(0,255,136,0.25) !important; }
.stTextArea textarea { background: var(--surface2) !important; border: 1px solid var(--border2) !important; border-radius: 12px !important; color: var(--text) !important; font-family: var(--font-b) !important; font-size: 0.93rem !important; line-height: 1.65 !important; }
.stTextArea textarea:focus { border-color: var(--neon) !important; box-shadow: 0 0 0 3px rgba(0,255,136,0.08) !important; }
.stTextArea textarea::placeholder { color: var(--text3) !important; }
.stButton > button { font-family: var(--font-b) !important; font-weight: 600 !important; border-radius: 9px !important; transition: all 0.2s !important; width: 100% !important; font-size: 0.88rem !important; padding: 0.55rem 1.1rem !important; }
.stButton:nth-of-type(1) > button { background: linear-gradient(135deg, #00FF88, #00CC6A) !important; color: #010F05 !important; border: none !important; box-shadow: 0 4px 20px rgba(0,255,136,0.3) !important; }
.stButton:nth-of-type(1) > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(0,255,136,0.45) !important; }
.stButton:nth-of-type(2) > button { background: var(--surface2) !important; color: var(--text2) !important; border: 1px solid var(--border2) !important; }
.stButton:nth-of-type(2) > button:hover { border-color: var(--neon2) !important; color: var(--neon2) !important; }
.stSelectbox > div > div { background: var(--surface2) !important; border: 1px solid var(--border2) !important; border-radius: 9px !important; color: var(--text) !important; font-family: var(--font-b) !important; }
.stMultiSelect > div { background: var(--surface2) !important; border: 1px solid var(--border2) !important; border-radius: 9px !important; }
.stMultiSelect span[data-baseweb="tag"] { background: rgba(0,255,136,0.1) !important; border: 1px solid rgba(0,255,136,0.3) !important; border-radius: 6px !important; color: var(--neon) !important; }
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] table { background: var(--surface) !important; font-family: var(--font-m) !important; font-size: 0.79rem !important; }
[data-testid="stDataFrame"] th { background: var(--surface3) !important; color: var(--neon) !important; font-weight: 600 !important; letter-spacing: 0.07em !important; text-transform: uppercase !important; font-size: 0.71rem !important; border-bottom: 1px solid var(--border2) !important; }
[data-testid="stDataFrame"] td { color: var(--text) !important; }
.streamlit-expanderHeader { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 9px !important; color: var(--text2) !important; font-size: 0.87rem !important; }
.streamlit-expanderContent { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 9px 9px !important; }
.stDownloadButton > button { background: rgba(0,255,136,0.05) !important; color: var(--neon) !important; border: 1px solid rgba(0,255,136,0.2) !important; border-radius: 9px !important; font-weight: 500 !important; font-size: 0.84rem !important; }
.stDownloadButton > button:hover { background: rgba(0,255,136,0.1) !important; }
.stSpinner > div { border-top-color: var(--neon) !important; }
.stWarning { background: rgba(255,184,0,0.07) !important; border-color: rgba(255,184,0,0.3) !important; }
.stError   { background: rgba(255,45,120,0.07) !important; border-color: rgba(255,45,120,0.3) !important; }
.stSuccess { background: rgba(0,255,136,0.07) !important; border-color: rgba(0,255,136,0.3) !important; }
code { font-family: var(--font-m) !important; color: var(--neon) !important; }
.stCode { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 9px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner('Initializing VeritasAI...'):
    all_models, load_errors = load_all_models()

cnn_loaded = all_models.get('Combined', {}).get('cnn') is not None
tok_loaded  = all_models.get('Combined', {}).get('tok') is not None
total_loaded = 8 if cnn_loaded else 0  # 1 model × 8 datasets

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='padding:1.4rem 0 1.2rem; border-bottom:1px solid #1A2840; margin-bottom:1.1rem;'>"
        "<div style='font-family:Bebas Neue,sans-serif; font-size:2rem; letter-spacing:0.15em;"
        " background:linear-gradient(135deg,#00FF88 0%,#7C3AED 55%,#FFB800 100%);"
        " -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>VERITAS AI</div>"
        "<div style='color:#2A3D5A; font-size:0.68rem; font-family:DM Mono,monospace;"
        " letter-spacing:0.15em; text-transform:uppercase; margin-top:0.25rem;'>Fake News Detection</div>"
        "</div>",
        unsafe_allow_html=True
    )

    sc = "#00FF88" if cnn_loaded and tok_loaded else "#FF2D78"
    st.markdown(
        "<div style='background:#0A1020; border:1px solid #1A2840; border-radius:11px;"
        " padding:1rem 1.1rem; margin-bottom:1.1rem;'>"
        "<div style='font-size:0.65rem; color:#2A3D5A; text-transform:uppercase;"
        " letter-spacing:0.12em; margin-bottom:0.7rem; font-family:DM Mono,monospace;'>System Status</div>"
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem;'>"
        "<span style='color:#6B82A8; font-size:0.8rem;'>CNN Model</span>"
        "<span style='font-family:Bebas Neue; font-size:1.2rem; letter-spacing:0.1em; color:" + sc + ";'>"
        + ("✓ LOADED" if cnn_loaded else "✗ FAILED") + "</span></div>"
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem;'>"
        "<span style='color:#6B82A8; font-size:0.8rem;'>Tokenizer</span>"
        "<span style='font-family:Bebas Neue; font-size:1.2rem; letter-spacing:0.1em; color:" + sc + ";'>"
        + ("✓ LOADED" if tok_loaded else "✗ FAILED") + "</span></div>"
        "<div style='display:flex; justify-content:space-between; align-items:center;'>"
        "<span style='color:#6B82A8; font-size:0.8rem;'>Training runs</span>"
        "<span style='font-family:Bebas Neue; font-size:1.2rem; letter-spacing:0.1em; color:#FFB800;'>32</span></div>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(lbl("Active Model"), unsafe_allow_html=True)
    meta = MODEL_META['CNN-Text']
    mc = meta['color']
    st.markdown(
        "<div style='display:flex; align-items:center; gap:0.7rem; padding:0.5rem 0.8rem;"
        " background:#0A1020; border:1px solid #1A2840; border-radius:8px;"
        " margin-bottom:0.35rem; border-left:2px solid " + mc + ";'>"
        "<span style='font-size:1rem;'>" + meta['icon'] + "</span>"
        "<div style='flex:1;'>"
        "<div style='font-size:0.8rem; font-weight:600; color:#F0F6FF;'>CNN-Text</div>"
        "<div style='font-size:0.67rem; color:#2A3D5A;'>" + meta['type'] + " · " + meta['desc'] + "</div>"
        "</div>"
        "<div style='font-family:DM Mono; font-size:0.65rem; color:" + mc + "; background:rgba(0,0,0,0.3);"
        " padding:0.1rem 0.4rem; border-radius:4px; border:1px solid " + mc + "33;'>8/8</div>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='margin-top:1.2rem; padding-top:1rem; border-top:1px solid #1A2840;'>"
        "<div style='font-size:0.65rem; color:#2A3D5A; text-transform:uppercase;"
        " letter-spacing:0.12em; margin-bottom:0.5rem; font-family:DM Mono,monospace;'>Project</div>"
        "<div style='font-size:0.77rem; color:#3A5070; line-height:1.9;'>"
        "BNU · CSC-233 AI Lab<br>Spring 2026 · Sections E and F<br>Instructor: Hasnain Ahmad & Saad Azhar"
        "</div></div>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='background:#0A1020; border:1px solid #1A2840; border-radius:18px;"
    " padding:2rem 2.4rem; margin-bottom:1.4rem;'>"
    "<div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1.2rem;'>"
    "<div>"
    "<div style='font-family:Bebas Neue,sans-serif; font-size:3rem; letter-spacing:0.12em;"
    " background:linear-gradient(135deg,#00FF88 0%,#7C3AED 45%,#FFB800 100%);"
    " -webkit-background-clip:text; -webkit-text-fill-color:transparent;"
    " line-height:1; margin-bottom:0.35rem;'>VERITAS AI</div>"
    "<div style='color:#3A5070; font-size:0.82rem; letter-spacing:0.08em;"
    " text-transform:uppercase; font-family:DM Mono,monospace;'>"
    "Cross-Dataset CNN-Text Fake News Intelligence</div>"
    "<div style='display:flex; gap:0.5rem; margin-top:0.8rem; flex-wrap:wrap;'>"
    "<span style='background:rgba(255,45,120,0.08); border:1px solid rgba(255,45,120,0.2);"
    " color:#FF2D78; font-size:0.68rem; padding:0.18rem 0.6rem; border-radius:20px;"
    " font-family:DM Mono,monospace;'>CNN-TEXT</span>"
    "<span style='background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.2);"
    " color:#A78BFA; font-size:0.68rem; padding:0.18rem 0.6rem; border-radius:20px;"
    " font-family:DM Mono,monospace;'>8 DATASETS</span>"
    "<span style='background:rgba(255,184,0,0.08); border:1px solid rgba(255,184,0,0.2);"
    " color:#FFB800; font-size:0.68rem; padding:0.18rem 0.6rem; border-radius:20px;"
    " font-family:DM Mono,monospace;'>~275K ARTICLES</span>"
    "</div></div>"
    "<div style='display:flex; gap:2rem; flex-wrap:wrap;'>"
    "<div style='text-align:center;'>"
    "<div style='font-family:Bebas Neue; font-size:2.2rem; letter-spacing:0.1em; color:#FF2D78;'>CNN</div>"
    "<div style='font-size:0.62rem; color:#2A3D5A; text-transform:uppercase;"
    " letter-spacing:0.1em; font-family:DM Mono,monospace;'>Architecture</div></div>"
    "<div style='text-align:center;'>"
    "<div style='font-family:Bebas Neue; font-size:2.2rem; letter-spacing:0.1em; color:#7C3AED;'>32</div>"
    "<div style='font-size:0.62rem; color:#2A3D5A; text-transform:uppercase;"
    " letter-spacing:0.1em; font-family:DM Mono,monospace;'>Training Runs</div></div>"
    "<div style='text-align:center;'>"
    "<div style='font-family:Bebas Neue; font-size:2.2rem; letter-spacing:0.1em; color:#FFB800;'>8</div>"
    "<div style='font-size:0.62rem; color:#2A3D5A; text-transform:uppercase;"
    " letter-spacing:0.1em; font-family:DM Mono,monospace;'>Datasets</div></div>"
    "</div></div></div>",
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Analyze Article",
    "📊  Model Comparison",
    "📈  Research Results",
    "👥  About"
])

# ════════ TAB 1 — ANALYZE ════════════════════════════════════════════════════
with tab1:
    selected_ds = 'Combined'

    st.markdown(lbl("Quick Examples"), unsafe_allow_html=True)
    example_choice = st.selectbox(
        "examples",
        [
            "-- Select an example --",
            "FAKE: Miracle cure suppressed",
            "REAL: Federal Reserve rate decision",
            "FAKE: Shocking government secret",
            "REAL: Climate report findings",
        ],
        label_visibility="collapsed",
        key="example_select"
    )

    EXAMPLES = {
        "FAKE: Miracle cure suppressed":
            "SHOCKING: Scientists reveal that the government has been hiding a miracle cure for cancer for 50 years. Big pharma doesn't want you to know! Share before this gets deleted!!! Anonymous sources confirm this bombshell discovery.",
        "REAL: Federal Reserve rate decision":
            "The Federal Reserve raised interest rates by 25 basis points on Wednesday, citing persistent inflation concerns. The unanimous decision brings the benchmark rate to its highest level in 15 years, according to official statements.",
        "FAKE: Shocking government secret":
            "BREAKING: Whistleblowers expose that top officials have been secretly meeting with foreign agents for years. Sources who cannot be named say the truth will shock everyone. Share this before it gets censored!",
        "REAL: Climate report findings":
            "A new report by the Intergovernmental Panel on Climate Change found that global temperatures have risen 1.1 degrees Celsius above pre-industrial levels. The report, authored by 234 scientists from 66 countries, warns of accelerating sea level rise.",
    }

    if example_choice in EXAMPLES:
        st.session_state['article_text'] = EXAMPLES[example_choice]

    st.markdown(
        "<div style='background:#060C1A; border:1px solid #1A2840; border-radius:9px;"
        " padding:0.55rem 0.9rem; margin:0.4rem 0 0.6rem; display:flex;"
        " align-items:center; gap:0.6rem; flex-wrap:wrap;'>"
        "<span style='font-size:0.68rem; color:#2A3D5A; font-family:DM Mono,monospace;"
        " text-transform:uppercase; letter-spacing:0.1em; white-space:nowrap;'>Running on Combined dataset:</span>"
        "<span style='background:rgba(0,0,0,0.3); border:1px solid #1A2840;"
        " border-radius:6px; padding:0.15rem 0.55rem; font-size:0.72rem; color:#FF2D78;"
        " font-family:DM Mono,monospace;'>🔬 CNN-Text</span>"
        "</div>",
        unsafe_allow_html=True
    )

    article = st.text_area(
        "article",
        value=st.session_state.get('article_text', ''),
        placeholder="Paste any English news article, headline, or political statement here...",
        height=155,
        key="article_input",
        label_visibility="collapsed"
    )

    b1, b2 = st.columns([3,1], gap="small")
    with b1:
        analyze_btn = st.button("🔍  Analyze with CNN-Text", use_container_width=True)
    with b2:
        clear_btn = st.button("✕  Clear", use_container_width=True)

    if clear_btn:
        st.session_state['article_text'] = ''
        st.session_state.pop('analysis_results', None)
        st.rerun()

    if analyze_btn and article and article.strip():
        st.session_state['article_text'] = article
        with st.spinner("Running CNN-Text on Combined dataset..."):
            preds, cleaned = predict_dataset(article.strip(), selected_ds, all_models)
        st.session_state['analysis_results'] = {
            'preds': preds, 'cleaned': cleaned, 'dataset': selected_ds
        }
    elif analyze_btn and not (article and article.strip()):
        st.warning("Please paste an article first.")

    if 'analysis_results' in st.session_state:
        res     = st.session_state['analysis_results']
        preds   = res['preds']
        ds_used = res['dataset']

        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

        prob = preds.get('CNN-Text')
        if prob is None:
            st.warning("CNN-Text model did not produce a result. Check that cnn_combined.keras and tokenizer_combined.json are uploaded to your Hugging Face repo.")
        else:
            verdict  = 'FAKE' if prob >= 0.5 else 'REAL'
            conf     = prob if prob >= 0.5 else 1 - prob
            b_bg     = 'rgba(255,45,120,0.08)' if verdict == 'FAKE' else 'rgba(0,255,136,0.08)'
            b_bdr    = '#FF2D78' if verdict == 'FAKE' else '#00FF88'
            conf_bar = int(conf * 100)

            st.markdown(
                "<div style='background:" + b_bg + "; border:1.5px solid " + b_bdr + ";"
                " border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1rem;"
                " display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;'>"
                "<div>"
                "<div style='font-family:Bebas Neue,sans-serif; font-size:2.6rem; color:" + b_bdr + ";"
                " letter-spacing:0.12em; line-height:1;'>" + verdict + " NEWS</div>"
                "<div style='color:#3A5070; font-size:0.79rem; margin-top:0.35rem; font-family:DM Mono,monospace;'>"
                "CNN-Text · " + str(round(conf*100, 1)) + "% confidence · Combined dataset"
                "</div></div>"
                "<div style='min-width:160px;'>"
                "<div style='display:flex; justify-content:space-between; margin-bottom:0.35rem;'>"
                "<span style='font-size:0.68rem; color:#2A3D5A; font-family:DM Mono,monospace; text-transform:uppercase;'>Confidence</span>"
                "<span style='font-size:0.68rem; color:" + b_bdr + "; font-family:DM Mono,monospace; font-weight:600;'>" + str(round(conf*100, 1)) + "%</span></div>"
                "<div style='background:#152040; border-radius:99px; height:6px; overflow:hidden;'>"
                "<div style='width:" + str(conf_bar) + "%; height:100%; border-radius:99px; background:" + b_bdr + ";'></div></div>"
                "<div style='text-align:right; margin-top:0.4rem;'>"
                "<span style='font-family:DM Mono,monospace; font-size:0.7rem; color:" + b_bdr + ";"
                " background:" + b_bg + "; padding:0.1rem 0.5rem; border-radius:5px;"
                " border:1px solid " + b_bdr + "44;'>Dataset: " + ds_used + "</span>"
                "</div></div></div>",
                unsafe_allow_html=True
            )

            with st.expander("🔬 View preprocessed text"):
                st.code(res['cleaned'][:500] + ('...' if len(res['cleaned']) > 500 else ''), language=None)

# ════════ TAB 2 — COMPARISON ═════════════════════════════════════════════════
with tab2:
    st.markdown(lbl("EDA — Class Distribution & Article Length"), unsafe_allow_html=True)
    eda_c1, eda_c2 = st.columns(2, gap="large")
    with eda_c1:
        p = os.path.join(BASE_DIR, 'charts', 'eda_class_distribution.png')
        if os.path.exists(p): st.image(p, use_container_width=True)
        else: st.info("eda_class_distribution.png not found in charts/")
    with eda_c2:
        p = os.path.join(BASE_DIR, 'charts', 'eda_length_distribution.png')
        if os.path.exists(p): st.image(p, use_container_width=True)
        else: st.info("eda_length_distribution.png not found in charts/")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown(lbl("Performance Heatmaps — 8 Datasets × 4 Models (32 Runs)"), unsafe_allow_html=True)
    hp = os.path.join(BASE_DIR, 'charts', 'heatmaps_accuracy_f1.png')
    if os.path.exists(hp): st.image(hp, use_container_width=True)
    else: st.info("heatmaps_accuracy_f1.png not found in charts/")

    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        st.markdown(lbl("Cross-Dataset Performance"), unsafe_allow_html=True)
        p = os.path.join(BASE_DIR, 'charts', 'cross_dataset_performance.png')
        if os.path.exists(p): st.image(p, use_container_width=True)
    with cc2:
        st.markdown(lbl("Training Time"), unsafe_allow_html=True)
        p = os.path.join(BASE_DIR, 'charts', 'training_time_comparison.png')
        if os.path.exists(p): st.image(p, use_container_width=True)

    for lbl_txt, fname in [
        ("Confusion Matrices", "confusion_matrices_all.png"),
        ("ROC Curves",         "roc_curves_all.png"),
    ]:
        st.markdown(lbl(lbl_txt), unsafe_allow_html=True)
        p = os.path.join(BASE_DIR, 'charts', fname)
        if os.path.exists(p): st.image(p, use_container_width=True)

# ════════ TAB 3 — RESULTS ════════════════════════════════════════════════════
with tab3:
    csv_path = os.path.join(BASE_DIR, 'master_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        acc_vals = pd.to_numeric(df['Accuracy'], errors='coerce')
        f1_vals  = pd.to_numeric(df['F1-Score'], errors='coerce')

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        metric_data = [
            (m1, "Training Runs",  "32",                                        "#00FF88"),
            (m2, "Total Articles", "~275K",                                     "#7C3AED"),
            (m3, "Best Accuracy",  str(round(acc_vals.max()*100, 2)) + "%",     "#FFB800"),
            (m4, "Best F1-Score",  str(round(f1_vals.max()*100, 2)) + "%",      "#00D4FF"),
        ]
        for col, label, val, clr in metric_data:
            with col:
                st.markdown(
                    "<div style='background:#0A1020; border:1px solid #1A2840;"
                    " border-radius:12px; padding:1.1rem 1rem; text-align:center;"
                    " margin-bottom:0.9rem; border-top:2px solid " + clr + ";'>"
                    "<div style='font-family:Bebas Neue,sans-serif; font-size:2rem;"
                    " color:" + clr + "; letter-spacing:0.08em;'>" + val + "</div>"
                    "<div style='color:#2A3D5A; font-size:0.67rem; text-transform:uppercase;"
                    " letter-spacing:0.1em; margin-top:0.2rem; font-family:DM Mono,monospace;'>"
                    + label + "</div></div>",
                    unsafe_allow_html=True
                )

        st.markdown(lbl("Master Results Table"), unsafe_allow_html=True)
        f1c, f2c = st.columns(2, gap="medium")
        with f1c:
            ds_filter = st.multiselect("Filter Dataset", options=df['Dataset'].unique().tolist(), default=df['Dataset'].unique().tolist())
        with f2c:
            model_filter = st.multiselect("Filter Model", options=df['Model'].unique().tolist(), default=df['Model'].unique().tolist())

        filtered = df[df['Dataset'].isin(ds_filter) & df['Model'].isin(model_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download Results CSV", data=df.to_csv(index=False).encode('utf-8'), file_name='veritas_ai_results.csv', mime='text/csv')

        st.markdown(lbl("Key Research Findings"), unsafe_allow_html=True)
        findings = [
            ("Dataset Quality Dominates at the Easy End",
             "All 4 models score 97-99% on ISOT. The difference between Logistic Regression and BiLSTM is negligible on clean single-source data."),
            ("Architecture Matters as Complexity Increases",
             "On LIAR (short political statements), the gap between classical ML and deep learning widens significantly."),
            ("Combined Dataset Improves Generalization",
             "Training on the combined dataset exposes models to all writing styles simultaneously."),
            ("Speed vs Accuracy Tradeoff",
             "LightGBM trains 10-20x faster than BiLSTM with only marginal accuracy loss on most datasets."),
            ("Newer Datasets Are Harder",
             "Models consistently score lower on newer datasets (ErfanM, Ultimate, GonzaloA) compared to ISOT."),
        ]
        for title, body in findings:
            with st.expander(title):
                st.markdown("<div style='color:#6B82A8; font-size:0.88rem; line-height:1.75; padding:0.4rem 0;'>" + body + "</div>", unsafe_allow_html=True)
    else:
        st.warning("master_results.csv not found. Run the Colab notebook first.")

# ════════ TAB 4 — ABOUT ══════════════════════════════════════════════════════
with tab4:
    a1, a2 = st.columns([1.1, 0.9], gap="large")

    with a1:
        st.markdown(lbl("About This Project"), unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#0A1020; border:1px solid #1A2840; border-radius:13px;"
            " padding:1.3rem 1.4rem; margin-bottom:1rem; border-left:2.5px solid #7C3AED;'>"
            "<div style='font-family:Bebas Neue,sans-serif; font-size:1.1rem;"
            " letter-spacing:0.08em; color:#F0F6FF; margin-bottom:0.55rem;'>Research Question</div>"
            "<div style='color:#6B82A8; font-size:0.87rem; line-height:1.75; font-style:italic;'>"
            "Does the age, size, and source diversity of a training dataset affect how well "
            "a model detects fake news, and do newer architectures handle this variation "
            "better than classical ones?"
            "</div></div>",
            unsafe_allow_html=True
        )

        st.markdown(lbl("Datasets"), unsafe_allow_html=True)
        ds_rows = [
            {'Dataset': ds, 'Period': info['period'], 'Articles': info['rows'], 'Source': info['source'], 'Type': info['tag']}
            for ds, info in DATASET_INFO.items()
        ]
        st.dataframe(pd.DataFrame(ds_rows), use_container_width=True, hide_index=True)

        st.markdown(lbl("Tech Stack"), unsafe_allow_html=True)
        techs = [
            ("Python 3.10","#00FF88"), ("TensorFlow/Keras","#FF2D78"),
            ("NLTK","#7C3AED"),        ("pandas","#00FF88"),
            ("NumPy","#00D4FF"),       ("Streamlit","#FF2D78"),
            ("Google Colab","#FFB800"),("HuggingFace","#7C3AED"),
            ("GitHub","#6B82A8"),
        ]
        badges = "".join(
            "<span style='background:rgba(0,0,0,0.3); border:1px solid #1A2840;"
            " border-radius:6px; padding:0.2rem 0.6rem; font-size:0.73rem; color:" + c + ";"
            " margin:0.12rem; display:inline-block; font-family:DM Mono,monospace;'>" + t + "</span>"
            for t, c in techs
        )
        st.markdown("<div style='line-height:2.1;'>" + badges + "</div>", unsafe_allow_html=True)

    with a2:
        st.markdown(lbl("Team"), unsafe_allow_html=True)
        for i, member in enumerate(TEAM):
            clr = MEMBER_COLORS[i % len(MEMBER_COLORS)]
            id_html = (
                "<div style='font-family:DM Mono,monospace; font-size:0.65rem; color:" + clr + ";"
                " flex-shrink:0; background:" + clr + "11; border:1px solid " + clr + "22;"
                " padding:0.12rem 0.4rem; border-radius:5px;'>" + member['id'] + "</div>"
            ) if member['id'] else ""
            st.markdown(
                "<div style='background:#0A1020; border:1px solid #1A2840; border-radius:11px;"
                " padding:0.85rem 1.05rem; margin-bottom:0.5rem; display:flex;"
                " align-items:center; gap:0.85rem; border-left:2px solid " + clr + ";'>"
                "<div style='width:38px; height:38px; border-radius:50%; flex-shrink:0;"
                " background:" + clr + "15; border:1px solid " + clr + "33;"
                " display:flex; align-items:center; justify-content:center; font-size:1rem;'>"
                + member['emoji'] + "</div>"
                "<div style='flex:1; min-width:0;'>"
                "<div style='font-weight:600; font-size:0.86rem; color:#F0F6FF;"
                " white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"
                + member['name'] + "</div>"
                "<div style='font-size:0.68rem; color:#2A3D5A; margin-top:0.1rem;"
                " font-family:DM Mono,monospace;'>" + member['role'] + "</div>"
                "</div>" + id_html + "</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            "<div style='background:#0A1020; border:1px solid #1A2840; border-radius:11px;"
            " padding:1.1rem 1.2rem; margin-top:0.4rem;'>"
            "<div style='font-size:0.65rem; color:#2A3D5A; text-transform:uppercase;"
            " letter-spacing:0.12em; margin-bottom:0.8rem; font-family:DM Mono,monospace;'>Project Info</div>"
            "<div style='display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;'>"
            "<div style='font-size:0.76rem; color:#2A3D5A; font-family:DM Mono,monospace;'>Course</div>"
            "<div style='font-size:0.76rem; color:#6B82A8; font-weight:500;'>CSC-233 AI Lab</div>"
            "<div style='font-size:0.76rem; color:#2A3D5A; font-family:DM Mono,monospace;'>Semester</div>"
            "<div style='font-size:0.76rem; color:#6B82A8; font-weight:500;'>Spring 2026</div>"
            "<div style='font-size:0.76rem; color:#2A3D5A; font-family:DM Mono,monospace;'>Instructor</div>"
            "<div style='font-size:0.76rem; color:#6B82A8; font-weight:500;'>Hasnain Ahmad & Saad Azhar</div>"
            "<div style='font-size:0.76rem; color:#2A3D5A; font-family:DM Mono,monospace;'>Section</div>"
            "<div style='font-size:0.76rem; color:#6B82A8; font-weight:500;'>E and F</div>"
            "<div style='font-size:0.76rem; color:#2A3D5A; font-family:DM Mono,monospace;'>University</div>"
            "<div style='font-size:0.76rem; color:#6B82A8; font-weight:500;'>BNU, Lahore</div>"
            "</div></div>",
            unsafe_allow_html=True
        )
