import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from scipy.stats import spearmanr

# ==========================================
# PAGE CONFIGURATION & PROFESSIONAL CSS
# ==========================================
st.set_page_config(
    page_title="Credit Risk XAI Research Framework",
    page_icon="🛡️",
    layout="wide"
)

# Professional CSS to fix readability and styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Typography & Background */
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        color: #1e293b !important; 
    }
    .stApp { background-color: #f8fafc; }

    /* Headers */
    h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; }

    /* FIX: Readable Tabs (No white font) */
    button[data-baseweb="tab"] {
        background-color: #f1f5f9 !important;
        border-radius: 4px 4px 0 0 !important;
        margin-right: 5px !important;
    }
    button[data-baseweb="tab"] div {
        color: #475569 !important; /* Dark Slate Font */
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #0284c7 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #ffffff !important; /* White text only for the active blue tab */
    }

    /* Metric & Leaderboard Cards */
    .metric-card { 
        background: white; border: 1px solid #e2e8f0; border-radius: 8px; 
        padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0369a1 !important; }
    .lb-card { 
        padding: 20px; border-radius: 12px; text-align: center; 
        box-shadow: 0 10px 15px rgba(0,0,0,0.1); border: 1px solid #cbd5e1;
    }
    
    /* Insight Box */
    .insight-box { 
        background-color: #ffffff; border-left: 5px solid #0369a1; 
        padding: 15px 20px; border-radius: 4px; margin: 15px 0; 
        color: #334155; font-size: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & REGISTRY (Exact Filenames)
# ==========================================
METHOD_COLORS = {
    'SHAP': '#64748b', 'Banzhaf': '#f59e0b', 'Myerson': '#10b981',
    'Owen-Domain': '#ef4444', 'Owen-Data': '#8b5cf6', 'Owen-Model': '#ec4899', 'R-Myerson': '#0284c7'
}

DATASET_REGISTRY = {
    "Coursera Loans": {
        "main": "Coursera_result.csv", "wil": "Coursera_result_wilcoxon.csv", 
        "nem": "Coursera_result_nemenyi.csv", "corr": "Coursera_result_correlation.csv", 
        "imb": 1.0, "desc": "Extreme 1% default rate. Tests attribution stability at the breakdown point."
    },
    "Lending Club A (10%)": {
        "main": "LC_result10.csv", "wil": "LC_result_wilcoxon.csv", 
        "nem": "LC_result_nemenyi.csv", "corr": "LC_result_correlation.csv", 
        "imb": 10.0, "desc": "Standard industry default rate (10%). Represents typical P2P lending risk."
    },
    "Lending Club B (4%)": {
        "main": "LC4_result(1).csv", "wil": "Lc66_wilcoxon_cliffs_results.csv", 
        "nem": "Lc66_nemenyi_results (1).csv", "corr": "Lc66_auc_I_correlation.csv", 
        "imb": 4.01, "desc": "Severe imbalance (4%). Tests robustness with high-dimensional engineered features."
    },
    "Taiwan Credit": {
        "main": "TW_result.csv", "wil": "TW_result_wilcoxon.csv", 
        "nem": "TW_result_nemenyi.csv", "corr": "TW_result_correlation.csv", 
        "imb": 22.12, "desc": "Temporal repayment data with moderate imbalance (22%)."
    },
    "German Credit": {
        "main": "Ger_result.csv", "wil": "Ger_result_wilcoxon.csv", 
        "nem": "Ger_result_nemenyi.csv", "corr": "Ger_result_correlation.csv", 
        "imb": 30.0, "desc": "Basel benchmark dataset with 30% default rate and categorical-heavy features."
    }
}

# ==========================================
# UTILITIES
# ==========================================
@st.cache_data
def load_and_clean(path, is_index=False):
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path, index_col=0 if is_index else None)
        if 'Sampler' in df.columns:
            df['Sampler'] = df['Sampler'].astype(str).replace(['nan','NaN','None',' '], 'None')
        for col in ['AUC', 'I', 'S(α=0.5)', 'Stability', 'Q']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except: return None

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🧭 Navigation")
selection = st.sidebar.radio("View Selection:", ["📊 Global Synthesis", "🏆 Leaderboards"] + list(DATASET_REGISTRY.keys()))

# ==========================================
# VIEW: GLOBAL SYNTHESIS
# ==========================================
if selection == "📊 Global Synthesis":
    st.title("Cross-Dataset Performance Synthesis")
    st.markdown("<div class='insight-box'>Observe XAI stability shift as default rates drop from 30% to 1%.</div>", unsafe_allow_html=True)
    
    all_data = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_and_clean(cfg['main'])
        if df is not None:
            df['Dataset'] = f"{name} ({cfg['imb']}%)"
            df['Imb'] = cfg['imb']
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data).sort_values('Imb', ascending=False)
        fig = px.scatter(combined, x="AUC", y="I", animation_frame="Dataset",
                         color="Method", size="S(α=0.5)", color_discrete_map=METHOD_COLORS,
                         range_y=[0, 1.1], template="plotly_white", height=600)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VIEW: LEADERBOARDS
# ==========================================
elif selection == "🏆 Leaderboards":
    st.title("🏆 Global XAI Rankings")
    global_list = [load_and_clean(cfg['main']) for cfg in DATASET_REGISTRY.values() if load_and_clean(cfg['main']) is not None]
    if global_list:
        combined = pd.concat(global_list)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Stability (I-Score)")
            st.dataframe(combined.groupby('Method')['I'].mean().sort_values(ascending=False).reset_index(), use_container_width=True)
        with c2:
            st.subheader("Predictive Integrity (S-Score)")
            st.dataframe(combined.groupby('Method')['S(α=0.5)'].mean().sort_values(ascending=False).reset_index(), use_container_width=True)

# ==========================================
# VIEW: INDIVIDUAL DATASET
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.title(f"{selection} Detailed View")
    st.markdown(f"<div class='insight-box'>{cfg['desc']}</div>", unsafe_allow_html=True)
    
    main_df = load_and_clean(cfg['main'])
    wil_df = load_and_clean(cfg['wil'])
    nem_df = load_and_clean(cfg['nem'], is_index=True)
    corr_df = load_and_clean(cfg['corr'])
    
    if main_df is not None:
        # Podium
        top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
        cols = st.columns(3)
        colors = ["#fef3c7", "#f1f5f9", "#ffedd5"]
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        for i in range(len(top3)):
            with cols[i]:
                st.markdown(f"<div class='lb-card' style='background:{colors[i]}'><h3>{medals[i]}</h3><b>{top3.loc[i, 'Method']}</b><br>{top3.loc[i, 'Model']}+{top3.loc[i, 'Sampler']}<br><div class='metric-value'>{top3.loc[i, 'S(α=0.5)']:.4f}</div></div>", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["🎯 Performance", "🔬 Statistics", "🗄️ Data"])
        
        with t1:
            st.plotly_chart(px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model', 
                                       color_discrete_map=METHOD_COLORS, size='S(α=0.5)', height=500), use_container_width=True)
            
            st.subheader("Spearman Rank Correlation (AUC vs I)")
            if corr_df is not None:
                # Handle different column naming in correlation files
                rho = corr_df.get('Spearman_rho', [0])[0]
                p = corr_df.get('Spearman_p', [1])[0]
            else:
                rho, p = spearmanr(main_df['AUC'].fillna(0), main_df['I'].fillna(0))
            
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{rho:.3f}</div><div class='metric-label'>p-value: {p:.4f} {'✅ (Significant)' if p < 0.05 else '❌'}</div></div>", unsafe_allow_html=True)
                
        with t2:
            if wil_df is not None and nem_df is not None:
                st.subheader("Statistical Consensus")
                res = []
                for _, r in wil_df.iterrows():
                    m1, m2 = r['Method1'], r['Method2']
                    p_nem = 1.0
                    try: p_nem = nem_df.loc[m1, m2]
                    except: 
                        try: p_nem = nem_df.loc[m2, m1]
                        except: pass
                    res.append({
                        "Comparison": f"{m1} vs {m2}", "Wilcoxon p": f"{r['p_value']:.4f}",
                        "Nemenyi p": f"{p_nem:.4f}", "Consensus Sig": "✅" if (r['p_value'] < 0.05 and p_nem < 0.05) else "❌"
                    })
                st.table(res)
        
        with t3:
            st.dataframe(main_df, use_container_width=True)
    else:
        st.error(f"Required file {cfg['main']} not found.")
