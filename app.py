import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# PAGE CONFIGURATION & PROFESSIONAL CSS
# ==========================================
st.set_page_config(
    page_title="Credit Risk XAI Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; font-weight: 700; letter-spacing: -0.5px; }
    
    .metric-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0369a1; margin: 10px 0; }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 500; text-transform: uppercase; }
    
    .lb-card { padding: 20px; border-radius: 12px; text-align: center; transition: transform 0.2s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .lb-card:hover { transform: translateY(-5px); }
    .lb-rank { font-size: 2rem; margin-bottom: 5px; color: #1e293b; }
    .lb-method { font-size: 1.3rem; font-weight: 700; color: #0f172a; margin: 0; }
    .lb-config { font-size: 0.9rem; color: #475569; margin: 5px 0 15px 0; }
    .lb-score { font-size: 1.5rem; font-weight: 700; color: #0369a1; margin: 0; }
    
    .insight-box { background-color: #ffffff; border-left: 4px solid #0369a1; padding: 15px 20px; border-radius: 6px; margin: 15px 0; color: #334155; font-size: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & UPDATED REGISTRY (Matched to actual files)
# ==========================================
METHOD_COLORS = {
    'SHAP': '#64748b', 'Banzhaf': '#d97706', 'Myerson': '#059669',
    'Owen-Domain': '#dc2626', 'Owen-Data': '#7c3aed', 'Owen-Model': '#db2777', 'R-Myerson': '#0284c7'
}

DATASET_REGISTRY = {
    "German Credit": {
        "main": "Ger_result.csv", "wilcoxon": "Ger_result_wilcoxon.csv",
        "nemenyi": "Ger_result_nemenyi.csv", "corr": "Ger_result_correlation.csv",
        "label": "Moderate Imbalance", "imb": 30.0
    },
    "Taiwan Credit": {
        "main": "TW_result.csv", "wilcoxon": "TW_result_wilcoxon.csv",
        "nemenyi": "TW_result_nemenyi.csv", "corr": "TW_result_correlation.csv",
        "label": "Moderate Imbalance", "imb": 22.12
    },
    "Lending Club A (10%)": {
        "main": "LC10pcdefault.csv", "wilcoxon": "lc10_wilcoxon_cliffs_results.csv",
        "nemenyi": "lc10_nemenyi_results.csv", "corr": "lc10_auc_I_correlation.csv",
        "label": "Industry Standard", "imb": 10.0
    },
    "Lending Club B (4%)": {
        "main": "LC4_result(1).csv", "wilcoxon": "Lc66_wilcoxon_cliffs_results.csv",
        "nemenyi": "Lc66_nemenyi_results (1).csv", "corr": "Lc66_auc_I_correlation.csv",
        "label": "Severe Imbalance", "imb": 4.01
    },
    "Coursera Loans": {
        "main": "coursera_loans_results_7methods.csv", "wilcoxon": "Coursera_result_wilcoxon.csv",
        "nemenyi": "Coursera_result_nemenyi.csv", "corr": "Coursera_result_correlation.csv",
        "label": "Extreme Imbalance", "imb": 1.0
    }
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
@st.cache_data
def load_data(path, is_index=False):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0 if is_index else None)
        if 'Sampler' in df.columns:
            df['Sampler'] = df['Sampler'].astype(str).replace(['nan', 'NaN', 'None', 'nan '], 'None')
        return df
    except Exception: return None

def get_wilcoxon_sig(sig_val, p_val):
    sig_str = str(sig_val).lower()
    if '✓' in sig_str or 'yes' in sig_str or 'true' in sig_str: return True
    try: return float(p_val) < 0.05
    except: return False

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("### 🧭 Navigation")
views = ["📊 Cross-Dataset Synthesis", "🏆 Leaderboards"] + list(DATASET_REGISTRY.keys())
selection = st.sidebar.radio("Select View:", views)

# ==========================================
# VIEW 1: SYNTHESIS
# ==========================================
if selection == "📊 Cross-Dataset Synthesis":
    st.title("Cross-Dataset Performance Analysis")
    st.markdown("<div class='insight-box'><b>Dashboard Overview:</b> Press <b>Play</b> to observe how XAI stability degrades as imbalance increases (30% → 1%).</div>", unsafe_allow_html=True)
    
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            df['Dataset'] = f"{name} ({cfg['imb']}%)"
            df['Imbalance'] = cfg['imb']
            df['Config'] = df['Method'] + "_" + df['Model'] + "_" + df['Sampler']
            global_results.append(df)
            
    if global_results:
        combined = pd.concat(global_results).sort_values('Imbalance', ascending=False)
        fig_anim = px.scatter(combined, x="AUC", y="I", animation_frame="Dataset", animation_group="Config",
                            color="Method", symbol="Model", size="S(α=0.5)",
                            color_discrete_map=METHOD_COLORS, range_y=[0, 1.1])
        fig_anim.update_layout(template="plotly_white", height=600)
        st.plotly_chart(fig_anim, use_container_width=True)

# ==========================================
# VIEW 3: DATASET DASHBOARD
# ==========================================
elif selection in DATASET_REGISTRY:
    cfg = DATASET_REGISTRY[selection]
    st.title(f"{selection} Analysis")
    
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is not None:
        # Top 3 Podium
        top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
        cols = st.columns(3)
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        colors = ["linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)", 
                  "linear-gradient(135deg, #e5e7eb 0%, #9ca3af 100%)", 
                  "linear-gradient(135deg, #fb923c 0%, #ea580c 100%)"]
        for i in range(len(top3)):
            with cols[i]:
                st.markdown(f"<div class='lb-card' style='background:{colors[i]}'><h3>{medals[i]}</h3><b>{top3.loc[i, 'Method']}</b><br>{top3.loc[i, 'Model']}+{top3.loc[i, 'Sampler']}<br><h2>{top3.loc[i, 'S(α=0.5)']:.4f}</h2></div>", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["🎯 Accuracy vs Interpretability", "🧩 Q vs I", "🔬 Statistics"])
        
        with t1:
            fig = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model', color_discrete_map=METHOD_COLORS)
            st.plotly_chart(fig, use_container_width=True)
            rho = main_df['AUC'].corr(main_df['I'], method='spearman')
            st.markdown(f"<div class='insight-box'><b>Trade-off Analysis:</b> Spearman ρ = {rho:.3f}.</div>", unsafe_allow_html=True)
            
        with t2:
            owen = main_df[main_df['Method'].str.contains('Owen')].dropna(subset=['Q'])
            if not owen.empty:
                st.plotly_chart(px.scatter(owen, x='Q', y='I', color='Method', symbol='Model'), use_container_width=True)
                q_rho = owen['Q'].corr(owen['I'], method='spearman')
                st.markdown(f"<div class='insight-box'><b>Derivation Method:</b> Spearman rank correlation is ρ = {q_rho:.3f}.</div>", unsafe_allow_html=True)
        
        with t3:
            if wil_df is not None and nem_df is not None:
                st.subheader("Statistical Consensus")
                res = []
                for _, r in wil_df.iterrows():
                    m1, m2 = r['Method1'], r['Method2']
                    w_sig = get_wilcoxon_sig(r.get('Significant',''), r.get('p_value', 1.0))
                    try: n_p = nem_df.loc[m1, m2]
                    except: n_p = nem_df.loc[m2, m1] if m2 in nem_df.index else 1.0
                    res.append({"Comparison": f"{m1} vs {m2}", "Wilcoxon": "✓" if w_sig else "✗", "Nemenyi p": f"{n_p:.3f}", "Consensus": "✓" if (w_sig and n_p < 0.05) else "✗"})
                st.table(res)
    else:
        st.error(f"Files for {selection} missing. Please ensure {cfg['main']} is uploaded.")
