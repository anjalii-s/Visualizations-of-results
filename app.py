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

# Professional UI Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Force Dark Text globally for readability */
    html, body, [class*="css"], .stMarkdown, p, span { 
        font-family: 'Inter', sans-serif; 
        color: #1e293b !important; 
    }

    /* Professional Background */
    .main { background-color: #f8fafc; }

    /* Fix unreadable white tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
    }
    button[data-baseweb="tab"] div {
        color: #64748b !important; /* Inactive tab font color */
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #0284c7 !important; /* Active tab font color */
        border-bottom: 2px solid #0284c7 !important;
    }

    /* Headers */
    h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; }
    
    /* Metric Cards */
    .metric-card { 
        background-color: #ffffff; 
        border: 1px solid #e2e8f0; 
        border-radius: 8px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); 
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0369a1; }
    .metric-label { font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    
    /* Leaderboard Cards */
    .lb-card { 
        padding: 20px; 
        border-radius: 12px; 
        text-align: center; 
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    .lb-rank { font-size: 1.8rem; font-weight: 800; margin-bottom: 5px; }
    .lb-method { font-size: 1.2rem; font-weight: 700; color: #0f172a; }
    .lb-config { font-size: 0.85rem; color: #475569; margin-bottom: 10px; }
    .lb-score { font-size: 1.4rem; font-weight: 700; color: #0369a1; }
    
    /* Insight Box */
    .insight-box { 
        background-color: #f0f9ff; 
        border-left: 5px solid #0ea5e9; 
        padding: 15px 20px; 
        border-radius: 4px; 
        margin: 15px 0; 
        font-size: 0.95rem; 
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# DATASET REGISTRY (Matched to your files)
# ==========================================
METHOD_COLORS = {
    'SHAP': '#94a3b8', 'Banzhaf': '#f59e0b', 'Myerson': '#10b981',
    'Owen-Domain': '#ef4444', 'Owen-Data': '#8b5cf6', 'Owen-Model': '#ec4899', 'R-Myerson': '#0ea5e9'
}

REGISTRY = {
    "German Credit": {
        "main": "Ger_result.csv", "wil": "Ger_result_wilcoxon.csv", 
        "nem": "Ger_result_nemenyi.csv", "corr": "Ger_result_correlation.csv", "imb": 30.0
    },
    "Taiwan Credit": {
        "main": "TW_result.csv", "wil": "TW_result_wilcoxon.csv", 
        "nem": "TW_result_nemenyi.csv", "corr": "TW_result_correlation.csv", "imb": 22.12
    },
    "Lending Club A (10%)": {
        "main": "LC_result10.csv", "wil": "LC_result_wilcoxon.csv", 
        "nem": "LC_result_nemenyi.csv", "corr": "LC_result_correlation.csv", "imb": 10.0
    },
    "Lending Club B (4%)": {
        "main": "LC4_result(1).csv", "wil": "Lc66_wilcoxon_cliffs_results.csv", 
        "nem": "Lc66_nemenyi_results (1).csv", "corr": "Lc66_auc_I_correlation.csv", "imb": 4.01
    },
    "Coursera Loans": {
        "main": "coursera_loans_results_7methods.csv", "wil": "Coursera_result_wilcoxon.csv", 
        "nem": "Coursera_result_nemenyi.csv", "corr": "Coursera_result_correlation.csv", "imb": 1.0
    }
}

# ==========================================
# HELPERS
# ==========================================
@st.cache_data
def load_file(fname, is_index=False):
    if not os.path.exists(fname): return None
    try:
        df = pd.read_csv(fname, index_col=0 if is_index else None)
        if 'Sampler' in df.columns:
            df['Sampler'] = df['Sampler'].astype(str).replace(['nan','NaN','None',' '], 'None')
        return df
    except: return None

def get_sig_icon(p):
    try: return "✅" if float(p) < 0.05 else "❌"
    except: return "❌"

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📊 XAI Navigation")
view = st.sidebar.radio("Go to:", ["🌎 Cross-Dataset View", "🏆 Leaderboards"] + list(REGISTRY.keys()))

# ==========================================
# VIEW: CROSS-DATASET
# ==========================================
if view == "🌎 Cross-Dataset View":
    st.title("Framework Performance Synthesis")
    st.markdown("<div class='insight-box'><b>Research Goal:</b> Evaluating XAI stability across the 'Imbalance Spectrum'. Bubble size indicates the integrated S(α) trade-off score.</div>", unsafe_allow_html=True)
    
    all_data = []
    for name, files in REGISTRY.items():
        df = load_file(files['main'])
        if df is not None:
            df['Dataset'] = f"{name} ({files['imb']}%)"
            df['Imbalance'] = files['imb']
            all_data.append(df)
    
    if all_data:
        full_df = pd.concat(all_data).sort_values('Imbalance', ascending=False)
        fig = px.scatter(full_df, x="AUC", y="I", animation_frame="Dataset",
                         color="Method", size="S(α=0.5)", hover_name="Method",
                         color_discrete_map=METHOD_COLORS, range_y=[0, 1.1])
        fig.update_layout(template="plotly_white", height=600)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VIEW: LEADERBOARDS
# ==========================================
elif view == "🏆 Leaderboards":
    st.title("🏆 Rankings & Benchmarks")
    
    summary_list = []
    for name, files in REGISTRY.items():
        df = load_file(files['main'])
        if df is not None:
            df['Dataset'] = name
            summary_list.append(df)
            
    if summary_list:
        combined = pd.concat(summary_list)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Performers (Global S-Score)")
            top_global = combined.groupby(['Method'])['S(α=0.5)'].mean().sort_values(ascending=False).reset_index()
            st.dataframe(top_global.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        with col2:
            st.subheader("Stability Rankings (Global I-Score)")
            top_i = combined.groupby(['Method'])['I'].mean().sort_values(ascending=False).reset_index()
            st.dataframe(top_i.style.background_gradient(cmap='Greens'), use_container_width=True)

# ==========================================
# VIEW: INDIVIDUAL DATASETS
# ==========================================
else:
    files = REGISTRY[view]
    st.title(f"{view} Analysis")
    st.caption(f"Class Imbalance: {files['imb']}% default rate")
    
    m_df = load_file(files['main'])
    w_df = load_file(files['wil'])
    n_df = load_file(files['nem'], is_index=True)
    c_df = load_file(files['corr'])
    
    if m_df is not None:
        # Podium
        st.markdown("### 🏅 Top 3 Configurations")
        top3 = m_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index()
        p_cols = st.columns(3)
        p_styles = ["background:#fef08a", "background:#e2e8f0", "background:#fed7aa"]
        labels = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        
        for i in range(len(top3)):
            with p_cols[i]:
                st.markdown(f"""<div class='lb-card' style='{p_styles[i]}'>
                    <div class='lb-rank'>{labels[i]}</div>
                    <div class='lb-method'>{top3.loc[i, 'Method']}</div>
                    <div class='lb-config'>{top3.loc[i, 'Model']} + {top3.loc[i, 'Sampler']}</div>
                    <div class='lb-score'>{top3.loc[i, 'S(α=0.5)']:.4f}</div>
                </div>""", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔬 Statistical Tests", "📋 Data"])
        
        with tab1:
            st.plotly_chart(px.scatter(m_df, x='AUC', y='I', color='Method', symbol='Model', 
                                       color_discrete_map=METHOD_COLORS, size='S(α=0.5)', height=500), use_container_width=True)
            if c_df is not None:
                st.info(f"Spearman Correlation (AUC vs I): **{c_df['Spearman_rho'].iloc[0]:.3f}** (p={c_df['Spearman_p'].iloc[0]:.3f})")
                
        with tab2:
            if w_df is not None and n_df is not None:
                st.subheader("Statistical Consensus")
                st.caption("A checkmark indicates BOTH Wilcoxon and Nemenyi tests agree on significance.")
                con_data = []
                for _, row in w_df.iterrows():
                    m1, m2 = row['Method1'], row['Method2']
                    p_nem = 1.0
                    try: p_nem = n_df.loc[m1, m2]
                    except: 
                        try: p_nem = n_df.loc[m2, m1]
                        except: pass
                    con_data.append({
                        "Comparison": f"{m1} vs {m2}", "Wilcoxon Sig": get_sig_icon(row['p_value']),
                        "Nemenyi p": f"{p_nem:.3f}", "Effect Size": row['Effect_size'],
                        "CONSENSUS": "✅" if (row['p_value'] < 0.05 and p_nem < 0.05) else "❌"
                    })
                st.table(con_data)
        
        with tab3:
            st.dataframe(m_df, use_container_width=True)
    else:
        st.error(f"Missing core data file: {files['main']}")
