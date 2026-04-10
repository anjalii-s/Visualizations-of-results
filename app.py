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
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        background-color: #f8fafc; 
    }
    
    /* Headers */
    h1, h2, h3 { 
        color: #0f172a; 
        font-weight: 700; 
        letter-spacing: -0.5px; 
    }
    
    /* Custom Metric Cards */
    .metric-card { 
        background-color: #ffffff; 
        border: 1px solid #e2e8f0; 
        border-radius: 8px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
    }
    .metric-value { 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: #0369a1; 
        margin: 10px 0; 
    }
    .metric-label { 
        font-size: 0.9rem; 
        color: #64748b; 
        font-weight: 500; 
        text-transform: uppercase; 
    }
    
    /* Leaderboard Cards */
    .lb-card { 
        padding: 20px; 
        border-radius: 12px; 
        text-align: center; 
        transition: transform 0.2s ease; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
    }
    .lb-card:hover { 
        transform: translateY(-5px); 
    }
    .lb-rank { 
        font-size: 2rem; 
        margin-bottom: 5px; 
    }
    .lb-method { 
        font-size: 1.3rem; 
        font-weight: 700; 
        color: #0f172a; 
        margin: 0; 
    }
    .lb-config { 
        font-size: 0.9rem; 
        color: #475569; 
        margin: 5px 0 15px 0; 
    }
    .lb-score { 
        font-size: 1.5rem; 
        font-weight: 700; 
        color: #0369a1; 
        margin: 0; 
    }
    
    /* Insight Box */
    .insight-box { 
        background-color: #ffffff; 
        border-left: 4px solid #0369a1; 
        padding: 15px 20px; 
        border-radius: 6px; 
        margin: 15px 0; 
        color: #334155; 
        font-size: 1rem; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
# Professional color palette optimized for both light and dark themes
METHOD_COLORS = {
    'SHAP': '#64748b',         # Slate Gray
    'Banzhaf': '#f59e0b',      # Amber
    'Myerson': '#10b981',      # Emerald
    'Owen-Domain': '#ef4444',  # Red
    'Owen-Data': '#8b5cf6',    # Violet
    'Owen-Model': '#ec4899',   # Pink
    'R-Myerson': '#3b82f6'     # Blue
}

# Updated dataset registry with correct file names
DATASET_REGISTRY = {
    "German Credit": {
        "main": "Ger_result.csv",
        "wilcoxon": "Ger_result_wilcoxon.csv",
        "nemenyi": "Ger_result_nemenyi.csv",
        "corr": "Ger_result_correlation.csv",
        "label": "Moderate Imbalance",
        "imb": 30.0,
        "description": "German credit dataset with 30% default rate, representing a moderately imbalanced credit risk scenario commonly found in European markets."
    },
    "Taiwan Credit": {
        "main": "TW_result.csv",
        "wilcoxon": "TW_result_wilcoxon.csv",
        "nemenyi": "TW_result_nemenyi.csv",
        "corr": "TW_result_correlation.csv",
        "label": "Moderate Imbalance",
        "imb": 22.12,
        "description": "Taiwan credit card default dataset with 22.12% default rate, capturing payment behavior patterns in Asian credit markets."
    },
    "Lending Club A (10%)": {
        "main": "LC_result10.csv",
        "wilcoxon": "LC_result_wilcoxon.csv",
        "nemenyi": "LC_result_nemenyi.csv",
        "corr": "LC_result_correlation.csv",
        "label": "Industry Standard",
        "imb": 10.0,
        "description": "Lending Club dataset with 10% default rate, representing the industry-standard imbalance level typical in peer-to-peer lending platforms."
    },
    "Lending Club B (4%)": {
        "main": "LC4_result(1).csv",
        "wilcoxon": "Lc66_wilcoxon_cliffs_results.csv",
        "nemenyi": "Lc66_nemenyi_results (1).csv",
        "corr": "Lc66_auc_I_correlation.csv",
        "label": "Severe Imbalance",
        "imb": 4.01,
        "description": "Lending Club dataset with 4% default rate, representing a severely imbalanced scenario where traditional XAI methods begin to show significant instability."
    },
    "Coursera Loans": {
        "main": "Coursera_result.csv",
        "wilcoxon": "Coursera_result_wilcoxon.csv",
        "nemenyi": "Coursera_result_nemenyi.csv",
        "corr": "Coursera_result_correlation.csv",
        "label": "Extreme Imbalance",
        "imb": 1.0,
        "description": "Educational loan dataset with 1% default rate, representing an extreme imbalance scenario that critically tests the robustness of XAI frameworks."
    }
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
@st.cache_data
def load_data(path, is_index=False):
    """Robust file loader handling missing files and cleaning 'nan' Samplers."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0 if is_index else None)
        if 'Sampler' in df.columns:
            df['Sampler'] = df['Sampler'].astype(str).replace(['nan', 'NaN', 'None', 'nan '], 'None')
        return df
    except Exception:
        return None

def color_effect(val):
    """Color coding for effect sizes"""
    v = str(val).lower()
    if v == 'large': 
        return 'color: #10b981; font-weight: bold;'
    if v == 'medium': 
        return 'color: #f59e0b; font-weight: bold;'
    return 'color: #64748b;'

def color_consensus(val):
    """Color coding for consensus results"""
    if '✓' in str(val): 
        return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
    return 'color: #94a3b8;'

def get_wilcoxon_sig(sig_val, p_val):
    """Safely determines if Wilcoxon is significant."""
    sig_str = str(sig_val).lower()
    if '✓' in sig_str or 'yes' in sig_str or 'true' in sig_str: 
        return True
    try:
        return float(p_val) < 0.05
    except:
        return False

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("### 🧭 Navigation")
views = ["📊 Cross-Dataset Synthesis", "🏆 Leaderboards"] + list(DATASET_REGISTRY.keys())
selection = st.sidebar.radio("Select View:", views)
st.sidebar.markdown("---")
st.sidebar.caption("Ensemble Learning & Coalition-aware Explainability for Imbalanced Credit Default Prediction")

# ==========================================
# VIEW 1: CROSS-DATASET SYNTHESIS
# ==========================================
if selection == "📊 Cross-Dataset Synthesis":
    st.title("Ensemble Learning and Coalition-aware Explainability for Imbalanced Credit Default")
    
    st.markdown("""
    <div class='insight-box'>
    <b>Executive Summary:</b> This dashboard unifies the results of seven attribution methods across five financial datasets. 
    By pressing the <b>Play</b> button below, you can visually track how standard Explainable AI (XAI) methods degrade as the dataset becomes increasingly imbalanced (from 30% down to 1% default rate), highlighting the robustness of the <b>R-Myerson</b> algorithm.
    </div>
    """, unsafe_allow_html=True)
    
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            df_copy = df.copy()
            df_copy['Imbalance'] = cfg['imb']
            df_copy['Dataset'] = f"{name} ({cfg['imb']}%)"
            df_copy['Config'] = df_copy['Method'] + "_" + df_copy['Model'] + "_" + df_copy['Sampler']
            global_results.append(df_copy)
            
    if global_results:
        combined = pd.concat(global_results).sort_values('Imbalance', ascending=False)
        
        st.subheader("Dynamic Pareto Front Shift Across Imbalance Levels")
        st.caption("Watch how the accuracy-interpretability trade-off evolves as class imbalance increases. Press ▶ to animate.")
        
        # Animated Bubble Chart
        fig_anim = px.scatter(
            combined, 
            x="AUC", y="I", 
            animation_frame="Dataset", 
            animation_group="Config",
            color="Method", 
            symbol="Model",
            size="S(α=0.5)",
            hover_name="Sampler",
            color_discrete_map=METHOD_COLORS,
            range_x=[combined['AUC'].min() - 0.02, combined['AUC'].max() + 0.02],
            range_y=[0.0, 1.05]
        )
        
        fig_anim.update_traces(marker=dict(line=dict(width=1, color='white')), opacity=0.85)
        fig_anim.update_layout(
            template="plotly_white", 
            height=600,
            xaxis_title="Predictive Accuracy (AUC)",
            yaxis_title="Interpretability (I-Score)"
        )
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
        st.plotly_chart(fig_anim, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Performance-Interpretability (Mean S-Score)")
        st.caption("Bar heights represent the **average** S-score for a given method across all base models and samplers for that dataset.")
        
        # Aggregate mean values
        summary_df = combined.groupby(['Dataset', 'Method', 'Imbalance'])['S(α=0.5)'].mean().reset_index()
        summary_df = summary_df.sort_values('Imbalance', ascending=False)
        
        # Calculate method order
        method_order = summary_df.groupby('Method')['S(α=0.5)'].mean().sort_values(ascending=False).index.tolist()
        
        fig_bar = px.bar(
            summary_df, 
            x='Dataset', 
            y='S(α=0.5)', 
            color='Method', 
            barmode='group',
            color_discrete_map=METHOD_COLORS,
            category_orders={
                "Dataset": summary_df['Dataset'].unique().tolist(),
                "Method": method_order
            }
        )
        
        fig_bar.update_traces(marker_line_width=1, marker_line_color="white")
        fig_bar.update_layout(
            xaxis_title="Datasets (Decreasing Default Rate →)", 
            yaxis_title="Mean S(α=0.5) Score",
            template="plotly_white",
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# VIEW 2: LEADERBOARDS
# ==========================================
elif selection == "🏆 Leaderboards":
    st.title("🏆 Global & Per-Dataset Leaderboards")
    st.markdown("Dynamic rankings of top model-sampler configurations across all datasets and methods.")
    
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            df_copy = df.copy()
            df_copy['Dataset_Name'] = name
            df_copy['Imbalance'] = cfg['imb']
            df_copy['Config'] = df_copy['Model'] + '–' + df_copy['Sampler'].fillna('None')
            global_results.append(df_copy)
            
    if global_results:
        combined = pd.concat(global_results)
        metrics_of_interest = ['AUC', 'I', 'S(α=0.5)']
        
        st.markdown("### 🌍 Top Configurations Aggregated Across All Datasets")
        st.caption("Calculated by **averaging** performance across all 5 datasets and all 7 explainability methods. Ranked in descending order.")
        
        overall_df = pd.DataFrame({"Rank": range(1, 6)})
        for metric in metrics_of_interest:
            top_5 = combined.groupby('Config')[metric].mean().reset_index()
            top_5 = top_5.sort_values(by=metric, ascending=False).head(5).reset_index(drop=True)
            overall_df[f"Config ({metric})"] = top_5['Config']
            overall_df[f"{metric} Score"] = top_5[metric].apply(lambda x: f"{x:.3f}")
            
        st.dataframe(overall_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Top Configurations Per Dataset")
        st.caption("Calculated by **averaging** the performance of Model-Sampler configurations across all 7 explainability methods for each specific dataset. Ranked in descending order.")
        
        for ds_name in DATASET_REGISTRY.keys():
            ds_matches = combined[combined['Dataset_Name'] == ds_name]
            if not ds_matches.empty:
                st.markdown(f"#### {ds_name} *(Default Rate: {DATASET_REGISTRY[ds_name]['imb']}%)*")
                st.caption(DATASET_REGISTRY[ds_name]['description'])
                ds_df = pd.DataFrame({"Rank": range(1, 4)})
                
                for metric in metrics_of_interest:
                    top_3 = ds_matches.groupby('Config')[metric].mean().reset_index()
                    top_3 = top_3.sort_values(by=metric, ascending=False).head(3).reset_index(drop=True)
                    ds_df[f"Config ({metric})"] = top_3['Config']
                    ds_df[f"{metric} Score"] = top_3[metric].apply(lambda x: f"{x:.3f}")
                    
                st.dataframe(ds_df, hide_index=True, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# VIEW 3: SPECIFIC DATASET DASHBOARD
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.title(f"{selection}")
    st.caption(f"**{cfg['label']}** — Default Rate: {cfg['imb']}%")
    st.markdown(f"<div class='insight-box'>{cfg['description']}</div>", unsafe_allow_html=True)
    
    # Load all files
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is None:
        st.error(f"⚠️ Primary results file ({cfg['main']}) not found. Please ensure the file is uploaded.")
        st.stop()

    # --- TOP 3 PODIUM ---
    st.markdown("### 🏆 Top 3 Configurations by S-Score")
    st.caption("Ranked by the single highest **absolute peak** S(α=0.5) score achieved by any specific row (Method + Model + Sampler combination).")
    
    top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
    
    cols = st.columns(3)
    bg_styles = [
        "background: linear-gradient(180deg, #fef3c7 0%, #ffffff 100%); border: 2px solid #fbbf24;",  # Gold
        "background: linear-gradient(180deg, #e5e7eb 0%, #ffffff 100%); border: 2px solid #9ca3af;",  # Silver
        "background: linear-gradient(180deg, #fed7aa 0%, #ffffff 100%); border: 2px solid #fb923c;"   # Bronze
    ]
    medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    
    for i in range(len(top3)):
        with cols[i]:
            st.markdown(f"""
            <div class='lb-card' style='{bg_styles[i]}'>
                <div class='lb-rank' style='color: #0f172a;'>{medals[i]}</div>
                <h3 class='lb-method' style='color: #0f172a;'>{top3.loc[i, 'Method']}</h3>
                <p class='lb-config' style='color: #475569;'>{top3.loc[i, 'Model']} + {top3.loc[i, 'Sampler']}</p>
                <h2 class='lb-score' style='color: #0369a1;'>{top3.loc[i, 'S(α=0.5)']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS ---
    t1, t2, t3, t4, t5 = st.tabs([
        "🎯 Accuracy vs Interpretability", 
        "🧩 Q vs I Analysis", 
        "🔬 Statistical Significance", 
        "🏅 Top Model-Samplers",
        "🗄️ Raw Data"
    ])
    
    # TAB 1: AUC vs I
    with t1:
        c1, c2 = st.columns([1.8, 1])
        with c1:
            fig_p = px.scatter(
                main_df, 
                x='AUC', 
                y='I', 
                color='Method', 
                symbol='Model',
                hover_data=['Sampler'], 
                color_discrete_map=METHOD_COLORS,
                title="Pareto Front: Accuracy vs. Interpretability"
            )
            fig_p.update_traces(marker=dict(size=14, opacity=0.85, line=dict(width=1, color='white')))
            fig_p.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            try:
                clean_df = main_df.dropna(subset=['AUC', 'I'])
                if len(clean_df) >= 3:
                    auc_i_rho = clean_df['AUC'].corr(clean_df['I'], method='spearman')
                else:
                    auc_i_rho = 0.0
            except:
                auc_i_rho = 0.0
                
            st.markdown(f"""
            <div class='insight-box'>
            <b>Trade-Off Analysis:</b><br><br>
            The Spearman rank correlation between Predictive Accuracy (AUC) and Interpretability (I-Score) is <b>ρ = {auc_i_rho:.3f}</b>.<br><br>
            <i>Interpretation:</i> {'A negative correlation indicates a classical trade-off: highly accurate models tend to have less stable explanations.' if auc_i_rho < -0.1 else 'A positive or near-zero correlation suggests that for this dataset, we can maintain stable explanations without sacrificing predictive power.'}
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: Q vs I
    with t2:
        st.markdown("### Does Better Feature Grouping Lead to Better Explanations?")
        st.caption("Analyzing the relationship between coalition quality (Q) and interpretability (I) for Owen-based methods.")
        
        owen_df = main_df[main_df['Method'].isin(['Owen-Domain', 'Owen-Data', 'Owen-Model'])].copy()
        owen_clean = owen_df.dropna(subset=['Q', 'I']).copy()
        
        if len(owen_clean) >= 3:
            qc1, qc2 = st.columns([1.8, 1])
            with qc1:
                fig_q = px.scatter(
                    owen_clean, 
                    x='Q', 
                    y='I', 
                    color='Method',
                    symbol='Model', 
                    hover_data=['Sampler'], 
                    color_discrete_map=METHOD_COLORS,
                    title="Group Quality (Q) vs Interpretability (I)"
                )
                fig_q.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
                fig_q.update_layout(
                    template="plotly_white", 
                    height=450,
                    hovermode="x unified"  # <-- This forces Plotly to show all overlapping dots
                )
                st.plotly_chart(fig_q, use_container_width=True)
            
            with qc2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                q_rho = owen_clean['Q'].corr(owen_clean['I'], method='spearman')
                if np.isnan(q_rho): 
                    q_rho = 0.0
                
                st.markdown(f"""
                <div class='insight-box'>
                <b>Derivation Method:</b><br>
                This relationship uses the <b>Spearman rank correlation (ρ)</b>. We pair Group Quality (Q) and Interpretability (I) scores for each Owen variant configuration, rank them, and measure their monotonic relationship.<br><br>
                <b>Analysis:</b><br>
                Spearman ρ = <b>{q_rho:.3f}</b><br><br>
                <i>Interpretation:</i> {'A strong positive relationship confirms that algorithmically defining better feature coalitions (higher Q) directly leads to more stable attributions (higher I).' if q_rho > 0.3 else 'The relationship is weak, indicating that baseline distribution rules impact stability more than coalition boundaries.'}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("⚠️ Insufficient data points to compute Q vs I correlation for Owen variants in this dataset.")

    # TAB 3: STATISTICAL SIGNIFICANCE
    with t3:
        st.markdown("### Statistical Rigor: Correlation & Pairwise Comparisons")
        
        if corr_df is not None and not corr_df.empty:
            c_rho, c_p = corr_df['Spearman_rho'].iloc[0], corr_df['Spearman_p'].iloc[0]
            k_tau, k_p = corr_df['Kendall_tau'].iloc[0], corr_df['Kendall_p'].iloc[0]
            
            st.markdown("#### Accuracy vs. Interpretability Correlation")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Spearman ρ</div>
                    <div class='metric-value'>{c_rho:.3f}</div>
                    <div style='color: #64748b; font-size:0.85rem;'>p-value: {c_p:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Kendall τ</div>
                    <div class='metric-value'>{k_tau:.3f}</div>
                    <div style='color: #64748b; font-size:0.85rem;'>p-value: {k_p:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### Rigorous Pairwise Comparison")
        st.caption("A *True Consensus Difference* is established only if BOTH the Wilcoxon test AND the Nemenyi post-hoc test confirm significance (p < 0.05).")
        
        sc1, sc2 = st.columns([1.1, 1])
        
        with sc1:
            st.markdown("##### Wilcoxon & Consensus Table")
            if wil_df is not None and nem_df is not None:
                consensus_data = []
                for _, row in wil_df.iterrows():
                    m1, m2 = row['Method1'], row['Method2']
                    eff = row.get('Effect_size', 'N/A')
                    
                    w_is_sig = get_wilcoxon_sig(row.get('Significant', ''), row.get('p_value', 1.0))
                    w_display = "✓ Yes" if w_is_sig else "✗ No"
                    
                    n_p = 1.0
                    try: 
                        n_p = float(nem_df.loc[m1, m2])
                    except KeyError:
                        try: 
                            n_p = float(nem_df.loc[m2, m1])
                        except KeyError: 
                            pass
                    
                    n_sig_bool = n_p < 0.05
                    consensus = "✓ Yes" if (w_is_sig and n_sig_bool) else "✗ No"
                    
                    consensus_data.append({
                        "Method 1": m1, 
                        "Method 2": m2,
                        "Wilcoxon Sig.": w_display,
                        "Effect Size": str(eff).title(),
                        "Consensus Diff": consensus
                    })
                
                st.dataframe(
                    pd.DataFrame(consensus_data).style
                    .map(color_effect, subset=['Effect Size'])
                    .map(color_consensus, subset=['Wilcoxon Sig.', 'Consensus Diff']),
                    hide_index=True, 
                    height=450, 
                    use_container_width=True
                )
            else: 
                st.warning("⚠️ Both Wilcoxon and Nemenyi files are required to display the consensus table.")
            
        with sc2:
            st.markdown("##### Nemenyi Post-hoc Heatmap")
            if nem_df is not None:
                colorscale = [
                    [0.0, '#10b981'],
                    [0.049, '#10b981'],
                    [0.05, '#f1f5f9'],
                    [1.0, '#f1f5f9']
                ]
                fig_nem = px.imshow(
                    nem_df, 
                    text_auto=".3f", 
                    color_continuous_scale=colorscale, 
                    zmin=0, 
                    zmax=1.0
                )
                fig_nem.update_layout(
                    height=450, 
                    margin=dict(t=10, b=0, l=0, r=0), 
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_nem, use_container_width=True)
                st.markdown("<small><b>Reading Guide:</b> <span style='color:#10b981; font-weight:bold;'>Green cells (p < 0.05)</span> indicate statistically significant differences between methods. Light gray cells indicate no significant difference.</small>", unsafe_allow_html=True)
            else: 
                st.warning("⚠️ Nemenyi data file not found.")

    # TAB 4: TOP MODEL-SAMPLERS
    with t4:
        st.markdown("### Top 5 Model-Sampler Configurations")
        st.caption("Ranked by their **average** score across all 7 XAI methods. This identifies the most consistently robust predictive pipelines.")
        
        main_df['Model_Sampler'] = main_df['Model'] + '_' + main_df['Sampler'].fillna('None')
        metrics_of_interest = ['AUC', 'I', 'S(α=0.5)']
        
        m_cols = st.columns(3)
        for i, metric in enumerate(metrics_of_interest):
            with m_cols[i]:
                st.markdown(f"<h4 style='text-align: center; color: #0f172a;'>Top 5 by {metric}</h4>", unsafe_allow_html=True)
                top_5 = main_df.groupby('Model_Sampler')[metric].mean().reset_index()
                top_5 = top_5.sort_values(by=metric, ascending=False).head(5)
                top_5 = top_5.rename(columns={'Model_Sampler': 'Configuration', metric: 'Avg Score'})
                
                st.dataframe(
                    top_5.style.format({'Avg Score': "{:.4f}"}), 
                    hide_index=True, 
                    use_container_width=True
                )
                
                avg_best = top_5['Avg Score'].mean()
                st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #475569; margin-top: 10px;'><b>Average of Top 5:</b> {avg_best:.4f}</div>", unsafe_allow_html=True)

    # TAB 5: RAW DATA
    with t5:
        st.markdown("### Complete Dataset")
        st.caption("View and download the complete raw analytical data for this dataset.")
        st.dataframe(main_df, use_container_width=True)
        
        csv_data = main_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data, 
            file_name=f"{selection.replace(' ', '_')}_data.csv", 
            mime="text/csv"
        )
