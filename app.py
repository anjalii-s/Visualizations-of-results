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
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    /* Headers */
    h1, h2, h3 { color: #0f172a; font-weight: 700; letter-spacing: -0.5px; }
    
    /* Custom Metric Cards */
    .metric-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0369a1; margin: 10px 0; }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 500; text-transform: uppercase; }
    
    /* Leaderboard Cards */
    .lb-card { padding: 20px; border-radius: 12px; text-align: center; transition: transform 0.2s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .lb-card:hover { transform: translateY(-5px); }
    .lb-rank { font-size: 2rem; margin-bottom: 5px; }
    .lb-method { font-size: 1.3rem; font-weight: 700; color: #0f172a; margin: 0; }
    .lb-config { font-size: 0.9rem; color: #475569; margin: 5px 0 15px 0; }
    .lb-score { font-size: 1.5rem; font-weight: 700; color: #0369a1; margin: 0; }
    
    /* Insight Box */
    .insight-box { background-color: #ffffff; border-left: 4px solid #0369a1; padding: 15px 20px; border-radius: 6px; margin: 15px 0; color: #334155; font-size: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    
    /* Warning Box */
    .warning-box { background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px 20px; border-radius: 6px; margin: 15px 0; color: #92400e; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
METHOD_COLORS = {
    'SHAP': '#64748b',         # Slate (Baseline)
    'Banzhaf': '#d97706',      # Amber
    'Myerson': '#059669',      # Emerald
    'Owen-Domain': '#dc2626',  # Red
    'Owen-Data': '#7c3aed',    # Purple
    'Owen-Model': '#db2777',   # Fuchsia
    'R-Myerson': '#0284c7'     # Sky Blue (Proposed)
}

DATASET_REGISTRY = {
    "German Credit": {
        "main": "Ger_result.csv",
        "wilcoxon": "Ger_result_wilcoxon.csv",
        "nemenyi": "Ger_result_nemenyi.csv",
        "corr": "Ger_result_correlation.csv",
        "label": "Moderate Imbalance",
        "imb": 30.0,
        "note": "Categorical-heavy features"
    },
    "Taiwan Credit": {
        "main": "TW_result.csv",
        "wilcoxon": "TW_result_wilcoxon.csv",
        "nemenyi": "TW_result_nemenyi.csv",
        "corr": "TW_result_correlation.csv",
        "label": "Moderate Imbalance",
        "imb": 22.12,
        "note": "Temporal repayment features"
    },
    "Lending Club A": {
        "main": "LC_result10.csv",
        "wilcoxon": "LC_result_wilcoxon.csv",
        "nemenyi": "LC_result_nemenyi.csv",
        "corr": "LC_result_correlation.csv",
        "label": "Industry Standard",
        "imb": 10.0,
        "note": "Geographic dummy features"
    },
    "Lending Club B": {
        "main": "LC4_result__1_.csv",
        "wilcoxon": "Lc66_wilcoxon_cliffs_results.csv",
        "nemenyi": "Lc66_nemenyi_results__1_.csv",
        "corr": "Lc66_auc_I_correlation.csv",
        "label": "Severe Imbalance",
        "imb": 4.01,
        "note": "High-dimensional engineered features"
    },
    "Coursera Loans": {
        "main": "Coursera_result.csv",
        "wilcoxon": "Coursera_result_wilcoxon.csv",
        "nemenyi": "Coursera_result_nemenyi.csv",
        "corr": "Coursera_result_correlation.csv",
        "label": "Extreme Imbalance",
        "imb": 1.0,
        "note": "Low-dimensional, clean features"
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
            # Clean the Sampler column strictly
            df['Sampler'] = df['Sampler'].astype(str).replace(['nan', 'NaN', 'None', '', ' '], 'None')
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {str(e)}")
        return None

def color_effect(val):
    v = str(val).lower()
    if v == 'large': return 'color: #059669; font-weight: bold;'
    if v == 'medium': return 'color: #d97706; font-weight: bold;'
    return 'color: #64748b;'

def color_consensus(val):
    if '✓' in str(val): return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
    return 'color: #94a3b8;'

def get_wilcoxon_sig(sig_val, p_val):
    """Safely determines if Wilcoxon is significant."""
    sig_str = str(sig_val).lower()
    if '✓' in sig_str or 'yes' in sig_str or 'true' in sig_str: return True
    try:
        return float(p_val) < 0.05
    except:
        return False

def interpret_correlation(rho, p_val):
    """Interpret correlation strength and significance."""
    if p_val >= 0.05:
        return f"**No significant correlation** (ρ = {rho:.3f}, p = {p_val:.3f}). AUC and interpretability are **statistically independent** for this dataset."
    
    strength = abs(rho)
    if strength < 0.1:
        return f"**Negligible correlation** (ρ = {rho:.3f}, p = {p_val:.3f}). Despite statistical significance, the relationship is practically meaningless."
    elif strength < 0.3:
        return f"**Weak {'negative' if rho < 0 else 'positive'} correlation** (ρ = {rho:.3f}, p = {p_val:.3f}). A slight tendency exists but is not robust."
    elif strength < 0.5:
        return f"**Moderate {'negative' if rho < 0 else 'positive'} correlation** (ρ = {rho:.3f}, p = {p_val:.3f}). A meaningful relationship exists."
    else:
        return f"**Strong {'negative' if rho < 0 else 'positive'} correlation** (ρ = {rho:.3f}, p = {p_val:.3f}). A robust relationship."

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
    st.title("Cross-Dataset Performance Analysis")
    
    st.markdown("""
    <div class='insight-box'>
    <b>Dashboard Overview:</b> This interactive visualization synthesizes results from seven game-theoretic attribution methods 
    across five credit risk datasets with varying imbalance ratios (30% → 1% default rate). 
    Use the animation controls to observe how method performance shifts as class imbalance increases.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='warning-box'>
    ⚠️ <b>Important Methodological Notes:</b>
    <ul>
    <li><b>R-Myerson's perfect scores</b> (Kuncheva=1.0, Cosine=1.0) are guaranteed by its uniform redistribution design (Eq. 2.21), not empirical discovery.</li>
    <li><b>Banzhaf approximation</b> used conservative sampling (n_inst=2, n_coal=2). Performance reflects this budget constraint.</li>
    <li><b>AUC-I correlations</b> vary substantially across datasets (ρ from -0.39 to +0.04), indicating dataset-conditional relationships.</li>
    </ul>
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
        
        st.subheader("Animated Pareto Front: Accuracy vs Interpretability")
        st.caption("Press ▶️ to animate across datasets ordered by decreasing default rate. Bubble size represents S(α=0.5) trade-off score.")
        
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
            hover_data={"AUC": ":.3f", "I": ":.3f", "S(α=0.5)": ":.3f"},
            color_discrete_map=METHOD_COLORS,
            range_x=[combined['AUC'].min() - 0.02, combined['AUC'].max() + 0.02],
            range_y=[0.0, 1.05]
        )
        
        fig_anim.update_traces(marker=dict(line=dict(width=1, color='white')), opacity=0.85)
        fig_anim.update_layout(
            template="plotly_white", 
            height=600,
            xaxis_title="Predictive Performance (AUC-ROC)",
            yaxis_title="Interpretability Score (I)",
            font=dict(size=12)
        )
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
        st.plotly_chart(fig_anim, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Mean Trade-off Score by Method Across Datasets")
        st.caption("Bar heights represent average S(α=0.5) across all model-sampler configurations for each dataset.")
        
        summary_df = combined.groupby(['Dataset', 'Method', 'Imbalance'])['S(α=0.5)'].mean().reset_index()
        summary_df = summary_df.sort_values('Imbalance', ascending=False)
        
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
            xaxis_title="Datasets (Ordered by Decreasing Default Rate →)", 
            yaxis_title="Mean S(α=0.5) Score",
            template="plotly_white",
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Correlation Summary Table
        st.markdown("---")
        st.subheader("AUC-Interpretability Correlation Summary")
        st.caption("Statistical relationship between predictive performance and explanation quality varies by dataset structure.")
        
        corr_summary = []
        for name, cfg in DATASET_REGISTRY.items():
            corr_df = load_data(cfg['corr'])
            if corr_df is not None and not corr_df.empty:
                corr_summary.append({
                    "Dataset": name,
                    "Imbalance": f"{cfg['imb']}%",
                    "Spearman ρ": f"{corr_df['Spearman_rho'].iloc[0]:.3f}",
                    "p-value": f"{corr_df['Spearman_p'].iloc[0]:.4f}",
                    "Significant": "✓" if corr_df['Spearman_p'].iloc[0] < 0.05 else "✗",
                    "Interpretation": "Negative" if corr_df['Spearman_rho'].iloc[0] < -0.1 else "Positive" if corr_df['Spearman_rho'].iloc[0] > 0.1 else "None"
                })
        
        if corr_summary:
            st.dataframe(pd.DataFrame(corr_summary), hide_index=True, use_container_width=True)
            
            st.markdown("""
            <div class='insight-box'>
            <b>Key Finding:</b> Only 3 datasets show significant correlations, and their directions vary. 
            This heterogeneity confirms that <b>AUC and interpretability are largely orthogonal objectives</b> 
            rather than exhibiting a universal trade-off. Dataset structure (feature types, correlations, dimensionality) 
            moderates any relationship.
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# VIEW 2: LEADERBOARDS
# ==========================================
elif selection == "🏆 Leaderboards":
    st.title("🏆 Performance Rankings")
    st.markdown("Dynamic rankings of model-sampler configurations across metrics.")
    
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
        st.caption("Averaged performance across all 5 datasets and all 7 explainability methods. Ranked in descending order.")
        
        overall_df = pd.DataFrame({"Rank": range(1, 6)})
        for metric in metrics_of_interest:
            top_5 = combined.groupby('Config')[metric].mean().reset_index()
            top_5 = top_5.sort_values(by=metric, ascending=False).head(5).reset_index(drop=True)
            overall_df[f"Config ({metric})"] = top_5['Config']
            overall_df[f"{metric} Score"] = top_5[metric].apply(lambda x: f"{x:.4f}")
            
        st.dataframe(overall_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Top Configurations Per Dataset")
        st.caption("Averaged across all 7 explainability methods for each specific dataset.")
        
        for ds_name in DATASET_REGISTRY.keys():
            ds_matches = combined[combined['Dataset_Name'] == ds_name]
            if not ds_matches.empty:
                st.markdown(f"#### {ds_name} *(Default Rate: {DATASET_REGISTRY[ds_name]['imb']}% | {DATASET_REGISTRY[ds_name]['note']})*")
                ds_df = pd.DataFrame({"Rank": range(1, 4)})
                
                for metric in metrics_of_interest:
                    top_3 = ds_matches.groupby('Config')[metric].mean().reset_index()
                    top_3 = top_3.sort_values(by=metric, ascending=False).head(3).reset_index(drop=True)
                    ds_df[f"Config ({metric})"] = top_3['Config']
                    ds_df[f"{metric} Score"] = top_3[metric].apply(lambda x: f"{x:.4f}")
                    
                st.dataframe(ds_df, hide_index=True, use_container_width=True)

# ==========================================
# VIEW 3: SPECIFIC DATASET DASHBOARD
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.title(f"{selection}")
    st.caption(f"{cfg['label']} ({cfg['imb']}% default rate) | {cfg['note']}")
    
    # Load all files for the dataset
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is None:
        st.error(f"⚠️ Primary results file ({cfg['main']}) not found.")
        st.stop()

    # --- TOP 3 LEADERBOARD (ABSOLUTE) ---
    st.markdown("### 🏆 Top 3 Configurations (Absolute Peak Scores)")
    st.caption("Ranked by the single highest S(α=0.5) score achieved by any specific configuration (Model + Sampler + Method).")
    top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
    
    cols = st.columns(3)
    bg_styles = [
        "background: linear-gradient(180deg, #fef08a 0%, #ffffff 100%); border: 2px solid #fde047;",
        "background: linear-gradient(180deg, #e2e8f0 0%, #ffffff 100%); border: 2px solid #cbd5e1;",
        "background: linear-gradient(180deg, #fed7aa 0%, #ffffff 100%); border: 2px solid #fdba74;"
    ]
    medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    
    for i in range(min(len(top3), 3)):
        with cols[i]:
            st.markdown(f"""
            <div class='lb-card' style='{bg_styles[i]}'>
                <div class='lb-rank'>{medals[i]}</div>
                <h3 class='lb-method'>{top3.loc[i, 'Method']}</h3>
                <p class='lb-config'>{top3.loc[i, 'Model']} + {top3.loc[i, 'Sampler']}</p>
                <h2 class='lb-score'>{top3.loc[i, 'S(α=0.5)']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS NAVIGATION ---
    t1, t2, t3, t4, t5 = st.tabs([
        "🎯 Accuracy vs Interpretability", 
        "🧩 Q vs I Analysis", 
        "🔬 Statistical Significance", 
        "🏅 Top Model-Samplers",
        "🗄️ Raw Data"
    ])
    
    # ==================================
    # TAB 1: AUC vs I (PARETO)
    # ==================================
    with t1:
        c1, c2 = st.columns([1.8, 1])
        with c1:
            fig_p = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model',
                             hover_data=['Sampler'], color_discrete_map=METHOD_COLORS,
                             title="Pareto Front: Predictive Performance vs Interpretability")
            fig_p.update_traces(marker=dict(size=14, opacity=0.85, line=dict(width=1, color='white')))
            fig_p.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if corr_df is not None and not corr_df.empty:
                rho = corr_df['Spearman_rho'].iloc[0]
                p_val = corr_df['Spearman_p'].iloc[0]
                interpretation = interpret_correlation(rho, p_val)
                
                st.markdown(f"""
                <div class='insight-box'>
                <b>Correlation Analysis:</b><br><br>
                {interpretation}
                <br><br>
                <i>Methodological Note:</i> Spearman's ρ measures monotonic (rank-order) relationships. 
                Values near zero indicate independence, not necessarily the absence of any relationship.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Correlation data not available for this dataset.")

    # ==================================
    # TAB 2: Q vs I (GROUP QUALITY)
    # ==================================
    with t2:
        st.markdown("### Group Quality vs Interpretability (Owen Variants Only)")
        
        owen_df = main_df[main_df['Method'].isin(['Owen-Domain', 'Owen-Data', 'Owen-Model'])].copy()
        
        # Check if Q column exists
        if 'Q' not in owen_df.columns:
            st.warning(f"⚠️ Group quality metric (Q) not found in {cfg['main']}. This analysis requires the Q column for Owen variants.")
        else:
            owen_clean = owen_df.dropna(subset=['Q', 'I']).copy()
            
            if len(owen_clean) >= 3:
                qc1, qc2 = st.columns([1.8, 1])
                with qc1:
                    fig_q = px.scatter(owen_clean, x='Q', y='I', color='Method',
                                       symbol='Model', hover_data=['Sampler'], 
                                       color_discrete_map=METHOD_COLORS,
                                       title="Group Quality (Q) vs Interpretability (I)")
                    fig_q.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
                    fig_q.update_layout(template="plotly_white", height=450)
                    st.plotly_chart(fig_q, use_container_width=True)
                
                with qc2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    q_rho = owen_clean['Q'].corr(owen_clean['I'], method='spearman')
                    if np.isnan(q_rho): q_rho = 0.0
                    
                    st.markdown(f"""
                    <div class='insight-box'>
                    <b>Spearman Correlation:</b> ρ = {q_rho:.3f}
                    <br><br>
                    <i>Interpretation:</i> {'Strong positive correlation (ρ > 0.5) indicates that better-defined feature coalitions directly improve attribution stability.' if q_rho > 0.5 else 'Moderate correlation (0.3 < ρ ≤ 0.5) suggests grouping quality influences but does not dominate interpretability.' if q_rho > 0.3 else 'Weak or no correlation (ρ ≤ 0.3) indicates that redistribution dynamics matter more than coalition boundaries for this dataset structure.'}
                    <br><br>
                    <b>Dataset Context:</b> {cfg['note']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⚠️ Insufficient valid Q and I values for correlation analysis (n < 3).")

    # ==================================
    # TAB 3: STATISTICAL SIGNIFICANCE
    # ==================================
    with t3:
        # Display AUC-I Correlations prominently
        if corr_df is not None and not corr_df.empty:
            c_rho, c_p = corr_df['Spearman_rho'].iloc[0], corr_df['Spearman_p'].iloc[0]
            k_tau, k_p = corr_df['Kendall_tau'].iloc[0], corr_df['Kendall_p'].iloc[0]
            
            st.markdown("### AUC vs Interpretability: Statistical Independence Test")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Spearman ρ</div>
                    <div class='metric-value'>{c_rho:.3f}</div>
                    <div style='color: #64748b; font-size:0.85rem;'>p = {c_p:.4f} {'(sig.)' if c_p < 0.05 else '(n.s.)'}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Kendall τ</div>
                    <div class='metric-value'>{k_tau:.3f}</div>
                    <div style='color: #64748b; font-size:0.85rem;'>p = {k_p:.4f} {'(sig.)' if k_p < 0.05 else '(n.s.)'}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Pairwise Method Comparison")
        st.markdown("**Consensus Difference** requires BOTH Wilcoxon signed-rank (p < 0.05) AND Nemenyi post-hoc (p < 0.05) agreement.")
        
        sc1, sc2 = st.columns([1.1, 1])
        
        with sc1:
            st.markdown("#### Wilcoxon Tests with Effect Sizes")
            if wil_df is not None and nem_df is not None:
                consensus_data = []
                for _, row in wil_df.iterrows():
                    m1, m2 = row['Method1'], row['Method2']
                    eff = row['Effect_size']
                    
                    w_is_sig = get_wilcoxon_sig(row.get('Significant', ''), row.get('p_value', 1.0))
                    w_display = "✓ Yes" if w_is_sig else "✗ No"
                    
                    # Safe Nemenyi lookup
                    n_p = 1.0
                    try: 
                        n_p = float(nem_df.loc[m1, m2])
                    except (KeyError, ValueError):
                        try: 
                            n_p = float(nem_df.loc[m2, m1])
                        except (KeyError, ValueError): 
                            pass
                    
                    n_sig_bool = n_p < 0.05
                    consensus = "✓ Yes" if (w_is_sig and n_sig_bool) else "✗ No"
                    
                    consensus_data.append({
                        "Method 1": m1, "Method 2": m2,
                        "Wilcoxon Sig.": w_display,
                        "Effect Size": str(eff).title(),
                        "Consensus Diff": consensus
                    })
                
                st.dataframe(
                    pd.DataFrame(consensus_data).style
                    .map(color_effect, subset=['Effect Size'])
                    .map(color_consensus, subset=['Wilcoxon Sig.', 'Consensus Diff']),
                    hide_index=True, height=450, use_container_width=True
                )
            else: 
                st.warning("Both Wilcoxon and Nemenyi CSV files required for consensus analysis.")
            
        with sc2:
            st.markdown("#### Nemenyi Post-hoc p-values")
            if nem_df is not None:
                # Green for significant, gray for non-significant
                colorscale = [
                    [0.0, '#10b981'],
                    [0.049, '#10b981'],
                    [0.05, '#f1f5f9'],
                    [1.0, '#f1f5f9']
                ]
                fig_nem = px.imshow(nem_df, text_auto=".3f", color_continuous_scale=colorscale, zmin=0, zmax=1.0)
                fig_nem.update_layout(height=450, margin=dict(t=10, b=0, l=0, r=0), coloraxis_showscale=False)
                st.plotly_chart(fig_nem, use_container_width=True)
                st.markdown("<small><b>Legend:</b> <span style='color:#10b981; font-weight:bold;'>Green (p < 0.05)</span> = statistically significant difference | Gray = no significant difference</small>", unsafe_allow_html=True)
            else: 
                st.warning("Nemenyi data not found.")

    # ==================================
    # TAB 4: TOP MODEL-SAMPLERS
    # ==================================
    with t4:
        st.markdown("### Top 5 Model-Sampler Configurations (Averaged Across Methods)")
        st.caption("Identifies the most robust predictive pipelines by averaging performance across all 7 XAI methods.")
        
        main_df['Model_Sampler'] = main_df['Model'] + '_' + main_df['Sampler'].fillna('None')
        metrics_of_interest = ['AUC', 'I', 'S(α=0.5)']
        
        m_cols = st.columns(3)
        for i, metric in enumerate(metrics_of_interest):
            with m_cols[i]:
                st.markdown(f"<h4 style='text-align: center; color: #0f172a;'>Top 5 by {metric}</h4>", unsafe_allow_html=True)
                top_5 = main_df.groupby('Model_Sampler')[metric].mean().reset_index()
                top_5 = top_5.sort_values(by=metric, ascending=False).head(5)
                
                st.dataframe(top_5.style.format({metric: "{:.4f}"}), hide_index=True, use_container_width=True)
                
                avg_best = top_5[metric].mean()
                st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #475569;'><b>Mean:</b> {avg_best:.4f}</div>", unsafe_allow_html=True)

    # ==================================
    # TAB 5: RAW DATA
    # ==================================
    with t5:
        st.markdown("### Complete Results Table")
        st.dataframe(main_df, use_container_width=True)
        csv_data = main_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV", 
            data=csv_data, 
            file_name=f"{selection.replace(' ', '_')}_results.csv", 
            mime="text/csv"
        )
