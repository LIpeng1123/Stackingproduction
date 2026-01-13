# prediction_app.py
# Sepsis-AMI Mortality Prediction - Fixed Version

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import json
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sepsis-AMI Mortality Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {text-align: center; color: #1f77b4; font-size: 2.5em; font-weight: bold; padding: 20px 0; border-bottom: 3px solid #1f77b4; margin-bottom: 30px;}
    .sub-title {text-align: center; color: #666; font-size: 1.2em; margin-bottom: 30px;}
    .result-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;}
    .result-value {font-size: 3em; font-weight: bold; margin: 10px 0;}
    .result-label {font-size: 1.2em; opacity: 0.9;}
    .risk-low {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);}
    .risk-medium {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
    .risk-high {background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);}
    .shap-section {background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;}
    .feature-positive {color: #d63384; font-weight: bold;}
    .feature-negative {color: #198754; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('stacking_model.pkl')
        scaler = joblib.load('scaler.pkl')
        metadata = joblib.load('model_metadata.pkl')
        try:
            shap_data = joblib.load('shap_background.pkl')
        except:
            shap_data = None
        try:
            with open('performance_summary.json', 'r', encoding='utf-8') as f:
                performance = json.load(f)
        except:
            performance = None
        return model, scaler, metadata, performance, shap_data
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None, None, None

@st.cache_resource
def create_shap_explainer(_model, _shap_data):
    if _shap_data is None:
        return None, None
    try:
        def model_predict(X):
            return _model.predict_proba(X)[:, 1]
        background_df = pd.DataFrame(_shap_data['background_samples'], columns=_shap_data['background_feature_names'])
        explainer = shap.KernelExplainer(model=model_predict, data=background_df, link="identity")
        expected_value = _shap_data['expected_value']
        return explainer, expected_value
    except Exception as e:
        return None, None

def predict_mortality(input_data, model, scaler, metadata):
    try:
        X = pd.DataFrame([input_data])[metadata['feature_names']]
        continuous_cols = metadata['continuous_features']
        X_scaled = X.copy()
        X_scaled[continuous_cols] = scaler.transform(X[continuous_cols])
        y_proba = model.predict_proba(X_scaled)[0, 1]
        threshold = metadata.get('threshold_best_f2', 0.11)
        y_pred = 1 if y_proba >= threshold else 0
        if y_proba < 0.05:
            risk_level, risk_color = 'Low Risk', 'risk-low'
        elif y_proba < 0.25:
            risk_level, risk_color = 'Medium Risk', 'risk-medium'
        else:
            risk_level, risk_color = 'High Risk', 'risk-high'
        return {'probability': float(y_proba), 'risk_level': risk_level, 'risk_color': risk_color, 'prediction': int(y_pred), 'threshold': float(threshold), 'X_scaled': X_scaled, 'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def compute_shap_values(explainer, X_scaled, expected_value, feature_names):
    try:
        shap_values = explainer.shap_values(X_scaled.values, nsamples=100)
        shap_df = pd.DataFrame({'Feature': feature_names, 'Feature_Value': X_scaled.values[0], 'SHAP_Value': shap_values[0]})
        shap_df['Abs_SHAP'] = np.abs(shap_df['SHAP_Value'])
        shap_df = shap_df.sort_values('Abs_SHAP', ascending=False)
        return {'shap_values': shap_values[0], 'expected_value': expected_value, 'shap_df': shap_df, 'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def create_shap_waterfall_figure(shap_values, expected_value, X_scaled, feature_names, max_display=15):
    try:
        explanation = shap.Explanation(values=shap_values, base_values=expected_value, data=X_scaled.values[0], feature_names=feature_names)
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        return fig
    except:
        return None

def create_shap_bar_plotly(shap_df, top_n=15):
    top_features = shap_df.head(top_n).copy().iloc[::-1]
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in top_features['SHAP_Value']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top_features['SHAP_Value'], y=top_features['Feature'], orientation='h', marker_color=colors, text=[f'{v:+.4f}' for v in top_features['SHAP_Value']], textposition='outside'))
    fig.update_layout(title='Feature Contributions', xaxis_title='SHAP Value', height=500, margin=dict(l=220, r=80, t=60, b=60), plot_bgcolor='white')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='black')
    return fig

def create_shap_summary_html(shap_df, expected_value, final_prob, input_data):
    top_pos = shap_df[shap_df['SHAP_Value'] > 0].head(5)
    top_neg = shap_df[shap_df['SHAP_Value'] < 0].head(5)
    change = final_prob - expected_value
    change_color = '#d62728' if change > 0 else '#2ca02c'
    change_sign = '+' if change > 0 else ''
    html = f'<div class="shap-section"><h4>🔍 Prediction Explanation</h4><p><strong>Base:</strong> {expected_value:.1%} → <strong>Final:</strong> {final_prob:.1%} (<span style="color: {change_color};">{change_sign}{change*100:.1f}%</span>)</p><div style="display: flex; gap: 20px; margin-top: 15px;"><div style="flex: 1; background: #ffe6e6; padding: 15px; border-radius: 8px; border-left: 4px solid #d62728;"><h5 style="color: #d62728; margin-top: 0;">🔺 Risk-Increasing</h5><ul style="margin-bottom: 0;">'
    for _, row in top_pos.iterrows():
        html += f"<li><span class='feature-positive'>{row['Feature']}</span>: <small>(+{row['SHAP_Value']:.4f})</small></li>"
    if len(top_pos) == 0:
        html += "<li><em>None</em></li>"
    html += '</ul></div><div style="flex: 1; background: #e6ffe6; padding: 15px; border-radius: 8px; border-left: 4px solid #2ca02c;"><h5 style="color: #2ca02c; margin-top: 0;">🔻 Risk-Decreasing</h5><ul style="margin-bottom: 0;">'
    for _, row in top_neg.iterrows():
        html += f"<li><span class='feature-negative'>{row['Feature']}</span>: <small>({row['SHAP_Value']:.4f})</small></li>"
    if len(top_neg) == 0:
        html += "<li><em>None</em></li>"
    html += "</ul></div></div></div>"
    return html

def create_gauge_chart(probability):
    if probability < 0.05:
        bar_color = "#11998e"
    elif probability < 0.25:
        bar_color = "#f5576c"
    else:
        bar_color = "#fa709a"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Mortality Risk", 'font': {'size': 20}}, number={'suffix': "%", 'font': {'size': 42, 'color': bar_color}}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': bar_color, 'thickness': 0.75}, 'steps': [{'range': [0, 5], 'color': 'rgba(56, 239, 125, 0.3)'}, {'range': [5, 25], 'color': 'rgba(245, 87, 108, 0.3)'}, {'range': [25, 100], 'color': 'rgba(250, 112, 154, 0.3)'}]}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    model, scaler, metadata, performance, shap_data = load_models()
    if model is None:
        st.error("Model loading failed!")
        return
    
    shap_explainer, expected_value = create_shap_explainer(model, shap_data)
    
    st.markdown('<h1 class="main-title">🏥 Sepsis-AMI Mortality Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">28-Day Mortality Risk with SHAP Explanation</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📊 Model Info")
        if performance:
            st.metric("AUC", performance.get('metrics', {}).get('AUC', 'N/A'))
        st.markdown("---")
        st.subheader("Risk Levels")
        st.markdown("🟢 Low: <5%")
        st.markdown("🟡 Medium: 5-25%")
        st.markdown("🔴 High: ≥25%")
        st.markdown("---")
        if shap_explainer:
            st.success("✅ SHAP Available")
        else:
            st.warning("⚠️ SHAP Unavailable")
        st.markdown("---")
        st.warning("⚠️ Research use only")
    
    st.header("📝 Patient Information")
    
    input_data = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data["Age, years"] = st.number_input("Age (years)", min_value=18.0, max_value=81.0, value=65.0, step=1.0)
        input_data["APS III score"] = st.number_input("APS III Score", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        input_data["Heart rate, beats/min"] = st.number_input("Heart Rate (beats/min)", min_value=30.0, max_value=200.0, value=80.0, step=1.0)
        input_data["pH"] = st.number_input("pH", min_value=6.0, max_value=8.0, value=7.4, step=0.01)
        input_data["WBC count, 10³/μL"] = st.number_input("WBC Count (×10³/μL)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        input_data["Anion gap, mmol/L"] = st.number_input("Anion Gap (mmol/L)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
    
    with col2:
        input_data["Norepinephrine use"] = st.number_input("Norepinephrine use (μg/kg/min)", min_value=0.0, max_value=10.0, value=0.1, step=1.0)
        input_data["SAPS II score"] = st.number_input("SAPS II Score", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        input_data["Temperature, °C"] = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        input_data["Lactate, mmol/L"] = st.number_input("Lactate (mmol/L)", min_value=0.0, max_value=25.0, value=2.0, step=0.1)
        input_data["RBC count, 10⁶/μL"] = st.number_input("RBC Count (×10⁶/μL)", min_value=1.0, max_value=10.0, value=4.5, step=0.1)
        input_data["Blood urea nitrogen, mg/dL"] = st.number_input("BUN (mg/dL)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
    
    with col3:
        acei = st.selectbox("ACEI Use", options=["No", "Yes"])
        input_data["ACEI use"] = 1 if acei == "Yes" else 0
        input_data["LODS score"] = st.number_input("LODS Score", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
        input_data["SpO₂, %"] = st.number_input("SpO2 (%)", min_value=60.0, max_value=100.0, value=95.0, step=1.0)
        input_data["Hemoglobin, g/dL"] = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
        input_data["aPTT, s"] = st.number_input("aPTT (seconds)", min_value=10.0, max_value=200.0, value=30.0, step=1.0)
        input_data["Calcium, mg/dL"] = st.number_input("Calcium (mg/dL)", min_value=0.0, max_value=20.0, value=9.0, step=0.1)
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔍 Calculate Mortality Risk", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner('Analyzing...'):
            result = predict_mortality(input_data, model, scaler, metadata)
        
        if result['success']:
            st.success("✅ Prediction completed!")
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown(f'<div class="result-box {result["risk_color"]}"><div class="result-label">28-Day Mortality Risk</div><div class="result-value">{result["probability"]*100:.1f}%</div><div class="result-label">{result["risk_level"]}</div></div>', unsafe_allow_html=True)
                if result['prediction'] == 1:
                    st.error(f"⚠️ High-risk (prob ≥ {result['threshold']:.1%})")
                else:
                    st.success(f"✓ Lower-risk (prob < {result['threshold']:.1%})")
            
            with res_col2:
                st.plotly_chart(create_gauge_chart(result['probability']), use_container_width=True)
            
            if shap_explainer:
                st.markdown("---")
                st.markdown("## 🔬 SHAP Explanation")
                with st.spinner('Computing SHAP values (10-30 seconds)...'):
                    shap_result = compute_shap_values(shap_explainer, result['X_scaled'], expected_value, metadata['feature_names'])
                
                if shap_result['success']:
                    st.markdown(create_shap_summary_html(shap_result['shap_df'], expected_value, result['probability'], input_data), unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🌊 Waterfall", "📋 Table"])
                    
                    with tab1:
                        st.plotly_chart(create_shap_bar_plotly(shap_result['shap_df']), use_container_width=True)
                        st.caption("🔴 Red = increases risk | 🟢 Green = decreases risk")
                    
                    with tab2:
                        fig = create_shap_waterfall_figure(shap_result['shap_values'], expected_value, result['X_scaled'], metadata['feature_names'])
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.warning("Waterfall plot unavailable")
                    
                    with tab3:
                        df_display = shap_result['shap_df'].copy()
                        df_display['Effect'] = df_display['SHAP_Value'].apply(lambda x: '🔺 Risk↑' if x > 0 else '🔻 Risk↓')
                        df_display['SHAP_Value'] = df_display['SHAP_Value'].apply(lambda x: f'{x:+.4f}')
                        st.dataframe(df_display[['Feature', 'SHAP_Value', 'Effect']], use_container_width=True, hide_index=True)
                else:
                    st.warning(f"SHAP failed: {shap_result.get('error')}")
            
            st.markdown("---")
            st.markdown("## 💡 Clinical Recommendations")
            if result['risk_level'] == 'High Risk':
                st.error("### ⚠️ High Risk Patient\n- Consider ICU admission\n- Close monitoring\n- Early intervention")
            elif result['risk_level'] == 'Medium Risk':
                st.warning("### ⚡ Medium Risk Patient\n- Enhanced monitoring\n- Optimize treatment\n- Regular reassessment")
            else:
                st.success("### ✅ Lower Risk Patient\n- Standard monitoring\n- Continue current plan")
        else:
            st.error(f"Prediction failed: {result.get('error')}")
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;padding:20px;'>© 2026 Sepsis-AMI Mortality Prediction | Research Use Only | The First Hospital of LanZhou University</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
