import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import boto3
import os
from pathlib import Path

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://0.0.0.0:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "income-prediction-ine")
REGION = os.getenv("AWS_REGION", "eu-north-1")

s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# Paths (ensure available locally by fetching from S3 if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)

# ============================
# Feature label mappings
# ============================
FEATURE_LABELS = {
    'age': 'Age',
    'studies': 'Education Level',
    'reg_living': 'Living Region',
    'ever_married': 'Marital Status',
    'number_children': 'Number of Children',
    'number_living': 'Household Size',
    'type_contract': 'Contract Type'
}

# Categorical decodings (update these based on your actual encodings)
CATEGORY_MAPPINGS = {
    'ever_married': {0: 'Never Married', 1: 'Ever Married'},
    'reg_living': {0: 'Region A', 1: 'Region B', 2: 'Region C'},
    'type_contract': {0: 'Permanent', 1: 'Temporary', 2: 'Freelance'},
    'studies': {
        0: 'No Studies', 1: 'Primary', 2: 'Secondary',
        3: 'High School', 4: 'Technical', 5: 'Bachelor',
        6: 'Master', 7: 'PhD', 8: 'Postdoc'
    }
}

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    """Load holdout data and create display dataframe"""
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    
    # Create display dataframe with additional info
    disp = pd.DataFrame(index=fe.index)
    disp['id'] = range(len(fe))
    
    # Add decoded categorical features for display
    for col, mapping in CATEGORY_MAPPINGS.items():
        if col in fe.columns:
            disp[f'{col}_label'] = fe[col].map(mapping)
    
    # Add numeric features
    disp['age'] = fe['age']
    disp['studies'] = fe['studies']
    disp['number_children'] = fe['number_children']
    disp['number_living'] = fe['number_living']
    disp['actual_income'] = fe['income']
    
    # Create income range categories for filtering
    disp['income_range'] = pd.cut(
        fe['income'],
        bins=[0, 500, 1000, 1500, 2000, 10000],
        labels=['€0-500', '€500-1k', '€1k-1.5k', '€1.5k-2k', '€2k+']
    )
    
    return fe, disp

fe_df, disp_df = load_data()

# ============================
# UI Configuration
# ============================
st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Income Prediction — Holdout Explorer")
st.markdown("Explore predictions vs actual income on holdout data")

# ============================
# Sidebar Filters
# ============================
st.sidebar.header("🔍 Filters")

# Age range filter
age_min, age_max = int(disp_df['age'].min()), int(disp_df['age'].max())
age_range = st.sidebar.slider(
    "Age Range",
    age_min, age_max,
    (age_min, age_max)
)

# Education level filter
education_levels = sorted(disp_df['studies'].unique())
selected_education = st.sidebar.multiselect(
    "Education Level",
    options=education_levels,
    default=education_levels,
    format_func=lambda x: CATEGORY_MAPPINGS['studies'].get(x, f'Level {x}')
)

# Marital status filter
marital_options = ['All'] + list(CATEGORY_MAPPINGS['ever_married'].values())
marital_status = st.sidebar.selectbox("Marital Status", marital_options)

# Contract type filter
contract_options = ['All'] + list(CATEGORY_MAPPINGS['type_contract'].values())
contract_type = st.sidebar.selectbox("Contract Type", contract_options)

# Income range filter
income_ranges = ['All'] + list(disp_df['income_range'].cat.categories)
income_range = st.sidebar.selectbox("Income Range", income_ranges)

# Number of records to predict
n_samples = st.sidebar.slider(
    "Number of samples to predict",
    min_value=10,
    max_value=min(500, len(disp_df)),
    value=min(100, len(disp_df)),
    step=10
)

# ============================
# Apply Filters
# ============================
mask = (disp_df['age'] >= age_range[0]) & (disp_df['age'] <= age_range[1])

if selected_education:
    mask &= disp_df['studies'].isin(selected_education)

if marital_status != 'All':
    mask &= disp_df['ever_married_label'] == marital_status

if contract_type != 'All':
    mask &= disp_df['type_contract_label'] == contract_type

if income_range != 'All':
    mask &= disp_df['income_range'] == income_range

filtered_indices = disp_df.index[mask]

# Show filter stats
st.sidebar.markdown("---")
st.sidebar.metric("Filtered Records", f"{len(filtered_indices):,}")
st.sidebar.metric("% of Total", f"{len(filtered_indices)/len(disp_df)*100:.1f}%")

# ============================
# Main Content
# ============================

if len(filtered_indices) == 0:
    st.warning("⚠️ No data matches your filters. Please adjust the criteria.")
    st.stop()

# Sample from filtered data
sample_indices = filtered_indices[:n_samples]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Filtered Data Summary")
    summary_df = disp_df.loc[sample_indices, ['age', 'studies', 'number_children', 'actual_income']].describe()
    st.dataframe(summary_df.style.format("{:.2f}"), use_container_width=True)

with col2:
    st.subheader("📈 Quick Stats")
    st.metric("Average Income", f"€{disp_df.loc[sample_indices, 'actual_income'].mean():,.2f}")
    st.metric("Median Income", f"€{disp_df.loc[sample_indices, 'actual_income'].median():,.2f}")
    st.metric("Income Std Dev", f"€{disp_df.loc[sample_indices, 'actual_income'].std():,.2f}")

# ============================
# Prediction Button
# ============================
if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
    
    with st.spinner(f"Running predictions for {len(sample_indices)} samples..."):
        
        # Prepare payload
        payload = fe_df.loc[sample_indices].to_dict(orient="records")
        
        try:
            # Call API
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            
            # Extract predictions
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)
            
            if not preds:
                st.error("❌ No predictions returned from API")
                st.stop()
            
            # Create results dataframe
            results = disp_df.loc[sample_indices].copy()
            results['predicted_income'] = pd.Series(preds, index=sample_indices).astype(float)
            
            if actuals and len(actuals) == len(sample_indices):
                results['actual_income'] = pd.Series(actuals, index=sample_indices).astype(float)
            
            # Calculate errors
            results['error'] = results['predicted_income'] - results['actual_income']
            results['abs_error'] = results['error'].abs()
            results['pct_error'] = (results['error'] / results['actual_income'] * 100).abs()
            
            # ============================
            # Metrics
            # ============================
            st.subheader("📊 Model Performance")
            
            mae = results['abs_error'].mean()
            rmse = (results['error'] ** 2).mean() ** 0.5
            mape = results['pct_error'].mean()
            r2 = 1 - (results['error'] ** 2).sum() / ((results['actual_income'] - results['actual_income'].mean()) ** 2).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"€{mae:,.2f}")
            with col2:
                st.metric("RMSE", f"€{rmse:,.2f}")
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            with col4:
                st.metric("R² Score", f"{r2:.4f}")
            
            # ============================
            # Visualizations
            # ============================
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "📉 Errors", "🎯 By Category", "📋 Data Table"])
            
            with tab1:
                st.subheader("Actual vs Predicted Income")
                
                # Scatter plot
                fig_scatter = px.scatter(
                    results,
                    x='actual_income',
                    y='predicted_income',
                    color='pct_error',
                    color_continuous_scale='RdYlGn_r',
                    hover_data=['age', 'studies', 'number_children'],
                    labels={
                        'actual_income': 'Actual Income (€)',
                        'predicted_income': 'Predicted Income (€)',
                        'pct_error': 'Error %'
                    },
                    title='Predictions vs Actuals'
                )
                
                # Add perfect prediction line
                min_val = min(results['actual_income'].min(), results['predicted_income'].min())
                max_val = max(results['actual_income'].max(), results['predicted_income'].max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab2:
                st.subheader("Prediction Errors")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Error distribution
                    fig_error_dist = px.histogram(
                        results,
                        x='error',
                        nbins=30,
                        title='Error Distribution',
                        labels={'error': 'Prediction Error (€)'}
                    )
                    fig_error_dist.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_error_dist, use_container_width=True)
                
                with col2:
                    # Absolute error by income range
                    fig_error_range = px.box(
                        results,
                        x='income_range',
                        y='abs_error',
                        title='Absolute Error by Income Range',
                        labels={'abs_error': 'Absolute Error (€)', 'income_range': 'Income Range'}
                    )
                    st.plotly_chart(fig_error_range, use_container_width=True)
            
            with tab3:
                st.subheader("Performance by Categories")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # By education level
                    education_perf = results.groupby('studies').agg({
                        'abs_error': 'mean',
                        'pct_error': 'mean',
                        'predicted_income': 'count'
                    }).reset_index()
                    education_perf.columns = ['Education Level', 'MAE', 'MAPE', 'Count']
                    education_perf['Education Level'] = education_perf['Education Level'].map(
                        lambda x: CATEGORY_MAPPINGS['studies'].get(x, f'Level {x}')
                    )
                    
                    fig_education = px.bar(
                        education_perf,
                        x='Education Level',
                        y='MAE',
                        text='Count',
                        title='MAE by Education Level',
                        labels={'MAE': 'Mean Absolute Error (€)'}
                    )
                    st.plotly_chart(fig_education, use_container_width=True)
                
                with col2:
                    # By age group
                    results['age_group'] = pd.cut(results['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
                    age_perf = results.groupby('age_group').agg({
                        'abs_error': 'mean',
                        'pct_error': 'mean',
                        'predicted_income': 'count'
                    }).reset_index()
                    age_perf.columns = ['Age Group', 'MAE', 'MAPE', 'Count']
                    
                    fig_age = px.bar(
                        age_perf,
                        x='Age Group',
                        y='MAE',
                        text='Count',
                        title='MAE by Age Group',
                        labels={'MAE': 'Mean Absolute Error (€)'}
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
            
            with tab4:
                st.subheader("Detailed Results")
                
                # Prepare display columns
                display_cols = [
                    'id', 'age', 'studies', 'ever_married_label',
                    'number_children', 'type_contract_label',
                    'actual_income', 'predicted_income', 'error', 'pct_error'
                ]
                
                display_results = results[display_cols].copy()
                display_results.columns = [
                    'ID', 'Age', 'Education', 'Marital Status',
                    'Children', 'Contract', 'Actual (€)', 'Predicted (€)',
                    'Error (€)', 'Error %'
                ]
                
                # Sort by absolute error
                display_results = display_results.sort_values('Error (€)', key=abs, ascending=False)
                
                st.dataframe(
                    display_results.style.format({
                        'Actual (€)': '{:,.2f}',
                        'Predicted (€)': '{:,.2f}',
                        'Error (€)': '{:+,.2f}',
                        'Error %': '{:.2f}%'
                    }).background_gradient(subset=['Error %'], cmap='RdYlGn_r'),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = display_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="income_predictions.csv",
                    mime="text/csv"
                )
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API call failed: {e}")
            st.exception(e)
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.exception(e)

else:
    st.info("👆 Adjust filters in the sidebar and click **Run Predictions** to start")
    
    # Show sample data
    with st.expander("📋 Preview Sample Data"):
        preview = disp_df.loc[sample_indices[:10], [
            'id', 'age', 'studies', 'ever_married_label',
            'number_children', 'type_contract_label', 'actual_income'
        ]]
        st.dataframe(preview, use_container_width=True)