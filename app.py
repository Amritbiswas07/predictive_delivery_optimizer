import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Predictive Delivery Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .risk-high {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 15px;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# ERROR HANDLING UTILITIES
# ============================================

class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""
    pass


def safe_file_load(filepath, file_type='csv'):
    """
    Safely load a file with comprehensive error handling

    Args:
        filepath: Path to the file
        file_type: Type of file ('csv', 'pkl', etc.)

    Returns:
        Loaded data or None if error occurs
    """
    try:
        if file_type == 'csv':
            data = pd.read_csv(filepath)
            if data.empty:
                st.warning(f"⚠️ Warning: {filepath} is empty")
                return None
            return data
        elif file_type == 'pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except FileNotFoundError:
        st.error(f"❌ Error: File not found - {filepath}")
        st.info("Please ensure the file exists in the correct directory")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"❌ Error: {filepath} is empty or corrupted")
        return None
    except Exception as e:
        st.error(f"❌ Error loading {filepath}: {str(e)}")
        return None


def validate_dataframe(df, required_columns, df_name):
    """
    Validate dataframe structure and required columns

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name of the dataframe for error messages

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if df is None:
            st.error(f"❌ {df_name}: DataFrame is None")
            return False

        if df.empty:
            st.warning(f"⚠️ {df_name}: DataFrame is empty")
            return False

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            st.error(f"❌ {df_name}: Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
            return False

        return True
    except Exception as e:
        st.error(f"❌ Error validating {df_name}: {str(e)}")
        return False


# ============================================
# DATA LOADING WITH ERROR HANDLING
# ============================================

@st.cache_data
def load_all_data():
    """Load all data files with comprehensive error handling"""

    with st.spinner("Loading data files..."):
        files = {
            'orders': 'orders.csv',
            'performance': 'delivery_performance.csv',
            'routes': 'routes_distance.csv',
            'costs': 'cost_breakdown.csv',
            'feedback': 'customer_feedback.csv',
            'fleet': 'vehicle_fleet.csv',
            'inventory': 'warehouse_inventory.csv'
        }

        data = {}
        load_errors = []

        for key, filename in files.items():
            df = safe_file_load(filename, 'csv')
            if df is not None:
                data[key] = df
                st.success(f"✅ Loaded {filename}: {len(df)} records")
            else:
                load_errors.append(filename)

        if load_errors:
            st.warning(f"⚠️ Failed to load: {', '.join(load_errors)}")
            st.info("The app will continue with available data")

        return data


@st.cache_resource
def load_model_and_explainer():
    """Load trained model with error handling"""

    try:
        with st.spinner("Loading ML model..."):
            model = safe_file_load('lightgbm_model.pkl', 'pkl')

            if model is None:
                st.error("❌ Failed to load model. Using demo mode.")
                return None, None

            # Try to load SHAP explainer if available
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                st.success("✅ Model and SHAP explainer loaded successfully")
                return model, explainer
            except ImportError:
                st.warning("⚠️ SHAP not available. Install with: pip install shap")
                return model, None
            except Exception as e:
                st.warning(f"⚠️ SHAP explainer creation failed: {str(e)}")
                return model, None

    except Exception as e:
        st.error(f"❌ Critical error loading model: {str(e)}")
        return None, None


# ============================================
# DATA PROCESSING PIPELINE
# ============================================

def safe_merge(left_df, right_df, on, how='left', df_name=''):
    """Safely merge dataframes with error handling"""
    try:
        if left_df is None or right_df is None:
            raise DataProcessingError(f"Cannot merge: One or both DataFrames are None")

        if on not in left_df.columns:
            raise DataProcessingError(f"Key column '{on}' not found in left DataFrame")
        if on not in right_df.columns:
            raise DataProcessingError(f"Key column '{on}' not found in right DataFrame")

        merged = left_df.merge(right_df, on=on, how=how, suffixes=('', '_dup'))

        # Log merge statistics
        if how == 'left':
            match_rate = (merged[on].notna().sum() / len(merged)) * 100
            st.info(f"📊 {df_name} merge: {match_rate:.1f}% match rate")

        return merged
    except Exception as e:
        st.error(f"❌ Merge error ({df_name}): {str(e)}")
        return left_df


def clean_text_data(df, text_columns):
    """Clean text columns with error handling"""
    try:
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace('\n', ' ').str.replace('\r', '')
        return df
    except Exception as e:
        st.warning(f"⚠️ Text cleaning warning: {str(e)}")
        return df


def convert_date_columns(df, date_columns):
    """Convert date columns with error handling"""
    try:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                invalid_dates = df[col].isna().sum()
                if invalid_dates > 0:
                    st.warning(f"⚠️ {col}: {invalid_dates} invalid dates converted to NaT")
        return df
    except Exception as e:
        st.warning(f"⚠️ Date conversion warning: {str(e)}")
        return df


def create_unified_dataset(data):
    """Create unified dataset with comprehensive error handling"""

    try:
        st.subheader("🔄 Data Integration Pipeline")

        # Validate required datasets
        if 'orders' not in data or data['orders'] is None:
            raise DataProcessingError("Orders data is required but not available")

        df = data['orders'].copy()
        st.success(f"✅ Starting with {len(df)} orders")

        # Merge performance data
        if 'performance' in data and data['performance'] is not None:
            df = safe_merge(df, data['performance'], on='Order_ID', df_name='Performance')
            df['In_Transit_Flag'] = df['Delivery_Status'].isna().astype(int)
        else:
            st.warning("⚠️ Performance data not available, setting all as in-transit")
            df['In_Transit_Flag'] = 1

        # Merge routes data
        if 'routes' in data and data['routes'] is not None:
            df = safe_merge(df, data['routes'], on='Order_ID', df_name='Routes')

        # Merge costs data
        if 'costs' in data and data['costs'] is not None:
            df = safe_merge(df, data['costs'], on='Order_ID', df_name='Costs')

        # Merge feedback data
        if 'feedback' in data and data['feedback'] is not None:
            df = safe_merge(df, data['feedback'], on='Order_ID', df_name='Feedback')

        # Clean text columns
        text_cols = ['Feedback_Text', 'Product_Category', 'Customer_Segment']
        df = clean_text_data(df, text_cols)

        # Convert date columns
        date_cols = ['Order_Date', 'Feedback_Date']
        df = convert_date_columns(df, date_cols)

        st.success(f"✅ Unified dataset created: {len(df)} rows × {len(df.columns)} columns")

        return df

    except DataProcessingError as e:
        st.error(f"❌ Data processing error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error in data integration: {str(e)}")
        st.exception(e)
        return None


def engineer_features(df, data):
    """Engineer features with comprehensive error handling"""

    try:
        st.subheader("⚙️ Feature Engineering")

        features_created = []

        # Target variable
        if 'Delivery_Status' in df.columns:
            df['Is_Delayed'] = df['Delivery_Status'].isin(['Slightly-Delayed', 'Severely-Delayed']).astype(int)
            features_created.append('Is_Delayed')

        # Route features
        try:
            if 'Route' in df.columns:
                df[['Origin_City', 'Destination_City']] = df['Route'].str.split('-', expand=True)
                international_cities = ['Singapore', 'Dubai', 'Hong Kong', 'Bangkok']
                df['Is_International'] = df['Destination_City'].isin(international_cities).astype(int)
                features_created.extend(['Origin_City', 'Destination_City', 'Is_International'])
        except Exception as e:
            st.warning(f"⚠️ Route feature creation warning: {str(e)}")

        # Cost per KM
        try:
            if 'Delivery_Cost_INR' in df.columns and 'Distance_KM' in df.columns:
                df['Cost_per_KM'] = df['Delivery_Cost_INR'] / df['Distance_KM'].replace(0, 1)
                features_created.append('Cost_per_KM')
        except Exception as e:
            st.warning(f"⚠️ Cost_per_KM calculation warning: {str(e)}")

        # Traffic impact ratio
        try:
            if 'Traffic_Delay_Minutes' in df.columns and 'Distance_KM' in df.columns:
                df['Traffic_Impact_Ratio'] = df['Traffic_Delay_Minutes'] / df['Distance_KM'].replace(0, 1)
                features_created.append('Traffic_Impact_Ratio')
        except Exception as e:
            st.warning(f"⚠️ Traffic_Impact_Ratio calculation warning: {str(e)}")

        # Temporal features
        try:
            if 'Order_Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Order_Date']):
                df['Order_Month'] = df['Order_Date'].dt.month
                df['Order_Day_of_Week'] = df['Order_Date'].dt.dayofweek
                df['Order_Week_of_Year'] = df['Order_Date'].dt.isocalendar().week
                features_created.extend(['Order_Month', 'Order_Day_of_Week', 'Order_Week_of_Year'])
        except Exception as e:
            st.warning(f"⚠️ Temporal feature creation warning: {str(e)}")

        # Value per day promised
        try:
            if 'Order_Value_INR' in df.columns and 'Promised_Delivery_Days' in df.columns:
                df['Value_per_Day_Promised'] = df['Order_Value_INR'] / df['Promised_Delivery_Days'].replace(0, 1)
                features_created.append('Value_per_Day_Promised')
        except Exception as e:
            st.warning(f"⚠️ Value_per_Day_Promised calculation warning: {str(e)}")

        # Carrier performance features
        try:
            if 'performance' in data and 'Carrier' in df.columns:
                perf = data['performance']
                if 'Carrier' in perf.columns:
                    carrier_stats = perf.groupby('Carrier').agg({
                        'Actual_Delivery_Days': 'mean',
                        'Delivery_Status': lambda x: (x != 'On-Time').mean()
                    }).reset_index()
                    carrier_stats.columns = ['Carrier', 'Carrier_Avg_Delay_Days', 'Carrier_Delay_Rate']

                    df = safe_merge(df, carrier_stats, on='Carrier', df_name='Carrier Stats')
                    features_created.extend(['Carrier_Avg_Delay_Days', 'Carrier_Delay_Rate'])
        except Exception as e:
            st.warning(f"⚠️ Carrier feature creation warning: {str(e)}")

        # Stock pressure
        try:
            if 'inventory' in data and 'Origin_City' in df.columns and 'Product_Category' in df.columns:
                inv = data['inventory']
                if 'Location' in inv.columns and 'Product_Category' in inv.columns:
                    inv['Stock_Pressure'] = inv['Current_Stock_Units'] / inv['Reorder_Level'].replace(0, 1)
                    df = safe_merge(
                        df,
                        inv[['Location', 'Product_Category', 'Stock_Pressure']],
                        left_on=['Origin_City', 'Product_Category'],
                        right_on=['Location', 'Product_Category'],
                        df_name='Inventory'
                    )
                    features_created.append('Stock_Pressure')
        except Exception as e:
            st.warning(f"⚠️ Stock pressure calculation warning: {str(e)}")

        st.success(f"✅ Created {len(features_created)} features: {', '.join(features_created[:5])}...")

        return df

    except Exception as e:
        st.error(f"❌ Feature engineering error: {str(e)}")
        st.exception(e)
        return df


# ============================================
# SHAP VISUALIZATION
# ============================================

def plot_shap_waterfall(shap_values, feature_names, base_value, order_id):
    """Create SHAP waterfall plot with error handling"""
    try:
        import shap

        fig = go.Figure()

        # Sort by absolute impact
        impacts = [(feat, val) for feat, val in zip(feature_names, shap_values)]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        # Take top 10
        top_impacts = impacts[:10]

        cumulative = base_value
        y_pos = []
        colors = []

        for i, (feat, val) in enumerate(top_impacts):
            y_pos.append(cumulative)
            cumulative += val
            colors.append('#ef4444' if val > 0 else '#10b981')

        feature_labels = [f"{feat}: {val:+.3f}" for feat, val in top_impacts]

        fig.add_trace(go.Waterfall(
            orientation="h",
            y=feature_labels,
            x=[impact[1] for impact in top_impacts],
            base=base_value,
            marker={"color": colors},
            connector={"line": {"color": "gray", "dash": "dot"}}
        ))

        fig.update_layout(
            title=f"SHAP Analysis: {order_id}",
            xaxis_title="Impact on Delay Probability",
            yaxis_title="Features",
            height=500,
            showlegend=False
        )

        return fig

    except ImportError:
        st.warning("⚠️ SHAP visualization requires 'shap' package")
        return None
    except Exception as e:
        st.error(f"❌ SHAP visualization error: {str(e)}")
        return None


# ============================================
# CORRECTIVE ACTIONS
# ============================================

def get_corrective_actions(top_factors):
    """Generate corrective actions based on delay drivers"""

    action_playbook = {
        'Carrier_Delay_Rate': {
            'tier1': 'Flag order for priority tracking in TMS',
            'tier2': 'Alert manager to consider carrier swap if feasible',
            'tier3': 'Add incident to carrier quarterly performance scorecard'
        },
        'Weather_Impact': {
            'tier1': 'Automatically check for alternative route availability',
            'tier2': 'Alert manager and customer of potential weather delay',
            'tier3': 'Analyze route for historical weather vulnerability'
        },
        'Traffic_Impact_Ratio': {
            'tier1': 'Send real-time traffic alert to driver device',
            'tier2': 'Authorize driver to use toll roads to bypass congestion',
            'tier3': 'Identify consistently congested routes for scheduling adjustments'
        },
        'Is_International': {
            'tier1': 'Flag order for automated customs documentation check',
            'tier2': 'Assign to senior logistics coordinator for oversight',
            'tier3': 'Review customs brokerage process for destination country'
        },
        'Stock_Pressure': {
            'tier1': 'Alert origin warehouse to prioritize picking for this order',
            'tier2': 'Notify customer of potential dispatch delay with revised ETA',
            'tier3': 'Investigate root cause of low inventory for this product'
        }
    }

    actions = []

    for factor in top_factors:
        factor_key = factor['name']

        # Match factor to playbook (fuzzy matching)
        for key in action_playbook.keys():
            if key.lower() in factor_key.lower() or factor_key.lower() in key.lower():
                playbook = action_playbook[key]
                actions.append({
                    'factor': factor_key,
                    'tier1': playbook['tier1'],
                    'tier2': playbook['tier2'],
                    'tier3': playbook['tier3']
                })
                break

    return actions


# ============================================
# MAIN APP
# ============================================

def main():
    st.title("📦 Predictive Delivery Optimizer")
    st.markdown("### NexGen Logistics - Real-time Risk Monitoring & Intervention")

    # Load data
    data = load_all_data()
    model, explainer = load_model_and_explainer()

    if not data or 'orders' not in data:
        st.error("❌ Critical: Unable to load required data files")
        st.stop()

    # Create unified dataset
    with st.expander("📊 Data Processing Pipeline", expanded=False):
        df = create_unified_dataset(data)
        if df is not None:
            df = engineer_features(df, data)

    if df is None:
        st.error("❌ Data processing failed. Cannot continue.")
        st.stop()

    # Filter in-transit orders
    in_transit = df[df['In_Transit_Flag'] == 1].copy()

    # Demo mode if no model
    if model is None:
        st.warning("⚠️ Running in DEMO mode (model not loaded)")
        in_transit['Predicted_Risk'] = np.random.uniform(10, 90, len(in_transit))
    else:
        # Make predictions (implement your actual prediction logic here)
        st.info("🔮 Generating predictions...")
        in_transit['Predicted_Risk'] = np.random.uniform(10, 90, len(in_transit))

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total In-Transit", len(in_transit))
    with col2:
        high_risk = len(in_transit[in_transit['Predicted_Risk'] >= 70])
        st.metric("High Risk Orders", high_risk, delta=None, delta_color="inverse")
    with col3:
        avg_risk = in_transit['Predicted_Risk'].mean()
        st.metric("Avg Delay Risk", f"{avg_risk:.1f}%")
    with col4:
        st.metric("Actions Taken Today", 12)

    st.markdown("---")

    # Main layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🚨 At-Risk Shipments Watchlist")

        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            risk_filter = st.selectbox("Filter by Risk", ["All", "High (70%+)", "Medium (30-70%)", "Low (<30%)"])
        with filter_col2:
            dest_filter = st.selectbox("Filter by Destination", ["All"] + sorted(in_transit[
                                                                                     'Destination_City'].dropna().unique().tolist()) if 'Destination_City' in in_transit.columns else [
                "All"])

        # Apply filters
        filtered = in_transit.copy()
        if risk_filter == "High (70%+)":
            filtered = filtered[filtered['Predicted_Risk'] >= 70]
        elif risk_filter == "Medium (30-70%)":
            filtered = filtered[(filtered['Predicted_Risk'] >= 30) & (filtered['Predicted_Risk'] < 70)]
        elif risk_filter == "Low (<30%)":
            filtered = filtered[filtered['Predicted_Risk'] < 30]

        if dest_filter != "All" and 'Destination_City' in filtered.columns:
            filtered = filtered[filtered['Destination_City'] == dest_filter]

        # Display orders
        for idx, row in filtered.head(10).iterrows():
            risk = row['Predicted_Risk']

            if risk >= 70:
                css_class = "risk-high"
                badge = "🔴 HIGH RISK"
            elif risk >= 30:
                css_class = "risk-medium"
                badge = "🟡 MEDIUM RISK"
            else:
                css_class = "risk-low"
                badge = "🟢 LOW RISK"

            with st.container():
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)

                order_col1, order_col2 = st.columns([3, 1])

                with order_col1:
                    st.markdown(f"**{row['Order_ID']}**")
                    customer = row.get('Customer_Segment', 'N/A')
                    dest = row.get('Destination_City', 'N/A')
                    st.markdown(f"*{customer} → {dest}*")

                with order_col2:
                    st.markdown(f"**{risk:.0f}%**")
                    st.markdown(f"*{badge}*")

                if st.button(f"View Details", key=f"btn_{row['Order_ID']}"):
                    st.session_state.selected_order = row['Order_ID']

                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")

    with col_right:
        st.subheader("🎯 Decision Cockpit")

        if 'selected_order' in st.session_state:
            order_id = st.session_state.selected_order
            order_data = in_transit[in_transit['Order_ID'] == order_id].iloc[0]

            # Risk display
            risk = order_data['Predicted_Risk']
            st.markdown(f"### Order: {order_id}")

            risk_color = "#ef4444" if risk >= 70 else "#f59e0b" if risk >= 30 else "#10b981"
            st.markdown(f"""
                <div style='background-color: {risk_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color}'>
                    <h2 style='color: {risk_color}; margin: 0;'>{risk:.1f}%</h2>
                    <p style='margin: 5px 0 0 0;'>Predicted Delay Risk</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Root Cause Analysis")

            # Mock SHAP values (replace with actual when model is available)
            top_factors = [
                {'name': 'Carrier_Delay_Rate', 'impact': 0.32},
                {'name': 'Is_International', 'impact': 0.28},
                {'name': 'Weather_Impact', 'impact': 0.15},
                {'name': 'Traffic_Impact_Ratio', 'impact': 0.12},
                {'name': 'Stock_Pressure', 'impact': -0.08}
            ]

            for factor in top_factors:
                impact_pct = factor['impact'] * 100
                color = "#ef4444" if factor['impact'] > 0 else "#10b981"
                bar_width = abs(impact_pct)

                st.markdown(f"**{factor['name']}**: {impact_pct:+.1f}%")
                st.progress(min(bar_width / 100, 1.0))
                st.markdown("")

            # Recommended actions
            st.markdown("#### ✅ Recommended Actions")
            actions = get_corrective_actions([{'name': f['name']} for f in top_factors[:3]])

            for action in actions:
                with st.expander(f"📋 {action['factor']}"):
                    st.markdown(f"**Tier 1 (Automated):** {action['tier1']}")
                    st.markdown(f"**Tier 2 (Manager Review):** {action['tier2']}")
                    st.markdown(f"**Tier 3 (Strategic):** {action['tier3']}")

            # What-If Simulator
            st.markdown("#### 🔮 What-If Simulator")

            col_sim1, col_sim2 = st.columns(2)

            with col_sim1:
                new_carrier = st.selectbox("Change Carrier",
                                           ["-- Current --", "QuickShip", "ReliableExpress", "FastTrack"])

            with col_sim2:
                new_priority = st.selectbox("Change Priority", ["-- Current --", "Express", "Standard", "Economy"])

            if new_carrier != "-- Current --" or new_priority != "-- Current --":
                # Simulate risk reduction
                new_risk = risk * 0.3  # Mock reduction

                st.success(f"**Simulated New Risk:** {new_risk:.1f}%")
                st.info(f"📉 Risk reduction: {risk - new_risk:.1f} percentage points")

        else:
            st.info("👈 Select an order from the watchlist to view details")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.exception(e)
        st.info("Please refresh the page or contact support")