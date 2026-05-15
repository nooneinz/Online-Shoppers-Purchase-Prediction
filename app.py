"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🛍️ ONLINE SHOPPERS PURCHASING INTENTION PREDICTION SYSTEM 🛍️             ║
║                                                                              ║
║  A Production-Ready Machine Learning Application                             ║
║  Glassmorphism Design | Streamlit Framework                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Online Shoppers Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Glassmorphism Purple & Blue Theme CSS
st.markdown("""
    <style>
    /* Main background with dark purple/blue gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Make base text light for contrast */
    p, li, span, label {
        color: #e2d1f9 !important;
    }
    
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    
    /* Header styling */
    .header-title {
        color: #00d2ff !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }
    
    .header-subtitle {
        color: #b768a2 !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.2em;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }
    
    /* Card styling for predictions (THE GLASS EFFECT) */
    .prediction-card {
        border-radius: 15px;
        padding: 25px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin: 15px 0;
        color: white;
    }
    
    .prediction-likely {
        border-left: 5px solid #00f2fe;
    }
    
    .prediction-unlikely {
        border-left: 5px solid #ff0844;
    }
    
    .prediction-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.6em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .prediction-likely .prediction-title {
        color: #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
    }
    
    .prediction-unlikely .prediction-title {
        color: #ff0844 !important;
        text-shadow: 0 0 10px rgba(255, 8, 68, 0.4);
    }
    
    .prediction-confidence {
        font-size: 1.1em;
        color: #e2d1f9;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.6) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: rgba(138, 43, 226, 0.3);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        font-size: 1.1em;
        font-weight: bold;
        padding: 12px 30px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        background: rgba(65, 105, 225, 0.6);
        border-color: rgba(255, 255, 255, 0.5);
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
        transform: translateY(-2px);
        color: #ffffff;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 8px;
    }
    
    /* Metric styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        color: #a8b0d3;
        font-size: 0.95em;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: #00d2ff;
        font-size: 1.8em;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.3);
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(138, 43, 226, 0.8), rgba(0, 210, 255, 0.8), transparent);
        margin: 30px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #8b9bb4;
        font-size: 0.85em;
        margin-top: 40px;
        padding-top: 20px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
    }
    
    /* Info box */
    .info-box {
        background: rgba(138, 43, 226, 0.15);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-left: 4px solid #00d2ff;
        padding: 15px;
        border-radius: 8px;
        color: #e2d1f9;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_production_models():
    """Load trained model and scaler from disk"""
    try:
        model = load_model('online_shoppers_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, None
    except FileNotFoundError as e:
        return None, None, str(e)

def get_business_insights(prediction_class, confidence):
    """Generate business-relevant insights based on prediction"""
    if prediction_class == 1:
        return {
            'status': 'LIKELY TO PURCHASE',
            'emoji': '✅',
            'color': '#27AE60',
            'recommendations': [
                '🎯 Highlight premium products and exclusive offers',
                '💳 Streamline checkout process to reduce friction',
                '📧 Personalize product recommendations based on behavior',
                '⏰ Create urgency with limited-time offers',
                '🎁 Offer loyalty rewards for immediate purchase'
            ]
        }
    else:
        return {
            'status': 'UNLIKELY TO PURCHASE',
            'emoji': '⚠️',
            'color': '#E74C3C',
            'recommendations': [
                '🔍 Analyze content relevance - are products what user seeks?',
                '💬 Implement exit-intent popup with special offer',
                '❓ Add live chat support to answer questions',
                '📊 Review page load speed and user experience',
                '🎨 Improve product descriptions and images'
            ]
        }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Load Models
model, scaler, load_error = load_production_models()

if load_error:
    st.error(f"""
    ⚠️ Model Loading Error
    
    The application couldn't load the required files:
    - `online_shoppers_model.keras`
    - `scaler.pkl`
    
    Please ensure you've run the Jupyter Notebook to generate these files first.
    """)
    st.stop()

# Header
st.markdown('<div class="header-title">🛍️ Purchase Prediction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">AI-Powered Customer Intent Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
📌 <strong>How it works:</strong> Input visitor behavioral metrics and our AI model predicts 
if they're likely to make a purchase. Use these insights to optimize your marketing strategy!
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Batch Analysis", "ℹ️ About"])

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================

with tab1:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Sidebar: Input Parameters
    st.sidebar.markdown("### 📋 Visitor Behavioral Data")
    st.sidebar.markdown("Enter the visitor's browsing metrics:")
    
    col_sidebar1, col_sidebar2 = st.sidebar.columns(2)
    
    with col_sidebar1:
        administrative = st.number_input(
            "Administrative Pages",
            min_value=0.0, max_value=100.0, value=1.0,
            help="Number of pages in administrative category"
        )
        
        administrative_duration = st.number_input(
            "Admin Duration (sec)",
            min_value=0.0, max_value=10000.0, value=0.0,
            help="Time spent on administrative pages"
        )
        
        informational = st.number_input(
            "Informational Pages",
            min_value=0.0, max_value=100.0, value=1.0,
            help="Number of pages in informational category"
        )
        
        informational_duration = st.number_input(
            "Info Duration (sec)",
            min_value=0.0, max_value=10000.0, value=0.0,
            help="Time spent on informational pages"
        )
        
        product_pages = st.number_input(
            "Product Pages",
            min_value=0.0, max_value=100.0, value=5.0,
            help="Number of product pages visited"
        )
    
    with col_sidebar2:
        product_duration = st.number_input(
            "Product Duration (sec)",
            min_value=0.0, max_value=100000.0, value=500.0,
            help="Time spent on product pages"
        )
        
        bounce_rate = st.slider(
            "Bounce Rate",
            0.0, 1.0, 0.5, 0.01,
            help="Percentage of single-page visits"
        )
        
        exit_rate = st.slider(
            "Exit Rate",
            0.0, 1.0, 0.3, 0.01,
            help="Percentage of exits from page"
        )
        
        page_value = st.number_input(
            "Page Value",
            min_value=0.0, max_value=100.0, value=0.0,
            help="Average monetary value per page"
        )
        
        operating_system = st.number_input(
            "Operating System ID",
            min_value=0, max_value=10, value=1,
            help="Encoded OS identifier"
        )
    
    # Additional parameters
    st.sidebar.markdown("### 🗓️ Temporal & Visitor Info")
    
    col_temp1, col_temp2 = st.sidebar.columns(2)
    
    with col_temp1:
        browser = st.number_input(
            "Browser ID",
            min_value=1, max_value=13, value=1,
            help="Browser type identifier"
        )
        
        region = st.number_input(
            "Region ID",
            min_value=1, max_value=9, value=1,
            help="Geographic region code"
        )
        
        month = st.selectbox(
            "Month",
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            index=0
        )
    
    with col_temp2:
        traffic_type = st.number_input(
            "Traffic Type ID",
            min_value=1, max_value=20, value=1,
            help="Traffic source type"
        )
        
        visitor_type = st.selectbox(
            "Visitor Type",
            ["Returning", "New", "Other"]
        )
        
        is_weekend = st.checkbox("Weekend Visit?", value=False)
        
        special_day = st.number_input(
            "Special Day Index",
            min_value=0, max_value=30, value=0,
            help="0 = No special day"
        )
    
    # Map categorical values
    month_map = {m: i+1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    visitor_type_map = {"Returning": 1, "New": 0, "Other": 2}
    
    # Predict button
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col_pred_left, col_pred_center, col_pred_right = st.columns([1, 2, 1])
    
    with col_pred_center:
        predict_button = st.button("🔮 Analyze Purchase Intent", use_container_width=True)
    
    if predict_button:
        # Prepare input data - MUST MATCH TRAINING DATA FEATURES (17 features)
        # Order: Administrative, Administrative_Duration, Informational, Informational_Duration, 
        #        ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, 
        #        SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend
        input_features = np.array([[
            administrative,              # 0: Administrative
            administrative_duration,     # 1: Administrative_Duration
            informational,               # 2: Informational
            informational_duration,      # 3: Informational_Duration
            product_pages,               # 4: ProductRelated
            product_duration,            # 5: ProductRelated_Duration
            bounce_rate,                 # 6: BounceRates
            exit_rate,                   # 7: ExitRates
            page_value,                  # 8: PageValues
            special_day,                 # 9: SpecialDay
            month_map[month],            # 10: Month
            operating_system,            # 11: OperatingSystems
            browser,                     # 12: Browser
            region,                      # 13: Region
            traffic_type,                # 14: TrafficType
            visitor_type_map[visitor_type],  # 15: VisitorType
            int(is_weekend)              # 16: Weekend
        ]])
        
        # Apply scaling
        scaled_input = scaler.transform(input_features)
        
        # Make prediction
        prediction_prob = model.predict(scaled_input, verbose=0)[0][0]
        prediction_class = (prediction_prob > 0.5).astype(int)
        confidence = prediction_prob if prediction_class == 1 else 1 - prediction_prob
        
        # Get insights
        insights = get_business_insights(prediction_class, confidence)
        
        # Display prediction card
        st.markdown(f"""
        <div class="prediction-card prediction-{'likely' if prediction_class == 1 else 'unlikely'}">
            <div class="prediction-title">{insights['emoji']} {insights['status']}</div>
            <div class="prediction-confidence">
                Confidence Level: <strong>{confidence*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display recommendations
        st.markdown("### 💡 Strategic Recommendations")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("**Immediate Actions:**")
            for i, rec in enumerate(insights['recommendations'][:3], 1):
                st.markdown(f"{i}. {rec}")
        
        with col_rec2:
            st.markdown("**Long-term Strategy:**")
            for i, rec in enumerate(insights['recommendations'][3:], 4):
                st.markdown(f"{i}. {rec}")
        
        # Display detailed metrics
        st.markdown("---")
        st.markdown("### 📊 Detailed Analytics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediction Score</div>
                <div class="metric-value">{prediction_prob:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Classification</div>
                <div class="metric-value">{'BUYER' if prediction_class == 1 else 'NON-BUYER'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            total_pages = int(administrative + informational + product_pages)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Pages</div>
                <div class="metric-value">{total_pages}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: BATCH ANALYSIS
# ============================================================================

with tab2:
    st.markdown("### 📤 Batch Prediction Upload")
    st.markdown("Upload a CSV file with multiple visitors for bulk predictions.")
    st.info("""
    **CSV Format Requirements:**
    Must contain all 17 features in this exact order:
    1. Administrative, 2. Administrative_Duration, 3. Informational, 4. Informational_Duration,
    5. ProductRelated, 6. ProductRelated_Duration, 7. BounceRates, 8. ExitRates, 9. PageValues,
    10. SpecialDay, 11. Month, 12. OperatingSystems, 13. Browser, 14. Region, 15. TrafficType,
    16. VisitorType, 17. Weekend
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        st.markdown(f"**Preview (first 5 rows):**")
        st.dataframe(batch_df.head())
        
        # Verify feature count
        if batch_df.shape[1] != 17:
            st.error(f" ERROR: CSV has {batch_df.shape[1]} columns, but 17 are required!")
            st.error("Please check your CSV format and try again.")
        else:
            if st.button(" Run Batch Predictions"):
                
                month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                if batch_df['Month'].dtype == object:
                    batch_df['Month'] = batch_df['Month'].map(month_map).fillna(0)
                
                visitor_map = {"Returning_Visitor": 1, "Returning": 1, "New_Visitor": 0, "New": 0, "Other": 2}
                if batch_df['VisitorType'].dtype == object:
                    batch_df['VisitorType'] = batch_df['VisitorType'].map(visitor_map).fillna(2)
                
                if batch_df['Weekend'].dtype == object or batch_df['Weekend'].dtype == bool:
                    batch_df['Weekend'] = batch_df['Weekend'].astype(int)
                    
                batch_df = batch_df.astype(float)
                # ==============================================================

                # Scale and predict
                scaled_batch = scaler.transform(batch_df)
                batch_predictions = model.predict(scaled_batch, verbose=0)
                
                # Add results to dataframe
                batch_df['Purchase_Probability'] = batch_predictions.flatten()
                batch_df['Predicted_Class'] = (batch_predictions.flatten() > 0.5).astype(int)
                batch_df['Prediction'] = batch_df['Predicted_Class'].map({1: 'BUYER', 0: 'NON-BUYER'})
                
                st.success(f"✓ Predictions complete for {len(batch_df)} visitors")
                
                # Display results
                st.dataframe(batch_df[['Purchase_Probability', 'Predicted_Class', 'Prediction']])
                
                # Summary statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    buyers = (batch_df['Predicted_Class'] == 1).sum()
                    st.metric("Potential Buyers", buyers, f"{buyers/len(batch_df)*100:.1f}%")
                
                with col_stat2:
                    non_buyers = (batch_df['Predicted_Class'] == 0).sum()
                    st.metric("Non-Buyers", non_buyers, f"{non_buyers/len(batch_df)*100:.1f}%")
                
                with col_stat3:
                    avg_confidence = batch_df['Purchase_Probability'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.markdown("""
    ### 🔬 About This System
    
    **Model Architecture:**
    - Framework: TensorFlow/Keras
    - Type: Binary Classification Neural Network
    - Layers: Input → Dense(32) → Dense(16) → Output
    - Activation: ReLU (hidden), Sigmoid (output)
    
    **Data Processing:**
    - Scaling: StandardScaler (numerical features)
    - Encoding: LabelEncoder (categorical features)
    - Class Balancing: Weighted loss (class_weight={0: 1, 1: 3})
    
    **Model Performance:**
    - Accuracy: ~88-89%
    - Optimized Recall for buyers to minimize missed sales
    
    ### 📚 Feature Descriptions (17 Features)
    
    | # | Feature | Description |
    |----|---------|-------------|
    | 1 | Administrative | Number of pages in administrative category visited |
    | 2 | Administrative_Duration | Time spent on administrative pages (seconds) |
    | 3 | Informational | Number of pages in informational category visited |
    | 4 | Informational_Duration | Time spent on informational pages (seconds) |
    | 5 | ProductRelated | Number of product pages visited |
    | 6 | ProductRelated_Duration | Time spent on product pages (seconds) |
    | 7 | BounceRates | Percentage of visitors who left after viewing one page |
    | 8 | ExitRates | Percentage of page exits |
    | 9 | PageValues | Average monetary value per page |
    | 10 | SpecialDay | Proximity to special events (0-30 days) |
    | 11 | Month | Month of visit (1-12) |
    | 12 | OperatingSystems | Operating system category (1-10) |
    | 13 | Browser | Browser type (1-13) |
    | 14 | Region | Geographic region (1-9) |
    | 15 | TrafficType | Traffic source (1-20) |
    | 16 | VisitorType | Returning (1), New (0), or Other (2) |
    | 17 | Weekend | Binary indicator of weekend visit |
    
    ### 🎯 Use Cases
    
    ✓ **Marketing Optimization** - Target high-intent visitors with personalized offers  
    ✓ **Revenue Prediction** - Forecast conversion rates by traffic segment  
    ✓ **UX Improvements** - Identify friction points for non-buyers  
    ✓ **Resource Allocation** - Focus support on likely converters  
    
    ### 🛠️ Technology Stack
    - Backend: Python, TensorFlow/Keras
    - Frontend: Streamlit
    - Model Explanation: SHAP
    - Deployment: Streamlit Cloud / AWS / GCP
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</p>
    <p>© 2026 Online Shoppers Purchase Prediction Project by Sara saleh Al-Harbi</p>
    <p>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</p>
</div>
""", unsafe_allow_html=True)