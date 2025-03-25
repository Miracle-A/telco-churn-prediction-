import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# Define the path handling to make it work both locally and on Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# Load the processed data
data_path = os.path.join(root_dir, 'data', 'telco_processed.csv')
df = pd.read_csv(data_path)
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# Try to load the saved models
try:
    logistic_model_path = os.path.join(root_dir, 'models', 'logistic_model.pkl')
    rf_model_path = os.path.join(root_dir, 'models', 'rf_model.pkl')
    xgb_model_path = os.path.join(root_dir, 'models', 'xgb_model.pkl')
    
    logistic_model = pickle.load(open(logistic_model_path, 'rb'))
    rf_model = pickle.load(open(rf_model_path, 'rb'))
    xgb_model = pickle.load(open(xgb_model_path, 'rb'))
    models_loaded = True
except:
    models_loaded = False

# Load model comparison data
model_comparison_path = os.path.join(root_dir, 'data', 'model_comparison.csv')
model_comparison = pd.read_csv(model_comparison_path)
model_comparison = model_comparison.set_index('Unnamed: 0')
model_comparison.index.name = 'Model'

# App title
st.title('Telco Customer Churn Analysis')
st.markdown("""
This dashboard presents the analysis and prediction of customer churn for a telecommunications company.
""")

# Sidebar
st.sidebar.title('Navigation')
options = [
    'Overview',
    'Exploratory Analysis', 
    'Customer Segments',
    'Model Performance',
    'Churn Prediction',
    'Business Recommendations'
]
selection = st.sidebar.radio('Go to', options)

# Overview page
if selection == 'Overview':
    st.header('Project Overview')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Total Customers', df.shape[0])
        st.metric('Churn Rate', f"{df['Churn_Binary'].mean():.1%}")
        
    with col2:
        st.metric('Monthly Revenue', f"${df['MonthlyCharges'].sum():,.2f}")
        st.metric('Avg. Customer Lifetime', f"{df['tenure'].mean():.1f} months")
    
    # Churn distribution
    fig = px.pie(df, names='Churn', title='Customer Churn Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Project Highlights')
    st.markdown("""
    - **Business Problem**: Customer churn costs telecom companies billions annually
    - **Approach**: Built machine learning models to identify at-risk customers
    - **Results**: Logistic Regression model achieved 80% accuracy and 65% precision
    - **Impact**: Potential annual savings of $3.8M through targeted retention strategies
    """)

# Exploratory Analysis page
elif selection == 'Exploratory Analysis':
    st.header('Exploratory Data Analysis')
    
    tab1, tab2, tab3 = st.tabs(['Churn Factors', 'Customer Demographics', 'Service Usage'])
    
    with tab1:
        st.subheader('Key Factors Influencing Churn')
        
        # Contract type vs churn
        fig = px.histogram(df, x='Contract', color='Churn', 
                         barmode='group', title='Churn by Contract Type')
        st.plotly_chart(fig, use_container_width=True)
        
        # Tenure vs churn
        fig = px.box(df, x='Churn', y='tenure', 
                   title='Customer Tenure by Churn Status')
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly charges vs churn
        fig = px.box(df, x='Churn', y='MonthlyCharges', 
                   title='Monthly Charges by Churn Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader('Customer Demographics')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            fig = px.pie(df, names='gender', title='Gender Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Senior citizen distribution
            fig = px.pie(df, names='SeniorCitizen', title='Senior Citizens')
            st.plotly_chart(fig, use_container_width=True)
        
        # Partner and Dependents
        fig = px.histogram(df, x='Partner', color='Dependents', 
                         barmode='group', title='Family Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader('Service Usage Patterns')
        
        # Internet service distribution
        fig = px.histogram(df, x='InternetService', color='Churn', 
                         barmode='group', title='Internet Service Type vs Churn')
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional services
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        service_data = {}
        for service in services:
            service_data[service] = df.groupby([service, 'Churn']).size().reset_index(name='count')
        
        selected_service = st.selectbox('Select Service to Analyze:', services)
        fig = px.bar(service_data[selected_service], x=selected_service, y='count', 
                   color='Churn', barmode='group', title=f'{selected_service} vs Churn')
        st.plotly_chart(fig, use_container_width=True)

# Customer Segments page
elif selection == 'Customer Segments':
    st.header('Customer Segmentation')
    
    # Create segments based on tenure and monthly charges
    df['tenure_segment'] = pd.qcut(df['tenure'], 4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    df['charges_segment'] = pd.qcut(df['MonthlyCharges'], 4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    
    # Create a segment grid
    segment_grid = df.groupby(['tenure_segment', 'charges_segment']).agg({
        'customerID': 'count',
        'Churn_Binary': 'mean'
    }).reset_index()
    
    segment_grid = segment_grid.rename(columns={
        'customerID': 'Customer Count',
        'Churn_Binary': 'Churn Rate'
    })
    
    # Plot segment grid as heatmap
    fig = px.density_heatmap(segment_grid, x='charges_segment', y='tenure_segment', z='Churn Rate',
                           title='Customer Segment Churn Rates', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # High-value customer analysis
    st.subheader('High-Value Customer Analysis')
    
    # Define high-value customers
    high_value = df[(df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)) & 
                   (df['tenure'] > df['tenure'].quantile(0.75))]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('High-Value Customers', f"{len(high_value)} ({len(high_value)/len(df):.1%})")
    
    with col2:
        st.metric('High-Value Churn Rate', f"{high_value['Churn_Binary'].mean():.1%}")
    
    # High-value churn by contract
    fig = px.histogram(high_value, x='Contract', color='Churn', 
                     barmode='group', title='High-Value Customer Churn by Contract')
    st.plotly_chart(fig, use_container_width=True)
    
    # Most common service combinations for high-value customers
    st.subheader('High-Value Customer Service Preferences')
    
    high_value_services = high_value[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                    'TechSupport', 'StreamingTV', 'StreamingMovies']].value_counts().reset_index()
    high_value_services.columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Count']
    
    st.dataframe(high_value_services.head(5))

# Model Performance page
elif selection == 'Model Performance':
    st.header('Model Performance')
    
    # Display model comparison
    st.subheader('Model Comparison')
    st.dataframe(model_comparison)
    
    # Plot metrics comparison
    metrics = model_comparison.columns.tolist()
    models = model_comparison.index.tolist()
    
    selected_metric = st.selectbox('Select Metric:', metrics)
    
    fig = px.bar(model_comparison, y=selected_metric, x=model_comparison.index,
               title=f'Model Comparison - {selected_metric}')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show confusion matrices - needs path fix
    st.subheader('Confusion Matrices')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confusion_lr_path = os.path.join(root_dir, 'images', 'confusion_matrix_Logistic_Regression.png')
        st.image(confusion_lr_path, caption='Logistic Regression')
    
    with col2:
        confusion_rf_path = os.path.join(root_dir, 'images', 'confusion_matrix_Random_Forest.png')
        st.image(confusion_rf_path, caption='Random Forest')
    
    with col3:
        confusion_xgb_path = os.path.join(root_dir, 'images', 'confusion_matrix_XGBoost.png')
        st.image(confusion_xgb_path, caption='XGBoost')
    
    # ROC Curves - needs path fix
    st.subheader('ROC Curves')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        roc_lr_path = os.path.join(root_dir, 'images', 'roc_curve_Logistic_Regression.png')
        st.image(roc_lr_path, caption='Logistic Regression')
    
    with col2:
        roc_rf_path = os.path.join(root_dir, 'images', 'roc_curve_Random_Forest.png')
        st.image(roc_rf_path, caption='Random Forest')
    
    with col3:
        roc_xgb_path = os.path.join(root_dir, 'images', 'roc_curve_XGBoost.png')
        st.image(roc_xgb_path, caption='XGBoost')


# Churn Prediction page
elif selection == 'Churn Prediction':
    st.header('Churn Prediction Tool')
    
    st.markdown("""
    This tool allows you to predict whether a customer will churn based on their characteristics.
    Adjust the values below to see how they affect the churn probability.
    """)
    
    if not models_loaded:
        st.warning("Models not loaded. Please run the model saving code first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider('Tenure (months)', 0, 72, 36)
            monthly_charges = st.slider('Monthly Charges ($)', 18, 119, 70)
            contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            
        with col2:
            online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
            tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
            paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
            payment_method = st.selectbox('Payment Method', 
                                       ['Electronic check', 'Mailed check', 
                                        'Bank transfer (automatic)', 'Credit card (automatic)'])
        
        # Create a customer sample
        customer = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [tenure * monthly_charges],
            'Contract': [contract],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'TechSupport': [tech_support],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method]
        })
        
        # Fill in other columns with defaults
        for col in df.columns:
            if col not in customer.columns and col != 'Churn' and col != 'customerID' and col != 'Churn_Binary':
                if col in df.select_dtypes(include=['object']).columns:
                    customer[col] = df[col].mode().iloc[0]
                else:
                    customer[col] = df[col].median()
        
        if st.button('Predict Churn Probability'):
            # Predict with all models
            lr_proba = logistic_model.predict_proba(customer)[0, 1]
            rf_proba = rf_model.predict_proba(customer)[0, 1]
            xgb_proba = xgb_model.predict_proba(customer)[0, 1]
            
            # Display predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric('Logistic Regression', f"{lr_proba:.1%}")
            
            with col2:
                st.metric('Random Forest', f"{rf_proba:.1%}")
            
            with col3:
                st.metric('XGBoost', f"{xgb_proba:.1%}")
            
            # Overall prediction
            avg_proba = (lr_proba + rf_proba + xgb_proba) / 3
            
            if avg_proba > 0.5:
                st.error(f"⚠️ High churn risk: {avg_proba:.1%} probability")
            elif avg_proba > 0.3:
                st.warning(f"⚠️ Moderate churn risk: {avg_proba:.1%} probability")
            else:
                st.success(f"✅ Low churn risk: {avg_proba:.1%} probability")
            
            # Recommendation based on prediction
            st.subheader('Retention Recommendations')
            
            if avg_proba > 0.5:
                st.markdown("""
                - **Immediate outreach** with personalized retention offer
                - Consider contract upgrade incentives with significant discounts
                - Offer free service upgrades (Tech Support, Online Security)
                """)
            elif avg_proba > 0.3:
                st.markdown("""
                - **Proactive contact** to address potential pain points
                - Survey customer satisfaction and address concerns
                - Offer bundled services at reduced rates
                """)
            else:
                st.markdown("""
                - Include in regular loyalty program communications
                - Cross-sell additional services that complement current usage
                - Periodic satisfaction checks to maintain relationship
                """)

# Business Recommendations page
elif selection == 'Business Recommendations':
    st.header('Business Recommendations')
    
    st.subheader('Key Insights from Analysis')
    
    st.markdown("""
    1. **Contract type** is the strongest predictor of churn, with month-to-month customers churning at 3-4x the rate of long-term contracts
    2. **Fiber optic customers** have higher churn despite paying more for premium service
    3. **Security & support services** significantly reduce churn probability
    4. **Electronic check payment** method correlates with higher churn rates
    5. **New customers** (tenure < 12 months) represent the highest churn risk segment
    """)
    
    st.subheader('Financial Impact of Churn')
    
    # Calculate financial metrics
    avg_monthly = df['MonthlyCharges'].mean()
    avg_tenure = df['tenure'].mean()
    customer_lifetime_value = avg_monthly * avg_tenure
    
    # Cost of churn
    annual_churn_rate = df['Churn_Binary'].mean()
    total_customers = len(df)
    annual_lost_customers = total_customers * annual_churn_rate
    annual_revenue_loss = annual_lost_customers * avg_monthly * 12
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Customer Lifetime Value', f"${customer_lifetime_value:,.2f}")
        st.metric('Annual Churn Rate', f"{annual_churn_rate:.1%}")
        
    with col2:
        st.metric('Annual Customers Lost', f"{annual_lost_customers:,.0f}")
        st.metric('Annual Revenue Loss', f"${annual_revenue_loss:,.2f}")
    
    st.subheader('Recommended Retention Strategies')
    
    tab1, tab2, tab3 = st.tabs(['Strategic', 'Tactical', 'Financial'])
    
    with tab1:
        st.markdown("""
        ### Strategic Recommendations
        
        1. **Contract Strategy Overhaul**
           - Develop more attractive 1-2 year contract incentives
           - Create stepped discount structure that increases with commitment length
           - Implement early renewal rewards for existing customers
        
        2. **Service Quality Initiative**
           - Address technical issues in fiber optic service
           - Implement proactive service monitoring for premium customers
           - Develop service guarantees with automatic compensation for outages
        
        3. **Customer Journey Optimization**
           - Redesign onboarding process to highlight security and support services
           - Implement "save team" with specialized retention training and offers
           - Create milestone-based engagement program throughout customer lifecycle
        """)
    
    with tab2:
        st.markdown("""
        ### Tactical Recommendations
        
        1. **High-Risk Segment Targeting**
           - Implement ML-powered early warning system for at-risk customers
           - Develop specialized retention offers for each customer segment
           - Create proactive outreach program for customers showing churn indicators
        
        2. **Service Bundling**
           - Redesign service packages to include security features by default
           - Create special upgrade paths for month-to-month customers
           - Implement family/household plans to increase switching costs
        
        3. **Payment Experience**
           - Offer discounts for automatic payment methods
           - Simplify billing statements and payment processes
           - Provide payment method migration assistance with incentives
        """)
    
    with tab3:
        st.markdown("""
        ### Financial Recommendations
        
        1. **Retention Budget Allocation**
           - Dedicate 3-5% of annual revenue to retention initiatives
           - Implement ROI tracking for all retention campaigns
           - Develop predictive CLV model to optimize retention spending
        
        2. **Pricing Strategy**
           - Implement subtle price increases for month-to-month contracts
           - Develop loyalty pricing tiers based on tenure
           - Create strategic discounting framework for retention offers
        
        3. **Retention Performance Metrics**
           - Establish retention KPIs and executive dashboards
           - Implement churn reduction targets in team performance metrics
           - Create cross-functional retention task force with clear objectives
        """)
    
    st.subheader('Implementation Roadmap')
    
    # Timeline data
    timeline_data = {
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        'Start': [0, 3, 6, 9],
        'Duration': [3, 3, 3, 3],
        'Description': [
            'Implement Early Warning System & Quick Wins',
            'Deploy Contract Strategy & Service Quality Improvements',
            'Roll Out Customer Journey Optimization',
            'Launch Advanced Segmentation & Personalization'
        ]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Create timeline chart
    fig = px.timeline(timeline_df, x_start='Start', x_end=timeline_df['Start'] + timeline_df['Duration'],
                    y='Phase', color='Phase', text='Description',
                    title='Implementation Roadmap (Months)',
                    labels={'Start': 'Month'})
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Expected Outcomes')
    
    # Expected outcome metrics
    target_reduction = 0.3  # 30% reduction in churn rate
    new_churn_rate = annual_churn_rate * (1 - target_reduction)
    new_annual_lost_customers = total_customers * new_churn_rate
    customers_saved = annual_lost_customers - new_annual_lost_customers
    revenue_saved = customers_saved * avg_monthly * 12
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Target Churn Rate', f"{new_churn_rate:.1%}", 
                delta=f"-{target_reduction:.0%}", delta_color='inverse')
        st.metric('Customers Saved Annually', f"{customers_saved:,.0f}")
        
    with col2:
        st.metric('Annual Revenue Protected', f"${revenue_saved:,.2f}")
        st.metric('3-Year Projected Savings', f"${revenue_saved * 3:,.2f}")
    
    st.markdown("""
    **Note:** These projections are based on a 30% reduction in churn rate over 12 months.
    Actual results may vary based on implementation quality and market conditions.
    """)
