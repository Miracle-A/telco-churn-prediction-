# Telecom Customer Churn Prediction

## Live Dashboard

Explore the interactive dashboard here: [Telco Churn Prediction Dashboard](https://xxqtczzu5qrahlcsdmdcsf.streamlit.app/)

This project develops a machine learning solution to predict customer churn for a telecommunications company. By identifying at-risk customers before they leave, our model enables proactive retention strategies with potential annual savings of $3.8M.

![image](https://github.com/user-attachments/assets/ed8dc222-4389-443a-a501-26839b4f2375)
 <!-- You'll need to replace this with an actual screenshot link -->

## Features

- Data preprocessing and exploratory analysis
- Machine learning models (Logistic Regression, Random Forest, XGBoost)
- Interactive dashboard for exploring churn patterns
- Churn prediction tool for risk assessment
- Business recommendations and financial impact analysis

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 79.8% | 65.2% | 51.4% | 57.5% | 84.6% |
| Random Forest | 77.8% | 61.2% | 44.5% | 51.5% | 82.0% |
| XGBoost | 78.1% | 60.2% | 51.2% | 55.3% | 81.7% |

The **Logistic Regression model** performed best overall across key metrics, particularly in precision and F1 score, making it our chosen model for churn prediction.

## Business Impact

- **Churn Rate**: 26.5% annual customer churn
- **Annual Revenue Loss**: $1.45M from churned customers
- **Projected Savings**: $435K annually through targeted retention strategies (30% reduction in churn)
- **3-Year Projected Savings**: $1.3M

## Key Insights

1. **Contract type** is the strongest predictor of churn, with month-to-month customers churning at 3-4x the rate of long-term contracts
2. **Fiber optic customers** have higher churn despite paying more for premium service
3. **Security & support services** significantly reduce churn probability
4. **Electronic check payment** method correlates with higher churn rates
5. **New customers** (tenure < 12 months) represent the highest churn risk segment

## Live Dashboard

Explore the interactive dashboard here: [Telco Churn Prediction Dashboard](https://xxqtczzu5qrahlcsdmdcsf.streamlit.app/)

## Running Locally

1. **Clone the repository**
git clone https://github.com/Miracle-A/telco-churn-prediction.git
cd telco-churn-prediction
Copy
2. **Set up a virtual environment**
python -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate
Copy
3. **Install dependencies**
pip install -r requirements.txt
Copy
4. **Run the dashboard**
streamlit run src/dashboard.py
Copy
## Project Structure

- `data/`: Raw and processed datasets
- `models/`: Saved model files
- `src/`: Python scripts for data processing, modeling, and dashboard
- `images/`: Visualizations generated during analysis

## Technologies Used

- **Python**: Data analysis and modeling
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn/XGBoost**: Machine learning algorithms
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **Streamlit**: Interactive dashboard

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
