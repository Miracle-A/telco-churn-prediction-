import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set styling for better visualizations
plt.style.use('ggplot')
sns.set(style='whitegrid')

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic data inspection
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with MonthlyCharges * tenure
mask = df['TotalCharges'].isnull()
df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']

# Basic statistics for numerical columns
print("\nNumerical Features Summary:")
print(df.describe())

# Distribution of categorical features
print("\nCategorical Features Summary:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())
    
# Churn rate analysis
churn_rate = df['Churn'].value_counts(normalize=True)
print("\nOverall Churn Rate:")
print(churn_rate)

# Visualize churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.tight_layout()
plt.savefig('churn_distribution.png')

# Analyze churn by tenure
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure by Churn Status')
plt.tight_layout()
plt.savefig('tenure_by_churn.png')

# Analyze churn by contract type
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.tight_layout()
plt.savefig('churn_by_contract.png')

# Analyze churn by monthly charges
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn Status')
plt.tight_layout()
plt.savefig('charges_by_churn.png')

# Feature Engineering
# Create binary features from categorical variables
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# Create a tenure group feature
def create_tenure_group(tenure):
    if tenure <= 12:
        return '0-1 year'
    elif tenure <= 24:
        return '1-2 years'
    elif tenure <= 48:
        return '2-4 years'
    else:
        return '4+ years'

df['tenure_group'] = df['tenure'].apply(create_tenure_group)

# Count total services per customer
services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Create service binary flags
for col in services:
    df[col + '_Flag'] = df[col].apply(
        lambda x: 1 if x not in ['No', 'No internet service', 'No phone service'] else 0
    )

df['total_services'] = df[[col + '_Flag' for col in services]].sum(axis=1)

# Visualize total services by churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='total_services', data=df)
plt.title('Total Services by Churn Status')
plt.tight_layout()
plt.savefig('services_by_churn.png')

# Save the processed dataset
df.to_csv('telco_processed.csv', index=False)

print("\nData processing complete. Processed file saved as 'telco_processed.csv'")