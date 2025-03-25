import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Load the processed data
df = pd.read_csv('telco_processed.csv')

# Create target variable
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# Drop unnecessary columns
columns_to_drop = ['customerID', 'Churn']  # Add any other columns you don't need
df = df.drop(columns=columns_to_drop)

# Split categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('Churn_Binary')  # Remove target from numerical features

# Print features
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Split data into features and target
X = df.drop('Churn_Binary', axis=1)
y = df['Churn_Binary']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Create preprocessing pipeline
# Numerical features: impute missing values and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define model evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and print metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"\n{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    
    # Display ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name}.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Create and train models
# 1. Logistic Regression
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
log_reg_pipeline.fit(X_train, y_train)
log_reg_results = evaluate_model(log_reg_pipeline, X_test, y_test, 'Logistic_Regression')

# 2. Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
rf_pipeline.fit(X_train, y_train)
rf_results = evaluate_model(rf_pipeline, X_test, y_test, 'Random_Forest')

# 3. XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42))
])
xgb_pipeline.fit(X_train, y_train)
xgb_results = evaluate_model(xgb_pipeline, X_test, y_test, 'XGBoost')

# Compare models
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
results = [log_reg_results, rf_results, xgb_results]

# Create comparison table
comparison_df = pd.DataFrame(index=models, columns=metrics)
for i, model in enumerate(models):
    for metric in metrics:
        comparison_df.loc[model, metric] = results[i][metric]

print("\nModel Comparison:")
print(comparison_df)

# Save model comparison
comparison_df.to_csv('model_comparison.csv')

# Identify best model
best_model_idx = np.argmax([r['f1'] for r in results])
best_model_name = models[best_model_idx]
print(f"\nBest model by F1 score: {best_model_name}")

# Feature importance for tree-based models
if best_model_idx in [1, 2]:  # Random Forest or XGBoost
    best_pipeline = [rf_pipeline, xgb_pipeline][best_model_idx - 1]
    
    # Get feature names after preprocessing
    # This part can be tricky because of the preprocessing transformations
    try:
        # For categorical features, get the one-hot encoded feature names
        cat_features_encoded = best_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features)
        
        # Combine all feature names
        feature_names = np.array(numerical_features + list(cat_features_encoded))
        
        # Get importances
        importances = best_pipeline.named_steps['classifier'].feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {best_model_name}')
        plt.bar(range(min(20, len(indices))), importances[indices][:20], align='center')
        plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices][:20], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Print top 10 features
        print("\nTop 10 important features:")
        for i in range(min(10, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")
        print("This is often due to preprocessing transformations changing feature names.")

print("\nModel development complete. Results saved.")
# Save models to files
pickle.dump(log_reg_pipeline, open('logistic_model.pkl', 'wb'))
pickle.dump(rf_pipeline, open('rf_model.pkl', 'wb'))
pickle.dump(xgb_pipeline, open('xgb_model.pkl', 'wb'))

print("Models saved for dashboard use.")