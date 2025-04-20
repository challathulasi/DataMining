import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings("ignore")

def churn_pipeline(file):
    df = pd.read_csv(file.name)

    # Preprocessing
    df['TotalCharges'].replace(" ", 0, inplace=True)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'No' if x == 0 else 'Yes')
    df.rename(columns={'InternetService': 'InternetServiceTypes'}, inplace=True)
    df['InternetService'] = df['InternetServiceTypes'].apply(lambda x: 'Yes' if x in ['DSL', 'Fiber optic'] else 'No')
    df.drop(columns=['customerID'], inplace=True)

    df_dummy = pd.get_dummies(df, dtype='int', drop_first=True)
    X = df_dummy.drop(columns=['Churn_Yes'])
    y = df_dummy['Churn_Yes']

    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    models = {
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=100),
            "param_grid": {"max_depth": [6], "min_samples_leaf": [8]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=100),
            "param_grid": {"n_estimators": [100], "max_depth": [10]}
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=100, use_label_encoder=False, eval_metric='logloss'),
            "param_grid": {"learning_rate": [0.1], "max_depth": [5]}
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=100),
            "param_grid": {"n_estimators": [100], "learning_rate": [0.1]}
        },
        "Logistic Regression": {
            "model": LogisticRegression(solver='liblinear', random_state=100),
            "param_grid": {"C": [1], "penalty": ['l2']}
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "param_grid": {"n_neighbors": [5], "metric": ['euclidean']}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=100),
            "param_grid": {"C": [1], "kernel": ['linear'], "gamma": ['scale']}
        }
    }

    results = []

    for model_name, model_details in models.items():
        grid_search = GridSearchCV(model_details["model"], model_details["param_grid"], cv=5, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        result = f"### {model_name}\n"
        result += f"- **Best Parameters**: {grid_search.best_params_}\n"
        result += f"- **Accuracy**: {acc:.4f}\n"
        result += f"- **Precision**: {prec:.4f}\n"
        result += f"- **Recall**: {rec:.4f}\n"
        result += f"- **F1 Score**: {f1:.4f}\n"
        result += f"- **Confusion Matrix**:\n```\n{matrix}\n```\n\n"
        results.append(result)

    return "\n".join(results)


# Gradio Interface
iface = gr.Interface(
    fn=churn_pipeline,
    inputs=gr.File(label="Upload Telco Churn CSV"),
    outputs=gr.Markdown(label="Model Results"),
    title="Telco Churn Prediction with ML Models",
    description="Upload the Telco churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) and evaluate multiple ML models with GridSearchCV."
)

if __name__ == "__main__":
    iface.launch()
