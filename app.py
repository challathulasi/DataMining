import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")

# Store uploaded dataset globally
global_df = {}

def load_dataset(file):
    df = pd.read_csv(file.name)
    df['TotalCharges'].replace(" ", 0, inplace=True)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'No' if x == 0 else 'Yes')
    df.rename(columns={'InternetService': 'InternetServiceTypes'}, inplace=True)
    df['InternetService'] = df['InternetServiceTypes'].apply(lambda x: 'Yes' if x in ['DSL', 'Fiber optic'] else 'No')
    df.drop(columns=['customerID'], inplace=True)
    global_df['df'] = df
    return f"✅ Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.", gr.update(choices=df.select_dtypes(include=['int64', 'float64']).columns.tolist(), value=None)

def show_eda_plot(column):
    if not column or column not in global_df['df'].columns:
        return None  # Avoid plotting if invalid column
    df = global_df['df']
    fig, ax = plt.subplots()
    sns.boxplot(x=df['Churn'], y=df[column], ax=ax)
    ax.set_title(f'{column} vs Churn')
    return fig


def run_model(model_name):
    df = global_df['df']
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']

    sm = SMOTEENN(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model_dict = {
        "Decision Tree": (DecisionTreeClassifier(random_state=100), {"max_depth": [6, 10], "min_samples_leaf": [8, 12]}),
        "Random Forest": (RandomForestClassifier(random_state=100), {"n_estimators": [100], "max_depth": [10]}),
        "XGBoost": (XGBClassifier(random_state=100, use_label_encoder=False, eval_metric="logloss"), {"learning_rate": [0.1], "max_depth": [3]}),
        "AdaBoost": (AdaBoostClassifier(random_state=100), {"n_estimators": [100], "learning_rate": [0.1]}),
        "Logistic Regression": (LogisticRegression(solver='liblinear', random_state=100), {"C": [1], "penalty": ['l2']}),
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [5], "metric": ['euclidean']})
    }

    model, param_grid = model_dict[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    metrics = f"""
### 📊 Model: {model_name}

**Best Params**: `{grid_search.best_params_}`  
**Accuracy**: {acc:.4f}  
**Precision**: {prec:.4f}  
**Recall**: {recall:.4f}  
**F1-Score**: {f1:.4f}  
    """

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix - {model_name}")

    fig_imp = None
    try:
        result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        imp_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': result.importances_mean}).sort_values(by='Importance', ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=imp_df.head(10), ax=ax_imp)
        ax_imp.set_title("Top 10 Important Features")
    except:
        pass

    return metrics, fig_cm, fig_imp

# Gradio UI
with gr.Blocks() as churn_ui:
    gr.Markdown("# 📈 Telco Customer Churn Prediction with EDA & Model Analysis")

    with gr.Row():
        file_input = gr.File(label="Upload Telco Churn CSV")
        load_btn = gr.Button("Load Dataset")
        output_text = gr.Textbox(label="Status")

    column_dropdown = gr.Dropdown(label="Select Column for EDA", choices=[], interactive=True)
    eda_output = gr.Plot(label="EDA Plot: Column vs Churn")

    load_btn.click(load_dataset, inputs=file_input, outputs=[output_text, column_dropdown])
    column_dropdown.change(fn=show_eda_plot, inputs=column_dropdown, outputs=eda_output)

    gr.Markdown("---")

    with gr.Row():
        model_selector = gr.Dropdown(choices=["Decision Tree", "Random Forest", "XGBoost", "AdaBoost", "Logistic Regression", "KNN", "SVM"], label="Select Model")
        model_btn = gr.Button("Run Model")

    metrics_output = gr.Markdown()
    cm_output = gr.Plot()
    imp_output = gr.Plot()

    model_btn.click(fn=run_model, inputs=model_selector, outputs=[metrics_output, cm_output, imp_output])

churn_ui.launch(share=True)
