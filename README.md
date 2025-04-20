
# Telco Customer Churn Prediction 📊

This repository provides a complete pipeline for analyzing and predicting customer churn using machine learning. It includes an interactive **Gradio app**, an exploratory **Jupyter Notebook**, and all necessary preprocessing, training, and evaluation steps.

---

## 🔍 Contents

- `app.py` - Main Gradio interface to run the churn prediction UI.
- `Churn_Analysis.ipynb` - Jupyter notebook for full data exploration, feature engineering, and model evaluation.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Sample dataset from IBM Watson (Telco customer churn).
- `requirements.txt` - Python dependencies for running the project.
- `README.md` - Project overview and usage instructions.

---

## 💻 How to Run

### Option 1: Run Locally
1. Clone the repository:
```bash
git clone https://github.com/yourusername/telco-churn-predictor.git
cd telco-churn-predictor
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Launch the Gradio App:
```bash
python app.py
```

---

### Option 2: Run on Hugging Face Spaces
1. Upload the following to your Space:
   - `app.py`
   - `WA_Fn-UseC_-Telco-Customer-Churn.csv`
   - `requirements.txt`

2. Hugging Face will automatically install dependencies and run the UI.

3. Open your hosted URL. You will see:
   - A file upload section (upload the CSV file).
   - Dropdown to select a numeric column for EDA (e.g., `MonthlyCharges`).
   - Dropdown to select model (KNN, XGBoost, etc.)
   - View Confusion Matrix, Feature Importances, and Performance Metrics.

---

## 🧪 Dataset Description

- **Source**: IBM Watson
- **Records**: 7043 customers
- **Target**: `Churn` column (Yes/No)
- **Features**: Customer demographic, account, and service usage info

---

## 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
gradio
imbalanced-learn
```

---

## 📷 UI Walkthrough
- Hugging Face Link : https://huggingface.co/spaces/Challat/CustomerChurnPrediction
- Upload the Telco dataset.
- Select a numeric column to visualize EDA (boxplot vs churn).
- Choose a machine learning model to evaluate performance.
- The app automatically performs GridSearchCV and plots outputs.

---

## 📜 License

MIT License
