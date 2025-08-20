# 🏡 Housing Price Prediction with Streamlit Dashboard

This project predicts **house prices** using **Multiple Linear Regression** and provides an interactive **Streamlit web app** with dashboards, dataset exploration, and prediction features.

---

## 📌 Features
- **Dataset Overview**: View dataset structure and summary statistics.  
- **Correlation Heatmap**: Visualize correlations between features.  
- **Interactive Graphs**: Explore feature distributions and relationships.  
- **Model Evaluation**: View regression metrics like MAE, RMSE, and R².  
- **Price Prediction**: Input house features and get predicted price instantly.  

---

## ⚙️ Tech Stack
- **Language**: Python 3.8  
- **Libraries**:  
  - `pandas`, `numpy` – Data handling  
  - `scikit-learn` – ML model & metrics  
  - `matplotlib`, `seaborn` – Visualization  
  - `streamlit` – Interactive dashboard  

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

```bash
conda create -n mlproj python=3.8

```

```bash
conda activate mlproj
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```