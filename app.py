import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    return df

df = load_data()

# Encoding
label_encoder = LabelEncoder()
cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col in cols:
    df[col] = label_encoder.fit_transform(df[col])

# One-hot encoding
ohe = OneHotEncoder(sparse=False)
furnishing_status = df[['furnishingstatus']]
encoded = ohe.fit_transform(furnishing_status)
df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['furnishingstatus']), index=df.index)
df = df.drop('furnishingstatus', axis=1).join(df_encoded)

# Features 
X = df.drop(columns=['price'])
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Performance", "Prediction"])


if page == "Home":
    st.title("🏠 Housing Price Prediction Dashboard")
    st.header("📊 Dataset Overview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Dataset Info")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")


elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Spectral", ax=ax)  
    st.pyplot(fig)

    st.subheader("Distribution of Price")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax, color="orange")  
    st.pyplot(fig)

    st.subheader("Price vs Area (Scatterplot)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['area'], y=df['price'], ax=ax, color="purple", alpha=0.6)
    st.pyplot(fig)

    st.subheader("Boxplot of Price by Stories")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['stories'], y=df['price'], ax=ax, palette="Set2") 
    st.pyplot(fig)


elif page == "Model Performance":
    st.title("📈 Model Performance")

    st.write(f"**MAE:** {mae:,.2f}")
    st.write(f"**RMSE:** {rmse:,.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.subheader("Predicted vs Actual Prices")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, c="red", edgecolors="black")  
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Predicted vs Actual")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((6,6)) 

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    st.pyplot(fig)

    st.subheader("Residual Distribution")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax, color="green") 
    ax.set_title("Residual Distribution")
    st.pyplot(fig)


elif page == "Prediction":
    st.title("🔮 Predict House Price")

    user_input = {}

    
    user_input['area'] = st.number_input("Enter area (sq ft)", float(df['area'].min()), float(df['area'].max()), float(df['area'].mean()))
    user_input['bedrooms'] = st.number_input("Number of bedrooms", int(df['bedrooms'].min()), int(df['bedrooms'].max()), int(df['bedrooms'].mean()))
    user_input['bathrooms'] = st.number_input("Number of bathrooms", int(df['bathrooms'].min()), int(df['bathrooms'].max()), int(df['bathrooms'].mean()))
    user_input['stories'] = st.number_input("Number of stories", int(df['stories'].min()), int(df['stories'].max()), int(df['stories'].mean()))

    
    for col in cols:
        user_input[col] = st.selectbox(f"{col}", [0, 1])

    
    furnishing_choice = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])
    furnishing_encoded = ohe.transform([[furnishing_choice]])
    furnishing_encoded = pd.DataFrame(furnishing_encoded, columns=ohe.get_feature_names_out(['furnishingstatus']))

    
    input_df = pd.DataFrame([user_input])
    input_df = input_df.join(furnishing_encoded)

   
    input_encoded_final = input_df.reindex(columns=X.columns, fill_value=0)

    # Predict
    if st.button("Predict Price"):
        prediction = model.predict(input_encoded_final)[0]
        st.success(f"🏡 Predicted House Price: ₹ {prediction:,.2f}")
