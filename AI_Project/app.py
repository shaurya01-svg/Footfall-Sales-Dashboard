import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import os

# PAGE CONFIG
st.set_page_config(page_title="Footfall Dashboard", layout="wide")

# LOAD DATA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "Data", "cleaned_walmart.csv")

df = pd.read_csv(csv_path)

#DEBUG 
st.write("Base Directory:", BASE_DIR)
st.write("Files in AI_Project:", os.listdir(BASE_DIR))
# DATE CONVERSION
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# CREATE MONTH & YEAR
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# LOAD MODEL

model_path = os.path.join(BASE_DIR, "Notebook", "model.pkl")

model = pickle.load(open(model_path, "rb"))
# SIDEBAR
st.sidebar.title("🔎 Filters")

selected_store = st.sidebar.selectbox(
    "Select Store",
    sorted(df["Store"].unique())
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

# FILTER DATA
filtered_df = df[
    (df["Store"] == selected_store) &
    (df["Year"] == selected_year)
]

# TITLE
st.title("📊 Footfall vs Sales Impact Dashboard")

st.write(
    "Analyze how footfall affects Walmart weekly sales using Machine Learning and Explainable AI."
)

# METRICS
total_sales = filtered_df["Weekly_Sales"].sum()
avg_sales = filtered_df["Weekly_Sales"].mean()
avg_footfall = filtered_df["Footfall"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"₹ {total_sales:,.0f}")
col2.metric("📈 Avg Weekly Sales", f"₹ {avg_sales:,.0f}")
col3.metric("👥 Avg Footfall", f"{avg_footfall:,.0f}")

# DATASET PREVIEW
st.subheader("📋 Dataset Preview")
st.dataframe(filtered_df.head())

# MONTHLY SALES TREND
st.subheader("📈 Monthly Sales Trend")

monthly_sales = filtered_df.groupby("Month")["Weekly_Sales"].sum()

fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(monthly_sales.index, monthly_sales.values, marker='o')
ax1.set_xlabel("Month")
ax1.set_ylabel("Sales")
ax1.set_title("Monthly Sales Trend")

st.pyplot(fig1)

# FOOTFALL VS SALES
st.subheader("👥 Footfall vs Sales")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    data=filtered_df,
    x="Footfall",
    y="Weekly_Sales",
    ax=ax2
)

ax2.set_title("Footfall vs Weekly Sales")

st.pyplot(fig2)

# TEMPERATURE VS SALES
st.subheader("🌡 Temperature vs Sales")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    data=filtered_df,
    x="Temperature",
    y="Weekly_Sales",
    ax=ax3
)

ax3.set_title("Temperature vs Weekly Sales")

st.pyplot(fig3)

# HOLIDAY VS SALES
st.subheader("🎉 Holiday vs Sales")

fig4, ax4 = plt.subplots(figsize=(8,5))

sns.boxplot(
    data=filtered_df,
    x="Holiday_Flag",
    y="Weekly_Sales",
    ax=ax4
)

ax4.set_title("Holiday Impact on Sales")

st.pyplot(fig4)

# STORE COMPARISON
st.subheader("🏪 Store-wise Comparison")

store_sales = df.groupby("Store")["Weekly_Sales"].mean()

fig5, ax5 = plt.subplots(figsize=(12,5))

store_sales.plot(kind='bar', ax=ax5)

ax5.set_ylabel("Average Weekly Sales")
ax5.set_title("Store-wise Average Weekly Sales")

st.pyplot(fig5)

# HEATMAP
st.subheader("🔥 Correlation Heatmap")

fig6, ax6 = plt.subplots(figsize=(10,6))

corr = filtered_df.select_dtypes(include=['number']).corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax6
)

st.pyplot(fig6)

# ML PREDICTION SECTION
st.subheader("🤖 ML Prediction")

input_footfall = st.slider(
    "Select Footfall",
    int(df["Footfall"].min()),
    int(df["Footfall"].max()),
    5000
)

holiday_input = st.selectbox(
    "Holiday Week?",
    [0, 1]
)

# PREDICTION
prediction = model.predict([[
    input_footfall,
    holiday_input
]])

st.success(f"Predicted Weekly Sales: ₹ {prediction[0]:,.2f}")

# EXPLAINABLE AI
st.subheader("🧠 Explainable AI (SHAP)")

X = df[[
    "Footfall",
    "Holiday_Flag"
]]

explainer = shap.Explainer(model, X)

sample_data = pd.DataFrame([[
    input_footfall,
    holiday_input
]], columns=[
    "Footfall",
    "Holiday_Flag"
])

shap_values = explainer(sample_data)

st.write("### Feature Impact on Prediction")

fig_shap, ax = plt.subplots()

shap.plots.waterfall(
    shap_values[0],
    show=False
)

st.pyplot(fig_shap)

# INSIGHTS SECTION
st.subheader("📌 Insights")

st.markdown("""
- Higher footfall generally increases weekly sales.
- Holiday weeks often show higher sales performance.
- Some stores perform significantly better than others.
- Temperature has moderate impact on sales.
- Machine Learning model predicts sales using footfall and holiday information.
- SHAP Explainable AI shows which features influence prediction the most.
""")