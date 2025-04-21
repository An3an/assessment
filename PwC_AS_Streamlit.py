import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data and model summary
features_df = pd.read_csv("./cleaned_features.csv")
model_df = pd.read_csv("./model_summary.csv")

# Title and intro
st.set_page_config(page_title="PwC Credit Risk Model", layout="wide")
st.title("📊 PwC BA900 Credit Risk Analysis")
st.write("This dashboard shows how BA900 financial indicators relate to credit risk (NPL ratio).")

# Show input data
st.subheader("📁 Cleaned BA900 Financial Data")
st.dataframe(features_df.head(10))

# Show model summary
st.subheader("🧠 Model Coefficients")
st.dataframe(model_df)

# Plot coefficients
st.subheader("📈 Feature Impact (Coefficient Magnitude)")
fig, ax = plt.subplots()
ax.barh(model_df['Feature'], model_df['Coefficient'])
ax.set_xlabel("Coefficient")
ax.set_title("Linear Regression Coefficients")
st.pyplot(fig)

# Show interpretive text
st.markdown("""
### 🔍 Insights
- The NPL ratio was calculated at **4.16%**, based on total impairments and loans.
- Coefficients are near zero because the target was constant (one value for all).
""")
