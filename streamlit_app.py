import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded data
results_df = pd.read_csv("./results.csv").drop('Unnamed: 0', axis=1, errors='ignore')
summary_df = pd.read_csv("./model_summary_new.csv")

# Dynamically identify significant features based on summary_df (coefficients rounded to 2 decimals not equal to 0)
coef_threshold = 0.00
summary_df = summary_df[summary_df["Coefficient"].round(2) != coef_threshold]
significant_features = summary_df["Feature"].tolist()

# Set up Streamlit app
st.set_page_config(page_title="Bank Credit Risk Explorer", layout="wide")
st.title("🏦 Bank Credit Risk & Exposure Insights")

st.markdown("""
Explore how deposit structure and macroeconomic variables influence modeled credit exposure.
Use the filters and visualizations to dive into model behavior and performance.
""")

# Section: Summary Table
st.subheader("📊 Model Performance Summary")
st.dataframe(summary_df)

# Section: Feature Distribution
st.subheader("📈 Feature Distribution (Significant Features Only)")
feature_to_plot = st.selectbox("Choose a Feature to Visualize", options=significant_features)

fig, ax = plt.subplots()
sns.histplot(results_df[feature_to_plot], kde=True, ax=ax)
ax.set_title(f"Distribution of {feature_to_plot}")
st.pyplot(fig)

# Section: Actual vs Predicted
st.subheader("📉 Actual vs Predicted TOTAL Deposits")
if "Actual TOTAL" in results_df.columns and "Predicted TOTAL" in results_df.columns:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=results_df, x="Actual TOTAL", y="Predicted TOTAL", color="steelblue", ax=ax2)
    ax2.plot([results_df["Actual TOTAL"].min(), results_df["Actual TOTAL"].max()],
             [results_df["Actual TOTAL"].min(), results_df["Actual TOTAL"].max()],
             color='gray', linestyle='--')
    ax2.set_title("Actual vs Predicted TOTAL Deposits")
    ax2.set_xlabel("Actual TOTAL")
    ax2.set_ylabel("Predicted TOTAL")
    st.pyplot(fig2)
else:
    st.warning("Required columns 'Actual TOTAL' and 'Predicted TOTAL' not found in results_df.")

# Section: Residual Analysis
st.subheader("📊 Residual Analysis")
if "Actual TOTAL" in results_df.columns and "Predicted TOTAL" in results_df.columns:
    results_df["Residual"] = results_df["Actual TOTAL"] - results_df["Predicted TOTAL"]
    fig3, ax3 = plt.subplots()
    sns.histplot(results_df["Residual"], kde=True, ax=ax3)
    ax3.set_title("Prediction Residuals")
    ax3.set_xlabel("Residual (Actual - Predicted)")
    st.pyplot(fig3)



# Section: Interpretation of Predictions and Residuals
st.subheader("📌 Interpretation of Predictions and Residuals")
st.markdown("""
### 📘 Interpretation Guide
- **Predicted TOTAL Deposits** reflect model-estimated deposit exposure based on macroeconomic and structural bank data.
- **Residuals** represent the difference between actual and predicted values.

#### How to interpret:
- A **small residual** suggests the model accurately captured the deposit behavior.
- A **large positive residual** means the model **underestimated** exposure (risk may be understated).
- A **large negative residual** indicates **overestimation** by the model (could imply conservative assumptions or anomalies).

""")

# Section: Business Problem Insights
st.markdown("---")
st.subheader("📌 Business Problem Insights")

business_problem = st.radio("Choose a business question to explore:", [
    "Non-Performing Loan (NPL) Ratios",
    "Market Share and Credit Risk"
])

if business_problem == "Non-Performing Loan (NPL) Ratios":
    st.markdown("""
    ### 🧮 NPL Ratios Analysis
    Explore how macroeconomic indicators like interest rates, inflation, GDP growth, and household debt relate to predicted deposit values as a proxy for NPL exposure.

    - **Repo Rate and Inflation** may influence repayment capacity.
    - **Household debt to income** reflects borrower stress.
    - **GDP growth** often improves credit performance.

    Below we visualize residuals (model prediction errors) against macro indicators.
    """)
    macro_choices = [col for col in results_df.columns if col not in ["Actual TOTAL", "Predicted TOTAL", "Residual"] and results_df[col].dtype != 'O']
    macro_to_plot = st.selectbox("Choose a macroeconomic variable to compare with Residuals", options=macro_choices)
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=results_df, x=macro_to_plot, y="Residual", ax=ax4, color="darkorange")
    ax4.set_title(f"Residual vs {macro_to_plot}")
    st.pyplot(fig4)

elif business_problem == "Market Share and Credit Risk":
    st.markdown("""
    ### 📊 Market Share and Credit Risk
    We investigate how a bank's deposit volume (market share proxy) relates to model accuracy and potential lending risk.

    - **Large deposits + high residuals** could signal risk from aggressive lending.
    - Review scatterplot to spot outliers.
    """)
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=results_df, x="Predicted TOTAL", y="Residual", ax=ax5, color="green")
    ax5.set_title("Predicted TOTAL vs Residuals")
    ax5.set_xlabel("Predicted TOTAL Deposits")
    ax5.set_ylabel("Residual (Actual - Predicted)")
    st.pyplot(fig5)

st.markdown("---")
st.markdown("💡 **Insight Tip:** Large residuals or skewed distributions may indicate where model refinements or alternative strategies are needed.")