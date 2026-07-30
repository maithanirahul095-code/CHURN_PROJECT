import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.caption(
    "Enter a customer's details to estimate their churn risk and see which factors "
    "are driving that prediction."
)


@st.cache_resource
def load_pipeline():
    return joblib.load("churn_pipeline.joblib")


pipeline = load_pipeline()


def tenure_bucket(t):
    if t <= 12:
        return "0-12"
    elif t <= 24:
        return "13-24"
    elif t <= 48:
        return "25-48"
    elif t <= 60:
        return "49-60"
    else:
        return "61-72"


with st.form("customer_form"):
    st.subheader("Customer profile")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    with col2:
        phone_service = st.selectbox("Phone service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input(
        "Total charges to date ($)", min_value=0.0, value=float(monthly_charges * tenure)
    )

    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    service_flags = [
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
    ]
    num_services = sum(1 for s in service_flags if s == "Yes")

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "tenure_group": tenure_bucket(tenure),
        "avg_monthly_spend": total_charges / (tenure + 1),
        "num_services": num_services,
    }])

    proba = pipeline.predict_proba(customer)[0][1]

    st.subheader("Result")
    risk_label = "🔴 High risk" if proba >= 0.5 else "🟢 Low risk"
    st.metric("Predicted churn probability", f"{proba:.1%}", risk_label)
    st.progress(min(proba, 1.0))

    with st.expander("Why did the model predict this? (SHAP explanation)"):
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        customer_processed = preprocessor.transform(customer)
        customer_dense = customer_processed.toarray() if hasattr(customer_processed, "toarray") else customer_processed

        feature_names = (
            list(preprocessor.transformers_[0][2])
            + list(preprocessor.named_transformers_["cat"].get_feature_names_out(preprocessor.transformers_[1][2]))
        )

        background = np.zeros((1, customer_dense.shape[1]))
        explainer = shap.Explainer(model.predict_proba, background)
        explanation = explainer(customer_dense)

        values = explanation.values[0]
        values = values[:, 1] if values.ndim == 2 else values

        shap_df = pd.DataFrame({"feature": feature_names, "shap_value": values})
        shap_df["abs_val"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("abs_val", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#E63946" if v > 0 else "#2E86AB" for v in shap_df["shap_value"][::-1]]
        ax.barh(shap_df["feature"][::-1], shap_df["shap_value"][::-1], color=colors)
        ax.set_xlabel("Impact on churn probability")
        ax.set_title("Top factors driving this prediction")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)

        st.caption(
            "🔴 Red bars push the prediction toward **churn**. "
            "🔵 Blue bars push it toward **staying**."
        )

    st.divider()
    if proba >= 0.5:
        st.warning(
            "This customer is flagged as high-risk. Consider a retention offer "
            "(e.g. a discount or contract upgrade incentive)."
        )
    else:
        st.success("This customer looks likely to stay — no immediate action needed.")
