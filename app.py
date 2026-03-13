import streamlit as st
from infer.predict_freight_cost import predict_freight_cost
from infer.flag_invoice import flag_invoice

st.set_page_config(page_title="Ganakah", page_icon="🧮", layout="wide")

st.title("Ganakah (गणकः)")
st.header("Machine Learning for Procurement Intelligence")
st.markdown("""
*Ganakah* is a machine learning-powered analytics tool designed to analyze procurement and inventory data.  
It predicts *freight costs for purchase orders* and *flags potentially risky invoices* using trained models.

The application aims to support better decision-making by identifying unusual patterns and estimating logistics-related costs.
""")

st.divider()

st.sidebar.title("🧮")
st.sidebar.header("Analyze Procurement Data")
selected_model = st.sidebar.radio("Choose to", [
    "Predict Freight Cost",
    "Flag Invoices for Risk"
])
st.sidebar.markdown("""
Use *Ganakah* to analyze procurement data.

- *Predict Freight Cost* estimates logistics costs from purchase values.
- *Flag Invoices for Risk* identifies potentially suspicious invoices.
""")

if selected_model == "Predict Freight Cost":
    st.subheader("Predict Freight Costs")
    st.markdown("""
**Objective:** Predict *Freight Cost* for an *Invoice* based on quantity, and/or value to forecast, budget, and negotiate cost.
""")
    with st.form("predict_freight_cost_form"):
        col_1, col_2 = st.columns(2)
        with col_1:
            quantity = st.number_input("Quantity", min_value=1, value=5000)
        with col_2:
            dollars = st.number_input("Cost ($)", min_value=1.0, value=50000.0)
        submit_button = st.form_submit_button("🔮 Predict")
    if submit_button:
        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }
        predicted = predict_freight_cost(input_data)["PredictedFreight"].iloc[0]
        st.success("🟢 Predicted successfully!")
        st.metric(label="Predicted Freight Cost", value=f"${predicted:,.2f}")
else:
    st.subheader("Flag Invoices for Risk")
    st.markdown("""
**Objective:** Flag an *Invoice* for manual approval based on cost, and/or delay to reduce risk, improve efficiency, and prioritize human intervention.
""")
    with st.form("flag_invoice_form"):
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            invoice_quantity = st.number_input("Quantity", min_value=1, value=5000)
            invoice_dollars = st.number_input("Cost ($)", min_value=1.0, value=50000.0)
        with col_2:
            freight = st.number_input("Freight Cost ($)", min_value=0.0, value=250.0)
        with col_3:
            total_quantity = st.number_input("Total Quantity", min_value=1, value=50000)
            total_dollars = st.number_input("Total Cost ($)", min_value=1.0, value=500000.0)
        submit_button = st.form_submit_button("🚩 Flag")
    if submit_button:
        input_data = {
            "InvoiceQuantity": [invoice_quantity],
            "InvoiceDollars": [invoice_dollars],
            "Freight": [freight],
            "TotalQuantity": [total_quantity],
            "TotalDollars": [total_dollars]
        }
        flagged = flag_invoice(input_data)["Flagged"].iloc[0]
        is_flagged = bool(flagged)
        st.success("🟢 Flagged successfully!")
        if is_flagged:
            st.warning("❌ | This invoice has been flagged as *risky*. Manual approval is recommended.")
        else:
            st.success("✔️ | This invoice has been flagged as *safe*. Auto approval is permitted")