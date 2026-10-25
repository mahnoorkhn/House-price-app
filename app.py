
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to estimate its price.")

with st.form("house_form"):
    st.subheader("House details")

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        sqft_living = st.number_input("Living area (sq ft)", min_value=1, value=1800)
        sqft_lot = st.number_input("Lot area (sq ft)", min_value=1, value=5000)
        floors = st.number_input("Floors", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
        waterfront = st.selectbox("Waterfront", [0, 1])
        view = st.selectbox("View rating", [0, 1, 2, 3, 4])
        condition = st.selectbox("Condition rating", [1, 2, 3, 4, 5])

    with col2:
        sqft_above = st.number_input("Area above ground (sq ft)", min_value=0, value=1500)
        sqft_basement = st.number_input("Basement area (sq ft)", min_value=0, value=300)
        yr_built = st.number_input("Year built", min_value=1800, max_value=2026, value=2000)
        yr_renovated = st.number_input("Year renovated (0 if never)", min_value=0, max_value=2026, value=0)
        sale_year = st.number_input("Sale year", min_value=1900, max_value=2026, value=2014)
        sale_month = st.selectbox("Sale month", list(range(1, 13)), index=5)

    st.subheader("Location")
    street = st.text_input("Street", value="Enter street")
    city = st.text_input("City", value="Seattle")
    statezip = st.text_input("State and ZIP code", value="WA 98101")
    country = st.text_input("Country", value="USA")

    submitted = st.form_submit_button("Predict House Price")

if submitted:
    input_data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "sale_year": sale_year,
        "sale_month": sale_month,
        "street": street,
        "city": city,
        "statezip": statezip,
        "country": country
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated house price: ${prediction:,.2f}")
        st.caption("This is a model estimate, not a guaranteed market price.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
