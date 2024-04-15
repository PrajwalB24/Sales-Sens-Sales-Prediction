import streamlit as st
import numpy as np
import datetime as dt
import joblib
import matplotlib.pyplot as plt


current_year = dt.datetime.today().year

def predict_sales(p1, p2, p3, p4, p5):
    model = joblib.load('bigmart_sales_pred_model')
    result = model.predict(np.array([[p1, p2, p3, p4, p5]]))
    return result


st.title("SALE SENS : SALES PREDICTION")

st.sidebar.title("Input Parameters")

p1 = st.sidebar.number_input("Item_MRP")
p2 = st.sidebar.selectbox("Outlet_Identifier", ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
p3 = st.sidebar.selectbox("Outlet_Size", ['High', 'Medium', 'Small'])
p4 = st.sidebar.selectbox("Outlet_Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
p5 = st.sidebar.selectbox("Outlet_Establishment_Year", list(range(1950, current_year + 1)))

if st.sidebar.button("Predict"):
    p2 = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'].index(p2)
    p3 = ['High', 'Medium', 'Small'].index(p3)
    p4 = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'].index(p4)
    p5 = current_year - p5

    result = predict_sales(p1, p2, p3, p4, p5)
    lower_bound = result - 714.42
    upper_bound = result + 714.42

    st.markdown(f"Predicted Value: **${result[0]:.2f}**")

    st.markdown(f"Lower Bound Value: **${lower_bound[0]:.2f}**")
    st.markdown(f"Upper Bound Value: **${upper_bound[0]:.2f}**")
    #st.markdown(f"Sales Value is between **${float(lower_bound):.2f}  and   ${float(upper_bound):.2f}**")

    #st.markdown(f"Sales Value is between **${float(lower_bound):.2f} and ${float(upper_bound):.2f}**")

    #st.write(f"Predicted Value : ", result)
    #st.write(f"Sales Value is between {lower_bound} and {upper_bound}")

    x_values = ['Predicted', 'Lower Bound', 'Upper Bound']
    y_values = [result[0], lower_bound[0], upper_bound[0]]
    plt.bar(x_values, y_values, color=['blue', 'green', 'red'])
    plt.xlabel('Sales')
    plt.ylabel('Value')
    plt.title('Sales Prediction')
    st.pyplot(plt)
