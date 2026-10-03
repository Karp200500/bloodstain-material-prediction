
import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดลเดิม
saved = joblib.load("blood_material_model.pkl")

model = saved["model"]
features = saved["features"]

st.title("Bloodstain Material Prediction")
st.subheader("Random Forest Classification")

st.write("กรอกค่าคุณลักษณะคราบเลือดจาก ImageJ")

values = {}

for feature in features:
    values[feature] = st.number_input(
        f"{feature}",
        value=0.0,
        format="%.6f"
    )

if st.button("ทำนายชนิดวัสดุ"):

    new_data = pd.DataFrame(
        [values],
        columns=features
    )

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    st.success(f"วัสดุที่ทำนาย: {prediction}")

    st.write("คะแนนจากโมเดลแต่ละวัสดุ")

    result = pd.DataFrame({
        "วัสดุ": model.classes_,
        "คะแนน (%)": probabilities * 100
    })

    st.dataframe(result, use_container_width=True)
