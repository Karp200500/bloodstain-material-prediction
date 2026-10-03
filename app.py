
import streamlit as st
import pandas as pd
import joblib

# ==============================
# ตั้งค่าหน้าเว็บไซต์
# ==============================

st.set_page_config(
    page_title="Bloodstain Material Prediction",
    page_icon="🩸",
    layout="centered"
)

# ==============================
# ตกแต่งเว็บไซต์
# ==============================

st.markdown("""
<style>
.stApp {
    background-color: #0D1B2A;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.main-title {
    text-align: center;
    color: #E0E1DD;
    font-size: 32px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #A9BCD0;
    font-size: 16px;
}

.result-box {
    background-color: #1B263B;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #415A77;
    margin-top: 20px;
}

.result-name {
    color: #70E000;
    font-size: 32px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #415A77;
    color: white;
    border-radius: 10px;
    border: none;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #778DA9;
    color: white;
}

hr {
    border-color: #415A77;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# โหลดโมเดลเดิม
# ==============================

saved = joblib.load("blood_material_model.pkl")

model = saved["model"]
features = saved["features"]

# ==============================
# ส่วนหัวเว็บไซต์
# ==============================

st.markdown(
    '<div class="main-title">🩸 Bloodstain Material Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">ระบบทำนายชนิดวัสดุจากลักษณะคราบเลือดด้วย Random Forest</div>',
    unsafe_allow_html=True
)

st.markdown("---")

st.info(
    "กรอกค่าคุณลักษณะของคราบเลือดที่ได้จาก ImageJ "
    "เพื่อให้โมเดลทำนายชนิดวัสดุ"
)

# ==============================
# กรอกข้อมูล
# ==============================

st.subheader("ข้อมูลคุณลักษณะของคราบเลือด")

labels = {
    "Area": "Area (พื้นที่)",
    "Perim.": "Perimeter (เส้นรอบรูป)",
    "Major": "Major Axis",
    "Minor": "Minor Axis",
    "Angle": "Angle (องศา)",
    "Circ.": "Circularity",
    "AR": "Aspect Ratio",
    "Round": "Roundness",
    "Solidity": "Solidity"
}

values = {}

for i in range(0, len(features), 2):
    cols = st.columns(2)

    for j, col in enumerate(cols):
        index = i + j

        if index < len(features):
            feature = features[index]

            with col:
                values[feature] = st.number_input(
                    labels.get(feature, feature),
                    value=0.0,
                    format="%.6f",
                    key=feature
                )

st.markdown("---")

# ==============================
# ทำนายวัสดุ
# ==============================

if st.button("🔍 ทำนายชนิดวัสดุ"):

    new_data = pd.DataFrame(
        [values],
        columns=features
    )

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    st.markdown(
        f"""
        <div class="result-box">
            <p>ผลการทำนายวัสดุ</p>
            <div class="result-name">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("คะแนนจากโมเดลแต่ละวัสดุ")

    result = pd.DataFrame({
        "วัสดุ": model.classes_,
        "คะแนน (%)": probabilities * 100
    })

    result = result.sort_values(
        "คะแนน (%)",
        ascending=False
    )

    st.bar_chart(
        result.set_index("วัสดุ"),
        horizontal=True
    )

    st.dataframe(
        result.style.format({"คะแนน (%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "หมายเหตุ: คะแนนเป็นสัดส่วนการทำนายของโมเดล "
        "ไม่ใช่ค่าความแม่นยำของโมเดลหรือความน่าจะเป็นที่ผ่านการปรับเทียบ"
    )

# ==============================
# ส่วนท้ายเว็บไซต์
# ==============================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#A9BCD0;">
        งานวิจัยการจำแนกชนิดวัสดุจากลักษณะคราบเลือด<br>
        Machine Learning | Random Forest
    </div>
    """,
    unsafe_allow_html=True
)

