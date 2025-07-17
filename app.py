# Streamlit app สำหรับทำนายสายพันธุ์เพนกวิน
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib
import streamlit as st

model = load_model("penguin_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

num_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
cat_cols = ["island", "sex"]

st.title("Penguin Species Prediction")

st.write("กรอกข้อมูลลักษณะของเพนกวิน หรืออัปโหลดไฟล์ CSV เพื่อทำนายสายพันธุ์")

tab1, tab2 = st.tabs(["กรอกข้อมูลทีละตัว", "อัปโหลดไฟล์ CSV"])

with tab1:
    island = st.selectbox("Island", options=encoder.categories_[0])
    culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0, step=0.1)
    culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0, step=0.1)
    flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0, step=0.1)
    body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, step=0.1)
    sex = st.selectbox("Sex", options=encoder.categories_[1])

    if st.button("Predict (Single)"):
        input_single = pd.DataFrame({
            "culmen_length_mm": [culmen_length_mm],
            "culmen_depth_mm": [culmen_depth_mm],
            "flipper_length_mm": [flipper_length_mm],
            "body_mass_g": [body_mass_g],
            "island": [island],
            "sex": [sex]
        })
        X_num_new = scaler.transform(input_single[num_cols])
        X_cat_new = encoder.transform(input_single[cat_cols]).toarray()
        X_new_all = np.hstack([X_num_new, X_cat_new])
        y_pred_prob = model.predict(X_new_all)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        y_pred_label = label_encoder.inverse_transform(y_pred_class)
        st.success(f"Predicted Species: {y_pred_label[0]}")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นครบไหม
        required_cols = num_cols + cat_cols
        if all(col in df_upload.columns for col in required_cols):
            # กรองเฉพาะแถวที่ไม่มี missing ในคอลัมน์ที่จำเป็น
            df_valid = df_upload.dropna(subset=required_cols)
            if not df_valid.empty:
                X_num_upload = scaler.transform(df_valid[num_cols])
                X_cat_upload = encoder.transform(df_valid[cat_cols]).toarray()
                X_upload_all = np.hstack([X_num_upload, X_cat_upload])
                y_pred_prob = model.predict(X_upload_all)
                y_pred_class = np.argmax(y_pred_prob, axis=1)
                y_pred_label = label_encoder.inverse_transform(y_pred_class)
                df_valid["predicted_species"] = y_pred_label
                # รวมผลลัพธ์กลับกับแถวที่ขาดข้อมูล
                df_result = df_upload.copy()
                df_result.loc[df_valid.index, "predicted_species"] = df_valid["predicted_species"]
                st.dataframe(df_result)
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download prediction CSV",
                    data=csv,
                    file_name="penguins_prediction.csv",
                    mime="text/csv"
                )
            else:
                st.warning("ไม่มีข้อมูลที่ครบถ้วนสำหรับทำนาย")
        else:
            st.error(f"ไฟล์ต้องมีคอลัมน์: {required_cols}")
