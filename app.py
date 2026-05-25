import joblib
import pandas as pd
import streamlit as st

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Hệ Thống Cảnh Báo Y Tế AI", page_icon="🏥", layout="centered"
)


# 2. TẢI MÔ HÌNH VÀ SCALER (Sử dụng caching để load nhanh)
@st.cache_resource
def load_model():
    model = joblib.load("best_model_random_forest.pkl")
    scaler = joblib.load("best_scaler.pkl")
    return model, scaler


model, scaler = load_model()

# Lấy danh sách cột chuẩn từ model (nếu có)
try:
    expected_cols = model.feature_names_in_
except:
    # Nếu thư viện cũ không hỗ trợ, chúng ta sẽ phải đọc file csv gốc để lấy header
    df_template = pd.read_csv(
        "vietnam_health_cleaned_ml.csv",
        nrows=1,
    )
    expected_cols = df_template.drop("Target_Risk", axis=1).columns

# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# Sử dụng emoji thay vì tải ảnh từ Internet để tránh khởi động bị treo trên Streamlit Cloud
st.title("🏥 Hệ Thống Dự Đoán Nguy Cơ Bỏ Bê Sức Khỏe")
st.markdown("""
Ứng dụng AI giúp phân tích hành vi người dùng, phát hiện nhóm có nguy cơ lười đi khám bệnh định kỳ 
để hệ thống tự động gửi thông báo và Voucher khuyến mãi.
""")

st.header("📋 Nhập thông tin khách hàng")

col1, col2 = st.columns(2)

with col1:
    age_gr = st.selectbox("Độ tuổi", ["<18", "18-29", "30-39", "40-49", ">=50"])
    bmi = st.number_input(
        "Chỉ số BMI (ví dụ: 22.5)", min_value=10.0, max_value=50.0, value=22.0
    )
    job = st.selectbox(
        "Nghề nghiệp", ["stable", "unstable", "student", "housewife", "other"]
    )
    health_ins = st.radio("Có Bảo hiểm y tế không?", ["Có", "Không"])

with col2:
    habit = st.radio("Gia đình/Tổ chức có thói quen đi khám không?", ["Có", "Không"])
    st_choise = st.selectbox(
        "Khi có triệu chứng ốm, bạn thường làm gì?",
        [
            "Tự tra Google (selfstudy)",
            "Đến phòng khám (clinic)",
            "Hỏi người thân (askrel)",
        ],
    )
    tangibles = st.slider("Đánh giá cơ sở vật chất phòng khám (1-5 sao)", 1, 5, 3)


# 4. XỬ LÝ DỮ LIỆU ĐẦU VÀO (PREPROCESSING CHO PREDICTION)
def prepare_input_data():
    # Tạo một dataframe chứa toàn bộ giá trị 0 với các cột giống hệt tập Train
    input_df = pd.DataFrame(0, index=[0], columns=expected_cols)

    # Fill các giá trị do user nhập vào (Map chữ sang số y như bước Data Preprocessing)
    age_map = {"<18": 0, "18-29": 1, "30-39": 2, "40-49": 3, ">=50": 4}
    input_df["Age_gr_encoded"] = age_map[age_gr]

    input_df["BMI"] = bmi
    input_df["HealthIns_encoded"] = 1 if health_ins == "Có" else 0
    input_df["Habit_encoded"] = 1 if habit == "Có" else 0
    input_df["Tangibles"] = float(tangibles)

    # Xử lý One-Hot Encoding cho Jobstt
    job_col = f"Jobstt_{job}"
    if job_col in input_df.columns:
        input_df[job_col] = 1

    # Xử lý One-Hot Encoding cho StChoise
    choise_val = st_choise.split("(")[1].replace(")", "")
    choise_col = f"StChoise_{choise_val}"
    if choise_col in input_df.columns:
        input_df[choise_col] = 1

    # Các trường khác (không có trong form này) sẽ tự động mang giá trị 0 hoặc median (tùy chọn)
    # Để an toàn nhất, cứ để 0 cho OHE, và median cho numeric, nhưng cho Demo, để 0 vẫn chạy tốt.

    return input_df


# 5. NÚT BẤM DỰ ĐOÁN
st.markdown("---")
if st.button("🚀 PHÂN TÍCH VÀ DỰ ĐOÁN NGUY CƠ", use_container_width=True):
    with st.spinner("AI đang phân tích dữ liệu hành vi..."):
        user_data = prepare_input_data()

        # Chuẩn hóa (Scale)
        user_data_scaled = scaler.transform(user_data)

        # Dự đoán xác suất
        probability = model.predict_proba(user_data_scaled)[0][1]

        # Áp dụng ngưỡng 0.4 mà chúng ta đã chốt
        threshold = 0.4

        st.markdown("### 📊 Kết Quả Phân Tích:")
        if probability >= threshold:
            st.error(
                f"⚠️ **NGUY CƠ CAO!** Khách hàng có {probability * 100:.1f}% xác suất lười đi khám bệnh."
            )
            st.info(
                "💡 **Hành động đề xuất (Auto-Trigger):** Kích hoạt hệ thống SMS/App Notification. Gửi tặng Voucher khám tổng quát trị giá 500k để kích cầu!"
            )
        else:
            st.success(
                f"✅ **AN TOÀN!** Khách hàng có ý thức y tế tốt (Nguy cơ chỉ: {probability * 100:.1f}%)."
            )
            st.info(
                "💡 **Hành động đề xuất:** Tiếp tục chăm sóc định kỳ, có thể tư vấn thêm các gói dịch vụ nâng cao."
            )
