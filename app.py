import joblib
import pandas as pd
import streamlit as st

# 1. CẤU HÌNH TRANG WEB (Chuyển layout sang 'wide' để xem biểu đồ cho đẹp)
st.set_page_config(
    page_title="Hệ Thống Cảnh Báo Y Tế AI", page_icon="🏥", layout="wide"
)


# 2. TẢI MÔ HÌNH VÀ SCALER (Sử dụng caching để load nhanh)
@st.cache_resource
def load_model():
    model = joblib.load(r"D:\DOan\best_model_random_forest.pkl")
    scaler = joblib.load(r"D:\DOan\best_scaler.pkl")
    return model, scaler


model, scaler = load_model()

# Lấy danh sách cột chuẩn từ model (nếu có)
try:
    expected_cols = model.feature_names_in_
except:
    # Nếu thư viện cũ không hỗ trợ, chúng ta sẽ phải đọc file csv gốc để lấy header
    df_template = pd.read_csv(
        r"D:\DOan\vietnam_health_cleaned_ml.csv",
        nrows=1,
    )
    expected_cols = df_template.drop("Target_Risk", axis=1).columns

# ==========================================
# KHU VỰC GIAO DIỆN (UI) - TẠO TABS
# ==========================================
st.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100
)  # Icon y tế
st.title("🏥 Hệ Thống Dự Đoán Nguy Cơ Không Chủ Động Đi Khám Sức Khỏe")
st.markdown("""
Ứng dụng AI giúp phân tích hành vi người dùng, phát hiện nhóm có nguy cơ không chủ động đi khám bệnh định kỳ 
để hệ thống tự động gửi thông báo và Voucher khuyến mãi.
""")

# Chia giao diện làm 2 Tab chính
tab2, tab1 = st.tabs(["📊 Tổng Quan Dữ Liệu (Dashboard)", "🚀 Dự Đoán Nguy Cơ (AI)"])

# ------------------------------------------
# TAB 1: GIAO DIỆN DỰ ĐOÁN
# ------------------------------------------
with tab1:
    st.header("📋 Nhập thông tin khách hàng")

    col1, col2 = st.columns(2)

    with col1:
        age_gr = st.selectbox("Độ tuổi", ["<18", "18-29", "30-39", "40-49", ">=50"])

        # [CẬP NHẬT 1]: NHẬP CHIỀU CAO, CÂN NẶNG VÀ TỰ TÍNH BMI
        col_h, col_w = st.columns(2)
        with col_h:
            height_cm = st.number_input(
                "Chiều cao (cm)", min_value=50.0, max_value=250.0, value=165.0
            )
        with col_w:
            weight_kg = st.number_input(
                "Cân nặng (kg)", min_value=10.0, max_value=200.0, value=60.0
            )

        # Công thức tính BMI: Cân nặng (kg) / [Chiều cao (m)]^2
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.info(f"👉 **Chỉ số BMI của khách hàng:** {bmi:.2f}")

        # [CẬP NHẬT 2]: VIỆT HÓA NGHỀ NGHIỆP
        job_mapping = {
            "Công việc ổn định (Văn phòng, Nhà nước)": "stable",
            "Lao động tự do / Không ổn định": "unstable",
            "Học sinh / Sinh viên": "student",
            "Nội trợ": "housewife",
            "Khác": "other",
        }
        # Hiển thị list tiếng Việt cho người dùng chọn
        job_display = st.selectbox("Nghề nghiệp", list(job_mapping.keys()))
        # Map ngược lại giá trị tiếng Anh để đưa vào Model tính toán
        job = job_mapping[job_display]

        health_ins = st.radio("Có Bảo hiểm y tế không?", ["Có", "Không"])

    with col2:
        habit = st.radio(
            "Gia đình/Tổ chức có thói quen đi khám không?", ["Có", "Không"]
        )
        st_choise = st.selectbox(
            "Khi có triệu chứng ốm, bạn thường làm gì?",
            [
                "Tự tra Google (selfstudy)",
                "Đến phòng khám (clinic)",
                "Hỏi người thân (askrel)",
            ],
        )
        tangibles = st.slider("Mong muốn cơ sở vật chất phòng khám (1-5 sao)", 1, 5, 3)

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
                    f"⚠️ **NGUY CƠ CAO!** Khách hàng có {probability * 100:.1f}% xác suất không chủ động đi khám bệnh."
                )
            else:
                st.success(
                    f"✅ **AN TOÀN!** Khách hàng có ý thức chủ động đi khám sức khỏe (Nguy cơ chỉ: {probability * 100:.1f}%)."
                )

# ------------------------------------------
# TAB 2: GIAO DIỆN BIỂU ĐỒ (DASHBOARD)
# ------------------------------------------
with tab2:
    st.header("📊 Các Phát Hiện Quan Trọng (Data Insights)")
    st.markdown(
        "Dưới đây là các cơ sở dữ liệu và thuật toán đằng sau quyết định của AI."
    )

    # Chia làm 2 cột để biểu đồ hiển thị đẹp hơn
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("1. Khả năng thuyết phục người lười đi khám")
        st.markdown(
            "Ngay cả những người lười nhất (bỏ khám trên 2 năm), nếu có App cảnh báo rủi ro dựa trên dữ liệu cá nhân, **tỷ lệ lớn sẽ đồng ý đi khám ngay**."
        )
        # Ảnh dashboard không có sẵn trong môi trường triển khai nên hiển thị thông tin thay thế.
        st.markdown(
            "**Ảnh minh họa:** Hành động sau khi App cảnh báo (AfterIT) phân theo Tần suất khám."
        )
        st.info(
            "Hiển thị biểu đồ tại đây sau khi triển khai với dữ liệu thực tế hoặc upload file ảnh vào repository."
        )

    with chart_col2:
        st.subheader("2. Nghịch lý của thế hệ 'Bác sĩ Google'")
        st.markdown(
            "Những người hay tự tra cứu bệnh (Selfstudy) tuy rành công nghệ nhưng lại hoài nghi App y tế nhất. Tính năng dự đoán AI này chính là chìa khóa để thuyết phục họ."
        )
        # Ảnh dashboard không có sẵn trong môi trường triển khai nên hiển thị thông tin thay thế.
        st.markdown(
            "**Ảnh minh họa:** Sự sẵn sàng dùng App (UseIT) theo Hành vi đối phó (StChoise)."
        )
        st.info(
            "Hiển thị biểu đồ tại đây sau khi triển khai với dữ liệu thực tế hoặc upload file ảnh vào repository."
        )
