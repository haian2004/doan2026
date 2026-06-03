import os
from pathlib import Path
import urllib.request
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

RAW_BASE = "https://raw.githubusercontent.com/haian2004/doan2026/main"

# 1. CẤU HÌNH TRANG WEB (Chuyển layout sang 'wide' để xem biểu đồ cho đẹp)
st.set_page_config(
    page_title="Hệ Thống Cảnh Báo Y Tế AI", page_icon="🏥", layout="wide"
)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        raise RuntimeError(f"Không tải được file từ {url}: {exc}") from exc


# 2. TẢI MÔ HÌNH VÀ SCALER (Sử dụng caching để load nhanh)
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "best_model_random_forest.pkl"
    scaler_path = BASE_DIR / "best_scaler.pkl"

    if not model_path.exists():
        download_file(
            f"{RAW_BASE}/best_model_random_forest.pkl",
            model_path,
        )

    if not scaler_path.exists():
        download_file(
            f"{RAW_BASE}/best_scaler.pkl",
            scaler_path,
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# Lấy danh sách cột chuẩn (lazy load - chỉ tính khi cần)
@st.cache_data
def get_expected_cols():
    csv_path = BASE_DIR / "vietnam_health_cleaned_ml.csv"
    if not csv_path.exists():
        download_file(
            f"{RAW_BASE}/vietnam_health_cleaned_ml.csv",
            csv_path,
        )

    df_template = pd.read_csv(csv_path, nrows=1)
    return df_template.drop("Target_Risk", axis=1).columns.tolist()

# ==========================================
# KHU VỰC GIAO DIỆN (UI) - TẠO TABS
# ==========================================
st.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100
)  # Icon y tế
st.title("🏥 Hệ Thống Dự Đoán Nguy Cơ Không Chủ Động Đi Khám Sức Khỏe")
st.write("✅ **App loaded!** Chọn tab để bắt đầu")
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

    with st.form("prediction_form"):
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
            st.metric("Chỉ số BMI", f"{bmi:.2f}")

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
            expected_cols = get_expected_cols()
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

        st.markdown("---")
        submitted = st.form_submit_button("🚀 PHÂN TÍCH VÀ DỰ ĐOÁN NGUY CƠ")

    result_container = st.container()
    if submitted:
        with result_container:
            with st.spinner("AI đang phân tích dữ liệu hành vi..."):
                # Load model chỉ khi bấm nút
                model, scaler = load_model()
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
    else:
        result_container.info("Nhấn nút dự đoán để xem kết quả.")

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
        # Dùng ảnh local 11.jpg cho biểu đồ 1
        chart1_path = BASE_DIR / "11.jpg"
        if chart1_path.exists():
            st.image(chart1_path, width=580)
        else:
            st.warning(
                "Hình ảnh 11.jpg chưa có trong thư mục dự án. Vui lòng đặt file vào cùng thư mục với app."
            )
            st.image(
                "https://via.placeholder.com/700x400.png?text=Bi%E1%BB%83u+%C4%91%C3%B2+AfterIT+missing",
                width=580,
            )

    with chart_col2:
        st.subheader("2. Nghịch lý của thế hệ 'Bác sĩ Google'")
        st.markdown(
            "Những người hay tự tra cứu bệnh (Selfstudy) tuy rành công nghệ nhưng lại hoài nghi App y tế nhất. Tính năng dự đoán AI này chính là chìa khóa để thuyết phục họ."
        )
        # Dùng ảnh local 22.jpg cho biểu đồ 2
        chart2_path = BASE_DIR / "22.jpg"
        if chart2_path.exists():
            st.image(chart2_path, width=580)
        else:
            st.warning(
                "Hình ảnh 22.jpg chưa có trong thư mục dự án. Vui lòng đặt file vào cùng thư mục với app."
            )
            st.image(
                "https://via.placeholder.com/700x400.png?text=Bi%E1%BB%83u+%C4%91%C3%B2+UseIT+missing",
                width=580,
            )
