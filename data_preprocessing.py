import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load dữ liệu gốc
# -----------------------------
df_raw = pd.read_csv("student-scores.csv")  # original raw data
df = df_raw.copy()  # copy để xử lý

# -----------------------------
# Xử lý dữ liệu
# -----------------------------
# Drop thông tin cá nhân
df = df.drop(columns=["id", "first_name", "last_name", "email"]) 

# Chuyển boolean sang int
df["extracurricular_activities"] = df["extracurricular_activities"].astype(int)
df["part_time_job"] = df["part_time_job"].astype(int)

# Encode gender
df["gender"] = df["gender"].map({"male": 0, "female": 1})

# One-hot encode career aspirations
df = pd.get_dummies(df, columns=["career_aspiration"], prefix="career")

# -----------------------------
# Xử lý outliers (IQR)
# -----------------------------
def handle_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(series, lower, upper)

numeric_cols = [
    "absence_days", "weekly_self_study_hours",
    "math_score","history_score","physics_score",
    "chemistry_score","biology_score","english_score","geography_score"
]

for col in numeric_cols:
    df[col] = handle_outliers(df[col])

# Tính điểm trung bình khoa học và nhân văn
df["avg_science_score"] = df[["physics_score","chemistry_score","biology_score"]].mean(axis=1)
df["avg_humanities_score"] = df[["history_score","geography_score","english_score"]].mean(axis=1)

# -----------------------------
# Chuẩn hóa dữ liệu
# -----------------------------
df_scaled = df.copy()
scaler = StandardScaler()
df_scaled[numeric_cols + ["avg_science_score","avg_humanities_score"]] = scaler.fit_transform(
    df_scaled[numeric_cols + ["avg_science_score","avg_humanities_score"]]
)

# -----------------------------
# Chọn dữ liệu (raw hoặc scaled)
# -----------------------------
select_data = "raw"  # change to "raw" or "scaled"

if select_data == "raw":
    processed_data = df.copy()
elif select_data == "scaled":
    processed_data = df_scaled.copy()
else:
    raise ValueError("Please select 'raw' or 'scaled' for select_data")

# -----------------------------
# Làm tròn tất cả số float trong DataFrame (1 chữ số thập phân)
# -----------------------------
processed_data = processed_data.round(1)

# Tùy chọn hiển thị các số float dạng thập phân
# pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# processed_data sẵn sàng sử dụng cho EDA
