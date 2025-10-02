import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import df, df_scaled  # import cả raw và scaled

sns.set(style="whitegrid")

# -----------------------------
# Mapping columns -> Vietnamese labels
# -----------------------------
col_labels = {
    "math_score": "Toán",
    "history_score": "Lịch sử",
    "physics_score": "Vật lý",
    "chemistry_score": "Hóa học",
    "biology_score": "Sinh học",
    "english_score": "Tiếng Anh",
    "geography_score": "Địa lý",
    "avg_science_score": "Trung bình Khoa học",
    "avg_humanities_score": "Trung bình Nhân văn",
    "absence_days": "Số ngày vắng",
    "weekly_self_study_hours": "Giờ tự học/tuần",
    "part_time_job": "Làm thêm (0=Không, 1=Có)",
    "extracurricular_activities": "Hoạt động ngoại khóa (0=Không, 1=Có)",
    "gender": "Giới tính (0=Nam, 1=Nữ)"
}

subject_cols = [
    "math_score","history_score","physics_score",
    "chemistry_score","biology_score","english_score","geography_score"
]

# -----------------------------
# Hàm chọn dữ liệu
# -----------------------------
def choose_data():
    global processed_data
    select_data = input("Chọn dữ liệu hiển thị (raw / scaled): ").strip().lower()
    if select_data == "raw":
        processed_data = df.copy()
        print("Đang dùng dữ liệu RAW (điểm thực tế).")
    elif select_data == "scaled":
        processed_data = df_scaled.copy()
        print("Đang dùng dữ liệu SCALED (chuẩn hóa).")
    else:
        print("Lựa chọn không hợp lệ, mặc định dùng RAW.")
        processed_data = df.copy()
    processed_data["avg_score"] = processed_data[subject_cols].mean(axis=1)

# -----------------------------
# Các hàm thống kê / vẽ biểu đồ
# -----------------------------
def basic_statistics():
    print("=== Thống kê cơ bản ===")
    print(processed_data[subject_cols].describe().rename(columns=col_labels))
    print("\n=== Điểm trung bình mỗi môn ===")
    print(processed_data[subject_cols].mean().rename(index=col_labels))
    print("\n=== Điểm thấp nhất mỗi môn ===")
    print(processed_data[subject_cols].min().rename(index=col_labels))
    print("\n=== Điểm cao nhất mỗi môn ===")
    print(processed_data[subject_cols].max().rename(index=col_labels))

def plot_score_distribution():
    plt.figure(figsize=(12,5))
    sns.boxplot(data=processed_data[subject_cols])
    plt.title("Phân phối điểm từng môn")
    plt.xticks(ticks=range(len(subject_cols)), labels=[col_labels[c] for c in subject_cols])
    plt.show()

def plot_factors():
    plt.figure(figsize=(8,5))
    sns.boxplot(x="part_time_job", y="avg_score", data=processed_data)
    plt.title("Điểm trung bình theo làm thêm")
    plt.xlabel(col_labels["part_time_job"])
    plt.ylabel("Điểm trung bình")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.scatterplot(x="weekly_self_study_hours", y="avg_score", data=processed_data)
    plt.title("Điểm trung bình vs Giờ tự học/tuần")
    plt.xlabel(col_labels["weekly_self_study_hours"])
    plt.ylabel("Điểm trung bình")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.boxplot(x="extracurricular_activities", y="avg_score", data=processed_data)
    plt.title("Điểm trung bình theo hoạt động ngoại khóa")
    plt.xlabel(col_labels["extracurricular_activities"])
    plt.ylabel("Điểm trung bình")
    plt.show()

def plot_gender_trend():
    gender_avg = processed_data.groupby("gender")[subject_cols + ["avg_score"]].mean()
    print("\n=== Điểm trung bình theo giới tính ===")
    print(gender_avg.rename(index={0:"Nam",1:"Nữ"}).rename(columns=col_labels))

    plt.figure(figsize=(10,5))
    gender_avg[subject_cols].T.plot(kind="bar")
    plt.title("Điểm trung bình theo giới tính")
    plt.xlabel("Môn học")
    plt.ylabel("Điểm trung bình")
    plt.xticks(ticks=range(len(subject_cols)), labels=[col_labels[c] for c in subject_cols], rotation=45)
    plt.show()

def plot_correlation():
    plt.figure(figsize=(10,8))
    sns.heatmap(processed_data[subject_cols].corr(), annot=True, cmap="coolwarm",
                xticklabels=[col_labels[c] for c in subject_cols],
                yticklabels=[col_labels[c] for c in subject_cols])
    plt.title("Ma trận tương quan giữa các môn học")
    plt.show()

def plot_low_performers():
    low_performers = processed_data[processed_data["avg_score"] < processed_data["avg_score"].mean() - processed_data["avg_score"].std()]
    print("\n=== Học sinh cần hỗ trợ ===")
    print(low_performers[subject_cols + ["avg_score"]].rename(columns=col_labels))

    plt.figure(figsize=(8,5))
    sns.histplot(processed_data["avg_score"], bins=10, kde=True)
    plt.title("Phân bố điểm trung bình")
    plt.xlabel("Điểm trung bình")
    plt.ylabel("Số học sinh")
    plt.show()

# -----------------------------
# Main menu select option
# -----------------------------
menu = {
    "1": ("Chọn dữ liệu hiển thị (raw / scaled)", choose_data),
    "2": ("Thống kê cơ bản", basic_statistics),
    "3": ("Phân phối điểm từng môn", plot_score_distribution),
    "4": ("Yếu tố ảnh hưởng đến học tập", plot_factors),
    "5": ("Xu hướng theo giới tính", plot_gender_trend),
    "6": ("Ma trận tương quan giữa các môn", plot_correlation),
    "7": ("Học sinh cần hỗ trợ", plot_low_performers)
}

# -----------------------------
# Vòng lặp menu
# -----------------------------
while True:
    print("\n=== Chọn thống kê / biểu đồ muốn xem ===")
    for key, (name, _) in menu.items():
        print(f"{key}. {name}")
    print("0. Thoát")

    choice = input("Nhập lựa chọn: ").strip()
    if choice == "0":
        break
    elif choice in menu:
        menu[choice][1]()  # gọi hàm tương ứng
    else:
        print("Lựa chọn không hợp lệ, thử lại.")
