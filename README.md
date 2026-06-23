# 🎗️ Breast Cancer Classification

Dự án phân loại ung thư vú sử dụng **Logistic Regression**, xây dựng trên bộ dữ liệu `sklearn.datasets.load_breast_cancer`. Mô hình dự đoán khối u là **Benign (lành tính)** hay **Malignant (ác tính)** dựa trên 30 đặc trưng lâm sàng.

---

## 📁 Cấu trúc dự án

```
breast-cancer-project/
├── breast-cancer-project.ipynb   # Notebook chính
├── model.pkl                     # Mô hình đã huấn luyện
└── scaler.pkl                    # StandardScaler đã fit
```

---

## 📊 Dataset

- **Nguồn:** `sklearn.datasets.load_breast_cancer`
- **Số mẫu:** 569
- **Số đặc trưng:** 30 (bán kính, texture, chu vi, diện tích, độ mịn, v.v.)
- **Nhãn:**
  - `0` — Malignant (ác tính)
  - `1` — Benign (lành tính)

---

## ⚙️ Pipeline xử lý

### 1. Data Preprocessing
- Kiểm tra thông tin tổng quan (`df.info()`)
- Kiểm tra missing values và duplicate rows
- Thống kê mô tả và vẽ boxplot để phát hiện outlier
- Phân tích tương quan (correlation heatmap)
- Visualize phân phối nhãn (pie chart)

### 2. Feature Engineering
- Tách `X` (30 đặc trưng) và `y` (nhãn target)
- Chia tập train/test: **80/20** với `random_state=42`
- Chuẩn hóa dữ liệu bằng **StandardScaler**

### 3. Huấn luyện mô hình
- Thuật toán: **Logistic Regression** (`max_iter=10000`)
- Tuning hyperparameter `C` bằng **GridSearchCV** (5-fold cross-validation)
  - Các giá trị thử nghiệm: `[0.01, 0.1, 1, 10]`

### 4. Đánh giá mô hình
- **Classification Report** (Precision, Recall, F1-score)
- **ROC-AUC Score** và biểu đồ ROC Curve

### 5. Giải thích mô hình
- **SHAP LinearExplainer** — summary plot thể hiện tầm quan trọng của từng đặc trưng

---

## 🚀 Cài đặt & Chạy

### Yêu cầu

```bash
pip install scikit-learn pandas numpy matplotlib seaborn shap joblib
```

### Chạy notebook

```bash
jupyter notebook breast-cancer-project.ipynb
```

---

## 🔮 Dự đoán bệnh nhân mới

Sau khi huấn luyện, mô hình và scaler được lưu thành file `.pkl`. Dùng hàm `predict_patient()` để dự đoán:

```python
import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_patient(features):
    """
    features: list gồm 30 giá trị tương ứng với 30 đặc trưng
    """
    features_df = pd.DataFrame([features], columns=scaler.feature_names_in_)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    result = "Benign" if prediction == 1 else "Malignant"
    return {
        "prediction": result,
        "probability_class_1": round(float(probability), 4)
    }

# Ví dụ
sample = [
    14.5, 20.3, 95.2, 600.1, 0.10,
    0.15, 0.12, 0.08, 0.18, 0.06,
    0.5,  1.2,  3.1,  25.0, 0.004,
    0.02, 0.03, 0.01, 0.02, 0.003,
    16.5, 25.3, 105.2, 750.1, 0.14,
    0.25, 0.30, 0.15,  0.25, 0.09
]

print(predict_patient(sample))
# Output: {'prediction': 'Benign', 'probability_class_1': 0.xxxx}
```

---

## 🛠️ Công nghệ sử dụng

| Thư viện | Mục đích |
|---|---|
| `scikit-learn` | Mô hình ML, preprocessing, đánh giá |
| `pandas` / `numpy` | Xử lý dữ liệu |
| `matplotlib` / `seaborn` | Trực quan hóa |
| `shap` | Giải thích mô hình |
| `joblib` | Lưu/load model |
