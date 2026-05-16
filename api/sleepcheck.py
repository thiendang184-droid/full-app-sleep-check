"""
=============================================================
  DỰ ĐOÁN CHẤT LƯỢNG GIẤC NGỦ BẰNG PERCEPTRON NETWORK
=============================================================
INPUT  : Age, Gender, Sleep Duration, Physical Activity Level,
         Stress Level, BMI Category, Heart Rate, Daily Steps
OUTPUT : Quality of Sleep   (hồi quy → làm tròn)
         Sleep Disorder      (phân loại: None / Insomnia / Sleep Apnea)
         Fatigue Level       (sinh tổng hợp từ data, hồi quy → làm tròn)
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import os
from flask import Flask, request, jsonify

# Khởi tạo Flask App phục vụ Vercel API
app = Flask(__name__)

# ────────────────────────────────────────────────────────────
# 1. ĐỌC & TIỀN XỬ LÝ DỮ LIỆU
# ────────────────────────────────────────────────────────────

# Tự động định vị file CSV dù chạy trên máy hay trên Cloud Vercel
current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
csv_path = os.path.join(current_dir, "Sleep_health_and_lifestyle_dataset.csv")

df = pd.read_csv(csv_path)

# ----- Tạo cột Fatigue Level (tổng hợp) -----
df["Fatigue Level"] = (
    10 - df["Quality of Sleep"] + df["Stress Level"] / 2 - df["Sleep Duration"] / 2
).round().clip(1, 10).astype(int)

# ----- Encode Gender & BMI Category -----
le_gender = LabelEncoder()
le_bmi    = LabelEncoder()
le_disorder = LabelEncoder()

df["Gender_enc"]  = le_gender.fit_transform(df["Gender"])        # Male=1, Female=0
df["BMI_enc"]     = le_bmi.fit_transform(df["BMI Category"])     # Normal/Overweight/Obese/Normal Weight
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df["Disorder_enc"] = le_disorder.fit_transform(df["Sleep Disorder"])

# ----- Chọn features & targets -----
FEATURES = [
    "Age", "Gender_enc", "Sleep Duration", "Physical Activity Level",
    "Stress Level", "BMI_enc", "Heart Rate", "Daily Steps"
]

X = df[FEATURES].values

y_quality  = df["Quality of Sleep"].values           # hồi quy [4-9]
y_disorder = df["Disorder_enc"].values               # phân loại 0/1/2
y_fatigue  = df["Fatigue Level"].values              # hồi quy [1-10]

# ----- Chuẩn hoá input -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- Train / Test split -----
(X_train, X_test,
 yq_train, yq_test,
 yd_train, yd_test,
 yf_train, yf_test) = train_test_split(
    X_scaled, y_quality, y_disorder, y_fatigue,
    test_size=0.2, random_state=42
)


# ────────────────────────────────────────────────────────────
# 2. PERCEPTRON NETWORK (numpy thuần)
# ────────────────────────────────────────────────────────────

def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class PerceptronNet:
    def __init__(self, n_in, n_hidden, n_out, task="regression",
                 lr=0.01, n_epochs=500, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in),   (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden),(n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.task     = task       
        self.lr       = lr
        self.n_epochs = n_epochs
        self.losses   = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        if self.task == "classification":
            self.a2 = softmax(self.z2)
        else:
            self.a2 = self.z2          
        return self.a2

    def loss(self, y_pred, y_true):
        if self.task == "classification":
            n = len(y_true)
            log_p = np.log(y_pred[np.arange(n), y_true.astype(int)] + 1e-9)
            return -log_p.mean()
        else:
            return ((y_pred.flatten() - y_true) ** 2).mean()

    def backward(self, X, y_true):
        n = len(y_true)
        if self.task == "classification":
            dz2 = self.a2.copy()
            dz2[np.arange(n), y_true.astype(int)] -= 1
            dz2 /= n
        else:
            dz2 = 2 * (self.a2.flatten() - y_true).reshape(-1, 1) / n

        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, verbose=False, label=""):
        for epoch in range(1, self.n_epochs + 1):
            pred = self.forward(X)
            l = self.loss(pred, y)
            self.losses.append(l)
            self.backward(X, y)
            if verbose and epoch % 100 == 0:
                print(f"  [{label}] Epoch {epoch:4d}/{self.n_epochs} | Loss = {l:.4f}")

    def predict(self, X):
        out = self.forward(X)
        if self.task == "classification":
            return np.argmax(out, axis=1)
        else:
            return out.flatten()


# ────────────────────────────────────────────────────────────
# 3. HUẤN LUYỆN 3 MÔ HÌNH (Chạy ngay khi khởi tạo backend)
# ────────────────────────────────────────────────────────────

N_IN     = X_train.shape[1]   
N_HIDDEN = 16
LR       = 0.01
EPOCHS   = 600

model_quality = PerceptronNet(N_IN, N_HIDDEN, 1, task="regression", lr=LR, n_epochs=EPOCHS)
model_quality.fit(X_train, yq_train, label="Quality")

model_disorder = PerceptronNet(N_IN, N_HIDDEN, 3, task="classification", lr=LR, n_epochs=EPOCHS)
model_disorder.fit(X_train, yd_train, label="Disorder")

model_fatigue = PerceptronNet(N_IN, N_HIDDEN, 1, task="regression", lr=LR, n_epochs=EPOCHS)
model_fatigue.fit(X_train, yf_train, label="Fatigue")


# ────────────────────────────────────────────────────────────
# 5. HÀM DỰ ĐOÁN CHO NGƯỜI DÙNG MỚI
# ────────────────────────────────────────────────────────────

def predict_new(age, gender, sleep_duration, physical_activity,
                stress_level, bmi_category, heart_rate, daily_steps):
    gender_enc = le_gender.transform([gender])[0]
    bmi_enc    = le_bmi.transform([bmi_category])[0]

    x_raw = np.array([[age, gender_enc, sleep_duration, physical_activity,
                        stress_level, bmi_enc, heart_rate, daily_steps]], dtype=float)
    x_scaled = scaler.transform(x_raw)

    q = int(model_quality.predict(x_scaled).round().clip(4, 9)[0])
    d_idx = model_disorder.predict(x_scaled)[0]
    d = le_disorder.inverse_transform([d_idx])[0]
    f = int(model_fatigue.predict(x_scaled).round().clip(1, 10)[0])

    return {
        "Quality of Sleep (1-10)": q,
        "Sleep Disorder"          : d,
        "Fatigue Level (1-10)"   : f,
    }


# ────────────────────────────────────────────────────────────
# XỬ LÝ API ENDPOINT CHO VERCEL WEB INTERFACE
# ────────────────────────────────────────────────────────────

@app.route('/api/sleepcheck', methods=['POST'])
def web_api_predict():
    try:
        vals = request.get_json()
        
        # Parse toàn bộ dữ liệu từ frontend đẩy lên
        age = int(vals.get('age'))
        gender = vals.get('gender')
        sleep_duration = float(vals.get('sleep_duration'))
        physical_activity = int(vals.get('physical_activity'))
        stress_level = int(vals.get('stress_level'))
        bmi_category = vals.get('bmi_category')
        heart_rate = int(vals.get('heart_rate'))
        daily_steps = int(vals.get('daily_steps'))
        
        # Chạy dự đoán qua mạng Perceptron Network thuần
        res = predict_new(age, gender, sleep_duration, physical_activity,
                          stress_level, bmi_category, heart_rate, daily_steps)
        
        # Trích xuất kết quả dự đoán từ mạng AI
        q_out = res["Quality of Sleep (1-10)"]
        d_out = res["Sleep Disorder"]
        f_out = res["Fatigue Level (1-10)"]
        
        # Tạo câu tư vấn tiếng Việt tự động dựa trên kết quả đầu ra của Perceptron
        advice_msg = f"Hệ thống ghi nhận bạn ngủ {sleep_duration}h/đêm với chỉ số căng thẳng ở mức {stress_level}/10. "
        if d_out == "Insomnia":
            advice_msg += "Mạng thần kinh phát hiện bạn có rủi ro thuộc nhóm Mất ngủ. Hãy tối ưu phòng ngủ đủ tối, giảm stress và hạn chế uống caffeine sau 14h chiều."
        elif d_out == "Sleep Apnea":
            advice_msg += "Chỉ số BMI và thể trạng của bạn có dấu hiệu tương quan với nhóm Ngưng thở khi ngủ. Bạn nên thử thay đổi tư thế nằm nghiêng và theo dõi nhịp tim đều đặn."
        else:
            advice_msg += "Chúc mừng! Chỉ số phân tích cho thấy giấc ngủ của bạn đang duy trì ở trạng thái rất tốt. Hãy tiếp tục duy trì số bước chân và rèn luyện thể thao hàng ngày."

        # Đóng gói JSON trả về cho React Frontend nhận diện
        return jsonify({
            "status": "success",
            "data": {
                "quality": q_out,
                "disorder": d_out,
                "fatigue": f_out,
                "advice": advice_msg
            }
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ────────────────────────────────────────────────────────────
# 6. NHẬP INPUT THỦ CÔNG ĐỂ DỰ ĐOÁN TRÊN TERMINAL (LOCAL)
# ────────────────────────────────────────────────────────────

def get_input_int(prompt, lo, hi):
    while True:
        v_str = input(prompt).strip()
        if v_str.lower() == 'q': return 'q'
        try:
            v = int(v_str)
            if lo <= v <= hi: return v
            print(f"  ⚠  Vui lòng nhập số nguyên từ {lo} đến {hi}.")
        except ValueError:
            print("  ⚠  Vui lòng nhập số nguyên hợp lệ.")

def get_input_float(prompt, lo, hi):
    while True:
        v_str = input(prompt).strip()
        if v_str.lower() == 'q': return 'q'
        try:
            v = float(v_str)
            if lo <= v <= hi: return v
            print(f"  ⚠  Vui lòng nhập số từ {lo} đến {hi}.")
        except ValueError:
            print("  ⚠  Vui lòng nhập số thực hợp lệ.")

def get_input_choice(prompt, choices):
    choices_lower = [c.lower() for c in choices]
    while True:
        v = input(prompt).strip()
        if v.lower() == 'q': return 'q'
        if v.lower() in choices_lower:
            return choices[choices_lower.index(v.lower())]
        print(f"  ⚠  Vui lòng chọn một trong: {choices}")

if __name__ == '__main__':
    # Đoạn này đảm bảo khi ông chạy file này offline trên terminal (local cmd), nó vẫn chạy tiếp trình nhập tay
    import sys
    # Nếu chạy script không qua môi trường Vercel API, khởi động terminal loop
    if len(sys.argv) == 1 or sys.argv[0] != '-m':
        BMI_CLASSES     = list(le_bmi.classes_)       
        GENDER_CLASSES  = list(le_gender.classes_)    

        print("\n" + "=" * 55)
        print("  DỰ ĐOÁN CHẤT LƯỢNG GIẤC NGỦ — NHẬP TAY")
        print("=" * 55)

        while True:
            print("\nNhập thông tin cá nhân (gõ 'q' ở bất kỳ bước nào để thoát):\n")
            gender = get_input_choice(f"  Giới tính {GENDER_CLASSES}: ", GENDER_CLASSES)
            if gender == 'q': break
            age = get_input_int("  Tuổi (18-80): ", 18, 80)
            if age == 'q': break
            sleep_duration = get_input_float("  Số giờ ngủ mỗi đêm (4.0 - 12.0): ", 4.0, 12.0)
            if sleep_duration == 'q': break
            physical_activity = get_input_int("  Mức độ hoạt động thể chất (0-100): ", 0, 100)
            if physical_activity == 'q': break
            stress_level = get_input_int("  Mức độ stress (1-10): ", 1, 10)
            if stress_level == 'q': break
            print(f"  Các lựa chọn BMI: {BMI_CLASSES}")
            bmi_category = get_input_choice("  BMI Category: ", BMI_CLASSES)
            if bmi_category == 'q': break
            heart_rate = get_input_int("  Nhịp tim lúc nghỉ (40-120 bpm): ", 40, 120)
            if heart_rate == 'q': break
            daily_steps = get_input_int("  Số bước chân mỗi ngày (0-30000): ", 0, 30000)
            if daily_steps == 'q': break

            result = predict_new(age, gender, sleep_duration, physical_activity,
                                 stress_level, bmi_category, heart_rate, daily_steps)

            print("\n" + "─" * 40)
            print("  KẾT QUẢ DỰ ĐOÁN")
            print("─" * 40)
            stars_q = "⭐" * result["Quality of Sleep (1-10)"]
            stars_f = "🔥" * result["Fatigue Level (1-10)"]
            print(f"  Chất lượng giấc ngủ : {result['Quality of Sleep (1-10)']}/10  {stars_q}")
            print(f"  Rối loạn giấc ngủ   : {result['Sleep Disorder']}")
            print(f"  Mức độ mệt mỏi      : {result['Fatigue Level (1-10)']}/10  {stars_f}")
            print("─" * 40)

            again = input("\n  Dự đoán tiếp? (y/n): ").strip().lower()
            if again != "y":
                break
        print("\n👋 Tạm biệt!")