"""
=============================================================
  DỰ ĐOÁN CHẤT LƯỢNG GIẤC NGỦ BẰNG PERCEPTRON NETWORK
=============================================================
INPUT  : Age, Gender, Sleep Duration, Physical Activity Level,
         Stress Level, BMI Category, Heart Rate, Daily Steps
OUTPUT : Quality of Sleep   (hồi quy → làm tròn)
         Sleep Disorder      (phân loại: None / Insomnia / Sleep Apnea)
         Fatigue Level       (sinh tổng hợp từ data, hồi quy → làm tròn)

FIXES:
  - Weighted cross-entropy loss để xử lý class imbalance
  - Tăng N_HIDDEN lên 64, EPOCHS lên 1000
  - Rule-based override cho trường hợp cực đoan rõ ràng
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from collections import Counter
import os
from flask import Flask, request, jsonify

# Khởi tạo Flask App phục vụ Vercel API
app = Flask(__name__)

# ────────────────────────────────────────────────────────────
# 1. ĐỌC & TIỀN XỬ LÝ DỮ LIỆU
# ────────────────────────────────────────────────────────────

current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
csv_path = os.path.join(current_dir, "Sleep_health_and_lifestyle_dataset.csv")

df = pd.read_csv(csv_path)

# ----- Tạo cột Fatigue Level (tổng hợp) -----
df["Fatigue Level"] = (
    10 - df["Quality of Sleep"] + df["Stress Level"] / 2 - df["Sleep Duration"] / 2
).round().clip(1, 10).astype(int)

# ----- Encode Gender & BMI Category -----
le_gender   = LabelEncoder()
le_bmi      = LabelEncoder()
le_disorder = LabelEncoder()

df["Gender_enc"]   = le_gender.fit_transform(df["Gender"])
df["BMI_enc"]      = le_bmi.fit_transform(df["BMI Category"])
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df["Disorder_enc"] = le_disorder.fit_transform(df["Sleep Disorder"])

# ----- Chọn features & targets -----
FEATURES = [
    "Age", "Gender_enc", "Sleep Duration", "Physical Activity Level",
    "Stress Level", "BMI_enc", "Heart Rate", "Daily Steps"
]

X = df[FEATURES].values

y_quality  = df["Quality of Sleep"].values
y_disorder = df["Disorder_enc"].values
y_fatigue  = df["Fatigue Level"].values

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
# FIX 1: Tính class weights để xử lý mất cân bằng dữ liệu
# ────────────────────────────────────────────────────────────

counts = Counter(yd_train)
total  = len(yd_train)
n_classes = len(counts)
# class_weights[c] = tổng mẫu / (số class * số mẫu của class c)
class_weights = {
    c: total / (n_classes * cnt)
    for c, cnt in counts.items()
}
print(f"[INFO] Class weights: { {le_disorder.inverse_transform([c])[0]: round(w, 3) for c, w in class_weights.items()} }")


# ────────────────────────────────────────────────────────────
# 2. PERCEPTRON NETWORK (numpy thuần) — đã thêm weighted loss
# ────────────────────────────────────────────────────────────

def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class PerceptronNet:
    def __init__(self, n_in, n_hidden, n_out, task="regression",
                 lr=0.01, n_epochs=500, seed=0, class_weights=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in),    (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.task          = task
        self.lr            = lr
        self.n_epochs      = n_epochs
        self.class_weights = class_weights   # dict {class_idx: weight} hoặc None
        self.losses        = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        if self.task == "classification":
            self.a2 = softmax(self.z2)
        else:
            self.a2 = self.z2
        return self.a2

    # ── FIX: weighted cross-entropy loss ──
    def loss(self, y_pred, y_true):
        if self.task == "classification":
            n = len(y_true)
            log_p = np.log(y_pred[np.arange(n), y_true.astype(int)] + 1e-9)
            if self.class_weights is not None:
                w = np.array([self.class_weights[int(y)] for y in y_true])
                return -(log_p * w).mean()
            return -log_p.mean()
        else:
            return ((y_pred.flatten() - y_true) ** 2).mean()

    def backward(self, X, y_true):
        n = len(y_true)
        if self.task == "classification":
            dz2 = self.a2.copy()
            dz2[np.arange(n), y_true.astype(int)] -= 1
            # Áp dụng sample weights vào gradient
            if self.class_weights is not None:
                w = np.array([self.class_weights[int(y)] for y in y_true]).reshape(-1, 1)
                dz2 = dz2 * w
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
# 3. HUẤN LUYỆN 3 MÔ HÌNH
#    FIX 2: Tăng N_HIDDEN lên 64, EPOCHS lên 1000
# ────────────────────────────────────────────────────────────

N_IN     = X_train.shape[1]
N_HIDDEN = 64      # tăng từ 16 → 64
LR       = 0.01
EPOCHS   = 1000    # tăng từ 600 → 1000

model_quality = PerceptronNet(N_IN, N_HIDDEN, 1, task="regression",
                               lr=LR, n_epochs=EPOCHS)
model_quality.fit(X_train, yq_train, verbose=True, label="Quality")

# FIX: truyền class_weights vào model disorder
model_disorder = PerceptronNet(N_IN, N_HIDDEN, 3, task="classification",
                                lr=LR, n_epochs=EPOCHS,
                                class_weights=class_weights)
model_disorder.fit(X_train, yd_train, verbose=True, label="Disorder")

model_fatigue = PerceptronNet(N_IN, N_HIDDEN, 1, task="regression",
                               lr=LR, n_epochs=EPOCHS)
model_fatigue.fit(X_train, yf_train, verbose=True, label="Fatigue")


# ────────────────────────────────────────────────────────────
# 4. ĐÁNH GIÁ MODEL (in ra terminal khi khởi động)
# ────────────────────────────────────────────────────────────

yq_pred = model_quality.predict(X_test).round().clip(4, 9)
yd_pred = model_disorder.predict(X_test)
yf_pred = model_fatigue.predict(X_test).round().clip(1, 10)

print("\n" + "=" * 50)
print("  ĐÁNH GIÁ MODEL TRÊN TẬP TEST")
print("=" * 50)
print(f"  Quality  MAE : {mean_absolute_error(yq_test, yq_pred):.3f}")
print(f"  Fatigue  MAE : {mean_absolute_error(yf_test, yf_pred):.3f}")
print(f"  Disorder ACC : {accuracy_score(yd_test, yd_pred):.3f}")
print()
print(classification_report(
    yd_test, yd_pred,
    target_names=le_disorder.classes_
))


# ────────────────────────────────────────────────────────────
# 5. HÀM DỰ ĐOÁN CHO NGƯỜI DÙNG MỚI
#    FIX 3: Rule-based override cho trường hợp cực đoan
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

    # ── FIX 3: Rule-based override ──────────────────────────
    # Ngưỡng rõ ràng về mặt y tế → override kết quả model nếu cần

    # Dấu hiệu mạnh của Insomnia
    insomnia_score = 0
    if sleep_duration <= 5.0:       insomnia_score += 2
    if sleep_duration <= 4.0:       insomnia_score += 1   # cực đoan
    if stress_level >= 8:           insomnia_score += 2
    if stress_level >= 9:           insomnia_score += 1   # cực đoan
    if physical_activity < 20:      insomnia_score += 1
    if heart_rate > 90:             insomnia_score += 1

    # Dấu hiệu mạnh của Sleep Apnea
    apnea_score = 0
    if bmi_category in ("Obese", "Overweight"):  apnea_score += 2
    if heart_rate > 85:                          apnea_score += 1
    if sleep_duration <= 5.5 and bmi_category in ("Obese", "Overweight"):
        apnea_score += 2

    # Override chỉ khi model dự đoán "None" nhưng điểm ngưỡng đủ cao
    if d == "None":
        if insomnia_score >= 4 and insomnia_score >= apnea_score:
            d = "Insomnia"
        elif apnea_score >= 4 and apnea_score > insomnia_score:
            d = "Sleep Apnea"
    # ────────────────────────────────────────────────────────

    return {
        "Quality of Sleep (1-10)": q,
        "Sleep Disorder"          : d,
        "Fatigue Level (1-10)"    : f,
    }


# ────────────────────────────────────────────────────────────
# 6. API ENDPOINT CHO VERCEL / FLASK
# ────────────────────────────────────────────────────────────

@app.route('/api/sleepcheck', methods=['POST'])
def web_api_predict():
    try:
        vals = request.get_json()

        age               = int(vals.get('age'))
        gender            = vals.get('gender')
        sleep_duration    = float(vals.get('sleep_duration'))
        physical_activity = int(vals.get('physical_activity'))
        stress_level      = int(vals.get('stress_level'))
        bmi_category      = vals.get('bmi_category')
        heart_rate        = int(vals.get('heart_rate'))
        daily_steps       = int(vals.get('daily_steps'))

        res = predict_new(age, gender, sleep_duration, physical_activity,
                          stress_level, bmi_category, heart_rate, daily_steps)

        q_out = res["Quality of Sleep (1-10)"]
        d_out = res["Sleep Disorder"]
        f_out = res["Fatigue Level (1-10)"]

        return jsonify({
            "status": "success",
            "data": {
                "quality" : q_out,
                "disorder": d_out,
                "fatigue" : f_out,
            }
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ────────────────────────────────────────────────────────────
# 7. NHẬP INPUT THỦ CÔNG TRÊN TERMINAL (LOCAL)
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
    import sys
    if len(sys.argv) == 1 or sys.argv[0] != '-m':
        BMI_CLASSES    = list(le_bmi.classes_)
        GENDER_CLASSES = list(le_gender.classes_)

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