import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv("C:/Users/mehdi/OneDrive/Masaüstü/Student_Performance.csv")

# Kategorik veriyi sayısala çevir
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Özellikler ve hedef değişken
feature_columns = [
    'Hours Studied',
    'Previous Scores',
    'Extracurricular Activities',
    'Sleep Hours',
    'Sample Question Papers Practiced'
]
target_column = 'Performance Index'

# X ve y oluştur
X = df[feature_columns].values
y = df[target_column].values

# Bias (sabit) sütunu ekle
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Kapalı formülle theta (ağırlıklar) hesapla
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Tahminleri üret
y_pred = X_b @ theta_best

# Ortalama kare hata (MSE)
mse = np.mean((y - y_pred) ** 2)

# === Sonuçları yazdır ===
print("=== WLSE Modeli Çıktıları ===\n")

print(f"θ₀ (bias): {theta_best[0]:.4f}")
for i in range(1, len(theta_best)):
    print(f"θ{i} (x{i} için katsayı): {theta_best[i]:.4f}")


print(f"\nOrtalama Kare Hata (MSE): {mse:.4f}")

# === Denklem yazdır (x₁, x₂... ile) ===
equation = f"y = {theta_best[0]:.4f} "
for i in range(1, len(theta_best)):
    sign = "+" if theta_best[i] >= 0 else "-"
    equation += f"{sign} {abs(theta_best[i]):.4f} * x{i} "
print("\nRegresyon Denklemi:")
print(equation)

# === Görselleştirme: Gerçek vs Tahmin ===
plt.scatter(y, y_pred, color='blue', label='Tahminler')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='y = y_pred')
plt.xlabel("Gerçek Performance Index")
plt.ylabel("Tahmin Edilen Performance Index")
plt.title("Gerçek vs Tahmin (WLSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
