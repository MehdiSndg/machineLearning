import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Veriyi yükle
df = pd.read_csv("C:/Users/mehdi/OneDrive/Masaüstü/Student_Performance.csv")

# Kategorik veriyi sayısala çevir
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Özellik ve hedef değişkenler
feature_columns = [
    'Hours Studied',
    'Previous Scores',
    'Extracurricular Activities',
    'Sleep Hours',
    'Sample Question Papers Practiced'
]
target_column = 'Performance Index'

X = df[feature_columns].values
y = df[target_column].values

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X, y)

# Ağırlıkları al
intercept = model.intercept_
coefficients = model.coef_

# θ0, θ1, θ2... olarak yazdır
print("=== Scikit-learn Lineer Regresyon Modeli ===\n")
print(f"θ₀ (bias): {intercept:.4f}")
for i, coef in enumerate(coefficients):
    print(f"θ{i+1} (x{i+1} için katsayı): {coef:.4f}")

# Ortalama Kare Hata (MSE)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"\nOrtalama Kare Hata (MSE): {mse:.4f}")

# Denklem yazdır
equation = f"y = {intercept:.4f} "
for i, coef in enumerate(coefficients):
    sign = "+" if coef >= 0 else "-"
    equation += f"{sign} {abs(coef):.4f} * x{i+1} "
print("\nModelin Regresyon Denklemi:")
print(equation)

# Gerçek vs Tahmin grafiği
plt.scatter(y, y_pred, color='green', label='Tahminler')
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='y = y_pred')
plt.xlabel("Gerçek Performance Index")
plt.ylabel("Tahmin Edilen Performance Index")
plt.title("Gerçek vs Tahmin (Scikit-learn)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
