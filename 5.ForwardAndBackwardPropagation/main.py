import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Neural Network Sınıfını İçe Aktar ===
from neural_network import NeuralNetwork

# === 2. Veri Yükleme ===
df = pd.read_csv(r"C:\Users\mehdi\OneDrive\Masaüstü\Raisin_Dataset.csv")

# === 3. Etiketleri Sayısala Çevirme (Besni → 0, Kecimen → 1) ===
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

# === 4. Özellik / Etiket Ayırma ===
X = df.drop('Class', axis=1).values
y = df['Class'].values.reshape(-1, 1)  # 900x1 olacak şekilde

# === 5. Özellikleri [0-1] Arasına Normalize Etme ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 6. Eğitim / Test Setine Bölme (%80 - %20) ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === 7. Sinir Ağını Oluşturma ve Eğitme ===
nn = NeuralNetwork(input_size=7, hidden_size=10, learning_rate=0.1)
losses = nn.fit(X_train, y_train, epochs=100)

# === 8. Test Setinde Tahmin ===
y_pred_test = nn.predict(X_test)

# === 9. Doğruluk Skoru Hesaplama ===
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy:.4f}")

# === 10. Confusion Matrix Görselleştirme ===
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === 11. Loss vs Epoch Grafiği ===
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
