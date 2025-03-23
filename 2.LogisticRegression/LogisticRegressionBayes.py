import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Veriyi yükleyin
df = pd.read_csv(r"C:\Users\mehdi\OneDrive\Masaüstü\gender_classification_v7.csv")

# Hedef değişkenin 'gender' olduğunu varsayıyoruz.
# 'gender' sütunu metinsel ('male' / 'female'), binary formata çevirelim.
if df['gender'].dtype == object:
    df['gender'] = df['gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)

# Özellikler (X) ve hedef (y) belirleniyor.
X = df.drop('gender', axis=1).values
y = df['gender'].values

# Veriyi karıştırıp %70 eğitim, %30 test olacak şekilde bölüyoruz.
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Bias (sabit terim) eklemek için her iki veri setine birler sütunu ekleyelim.
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias  = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Sigmoid fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Eğitim fonksiyonu: Gradient descent ile ağırlıkları güncelleyelim.
def train_logistic_regression(X, y, learning_rate=0.01, n_iterations=10000):
    weights = np.zeros(X.shape[1])
    for i in range(n_iterations):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / len(y)
        weights -= learning_rate * gradient
    return weights

# Test verisi için tahmin fonksiyonu
def predict(X, weights):
    z = np.dot(X, weights)
    probs = sigmoid(z)
    return np.where(probs >= 0.5, 1, 0)

# Karmaşıklık (confusion) matrisi hesaplama fonksiyonu
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

# Eğitim süresini ölçüyoruz
start_train = time.perf_counter()
weights = train_logistic_regression(X_train_bias, y_train, learning_rate=0.01, n_iterations=10000)
end_train = time.perf_counter()
train_time = end_train - start_train

# Tahmin süresini ölçüyoruz
start_pred = time.perf_counter()
y_pred = predict(X_test_bias, weights)
end_pred = time.perf_counter()
pred_time = end_pred - start_pred

# Eğitim ve tahmin süreleri
print(f"\nEğitim Süresi: {train_time:.6f} saniye")
print(f"Tahmin Süresi: {pred_time:.6f} saniye")

# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)
print("\nKarmaşıklık Matrisi:")
print(cm)

# Değerlendirme metriklerini hesaplama
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Metrik değerlerini yazdırma
print(f"\nDoğruluk (Accuracy): {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")

# Karmaşıklık Matrisi Görselleştirme
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Karmaşıklık Matrisi")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Female (0)', 'Male (1)'])
plt.yticks(tick_marks, ['Female (0)', 'Male (1)'])
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")

# Hücrelerde sayıların gösterilmesi
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.show()


