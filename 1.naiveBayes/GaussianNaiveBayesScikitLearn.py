import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score,precision_score, recall_score, f1_score)

# ---------------------------
# 1. Veri Seti Yükleme ve İnceleme
# ---------------------------

df = pd.read_excel(r"C:\Users\mehdi\OneDrive\Masaüstü\diabetes.xlsx")

# Veri setinin boyutunu ve ilk birkaç satırını ekrana yazdırıyoruz.
print("Veri seti boyutu:", df.shape)
print(df.head())

# ---------------------------
# 2. Özellikler ve Hedef Değişkenin Ayrılması
# ---------------------------

# 'Outcome' sütunu hedef değişkenimizdir (0: diyabet yok, 1: diyabet var).
X = df.drop("Outcome", axis=1)  # Özellikler: tüm sütunlar, Outcome hariç
y = df["Outcome"]

# ---------------------------
# 3. Eğitim ve Test Setlerine Ayırma
# ---------------------------
# Veri setini %80 eğitim ve %20 test olarak bölüyoruz.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Gaussian Naive Bayes Modelinin Oluşturulması ve Eğitilmesi
# ---------------------------

gnb = GaussianNB()

# Modelin eğitim süresini ölçüyoruz.
start_train = time.perf_counter()
gnb.fit(X_train, y_train)   # Modeli eğitim verileriyle eğitiyoruz.
end_train = time.perf_counter()
training_time = end_train - start_train

# ---------------------------
# 5. Test Seti Üzerinde Tahmin Yapma ve Süre Ölçümü
# ---------------------------

# Test seti üzerinde tahmin yapma süresini ölçüyoruz.
start_pred = time.perf_counter()
y_pred = gnb.predict(X_test)
end_pred = time.perf_counter()
prediction_time = end_pred - start_pred

# ---------------------------
# 6. Performans Metriklerinin Hesaplanması
# ---------------------------

# Confusion matrix oluşturuyoruz.
cm = confusion_matrix(y_test, y_pred)
# cm yapısı:
# [[TN, FP],
#  [FN, TP]]
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

# Accuracy: Doğru tahminlerin oranı.
accuracy = accuracy_score(y_test, y_pred)

# Precision: Pozitif tahminlerden, gerçekten pozitif olanların oranı.
precision = precision_score(y_test, y_pred)

# Recall: Gerçek pozitiflerin ne kadarının doğru tahmin edildiğini gösterir.
recall = recall_score(y_test, y_pred)

# Spesifisite: Negatiflerin ne kadarının doğru tanımlandığı (True Negative Rate).
specificity = recall_score(y_test, y_pred, pos_label=0)

# F1 Score: Precision ve Recall'un harmonik ortalaması.
f1 = f1_score(y_test, y_pred)

# ---------------------------
# 7. Sonuçların Yazdırılması
# ---------------------------

print("Gaussian Naive Bayes Performance (Scikit-learn ile):")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("Eğitim süresi: {:.6f} saniye".format(training_time))
print("Tahmin süresi: {:.6f} saniye".format(prediction_time))
print("Karmaşıklık Matrisi:")
print(cm)

# ---------------------------
# 8. Karmaşıklık Matrisinin Görselleştirilmesi
# ---------------------------

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Karmaşıklık Matrisi")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Diabetic (0)', 'Diabetic (1)'], rotation=45)
plt.yticks(tick_marks, ['Non-Diabetic (0)', 'Diabetic (1)'])
plt.ylabel("Gerçek Değer")
plt.xlabel("Tahmin Edilen Değer")

# Her hücredeki sayıyı ekrana yazdırıyoruz.
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()