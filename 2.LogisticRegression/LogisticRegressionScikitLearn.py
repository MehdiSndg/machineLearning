import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# CSV dosyasını yükleme
df = pd.read_csv(r"C:\Users\mehdi\OneDrive\Masaüstü\gender_classification_v7.csv")

# Hedef değişkeni numeric'e çeviriyoruz (Male: 1, Female: 0)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Özellikler ve hedef ayrımı
X = df.drop('gender', axis=1)
y = df['gender']

# Eğitim ve test setlerine ayırma (%70 eğitim, %30 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scikit-learn Logistic Regression Modeli Eğitimi
model = LogisticRegression()

# Eğitim süresini ölçme
start_time_train = time.time()
model.fit(X_train, y_train)
end_time_train = time.time()
train_time = end_time_train - start_time_train

# Test seti üzerinde tahmin ve tahmin süresini ölçme
start_time_pred = time.time()
y_pred = model.predict(X_test)
end_time_pred = time.time()
pred_time = end_time_pred - start_time_pred

# Model değerlendirme metriklerini hesaplama
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Specificity: TN / (TN + FP) hesaplanıyor
TN, FP, FN, TP = cm.ravel()
spec = TN / (TN + FP) if (TN + FP) != 0 else 0

# Sonuçları yazdırma
print("\nEğitim Süresi: {:.6f} saniye".format(train_time))
print("Tahmin Süresi: {:.6f} saniye".format(pred_time))
print("\nKarmaşıklık Matrisi:\n", cm)
print("\nAccuracy: {:.2f}%".format(acc * 100))
print("Precision: {:.2f}%".format(prec * 100))
print("Recall: {:.2f}%".format(rec * 100))
print("Specificity: {:.2f}%".format(spec * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))

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

