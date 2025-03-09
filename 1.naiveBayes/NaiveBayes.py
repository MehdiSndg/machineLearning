import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------------------
# 1. Veri Seti Yükleme ve Bölme
# ---------------------------

# Excel dosyasını yükleme
df = pd.read_excel(r"C:\Users\mehdi\OneDrive\Masaüstü\diabetes.xlsx")

# Özellikler (X) ve hedef (y) değişkeni
X = df.drop("Outcome", axis=1)
y = df["Outcome"].values  # Hedef değerleri NumPy dizisine çeviriyoruz


# Özel train-test bölme fonksiyonu (%20 test)
def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    test_count = int(len(y) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    # Index bazlı seçim yaparak eğitim ve test setlerini oluşturuyoruz.
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test


# Veri setini %80 eğitim, %20 test olarak bölüyoruz.
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------
# 2. Custom Gaussian Naive Bayes Model (Scikit-learn kullanmadan)
# ---------------------------

class CustomGaussianNB:
    def __init__(self):
        self.classes = None  # Sınıfları tutacak
        self.mean = {}  # Her sınıf için özellik ortalamaları
        self.var = {}  # Her sınıf için özellik varyansları
        self.priors = {}  # Her sınıfın öncelik (prior) olasılığı

    def fit(self, X, y):
        # Giriş verisinin DataFrame olduğundan emin oluyoruz
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.classes = np.unique(y)
        # Her sınıf için ortalama, varyans ve prior değerlerini hesaplıyoruz
        for c in self.classes:
            X_c = X[y == c]  # Sadece c sınıfına ait örnekler
            self.mean[c] = X_c.mean(axis=0)
            # Varyans değerine çok küçük bir sabit ekleyerek bölme hatalarını önlüyoruz.
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / float(X.shape[0])

    def gaussian_log_prob(self, x, mean, var):
        # Her bir özellik için normal dağılımın log olasılığını hesaplar
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def predict(self, X):
        # Giriş verisini DataFrame'e çeviriyoruz (eğer değilse)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        predictions = []
        # Her örnek için, her sınıfın log olasılığını hesaplayıp en yüksek değeri veren sınıfı seçiyoruz.
        for index, row in X.iterrows():
            class_log_probs = {}
            for c in self.classes:
                log_prob = np.log(self.priors[c])
                log_prob += np.sum(self.gaussian_log_prob(row, self.mean[c], self.var[c]))
                class_log_probs[c] = log_prob
            # En yüksek log olasılığa sahip sınıfı tahmin ediyoruz.
            predicted_class = max(class_log_probs, key=class_log_probs.get)
            predictions.append(predicted_class)
        return np.array(predictions)


# ---------------------------
# 3. Model Eğitimi ve Tahmin Sürelerinin Ölçümü
# ---------------------------

custom_model = CustomGaussianNB()

# Eğitim süresini ölçüyoruz.
start_train = time.perf_counter()
custom_model.fit(X_train, y_train)
end_train = time.perf_counter()
custom_training_time = end_train - start_train

# Tahmin süresini ölçüyoruz.
start_pred = time.perf_counter()
y_pred_custom = custom_model.predict(X_test)
end_pred = time.perf_counter()
custom_prediction_time = end_pred - start_pred


# ---------------------------
# 4. Performans Metriklerinin Hesaplanması (Scikit-learn kullanmadan)
# ---------------------------

def compute_confusion_matrix(y_true, y_pred):
    """
    Gerçek ve tahmin edilen etiketlere göre karmaşıklık matrisi oluşturur.
    Matriste:
      - [0,0] -> True Negatives (TN)
      - [0,1] -> False Positives (FP)
      - [1,0] -> False Negatives (FN)
      - [1,1] -> True Positives (TP)
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP],
                     [FN, TP]])


def compute_accuracy(cm):
    """Accuracy = (TP + TN) / Toplam örnek sayısı"""
    return (cm[0, 0] + cm[1, 1]) / np.sum(cm)


def compute_precision(cm):
    """Precision = TP / (TP + FP)"""
    if (cm[1, 1] + cm[0, 1]) == 0:
        return 0
    return cm[1, 1] / (cm[1, 1] + cm[0, 1])


def compute_recall(cm):
    """Recall = TP / (TP + FN)"""
    if (cm[1, 1] + cm[1, 0]) == 0:
        return 0
    return cm[1, 1] / (cm[1, 1] + cm[1, 0])


def compute_specificity(cm):
    """Specificity = TN / (TN + FP)"""
    if (cm[0, 0] + cm[0, 1]) == 0:
        return 0
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])


def compute_f1(precision, recall):
    """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


# Karmaşıklık matrisini hesaplayalım.
cm_custom = compute_confusion_matrix(y_test, y_pred_custom)
accuracy_custom = compute_accuracy(cm_custom)
precision_custom = compute_precision(cm_custom)
recall_custom = compute_recall(cm_custom)
specificity_custom = compute_specificity(cm_custom)
f1_custom = compute_f1(precision_custom, recall_custom)

# ---------------------------
# 5. Sonuçların Ekrana Yazdırılması ve Görselleştirilmesi
# ---------------------------

print("Custom Gaussian Naive Bayes Performance (Scikit-learn kullanmadan):")
print("Doğruluk (Accuracy):", accuracy_custom)
print("Precision:", precision_custom)
print("Recall:", recall_custom)
print("Spesifisite (Specificity):", specificity_custom)
print("F1 Score:", f1_custom)
print("Eğitim süresi: {:.6f} saniye".format(custom_training_time))
print("Tahmin süresi: {:.6f} saniye".format(custom_prediction_time))
print("Karmaşıklık Matrisi (Custom):")
print(cm_custom)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(6, 6))
plt.imshow(cm_custom, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Karmaşıklık Matrisi (Custom Gaussian NB)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Diabetic (0)', 'Diabetic (1)'], rotation=45)
plt.yticks(tick_marks, ['Non-Diabetic (0)', 'Diabetic (1)'])
plt.ylabel("Gerçek Değer")
plt.xlabel("Tahmin Edilen Değer")

# Her hücreye değerleri yazdırıyoruz.
thresh = cm_custom.max() / 2.0
for i in range(cm_custom.shape[0]):
    for j in range(cm_custom.shape[1]):
        plt.text(j, i, format(cm_custom[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_custom[i, j] > thresh else "black")

plt.tight_layout()
plt.show()


