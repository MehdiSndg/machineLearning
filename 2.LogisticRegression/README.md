**Gender Classification Projesi: Logistic Regression Uygulamaları**

**1. Problem Tanımı**
Bu projede, "gender\_classification\_v7.csv" veri seti kullanılarak cinsiyet sınıflandırması gerçekleştirilmiştir. Amaç, yüz özelliklerine dayanarak kişinin cinsiyetini (Male/Female) doğru şekilde tahmin edebilmektir.


**2. Veri**

- **Veri Seti:** gender\_classification\_v7.csv
- **Örnek Sayısı:** 5001
- **Özellik Sayısı:** 7
- **Özellikler:** 

long\_hair; (uzun saçlı -> 1 , kısa saçlı -> 0) 

forehead\_width\_cm; (alnın cm cinsinden genişliği)

forehead\_height\_cm; (alnın cm cinsinden yüksekliği) 

nose\_wide; (geniş burun -> 1 ,  geniş olmayan burun -> 0) 

nose\_long; (uzun burun -> 1 , uzun olmayan burun -> 0) 

lips\_thin; (ince dudaklar -> 1 , ince olmayan dudaklar  -> 0) 

distance\_nose\_to\_lip\_long; (burun ve dudaklar arası uzun mesafe -> 1 ,burun ve dudaklar arası kısa mesafe -> 0)

- **Hedef Değişken:** gender (Male = 1, Female = 0)


**3. Yöntem**
Projede aynı veri bölünmesi (%70 eğitim, %30 test) kullanılarak iki yaklaşım uygulanmıştır:

**A. Kütüphane Kullanılarak (Scikit-learn) Logistic Regression**

- Model: Scikit-learn LogisticRegression
- Ölçütler: Accuracy, Precision, Recall, Specificity, F1 Score, eğitim ve tahmin süreleri
- Görselleştirme: Confusion matrix

**B. Kütüphane Kullanılmadan (Custom Model) Logistic Regression**

- Model: Numpy ve Python ile sıfırdan oluşturulan algoritma (gradient descent, sigmoid, vb.)
- Ölçütler: Yukarıdaki metrikler aynı şekilde hesaplanmıştır


**4. Sonuçlar**

**Kütüphane Kullanılarak:**

- **Eğitim Süresi:** 0.011653 saniye
- **Tahmin Süresi:** 0.001052 saniye
- **Karmaşıklık Matrisi:** [[715, 24],[30, 732]]
- **Accuracy:** %96.40
- **Precision:** %96.83
- **Recall:** %96.06
- **Specificity:** %96.75
- **F1 Score:** %96.44

**Kütüphane Kullanılmadan:**

- **Eğitim Süresi:** 0.373174 saniye
- **Tahmin Süresi:** 0.000056 saniye
- **Karmaşıklık Matrisi:** [[740, 38],[33, 690]]
- **Accuracy:** %95.27
- **Precision:** %94.78
- **Recall:** %95.44
- **Specificity:** %95.12
- **F1 Score:** %95.11


**5. Yorum / Tartışma**

**Eğitim Süreleri:**

- **Kütüphane Kullanılarak:**
  Eğitim süresi 0.011653 saniye olarak ölçülmüştür. Bu hız, Scikit-learn’ün C tabanlı olmasından kaynaklanır.
- **Kütüphane Kullanılmadan:**
  Eğitim süresi 0.373174 saniye olarak hesaplanmıştır. Python’un yorumlanmış yapısı ve döngüsel hesaplama yöntemleri, optimizasyon açısından daha yavaş çalıştığı için bu süre daha uzundur.

**Tahmin Süreleri:**

- **Kütüphane Kullanılarak:**
  Tahmin süresi 0.001052 saniye. Scikit-learn'ün predict metodunda, verilerin doğrulanması, hata kontrolleri, veri tiplerinin uyumlu hale getirilmesi gibi ek işlemler bulunur. Bu ek işlemler, metodun esnekliğini ve güvenilirliğini artırır fakat her tahmin çağrısında küçük bir overhead (ek yük) oluşturur. Bu da tahmin süresinin biraz daha uzun olmasına neden olur.
- **Kütüphane Kullanılmadan:**
  Tahmin süresi 0.000056 saniye. Kendi yazdığınız modelde tahmin, sadece Numpy’nin vektörleştirilmiş matris çarpımı ve eşikleme işlemiyle yapılıyor. Bu işlemler doğrudan gerçekleştirilir ve fazladan ek kontrol ya da veri doğrulama adımları olmadığı için son derece hızlı çalışır.

**Accuracy ve Diğer Performans Metrikleri:**

- **Accuracy:**
  Kütüphane tabanlı model %96.40, custom model ise %95.27 doğruluk sağlamıştır. Küçük farklılık, modelin optimizasyon stratejileri (öğrenme oranı ayarlamaları vb.) ve veri bölünmesindeki  varyasyonlardan kaynaklanabilir.
- **Precision, Recall, Specificity, F1 Score:**
  Bu metrikler, modelin pozitif tahminlerinin doğruluğu (Precision), gerçek pozitifleri yakalama oranı (Recall), negatif örnekleri doğru sınıflandırma oranı (Specificity) ve genel performansın harmonik ortalaması (F1 Score) gibi farklı açılardan model performansını değerlendirmeye yarar. Her iki modelde de benzer değerler elde edilmiştir; bu durum, her iki yaklaşımın genel sınıflandırma yeteneğinin birbirine yakın olduğunu göstermektedir.

**Karmaşıklık Matrisi:**

- **Kütüphane Kullanılarak:**
  Matriste 715 doğru negatif, 732 doğru pozitif; 24 yanlış pozitif ve 30 yanlış negatif gözlemlenmiştir.
- **Kütüphane Kullanılmadan:**
  Matriste 740 doğru negatif, 690 doğru pozitif; 38 yanlış pozitif ve 33 yanlış negatif bulunmaktadır.
  Küçük farklılıklar, eğitim-test bölünmesindeki rastlantısal örnek seçimi ve modelin hesaplama yöntemlerindeki ince farklardan kaynaklanır.

**Genel Değerlendirme:**

- **Performans Metrikleri:**
  Kullanılan metrikler, modelin sadece genel doğruluk oranını (accuracy) değil; aynı zamanda pozitif ve negatif sınıflandırmalardaki başarısını da detaylandırmaktadır. Bu, özellikle dengesiz veri setlerinde model performansını daha doğru değerlendirmek için önemlidir.
- **Sonuçların Uygulanabilirliği:**
  Her iki yaklaşım da yüksek doğruluk ve benzer performans metrikleri sunmaktadır. Ancak, gerçek zamanlı uygulamalarda Scikit-learn’ün optimize edilmiş yapısı eğitim süresinin çok kısa olması açısından avantaj sağlar. Custom model ise, algoritmanın temel prensiplerini anlamak ve üzerinde ince ayar yapabilmeyi sağlar.

https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset





