
**Diyabet Sınıflandırması Projesi: Gaussian Naive Bayes Uygulamaları**

1\. Problem Tanımı

Bu projede, diyabet tanısı konusundaki klinik ölçümlere dayalı bir veri seti üzerinde ikili sınıflandırma gerçekleştirilmiştir. Amacımız, hastanın diyabetli olup olmadığını belirlemek için Gaussian Naive Bayes algoritmasını iki farklı yaklaşımla uygulamaktır:

2\. Veri

Veri Seti: Diyabet tanısı için kullanılan klinik ölçümleri içeren tabular veri seti.

Örnek Sayısı: 768

Özellik Sayısı: 9 (glukoz, kan basıncı, insülin, BMI, yaş vb.)

Hedef Değişken: `Outcome` sütunu  

- `1` → Diyabet var  

- `0` → Diyabet yok

3\. Yöntem

Proje, aynı veri bölünmesi kullanılarak iki farklı şekilde uygulanmıştır:

A. Gaussian Naive Bayes Scikit-learn

Model: `GaussianNB` kullanılarak uygulanmıştır.

Veri Bölünmesi: %80 eğitim, %20 test.

Performans Ölçümleri:  

- Accuracy, Precision, Recall, F1 Score, Specificity

- Eğitim ve tahmin süreleri 

- Karmaşıklık Matrisi oluşturulup görselleştirilmiştir.

B. Naive Bayes

Model: Gaussian Naive Bayes algoritması, tüm istatistiksel hesaplamalar (ortalama, varyans, log-olasılık) Python sınıfı olarak sıfırdan yazılmıştır.

Veri Bölünmesi: %80 eğitim, %20 test kullanılmıştır.

Performans Ölçümleri:  

- Kendi yazdığımız fonksiyonlarla Accuracy, Precision, Recall, F1 Score ve Specificity hesaplanmıştır.

- Eğitim ve tahmin süreleri 

- Karmaşıklık Matrisi hesaplanıp görselleştirilmiştir.

4\. Sonuçlar

Gaussian Naive Bayes Scikit-learn:

Accuracy: 0.7662337662337663  

Precision: 0.6610169491525424  

Recall: 0.7090909090909091  

Specificity: 0.797979797979798  

F1 Score: 0.6842105263157895  

Eğitim Süresi: 0.001139 saniye  

Tahmin Süresi: 0.000579 saniye  

Karmaşıklık Matrisi: [[79, 20], [16, 39]]

Naive Bayes:

Accuracy: 0.7647058823529411  

Precision: 0.6610169491525424  

Recall: 0.7090909090909091  

Specificity: 0.7959183673469388  

F1 Score: 0.6842105263157895  

Eğitim Süresi: 0.001163 saniye  

Tahmin Süresi: 0.086555 saniye  

Karmaşıklık Matrisi: [[78, 20], [16, 39]]


5\. Yorum / Tartışma

Accuracy Farkı:  

\- Scikit-learn uygulamasında accuracy değeri 0.76623 iken, custom modelde 0.76471 olarak gözükmektedir. Bu küçük fark, veri setinin eğitim-test bölünmesindeki yuvarlama farkları veya modelin hesaplama sürecinde ortaya çıkan küçük farklılıklardan kaynaklanabilir.

Precision, Recall, Specificity, F1 Score:  

\- Her iki modelde de precision, recall, specificity ve F1 score değerleri aynıdır. Bunun nedeni, iki model de genel olarak aynı sınıflandırma sonuçlarını üretmiş olmasıdır. Küçük farklılıklar, esas olarak overall accuracy veya karmaşıklık matrisindeki tekil örnek farklarından kaynaklanmaktadır.

Eğitim Süresi:  

\- Her iki modelde de eğitim süreleri çok kısa (yaklaşık 0.0011 saniye) olup, benzer performans göstermektedir.

Tahmin Süresi:  

\- Scikit-learn modelinde tahmin süresi 0.000579 saniyeyken, custom modelde 0.086555 saniye gözlemlenmiştir. Bu fark, Scikit-learn'ün optimize edilmiş (ve muhtemelen C tabanlı) implementasyonuyla, custom modelin ise Python’daki döngüsel hesaplama yapısı nedeniyle tahmin aşamasının daha yavaş gerçekleşmesinden kaynaklanmaktadır.

Karmaşıklık Matrisi Farkı:  

\- Scikit-learn uygulamasında TN değeri 79 iken, custom modelde 78 olarak hesaplanmıştır. Bu küçük fark, eğitim-test bölünmesi sırasında rastgele seçilen örneklerdeki ince farklılıklardan veya hesaplama yöntemindeki küçük farklardan kaynaklanabilir. Diğer metriklerin aynı çıkması, genel sınıflandırma performansının benzer olduğunu göstermektedir.


