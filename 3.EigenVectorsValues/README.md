## 1. Matris Manipülasyonu, Özdeğerler ve Özvektörlerin Makine Öğrenmesiyle İlişkisi

### 🔹 Matris Manipülasyonu Nedir?

Matris manipülasyonu, bir matris üzerinde yapılan temel işlemler bütünüdür. Bu işlemler arasında matris toplama, çarpma, transpoz alma, determinant ve tersini hesaplama gibi işlemler yer alır. Veri analizi ve makine öğrenmesinde veriler genellikle matris formatında temsil edilir; bu nedenle bu işlemler verinin işlenmesi, dönüştürülmesi ve modele uygun hale getirilmesi için temel bir yapı taşını oluşturur.

Örneğin, veri standardizasyonu, boyut indirgeme, ağırlık güncellemeleri veya sinir ağı katmanları arasında yapılan hesaplamalar hep matris manipülasyonları ile gerçekleştirilir.

---

### 🔹 Özdeğer ve Özvektör Nedir?

**Özdeğer (eigenvalue)**, bir kare matrisin uyguladığı dönüşüm sonucunda bir vektörün yalnızca ölçeklenmesini sağlayan skaler bir katsayıdır. **Özvektör (eigenvector)** ise bu dönüşümde yönü değişmeyen vektördür. Matematiksel olarak:

> A · v = λ · v

denkleminde `A` kare bir matris, `v` özvektör, `λ` ise özdeğerdir. Bu denklemin anlamı; `A` matrisinin `v` vektörünü yalnızca `λ` oranında büyütüp küçülmesidir.

---

### 🔹 Makine Öğrenmesi ile İlişkisi

Özdeğer ve özvektör kavramları, makine öğrenmesinde özellikle **veri dönüşümü** ve **boyut indirgeme** işlemlerinde kritik rol oynar. En yaygın kullanımı **Principal Component Analysis (PCA)** yöntemindedir:

- Veri setinin kovaryans matrisi oluşturulur.

- Bu matrisin özdeğer ve özvektörleri hesaplanır.

- En yüksek özdeğere sahip yönler (özvektörler) veri için en bilgilendirici doğrultuları gösterir.

- Veri bu yönlere projekte edilerek daha az boyutla temsil edilir.

### Bu kavramların kullanıldığı başlıca yöntem ve yaklaşımlar:

- Principal Component Analysis (PCA)

- Spektral Kümeleme (Spectral Clustering)

- Lineer diskriminant analizi (LDA)

- Görüntü sıkıştırma ve boyut indirgeme teknikleri

- Covariance Matrix analizleri


---

## 2. `numpy.linalg.eig` Fonksiyonunun Kullanımı ve İncelenmesi
  
`numpy.linalg.eig` fonksiyonu, bir kare matrisin özdeğerlerini (`eigenvalues`) ve sağ özvektörlerini (`eigenvectors`) döndürür.  
  
---  
  
### Kaynak Kod:  
  
```python  
def eig(a):  
```  
  
`a`: Girdi olarak kare bir NumPy dizisi (matris) alır.  
  
---  
  
###  İç İşleyiş:  
  
#### 1. Girdi kontrolü:  
  
```python  
a, wrap = _makearray(a)  
_assert_stacked_square(a)  
_assert_finite(a)  
```  
  
- `_makearray(a)`: `a` verisini düzgün bir NumPy dizisine dönüştürür.  
- `_assert_stacked_square(a)`: Matrisin kare (n x n) olup olmadığını kontrol eder.  
- `_assert_finite(a)`: `a` içinde NaN veya sonsuz değer olup olmadığını kontrol eder.  
  
  
#### 2. Veri tipi belirleme ve LAPACK signature seçimi:  
  
```python  
t, result_t = _commonType(a)  
signature = 'D->DD' if isComplexType(t) else 'd->DD'  
```  
  
- `t`: veri tipi (float, complex vs.)  
- `signature`: karmaşık mı gerçek mi olduğunu belirleyip LAPACK fonksiyonuna uygun çağrı yapılmasını sağlar.  
  
  
#### 3. Asıl hesaplama:  
  
```python  
w, vt = _umath_linalg.eig(a, signature=signature)  
```  
  
- NumPy’nin C arayüzü (`_umath_linalg`) üzerinden LAPACK’in `geev` rutinini çağırır.  
- `w`: özdeğerler  
- `vt`: özvektörlerin satır bazlı hali (her sütun bir özvektör)  
  
#### 4. Sonuçların dönüştürülmesi:  
  
```python  
if not isComplexType(t) and all(w.imag == 0.0):  
 w = w.real vt = vt.real result_t = _realType(result_t)else:  
 result_t = _complexType(result_t)  
vt = vt.astype(result_t, copy=False)  
```  
  
- Karmaşık olmayan durumlarda `.real` ile sadeleştirme yapılır.  
- `vt` matrisi uygun türe dönüştürülür.  
  
---  
  
###  Döndürülen Değer:  
  
```python  
return EigResult(w.astype(result_t, copy=False), wrap(vt))  
```  
  
Yani:  
  
- `w`: Özdeğerleri içeren bir NumPy dizisi  
- `vt`: Her sütunu bir özvektör olan bir NumPy matrisi  
- Bunlar `EigResult` adlı bir `namedtuple` yapısında döndürülür.  
  
---  
  
###  Özet:  
  
- `eig(a)`, `A·v = λ·v` denklemine göre `λ` ve `v` değerlerini bulur.  
- Sayısal doğrulamalar yapar.  
- Girdi matrisin yapısına göre otomatik veri türü seçimi yapar.  
- Yüksek verimlilik için LAPACK (geev) rutinini kullanır.  
- Karmaşık sayılarla çalışmaya da uygundur.

---
## 3. NumPy ve Custom Yöntemle Hesaplanan Özdeğer ve Özvektörlerin Karşılaştırması  
  

  
###  Özdeğer Karşılaştırması  
  
| NumPy Özdeğerleri | Custom Özdeğerleri |  
|-------------------|--------------------|  
| 5.0               | 7.0                |  
| 3.0               | 5.0                |  
| 7.0               | 3.0                |  
  
  Her iki yöntemde de **özdeğerler sayısal olarak aynıdır**, ancak **sıralamaları farklıdır**.  
- Bu fark sadece çıktının sıralama formatından kaynaklanmaktadır; matematiksel sonuçlarda bir fark yoktur.  
- Çünkü özdeğerlerin sırası lineer cebir açısından önemli değildir; vektörlere karşılık gelen sıralama doğru olduğu sürece geçerlidir.  
  
---  
  
###  Özvektör Karşılaştırması  
  
| NumPy Özvektörleri (yaklaşık)          | Custom Özvektörleri (yaklaşık)         |  
|----------------------------------------|----------------------------------------|  
| [0.7071, 0.3162, 0.5883]               | [ 0.5883, -0.7071, -0.3162]            |  
| [0.    , 0.    , 0.7845]               | [ 0.7845,  0.    ,  0.    ]            |  
| [0.7071, 0.9487, 0.1961]               | [ 0.1961, -0.7071, -0.9487]            |  
  
 Özvektörler **yön açısından farklı görünüyor** olabilir, çünkü özvektörler **skaler çarpanlara göre tanımsızdır**.    
  Yani bir özvektör `v` için `-v` de geçerli bir özvektördür.  
- NumPy tarafından döndürülen özvektörler genellikle normalize edilmiştir ve yönleri farklı olabilir.  
- Özellikle işaret farkları veya sıralama farkları olması özvektörlerin geçersiz olduğu anlamına gelmez.  
  
---  
  
###  Süre Karşılaştırması  
  
| Yöntem         | Süre (saniye)        |  
|----------------|----------------------|  
| NumPy          | 0.0002094000         |  
| Custom         | 0.0003252000         |  
  
- NumPy fonksiyonu, optimize edilmiş LAPACK altyapısını kullandığı için daha hızlıdır.  
- Custom fonksiyon temel Python işlemleriyle çalıştığından biraz daha yavaştır.  
- Bu fark küçük matrislerde belirgin olmasa da büyük matrislerde NumPy çok daha hızlıdır.  
  
---  
  
###  Genel Değerlendirme  
  
| Kriter            | NumPy                            | Custom                              |  
|------------------|----------------------------------|-------------------------------------|  
| Hız              |  Çok hızlı                      |  Göreceli olarak daha yavaş       |  
| Kolaylık         |  Tek satır fonksiyon            |  Daha fazla kod gerektirir        |   
| Hassasiyet       |  Yüksek                         |  Benzer doğruluk (küçük farklarla)|  
  
---  
  
###  Sonuç  
  
Her iki yöntem de doğru sonuçlar üretmiştir. NumPy yöntemi pratik uygulamalarda tercih edilirken, custom çözüm algoritmanın mantığını anlamak ve eğitim amaçlı çalışmalarda oldukça değerlidir. Bu karşılaştırma, hazır kütüphane kullanımı ile temel lineer cebir kavramlarının manuel uygulanışı arasındaki farkları net biçimde ortaya koymaktadır.

---

### Kullanılan Kaynaklar
[https://www.geeksforgeeks.org/eigen-values/](https://www.geeksforgeeks.org/eigen-values/)

https://www.geeksforgeeks.org/matrices-and-matrix-arithmetic-for-machine-learning/

[https://en.wikipedia.org/wiki/Principal_component_analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
