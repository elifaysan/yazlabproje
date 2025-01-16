
# USD/TL Zaman Serisi Analizi  

Bu proje, transformer tabanlı beş farklı modelin (Transformer, Informer, Reformer, Temporal Fusion Transformer (TFT), Autoformer) *Gram Altın/TL fiyat tahmini* üzerindeki performansını değerlendirmek için tasarlanmıştır.  

## İçerik  

- *Veri Toplama*  
  Veri seti Selenium kullanılarak Investing.com'dan elde edilmiştir (3866 kayıt). Veriler, ön işleme ve normalleştirme sonrasında modele uygun şekilde işlenmiştir.  

- *Kullanılan Modeller*  
  - *Transformer*: Çok başlı dikkat mekanizması.  
  - *Informer*: Uzun vadeli bağımlılıkları optimize eden model.  
  - *Reformer*: Daha hızlı encoder-decoder yapısı.  
  - *TFT*: Zaman serisi tahmini için optimize edilmiş model.  
  - *Autoformer*: Sezonsallık ve trend ayrışımıyla doğruluğu artırır.  

- *Değerlendirme Metrikleri*  
  - *MSE*: Ortalama kare hata.  
  - *MAE*: Ortalama mutlak hata.  
  - *RMSE*: Hata karekökü.  
  - *MAPE*: Yüzdelik hata.  
  - *R²*: Varyans açıklama oranı.  

## Sonuçlar  

- *En doğru model*: TFT (En düşük MSE ve RMSE değerleriyle).  
- *Verimlilik açısından öne çıkan*: Informer.  
- *Genel performans*: Tüm transformer modelleri, sezonsallık ve trend özelliklerini güçlü bir şekilde modelledi.  

## Gelecek Çalışmalar  
Ekonomik göstergeler ve dışsal faktörlerin entegrasyonu ile tahmin doğruluğu artırılabilir.  

![WhatsApp Image 2025-01-17 at 02 16 18](https://github.com/user-attachments/assets/03b0f28f-96dd-4b68-b0c3-df6459ee7ad4)
![WhatsApp Image 2025-01-17 at 02 16 18 (1)](https://github.com/user-attachments/assets/6d3336e6-b6f3-4a08-874c-33c1bec89aab)
![WhatsApp Image 2025-01-17 at 02 16 18 (2)](https://github.com/user-attachments/assets/194e00c7-f602-420d-bc66-8dfff2ad26a3)
![WhatsApp Image 2025-01-17 at 02 16 18 (3)](https://github.com/user-attachments/assets/ecb58516-faf1-41d5-a206-91f745cd55b1)
![WhatsApp Image 2025-01-17 at 02 16 27](https://github.com/user-attachments/assets/9fdcf6f1-9b99-48dd-be8d-e6d9fe412230)
