# Mini Yüzey Hatası Tespit Uygulaması

Bu proje, TÜBİTAK 2247-C STAR bursiyer başvurusu kapsamında hazırlanmış mini **alüminyum yüzey hatası tespit ve raporlama** prototipidir. Uygulama, 1505 kodlu sanayi projesinde ihtiyaç duyulan yüzey hata tespiti akışını küçük ve çalıştırılabilir bir demo olarak simüle eder. Amaç; kısa sürede çalıştırılabilir, okunabilir, Docker ile ayağa kalkabilen ve ileride PyTorch tabanlı hazır modelle değiştirilebilir bir yazılım mimarisi göstermektir.

## Özellikler

- Operatör ürün adı, ürün etiketi/ID ve eşik değeri girebilir.
- Başlat, Durdur, Resetle ve Rapor Oluştur işlemleri vardır.
- Başlat ile hata tespiti sonucu üretilir/gösterilir.
- Sonuçlar tabloda listelenir:
  - Hata tipi
  - Metre bilgisi
  - Güven skoru
  - Zaman bilgisi
  - Bounding box bilgisi
- JSON ve CSV rapor oluşturulur.
- Kod yapısı PyTorch/YOLO modeli entegrasyonuna uygundur.
- `models/best.pt` varsa gerçek model denenir; gerekli bağımlılık/model uyumsuzluğu olursa uygulama mock moda düşer.

## Mimari

```text
app/
  main.py                 # Streamlit arayüzü
  detection_service.py    # Mock veya PyTorch/YOLO tespit servisi
  report_service.py       # JSON/CSV raporlama servisi
models/
  best.pt                 # Hazır PyTorch/YOLO model dosyası
data/sample_images/       # Örnek test görselleri
reports/                  # Oluşturulan raporlar
Dockerfile
requirements.txt
```

Arayüz kodu model tahmin detaylarını bilmez. Tüm tespit işlemi `DetectionService.detect(image_path, threshold)` üzerinden yapılır. Böylece ileride farklı bir PyTorch modeli kullanılacaksa yalnızca `app/detection_service.py` değiştirilir.

## Lokal Çalıştırma

```bash
python3 -m venv tubitak_env
source tubitak_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```

Tarayıcıda genellikle şu adres açılır:

```text
http://localhost:8501
```

## Birim Testleri Çalıştırma

Kod kalitesini ve servis katmanlarının bütünlüğünü doğrulamak için aşağıdaki komut ile birim testlerini çalıştırabilirsiniz:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```


## Docker ile Çalıştırma

```bash
docker build -t surface-defect-demo .
docker run -p 8501:8501 surface-defect-demo
```

Sonra tarayıcıdan aç:

```text
http://localhost:8501
```

## PyTorch/YOLO Model Entegrasyonu

Model entegrasyon noktası:

```text
app/detection_service.py
```

Varsayılan model yolu:

```text
models/best.pt
```

`DetectionService` önce `ultralytics.YOLO` ile modeli yüklemeyi dener. Model yüklenirse tespit sonuçlarını şu ortak formata çevirir:

```json
{
  "defect_type": "Cizik",
  "meter": 12.4,
  "confidence": 91.2,
  "timestamp": "2026-06-06 10:15:00",
  "bbox": [10, 20, 80, 120],
  "source": "yolo-pytorch"
}
```

Model veya bağımlılık yüklenemezse uygulama otomatik mock sonuç üretir. Bu sayede demo ortamında model çalışmasa bile temel işlevler gösterilebilir.

## Önerilen Git Commit Sırası

```bash
git init
git add README.md .gitignore requirements.txt Dockerfile
git commit -m "Initialize project"

git add app/detection_service.py models/best.pt data/sample_images
git commit -m "Add modular detection service and sample assets"

git add app/main.py
git commit -m "Create Streamlit control panel UI"

git add app/report_service.py
git commit -m "Add JSON and CSV report generation"

git add Dockerfile README.md
git commit -m "Add Docker support and usage documentation"
```

## Demo Açıklaması

1. Ürün adı, ürün ID ve eşik değeri girilir.
2. Örnek görüntü seçilir veya kullanıcı görüntü yükler.
3. Başlat butonuna basılır.
4. Sistem sonuçları tabloda gösterir.
5. Rapor oluştur butonu ile JSON/CSV rapor alınır.
6. Durdur ve Resetle ile sistem kontrol edilir.

## Not

Bu uygulama gerçek üretim sistemi değildir; mülakat/demo için hazırlanmış küçük bir prototiptir. Ancak modüler yapı sayesinde hazır PyTorch modeliyle genişletilebilir.
