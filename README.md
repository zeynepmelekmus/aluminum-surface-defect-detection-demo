# Mini Yüzey Hatası Tespit Uygulaması

Bu proje, alüminyum profil imalatı aşamasında meydana gelen yüzey hatalarının derin öğrenme yöntemleriyle tespiti ve sınıflandırılması süreçlerini simüle eden mini bir **alüminyum yüzey hatası tespit ve raporlama** prototipidir. Amaç; kısa sürede çalıştırılabilir, okunabilir, Docker ile kolayca ayağa kaldırılabilen ve ileride PyTorch tabanlı hazır bir modelle doğrudan değiştirilebilecek modüler bir yazılım mimarisi göstermektir.

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

> [!NOTE]
> Uygulamanın ve kütüphane bağımlılıklarının (Pillow, Ultralytics vb.) sorunsuz kurulması için **Python 3.11** stabil sürümünün kullanılması önerilir. Python 3.14 gibi önizleme/geliştirme sürümleri derleme hatalarına sebep olabilir.

### macOS & Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```

### Windows
```cmd
:: Command Prompt (CMD)
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```
veya PowerShell kullanıyorsanız:
```powershell
# PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
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

- **macOS / Linux:**
  ```bash
  python3 -m unittest discover -s tests
  ```
- **Windows:**
  ```cmd
  python -m unittest discover -s tests
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

## Demo Açıklaması

1. Ürün adı, ürün ID ve eşik değeri girilir.
2. Örnek görüntü seçilir veya kullanıcı görüntü yükler.
3. Başlat butonuna basılır.
4. Sistem sonuçları tabloda gösterir.
5. Rapor oluştur butonu ile JSON/CSV rapor alınır.
6. Durdur ve Resetle ile sistem kontrol edilir.

## Not

Bu uygulama gerçek üretim sistemi değildir; mülakat/demo için hazırlanmış küçük bir prototiptir. Ancak modüler yapı sayesinde hazır PyTorch modeliyle genişletilebilir.
