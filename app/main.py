from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from detection_service import DetectionService
from report_service import build_report_payload, save_csv_report, save_json_report

st.set_page_config(page_title="Mini Yüzey Hatası Tespit", page_icon="🔎", layout="wide")

MODEL_PATH = Path("models/best.pt")
SAMPLE_DIR = Path("data/sample_images")

if "running" not in st.session_state:
    st.session_state.running = False
if "detections" not in st.session_state:
    st.session_state.detections = []
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None

st.title("🔎 Mini Yüzey Hatası Tespit Uygulaması")
st.caption("Alüminyum yüzey hatalarını tespit ve raporlama prototipi | Mock + PyTorch/YOLO entegrasyonuna hazır yapı")

with st.sidebar:
    st.header("Operatör Girişi")
    product_name = st.text_input("Ürün adı", value="Alüminyum Profil")
    product_id = st.text_input("Ürün etiketi / ürün ID", value="ALM-2026-001")
    threshold = st.slider("Eşik değeri (%)", min_value=0, max_value=100, value=60, step=1)

    st.divider()
    st.header("Görüntü Kaynağı")
    uploaded = st.file_uploader("Test görüntüsü yükle", type=["jpg", "jpeg", "png"])
    sample_images = sorted(SAMPLE_DIR.glob("*.jpg"))
    selected_sample = st.selectbox(
        "Veya örnek görüntü seç",
        options=["Seçilmedi"] + [p.name for p in sample_images],
    )

    st.divider()
    real_model_requested = st.toggle("Varsa PyTorch/YOLO modelini kullan", value=True)

service = DetectionService(MODEL_PATH, use_real_model=real_model_requested)

col_status_1, col_status_2, col_status_3 = st.columns(3)
col_status_1.metric("Sistem durumu", "Çalışıyor" if st.session_state.running else "Durdu")
col_status_2.metric("Model", "Yüklendi" if service.model_loaded else "Mock/Fallback")
col_status_3.metric("Son hata sayısı", len(st.session_state.detections))

if service.load_error and not service.model_loaded:
    st.info(f"Model mock moda düştü: {service.load_error}")

image_path = None
if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    image_path = Path(tmp.name)
elif selected_sample != "Seçilmedi":
    image_path = SAMPLE_DIR / selected_sample

if image_path and Path(image_path).exists():
    st.image(str(image_path), caption=f"Seçilen görüntü: {Path(image_path).name}", width=420)

btn1, btn2, btn3, btn4 = st.columns(4)
with btn1:
    if st.button("▶️ Başlat", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.session_state.last_image_path = str(image_path) if image_path else None
        detections = service.detect(image_path, threshold)
        st.session_state.detections = [d.to_dict() for d in detections]
        st.success("Hata tespiti tamamlandı.")
with btn2:
    if st.button("⏸️ Durdur", use_container_width=True):
        st.session_state.running = False
        st.warning("Sistem durduruldu.")
with btn3:
    if st.button("🔄 Resetle", use_container_width=True):
        st.session_state.running = False
        st.session_state.detections = []
        st.session_state.last_image_path = None
        st.success("Ekran ve sonuçlar resetlendi.")
with btn4:
    report_clicked = st.button("📄 Rapor oluştur", use_container_width=True)

st.subheader("Tespit Sonuçları")
if st.session_state.detections:
    df = pd.DataFrame(st.session_state.detections)
    st.dataframe(
        df[["defect_type", "meter", "confidence", "timestamp", "bbox", "source"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("Henüz sonuç yok. Başlat butonuna basınca sonuçlar burada listelenir.")

if report_clicked:
    payload = build_report_payload(product_name, product_id, threshold, st.session_state.detections)
    json_path = save_json_report(payload)
    csv_path = save_csv_report(payload)
    st.success("Raporlar oluşturuldu.")
    st.json(payload)
    st.download_button("JSON raporu indir", data=json_path.read_bytes(), file_name=json_path.name, mime="application/json")
    st.download_button("CSV raporu indir", data=csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv")

with st.expander("Mimari notu"):
    st.markdown(
        """
- Arayüz: `app/main.py`
- Hata tespit katmanı: `app/detection_service.py`
- Raporlama katmanı: `app/report_service.py`
- Model dosyası: `models/best.pt`

Arayüz doğrudan model kodu içermez. Gerçek PyTorch/YOLO modeli kullanılacaksa yalnızca `DetectionService` katmanı güncellenir veya mevcut `models/best.pt` ile çalıştırılır.
        """
    )
