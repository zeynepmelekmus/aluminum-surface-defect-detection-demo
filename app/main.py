from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from detection_service import DetectionService
from report_service import build_report_payload, save_csv_report, save_json_report

st.set_page_config(page_title="Mini Yüzey Hatası Tespit", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Global Fonts & Backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Card Glassmorphism */
    .header-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .header-card h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-card p {
        font-size: 1.05rem;
        color: #9ca3af;
        margin: 0;
    }

    /* Metric Card styling */
    div[data-testid="stMetric"] {
        background: rgba(31, 41, 55, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.15);
        border-color: rgba(0, 242, 254, 0.3) !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #00f2fe !important;
        font-weight: 700 !important;
    }

    /* Elegant sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Sidebar Text Contrast Fix */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] span[data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #f3f4f6 !important;
    }
    
    /* Keep dropdown select box text dark for readability */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #0f172a !important;
    }
    
    /* Keep file uploader helper text readable */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] * {
        color: inherit;
    }
    
    /* Make slider ticks, min/max values and current value light */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] span {
        color: #e5e7eb !important;
    }

    /* Hide Streamlit default Header (Deploy button, Status) and Footer */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }

    /* Buttons micro-interactions */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Play/Start button (Primary) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        border: none !important;
        color: #0f172a !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4) !important;
    }

    /* Secondary buttons hover */
    div.stButton > button[kind="secondary"]:hover {
        border-color: #00f2fe !important;
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.05) !important;
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)
MODEL_PATH = Path("models/best.pt")
SAMPLE_DIR = Path("data/sample_images")

if "running" not in st.session_state:
    st.session_state.running = False
if "detections" not in st.session_state:
    st.session_state.detections = []
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None

# Premium Glassmorphic Header Card
st.markdown(
    """
    <div class="header-card">
        <h1>Mini Yüzey Hatası Tespit Uygulaması</h1>
    </div>
    """,
    unsafe_allow_html=True
)

def draw_bounding_boxes(image_path: Path | str, detections: list[dict]) -> Image.Image:
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Load a default font with a larger size for high readability
        try:
            font = ImageFont.load_default(size=14)
        except Exception:
            font = ImageFont.load_default()
            
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            defect_type = det.get("defect_type", "Hata")
            confidence = det.get("confidence", 0.0)
            
            # Draw bbox rectangle
            draw.rectangle([x1, y1, x2, y2], outline="#ef4444", width=3)
            
            # Label background and text
            label = f" {defect_type} ({confidence}%) "
            
            # Get text bounding box to draw background card
            try:
                l, t, r, b = draw.textbbox((0, 0), label, font=font)
                text_w = r - l
                text_h = b - t
            except Exception:
                text_w = len(label) * 8
                text_h = 16
                
            # Place label above the bounding box if it fits, otherwise inside the box
            label_y = y1 - text_h - 6
            if label_y < 0:
                label_y = y1 + 3
                
            # Draw solid background for the text label
            draw.rectangle([x1, label_y, x1 + text_w, label_y + text_h + 4], fill="#ef4444")
            
            # Draw label text in white on top of the background
            draw.text((x1, label_y + 1), label, fill="#ffffff", font=font)
            
        return img
    except Exception as exc:
        return Image.open(image_path)

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

image_path = None
if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    image_path = Path(tmp.name)
elif selected_sample != "Seçilmedi":
    image_path = SAMPLE_DIR / selected_sample

# Setup Tabs
tab_analysis, tab_analytics, tab_arch = st.tabs([
    "Canlı Analiz Ekranı", 
    "Hata Analitiği & İstatistik", 
    "Sistem Mimarisi"
])

with tab_analysis:
    col_status_1, col_status_2, col_status_3 = st.columns(3)
    col_status_1.metric("Sistem durumu", "Çalışıyor" if st.session_state.running else "Durdu")
    col_status_2.metric("Model", "Yüklendi" if service.model_loaded else "Mock/Fallback")
    col_status_3.metric("Son hata sayısı", len(st.session_state.detections))

    if service.load_error and not service.model_loaded:
        st.info(f"Model mock moda düştü: {service.load_error}")

    if image_path and Path(image_path).exists():
        if st.session_state.detections and st.session_state.running:
            img_to_show = draw_bounding_boxes(image_path, st.session_state.detections)
            st.image(img_to_show, caption=f"Analiz Edilen Görüntü: {Path(image_path).name}", width=520)
        else:
            st.image(str(image_path), caption=f"Seçilen Görüntü: {Path(image_path).name}", width=520)

    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        if st.button("Başlat", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.session_state.last_image_path = str(image_path) if image_path else None
            detections = service.detect(image_path, threshold)
            st.session_state.detections = [d.to_dict() for d in detections]
            st.success("Hata tespiti tamamlandı.")
            st.rerun()
    with btn2:
        if st.button("Durdur", use_container_width=True):
            st.session_state.running = False
            st.warning("Sistem durduruldu.")
            st.rerun()
    with btn3:
        if st.button("Resetle", use_container_width=True):
            st.session_state.running = False
            st.session_state.detections = []
            st.session_state.last_image_path = None
            st.success("Ekran ve sonuçlar resetlendi.")
            st.rerun()
    with btn4:
        report_clicked = st.button("Rapor oluştur", use_container_width=True)

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

with tab_analytics:
    st.subheader("Canlı Hata Analitikleri")
    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections)
        
        # Defect counts by type
        defect_counts = df["defect_type"].value_counts().reset_index()
        defect_counts.columns = ["Hata Tipi", "Hata Adedi"]
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Hata Türü Dağılımı")
            st.bar_chart(defect_counts.set_index("Hata Tipi"), use_container_width=True)
        with col_c2:
            st.markdown("##### Bulunan Hataların Güven Skorları (%)")
            st.line_chart(df.set_index("defect_type")["confidence"], use_container_width=True)
            
        # Summary statistics
        st.markdown("##### Hata Özet Tablosu")
        summary_df = df.groupby("defect_type").agg(
            Adet=("confidence", "count"),
            Ortalama_Guven=("confidence", "mean"),
            Min_Metre=("meter", "min"),
            Max_Metre=("meter", "max")
        ).reset_index()
        summary_df["Ortalama_Guven"] = summary_df["Ortalama_Guven"].round(2)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("Analitik grafiklerini görüntülemek için önce analiz ekranından hata tespiti (Başlat) yapmalısınız.")

with tab_arch:
    st.subheader("Yazılım Mimarisi ve Entegrasyon")
    st.markdown(
        """
        - **Arayüz Katmanı (`app/main.py`):** Görsel tasarım, sekme yönetimi, kullanıcı girişi, görsel üzerine çerçeve çizimi ve analitik grafiklerinin oluşturulmasını üstlenir.
        - **Hata Tespit Katmanı (`app/detection_service.py`):** YOLOv8 modelinin PyTorch üzerinden yüklenmesi, tahmin yapılması ve çıktının normalize edilmesinden sorumludur. Bağımlılıkların eksik olması durumunda mock tohumlama mekanizmasıyla çalışır.
        - **Raporlama Katmanı (`app/report_service.py`):** JSON ve CSV raporlarını diske kaydeder ve indirme arabelleklerini hazırlar.
        - **Birim Testleri (`tests/test_app.py`):** Servis modüllerinin veri akışını doğrular.
        """
    )
