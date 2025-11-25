import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import torch

# --- MODEL MODULES ---
from modules.tf_model import load_tf_model_weights, preprocess_tf
from modules.torch_model import load_torch_model, preprocess_torch

# ==========================================
# 0. GLOBAL CONFIG
# ==========================================
st.set_page_config(
    page_title="Hybrid CNN VGG16 EfficientNetB0 dengan CBAM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stFileUploader { background-color: #262730; border-radius: 5px; padding: 20px; }
    div[data-testid="metric-container"] { background-color: #262730; border: 1px solid #464b5c; padding: 10px; border-radius: 5px; }
    .suggestion-box { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Curl Virus", 
    "Healthy Leaf", "Nutrition Deficiency", "White Spot"
]

# Dictionary Saran Penanganan
DISEASE_SOLUTIONS = {
    "Bacterial Spot": """
    **Penyebab:** Infeksi bakteri *Xanthomonas campestris* yang menyerang daun, batang, dan buah.
    Penyakit ini mudah menyebar melalui percikan air hujan, alat pertanian, dan daun saling bersentuhan.

    **Gejala Utama:**
    - Bercak kecokelatan dengan tepi kuning.
    - Daun berlubang seperti disemprot peluru (“shot-hole”).
    - Penyebaran cepat setelah hujan/kelembapan tinggi.

    **Saran Penanganan:**
    1. **Sanitasi:** Pangkas dan bakar daun/bagian tanaman yang terinfeksi untuk menghentikan sumber inokulum.
    2. **Penyemprotan:** Gunakan bakterisida berbahan aktif **Tembaga (Copper Hydroxide/ Copper Oxychloride)** atau antibiotik pertanian seperti **Streptomisin** atau **Kasugamisin**.
    3. **Lingkungan:** Hindari penyiraman dari atas (*overhead watering*) untuk mengurangi kelembapan daun.
    4. **Pencegahan:** Sterilkan alat pangkas, dan jaga jarak tanam supaya sirkulasi udara optimal.
    """,

    "Cercospora Leaf Spot": """
    **Penyebab:** Jamur *Cercospora capsici* yang berkembang pada kelembapan tinggi dan sirkulasi udara buruk.

    **Gejala Utama:**
    - Bercak bulat keabu-abuan bertepi ungu atau coklat gelap.
    - Daun menguning lalu gugur.
    - Biasanya menyerang daun tua terlebih dahulu.

    **Saran Penanganan:**
    1. **Fungisida Kontak:** Gunakan Mankozeb atau Klorotalonil untuk perlindungan permukaan daun.
    2. **Fungisida Sistemik:** Gunakan Azoksistrobin, Difenokonazol, atau Propikonazol jika infeksi sudah menyebar.
    3. **Drainase:** Pastikan aliran air baik dan lahan tidak menggenang.
    4. **Gulma:** Bersihkan gulma yang dapat menjadi inang penyakit.
    5. **Rotasi Tanaman:** Hindari menanam kembali tanaman Solanaceae pada area yang terinfeksi selama 1–2 musim.
    """,

    "Curl Virus": """
    **Penyebab:** Infeksi Curl Virus yang disebarkan oleh **kutu kebul (Bemisia tabaci)** dan kadang Thrips.

    **Gejala Utama:**
    - Daun mengeriting, menebal, dan pertumbuhan kerdil.
    - Warna daun pucat atau mosaik.
    - Tidak dapat disembuhkan setelah tanaman terinfeksi.

    **Saran Penanganan:**
    1. **Eradikasi:** Cabut dan bakar tanaman sangat parah untuk mencegah penyebaran virus.
    2. **Kendalikan Vektor:** Gunakan insektisida seperti **Imidakloprid**, **Abamektin**, **Pirimiphos**, atau pasang **yellow sticky trap**.
    3. **Pencegahan:** Gunakan varietas tahan virus dan lakukan rotasi tanaman non-Solanaceae.
    4. **Fisik:** Gunakan mulsa perak (silver mulch) untuk mengurangi kedatangan kutu kebul.
    """,

    "Healthy Leaf": """
    **Status:** Tanaman dalam kondisi sehat dan fisiologinya optimal.

    **Ciri Positif:**
    - Warna hijau merata.
    - Daun kokoh tanpa bercak.
    - Pertumbuhan simetris dan stabil.

    **Saran Perawatan:**
    1. **Pertahankan:** Lanjutkan pola penyiraman yang konsisten dan pemupukan teratur.
    2. **Monitoring:** Periksa hama (kutu kebul, thrips, tungau) setiap 3–5 hari.
    3. **Nutrisi:** Berikan pupuk seimbang (NPK 16-16-16 atau setara) untuk menjaga vigor.
    4. **Lingkungan:** Pastikan tanaman mendapat cahaya cukup dan ventilasi udara baik.
    """,

    "Nutrition Deficiency": """
    **Penyebab:** Kekurangan unsur hara makro atau mikro — paling sering Nitrogen (N), Magnesium (Mg), atau Kalium (K).
    Kesalahan pH tanah, tekstur tanah buruk, atau penyerapan akar tidak optimal memperburuk gejala.

    **Gejala Berdasarkan Unsur:**
    - **Nitrogen (N):** Daun kuning merata, tanaman kerdil.
    - **Magnesium (Mg):** Klorosis di antara tulang daun (daun belang kuning).
    - **Kalium (K):** Ujung daun mengering/coklat (scorching).

    **Saran Penanganan:**
    1. **Identifikasi Unsur:** Cocokkan gejala dengan warna dan pola klorosis.
    2. **Pemupukan:** Aplikasikan pupuk daun mikro atau kocor NPK seimbang + Magnesium sulfat (Kieserite).
    3. **Perbaikan Tanah:** Cek pH tanah. Jika <5.5, berikan kapur dolomit agar nutrisi lebih mudah diserap.
    4. **Organik:** Tambahkan kompos/arang sekam untuk memperbaiki struktur tanah.
    """,

    "White Spot": """
    **Penyebab:** Biasanya jamur **Embun Tepung (Powdery Mildew / Oidium)** atau serangan tungau (mite).

    **Gejala Utama:**
    - Bercak putih seperti tepung di permukaan daun.
    - Daun kusam, melengkung, dan pertumbuhan melambat.
    - Pada tungau: titik putih kecil (stippling), daun bertekstur kasar.

    **Saran Penanganan:**
    1. **Fungisida:** Gunakan belerang (Sulfur), Miklobutanil, atau Heksakonazol untuk embun tepung.
    2. **Organik:** Gunakan **Neem Oil**, larutan susu 10%, atau larutan soda kue (1 sdt baking soda + 1 liter air + 3 tetes sabun cair).
    3. **Sirkulasi Udara:** Pangkas daun yang terlalu rapat untuk meningkatkan penetrasi cahaya.
    4. **Jika Tungau:** Tambahkan akarisida (Abamektin, Fenpyroximate).
    """
}

# ==========================================
# 1. MAIN APP LOGIC
# ==========================================

# Sidebar
st.sidebar.title("Konfigurasi") 
model_choice = st.sidebar.selectbox("Pilih Model Backend:", ("PyTorch","TensorFlow (Keras)"))   

enable_mc = st.sidebar.checkbox("Aktifkan Uncertainty (MC Dropout)", value=False)
mc_iter = st.sidebar.slider("Sampel MC Dropout", 5, 50, 10, disabled=not enable_mc)

st.title(f"Deteksi Penyakit Cabai Model {model_choice}")

# Paths
TF_PATH = "Models/final_vgg_effattnnet.keras"
PT_PATH = "Models/best_model_pytorch.pth"

# Load Selected Model
active_model = None
error_msg = ""

if model_choice == "TensorFlow (Keras)":
    active_model, error_msg = load_tf_model_weights(TF_PATH)
else:
    active_model, error_msg = load_torch_model(PT_PATH)

# UI Status
if active_model is None:
    st.error(f"Gagal memuat model {model_choice}!")
    st.code(f"Error Details: {error_msg}")
else:
    st.success(f"Model {model_choice} siap digunakan.")

# Upload & Predict
uploaded_file = st.file_uploader("Unggah Daun Cabai (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file and active_model:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        if st.button("🔍 Analisis Daun"):
            start_time = time.time()
            probs = None
            uncertainty = 0.0
            
            try:
                # --- TENSORFLOW INFERENCE ---
                if model_choice == "TensorFlow (Keras)":
                    img_tensor = preprocess_tf(image)
                    
                    if enable_mc:
                        preds = []
                        bar = st.progress(0)
                        for i in range(mc_iter):
                            # training=False triggers MCDropout logic because we set call(training=True) in class
                            p = active_model(img_tensor, training=False).numpy() 
                            preds.append(p)
                            bar.progress((i+1)/mc_iter)
                        bar.empty()
                        preds = np.array(preds)
                        probs = preds.mean(axis=0)[0]
                        uncertainty = preds.std(axis=0).mean()
                    else:
                        probs = active_model.predict(img_tensor, verbose=0)[0]

                # --- PYTORCH INFERENCE ---
                else:
                    device = next(active_model.parameters()).device
                    img_tensor = preprocess_torch(image).to(device)
                    
                    if enable_mc:
                        active_model.train()
                        preds = []
                        bar = st.progress(0)
                        with torch.no_grad():
                            for i in range(mc_iter):
                                out = active_model(img_tensor)
                                p = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
                                preds.append(p)
                                bar.progress((i+1)/mc_iter)
                        bar.empty()
                        preds = np.array(preds)
                        probs = preds.mean(axis=0)[0]
                        uncertainty = preds.std(axis=0).mean()
                        active_model.eval()
                    else:
                        active_model.eval()
                        with torch.no_grad():
                            out = active_model(img_tensor)
                            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]

                end_time = time.time()
                
                # --- RESULT DISPLAY ---
                idx = np.argmax(probs)
                label = CLASS_NAMES[idx]
                conf = probs[idx]
                
                st.subheader(f"Diagnosa: {label}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{conf:.2%}")
                m2.metric("Inference Time", f"{end_time-start_time:.3f}s")
                if enable_mc:
                    m3.metric("Uncertainty", f"{uncertainty:.4f}")
                
                st.bar_chart(dict(zip(CLASS_NAMES, probs)))
                
                # ==========================================
                # 4. DISPLAY TREATMENT SUGGESTIONS (BARU)
                # ==========================================
                st.markdown("---")
                st.subheader("Saran Penanganan")
                
                suggestion_text = DISEASE_SOLUTIONS.get(label, "Belum ada data saran penanganan untuk kelas ini.")
                
                if label == "Healthy Leaf":
                    st.success(suggestion_text)
                else:
                    st.info(suggestion_text)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
