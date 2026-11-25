import os
import base64
import time
import cv2
from collections import deque
import numpy as np
import gradio as gr
from PIL import Image, ImageOps
import tensorflow as tf
model = tf.keras.models.load_model("egitilmis_mlp_modeli.keras")
CLASS_NAMES = ["T-shirt/top", "Pantolon", "Kazak", "Elbise", "Ceket","Sandalet", "Gömlek", "Spor Ayakkabı", "Çanta", "Yarım Bot"]
def ins_img(doc_name):
    return doc_name if os.path.exists(doc_name) else None
def arka_plan_css_olustur(doc_name):
    if not os.path.exists(doc_name):
        return ""
    with open(doc_name, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = doc_name.split(".")[-1].lower()
    mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
    return f"""
    body, .gradio-container {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.25), rgba(0, 0, 0, 0.25)), url("data:image/{mime};base64,{encoded}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        min-height: 100vh !important;
        max-width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 20px !important;
    }}
    .main-card {{
        width: 66vw !important;
        max-width: 1100px !important;
        min-width: 720px !important;
        margin: auto !important;
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.45) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
    }}
    .main-card .block {{
        background: rgba(255, 255, 255, 0.70) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
    }}
    .header-row .block, .header-row .image-container, .header-row .prose {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    .logo-img img {{
        mix-blend-mode: multiply !important;
    }}
    .header-row .icon-button-wrapper, .header-row .image-buttons {{
        display: none !important;
    }}
    footer {{
        display: none !important;
    }}
    """
def preprocess_image(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img.convert("RGB"))
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    h, w = img.shape[:2]
    MD = 220
    scale = min(1.0, MD / max(h, w))
    WW, WH = max(32, int(w * scale)), max(32, int(h * scale))
    WR = cv2.resize(img, (WW, WH), interpolation=cv2.INTER_AREA)
    WB = cv2.cvtColor(WR, cv2.COLOR_RGB2BGR)
    L = cv2.cvtColor(WR, cv2.COLOR_RGB2LAB).astype(np.float32)
    CH = max(4, int(WH * 0.08))
    CW = max(4, int(WW * 0.08))
    crns = [
        np.median(L[:CH, :CW].reshape(-1, 3), axis=0),
        np.median(L[:CH, -CW:].reshape(-1, 3), axis=0),
        np.median(L[-CH:, :CW].reshape(-1, 3), axis=0),
        np.median(L[-CH:, -CW:].reshape(-1, 3), axis=0),
    ]
    j = np.array([0.75, 1.45, 1.45], dtype=np.float32)
    BD = np.min([np.linalg.norm((L - c) * j, axis=2) for c in crns], axis=0)
    GM = np.full((WH, WW), cv2.GC_PR_BGD, dtype=np.uint8)
    FT = max(18.0, float(np.percentile(BD, 62)))
    SFT = max(28.0, float(np.percentile(BD, 78)))
    BH, BW = max(2, int(WH * 0.04)), max(2, int(WW * 0.04))
    IP = np.zeros((WH, WW), dtype=bool)
    IP[BH:-BH, BW:-BW] = True
    GM[(BD > FT) & IP] = cv2.GC_PR_FGD
    SH, SW = max(3, int(WH * 0.09)), max(3, int(WW * 0.09))
    IS = np.zeros((WH, WW), dtype=bool)
    IS[SH:-SH, SW:-SW] = True
    GM[(BD > SFT) & IS] = cv2.GC_FGD
    BB = np.zeros((WH, WW), dtype=bool)
    EH, EW = max(1, int(WH*0.02)), max(1, int(WW*0.02))
    BB[:EH, :] = True
    BB[-EH:, :] = True
    BB[:, :EW] = True
    BB[:, -EW:] = True
    GM[BB & (BD < 16.0)] = cv2.GC_BGD
    GM[:CH, :CW][BD[:CH, :CW] < 20.0] = cv2.GC_BGD
    GM[:CH, -CW:][BD[:CH, -CW:] < 20.0] = cv2.GC_BGD
    GM[-CH:, :CW][BD[-CH:, :CW] < 20.0] = cv2.GC_BGD
    GM[-CH:, -CW:][BD[-CH:, -CW:] < 20.0] = cv2.GC_BGD
    HF = np.any((GM == cv2.GC_FGD) | (GM == cv2.GC_PR_FGD))
    HB = np.any((GM == cv2.GC_BGD) | (GM == cv2.GC_PR_BGD))
    if HF and HB:
        BM = np.zeros((1, 65), np.float64)
        FM = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(WB, GM, None, BM, FM, 5, cv2.GC_INIT_WITH_MASK)
            FM = np.where((GM == cv2.GC_FGD) | (GM == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            FM = np.where(BD > FT, 255, 0).astype(np.uint8)
    else:
        FM = np.where(BD > FT, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    FM = cv2.morphologyEx(FM, cv2.MORPH_CLOSE, kernel, iterations=1)
    FR = np.mean(FM > 0)
    if FR < 0.03 or FR > 0.92:
        FM = np.where(BD > np.percentile(BD, 60), 255, 0).astype(np.uint8)
    GR = cv2.cvtColor(WR, cv2.COLOR_RGB2GRAY).astype(np.float32)
    BP = GR[FM == 0]
    FP = GR[FM > 0]
    if len(FP) == 0:
        C = np.zeros((28, 28), dtype=np.uint8)
        return C.astype(np.float32) / 255.0, canvas
    BL = float(np.median(BP)) if len(BP) > 0 else 255.0
    FL = float(np.median(FP))
    p5, p95 = np.percentile(FP, [5, 95])
    if p95 - p5 < 10.0:
        norm = np.full_like(GR, 0.65)
    else:
        norm = np.clip((GR - p5) / (p95 - p5), 0.0, 1.0)
        if FL < BL:
            norm = 1.0 - norm
    M = np.zeros_like(GR, dtype=np.uint8)
    M[FM > 0] = np.clip(188.0 + norm[FM > 0] * 52.0, 188, 242).astype(np.uint8)
    YS, XS = np.where(FM > 0)
    y1, y2 = YS.min(), YS.max() + 1
    x1, x2 = XS.min(), XS.max() + 1
    CC = M[y1:y2, x1:x2]
    CCH, CCW = CC.shape
    TB = 22
    r = TB / max(CCH, CCW)
    NW = max(1, int(round(CCW * r)))
    NH = max(1, int(round(CCH * r)))
    R = cv2.resize(CC, (NW, NH), interpolation=cv2.INTER_AREA)
    C = np.zeros((28, 28), dtype=np.uint8)
    YO = (28 - NH) // 2
    XO = (28 - NW) // 2
    C[YO:YO + NH, XO:XO + NW] = R
    PL = cv2.resize(C, (280, 280), interpolation=cv2.INTER_NEAREST)
    return C.astype(np.float32) / 255.0, PL
def predict_fashion(img):
    if img is None:
        return None, "bir kıyafet fotoğrafı yükle:", None
    ST = time.time()
    IA, PI = preprocess_image(img)
    IO = np.expand_dims(IA, axis=(0, -1))
    IF = np.expand_dims(np.fliplr(IA), axis=(0, -1))
    PO = model.predict(IO, verbose=0)[0]
    PF = model.predict(IF, verbose=0)[0]
    FI = [5, 7, 9]
    FO = float(np.sum(PO[FI]))
    FF = float(np.sum(PF[FI]))
    if FF > 0.60 and FF > FO + 0.25:
        P = PF
    elif FO > 0.60 and FO > FF + 0.25:
        P = PO
    else:
        P = (PO + PF) / 2.0
    elapsed_ms = (time.time() - ST) * 1000
    CO = {CLASS_NAMES[i]: float(P[i]) for i in range(len(CLASS_NAMES))}
    info_text = f"Tahmin Süresi: {elapsed_ms:.2f} ms"
    return CO, info_text, PI
custom_css = arka_plan_css_olustur("okul_resmi.jpg")
with gr.Blocks(css=custom_css, title="MNIST Görüntü Sınıflandırma Uygulaması") as demo:
    with gr.Column(elem_classes="main-card"):
        with gr.Row(equal_height=True, elem_classes="header-row"):
            with gr.Column(scale=1, min_width=120):
                gr.Image(
                    value=ins_img("okul_logo.png"),
                    show_label=False,
                    interactive=False,
                    height=120,
                    elem_classes="logo-img"
                )
            with gr.Column(scale=4):
                gr.Markdown(
                    """
                    # MNIST Görüntü Sınıflandırma Uygulaması (CNN)
                    **İstanbul Medeniyet Üniversitesi — Endüstri Mühendisliği**  
                    **Hazırlayan:** Cemre Sırma | **Öğrenci No:** 21120606021  
                    **E-posta:** [cemre.sirma.2002@gmail.com](mailto:cemre.sirma.2002@gmail.com) | **Model:** Özel CNN Mimarisi
                    """
                )
            with gr.Column(scale=1, min_width=120):
                gr.Image(
                    value=ins_img("benim_fotografim.jpg"),
                    show_label=False,
                    interactive=False,
                    height=120
                )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Obje Fotoğrafı Yükle:",
                    height=320
                )
                with gr.Row():
                    predict_btn = gr.Button("Tahmin Et", variant="primary")
                    clear_btn = gr.Button("Ekranı Temizle")
            with gr.Column(scale=1):
                output_label = gr.Label(
                    num_top_classes=3,
                    label="Model Tahminleri (İlk 3 Sınıf)"
                )
                output_info = gr.Textbox(
                    label="Performans Bilgisi",
                    interactive=False
                )
                preview_out = gr.Image(
                    label="Modelin Gördüğü 28×28 Girdi",
                    height=130,
                    interactive=False
                )
        predict_btn.click(
            fn=predict_fashion,
            inputs=input_image,
            outputs=[output_label, output_info, preview_out]
        )
        clear_btn.click(
            fn=lambda: (None, None, "", None),
            inputs=None,
            outputs=[input_image, output_label, output_info, preview_out]
        )
if __name__ == "__main__":
    demo.launch(inbrowser=True)