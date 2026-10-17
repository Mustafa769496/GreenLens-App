import base64
import html
import io
import re
from datetime import datetime

import streamlit as st
from PIL import Image

from disease_detection import predict_plant
from similarity import search_similar_images
from recommendations import generate_smart_recommendations


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="GreenLens - Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Session State
# --------------------------------------------------
_DEFAULTS = {
    "page": "home",
    "prediction_result": None,
    "home_similar": None,
    "home_sig": None,
    "show_all_similar": False,
    "top_k": 6,  # تم تثبيت عدد الصور المشابهة على 6 افتراضياً
    "ar_mode": False,
    "dark_mode": False,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

IS_AR = bool(st.session_state.ar_mode)
DARK = bool(st.session_state.dark_mode)
LANG = "ar" if IS_AR else "en"


# --------------------------------------------------
# Translations (UI text)
# --------------------------------------------------
TR = {
    "en": {
        "nav_home": "🏠  Home",
        "nav_about": "ℹ️  About",
        "brand_tag": "Healthy Plants &nbsp;•&nbsp; Better Tomorrow",
        "side_footer": "Healthy plants<br>support a healthier<br>planet 🌱",
        "side_prefs": "Preferences",
        "lang_toggle": "🌐 العربية",
        "dark_toggle": "🌙 Dark mode",
        "app_title": "Plant Disease Detection &amp; Care",
        "app_sub": "Identify plant diseases, find similar plants, and get care recommendations.",
        "det_title": "Disease Detection",
        "det_sub": "Upload a plant image to detect possible diseases and get instant results.",
        "uploader_label": "Choose a clear image of a plant leaf",
        "uploader_help": "Upload a leaf image to analyze it and receive care guidance.",
        "supports": "Supports: JPG, JPEG, PNG, WEBP",
        "uploaded_caption": "Uploaded image",
        "no_image": "No image yet<br>Upload a leaf image to start",
        "image_dims": "Image size",
        "analyze": "🔍  Analyze Plant",
        "pred_result": "📋 Prediction Result",
        "pred_empty": "Click <b>Analyze Plant</b> to see the predicted class and confidence.",
        "confidence": "Confidence",
        "conf_high": "High confidence",
        "conf_med": "Medium confidence",
        "conf_low": "Low confidence",
        "low_conf_warn": "The model is not very sure about this result. Try a sharper image with the leaf clearly visible.",
        "possible_disease": "⚠ Possible disease",
        "healthy": "Healthy",
        "other_classes": "Other Possible Classes",
        "analyzing": "Analyzing the plant image...",
        "searching": "Searching for similar plants...",
        "detect_failed": "Disease detection failed",
        "similar_failed": "Similarity search failed",
        "img_error": "Could not open this image. Please upload a valid image file.",
        "care_title": "Care Recommendations",
        "care_sub": "Based on the predicted class and plant condition.",
        "care_empty": "Care guidance will appear here after you analyze a plant.",
        "plant": "Plant",
        "condition": "Condition",
        "status": "Status",
        "unknown": "Unknown",
        "what_to_do": "What you can do",
        "no_recs": "No recommendations are available for this result.",
        "disclaimer": "AI suggestions are not a confirmed diagnosis. Model scores are not necessarily calibrated probabilities.",
        "rec_failed": "Could not generate care recommendations",
        "download": "⬇️  Download report",
        "sim_title": "Similar Plants",
        "sim_sub": "Find visually similar plants from our dataset.",
        "view_all": "View All  ›",
        "show_less": "Show Less",
        "sim_empty": "Similar images will appear here after you analyze a plant.",
        "sim_none": "No similar images were found. Check that the dataset is available and indexed.",
        "similarity": "similarity",
        "about_title": "About GreenLens",
        "about_sub": "AI-assisted plant health workspace.",
        "about_text": "GreenLens detects the most likely plant disease from a leaf image, gives practical care guidance, and finds visually similar images from the local dataset.",
        "about_b1": "🔬 **Disease detection** – estimates the most likely condition from the supported classes.",
        "about_b2": "🌱 **Care recommendations** – guidance based on the model prediction.",
        "about_b3": "🖼️ **Similar plants** – visual search over the local dataset.",
        "about_warn": "AI-assisted results should be verified before making treatment decisions.",
        "footer": "GreenLens | AI-assisted plant analysis. Predictions should be verified before making treatment decisions.",
        "report_title": "GreenLens - Plant Analysis Report",
        "report_date": "Date",
        "report_class": "Predicted class",
        "report_note": "Note: AI-assisted result, verify before making treatment decisions.",
    },
    "ar": {
        "nav_home": "🏠  الرئيسية",
        "nav_about": "ℹ️  عن التطبيق",
        "brand_tag": "نباتات صحية &nbsp;•&nbsp; غدٍ أفضل",
        "side_footer": "النباتات الصحية<br>تدعم كوكبًا<br>أكثر صحة 🌱",
        "side_prefs": "التفضيلات",
        "lang_toggle": "🌐 العربية",
        "dark_toggle": "🌙 الوضع الليلي",
        "app_title": "تشخيص أمراض النباتات والعناية بها",
        "app_sub": "اكتشف أمراض النباتات، وابحث عن نباتات مشابهة، واحصل على توصيات العناية.",
        "det_title": "تشخيص المرض",
        "det_sub": "ارفع صورة للنبات لاكتشاف الأمراض المحتملة والحصول على نتيجة فورية.",
        "uploader_label": "اختر صورة واضحة لورقة النبات",
        "uploader_help": "ارفع صورة ورقة لتحليلها والحصول على إرشادات العناية.",
        "supports": "الصيغ المدعومة: JPG, JPEG, PNG, WEBP",
        "uploaded_caption": "الصورة المرفوعة",
        "no_image": "لا توجد صورة بعد<br>ارفع صورة ورقة للبدء",
        "image_dims": "حجم الصورة",
        "analyze": "🔍  تحليل النبات",
        "pred_result": "📋 نتيجة التشخيص",
        "pred_empty": "اضغط على <b>تحليل النبات</b> لعرض التصنيف المتوقع ونسبة الثقة.",
        "confidence": "الثقة",
        "conf_high": "ثقة عالية",
        "conf_med": "ثقة متوسطة",
        "conf_low": "ثقة منخفضة",
        "low_conf_warn": "النموذج غير متأكد كثيرًا من هذه النتيجة. جرّب صورة أوضح تظهر فيها الورقة بشكل جيد.",
        "possible_disease": "⚠ مرض محتمل",
        "healthy": "سليم",
        "other_classes": "تصنيفات أخرى محتملة",
        "analyzing": "جارٍ تحليل صورة النبات...",
        "searching": "جارٍ البحث عن نباتات مشابهة...",
        "detect_failed": "فشل اكتشاف المرض",
        "similar_failed": "فشل البحث عن الصور المشابهة",
        "img_error": "تعذّر فتح هذه الصورة. من فضلك ارفع ملف صورة صالح.",
        "care_title": "توصيات العناية",
        "care_sub": "بناءً على التصنيف المتوقع وحالة النبات.",
        "care_empty": "ستظهر إرشادات العناية هنا بعد تحليل النبات.",
        "plant": "النبات",
        "condition": "الحالة",
        "status": "الوضع",
        "unknown": "غير معروف",
        "what_to_do": "ماذا يمكنك أن تفعل",
        "no_recs": "لا توجد توصيات متاحة لهذه النتيجة.",
        "disclaimer": "اقتراحات الذكاء الاصطناعي ليست تشخيصًا مؤكدًا، ودرجات النموذج ليست بالضرورة احتمالات معايرة.",
        "rec_failed": "تعذّر إنشاء توصيات العناية",
        "download": "⬇️  تحميل التقرير",
        "sim_title": "نباتات مشابهة",
        "sim_sub": "اعثر على نباتات مشابهة بصريًا من قاعدة البيانات.",
        "view_all": "عرض الكل  ‹",
        "show_less": "عرض أقل",
        "sim_empty": "ستظهر الصور المشابهة هنا بعد تحليل النبات.",
        "sim_none": "لم يتم العثور على صور مشابهة. تأكد من توفر قاعدة البيانات وفهرستها.",
        "similarity": "تشابه",
        "about_title": "عن GreenLens",
        "about_sub": "مساحة عمل لصحة النباتات بمساعدة الذكاء الاصطناعي.",
        "about_text": "يكتشف GreenLens المرض الأرجح للنبات من صورة الورقة، ويقدّم إرشادات عملية للعناية، ويبحث عن صور مشابهة بصريًا من قاعدة البيانات المحلية.",
        "about_b1": "🔬 **اكتشاف المرض** – يقدّر الحالة الأرجح من التصنيفات المدعومة.",
        "about_b2": "🌱 **توصيات العناية** – إرشادات مبنية على تنبؤ النموذج.",
        "about_b3": "🖼️ **نباتات مشابهة** – بحث بصري داخل قاعدة البيانات المحلية.",
        "about_warn": "يجب التحقق من نتائج الذكاء الاصطناعي قبل اتخاذ قرارات العلاج.",
        "footer": "GreenLens | تحليل النباتات بمساعدة الذكاء الاصطناعي. يجب التحقق من التنبؤات قبل اتخاذ قرارات العلاج.",
        "report_title": "GreenLens - تقرير تحليل النبات",
        "report_date": "التاريخ",
        "report_class": "التصنيف المتوقع",
        "report_note": "ملاحظة: نتيجة بمساعدة الذكاء الاصطناعي، تحقق منها قبل اتخاذ قرارات العلاج.",
    },
}


def t(key: str) -> str:
    return TR[LANG].get(key, TR["en"][key])


# --------------------------------------------------
# Arabic names for model output & Recommendations
# --------------------------------------------------
TERMS_AR = {
    # plants
    "apple": "تفاح", "blueberry": "توت أزرق", "cherry (including sour)": "كرز",
    "cherry": "كرز", "corn (maize)": "ذرة", "corn": "ذرة", "maize": "ذرة",
    "grape": "عنب", "orange": "برتقال", "peach": "خوخ", "pepper, bell": "فلفل حلو",
    "bell pepper": "فلفل حلو", "pepper": "فلفل", "potato": "بطاطس",
    "raspberry": "توت العليق", "soybean": "فول الصويا", "squash": "قرع",
    "strawberry": "فراولة", "tomato": "طماطم",
    # conditions
    "apple scab": "جرب التفاح", "scab": "الجرب", "black rot": "العفن الأسود",
    "cedar apple rust": "صدأ التفاح الأرزي", "healthy": "سليم",
    "powdery mildew": "البياض الدقيقي",
    "cercospora leaf spot gray leaf spot": "تبقع الأوراق السركوسبوري (التبقع الرمادي)",
    "cercospora leaf spot": "تبقع الأوراق السركوسبوري", "gray leaf spot": "التبقع الرمادي",
    "common rust": "الصدأ الشائع", "northern leaf blight": "لفحة الأوراق الشمالية",
    "esca (black measles)": "الإسكا (الحصبة السوداء)",
    "leaf blight (isariopsis leaf spot)": "لفحة الأوراق (تبقع إيزاريوبسيس)",
    "haunglongbing (citrus greening)": "اخضرار الحمضيات (هوانغلونغبينغ)",
    "bacterial spot": "التبقع البكتيري", "early blight": "اللفحة المبكرة",
    "late blight": "اللفحة المتأخرة", "leaf mold": "عفن الأوراق",
    "septoria leaf spot": "تبقع الأوراق السبتوري",
    "spider mites two-spotted spider mite": "العنكبوت الأحمر ذو البقعتين",
    "two-spotted spider mite": "العنكبوت الأحمر ذو البقعتين", "spider mites": "العنكبوت الأحمر",
    "target spot": "التبقع الهدفي",
    "tomato yellow leaf curl virus": "فيروس تجعد واصفرار أوراق الطماطم",
    "yellow leaf curl virus": "فيروس تجعد واصفرار الأوراق",
    "tomato mosaic virus": "فيروس موزاييك الطماطم", "mosaic virus": "فيروس الموزاييك",
    "leaf scorch": "احتراق الأوراق",
    "uncertain prediction": "تشخيص غير مؤكد",
}
_TERMS_SORTED = sorted(TERMS_AR.items(), key=lambda kv: -len(kv[0]))

ADVICE_AR = {
    "inspect lower leaves for dark spots with concentric rings":
        "افحص الأوراق السفلى بحثًا عن بقع داكنة تحتوي على حلقات مركزية.",
    "inspect leaves and fruits for olive-colored or dark lesions":
        "افحص الأوراق والثمار بحثًا عن بقع زيتونية اللون أو داكنة.",
    "remove severely affected leaves when appropriate":
        "أزل الأوراق المصابة بشدة متى كان ذلك مناسبًا.",
    "remove fallen infected leaves where practical":
        "أزل الأوراق المصابة المتساقطة متى كان ذلك عمليًا.",
    "avoid overhead watering where practical":
        "تجنب الري من الأعلى فوق الأوراق متى كان ذلك عمليًا.",
    "monitor whether symptoms spread":
        "راقب ما إذا كانت الأعراض تنتشر إلى باقي الأجزاء.",
    "improve airflow through appropriate pruning":
        "حسّن التهوية بين النباتات عن طريق التقليم المناسب.",
    "consult a local agricultural specialist about suitable disease management":
        "استشر مختصًا زراعيًا محليًا بشأن الإدارة المناسبة للمرض.",
    "take a clear photo in good lighting":
        "التقط صورة واضحة في إضاءة جيدة.",
    "focus on the affected leaf and avoid blurry images":
        "ركز على الورقة المصابة وتجنب الصور الضبابية.",
    "try another image showing the symptoms clearly":
        "جرب صورة أخرى تظهر فيها الأعراض بوضوح.",
    "consult an agricultural specialist if symptoms persist":
        "استشر أخصائيًا زراعيًا إذا استمرت الأعراض.",
    "low model confidence":
        "ثقة منخفضة بالنموذج",
    "possible disease": "مرض محتمل",
    "healthy": "سليم",
}


def pretty(name) -> str:
    return re.sub(r"\s+", " ", str(name).replace("_", " ")).strip()


def ar_term(text) -> str:
    text = pretty(text)
    if not IS_AR:
        return text
    out = text
    for en, ar in _TERMS_SORTED:
        out = re.sub(
            rf"(?<![A-Za-z]){re.escape(en)}(?![A-Za-z])", lambda _m, a=ar: a, out, flags=re.IGNORECASE
        )
    return out


def ar_advice(text) -> str:
    if not IS_AR:
        return str(text)
    clean_text = str(text).strip().rstrip(".").lower()
    return ADVICE_AR.get(clean_text, str(text))


# --------------------------------------------------
# Theme (light / dark) + RTL + CSS
# --------------------------------------------------
PALETTE = {
    False: {
        "bg": "#f3f8f4", "card": "#ffffff", "ink": "#14231a", "muted": "#607166",
        "line": "#e0ece3", "soft": "#f3f8f4", "acc-soft": "#dcfce7", "acc-text": "#166534",
        "ring": "#e3efe6", "warn-bg": "#fff1e0", "warn-fg": "#c2570c",
        "bad-bg": "#fde8e8", "bad-fg": "#b42318", "mid-bg": "#fef3c7", "mid-fg": "#92400e",
        "shadow": "rgba(20, 83, 45, .07)", "input": "#fafdfb", "dash": "#9ac8a6",
    },
    True: {
        "bg": "#0b1410", "card": "#121e18", "ink": "#e7f2eb", "muted": "#94a99c",
        "line": "#22342a", "soft": "#0e1913", "acc-soft": "#15351f", "acc-text": "#6ee7a0",
        "ring": "#22342a", "warn-bg": "#3a2410", "warn-fg": "#fdba74",
        "bad-bg": "#3b1616", "bad-fg": "#fca5a5", "mid-bg": "#3a2f0b", "mid-fg": "#fcd34d",
        "shadow": "rgba(0, 0, 0, .40)", "input": "#0e1913", "dash": "#2f5a3e",
    },
}
FONT = "'Cairo','Segoe UI',Tahoma,sans-serif" if IS_AR else "'Inter','Segoe UI',Roboto,sans-serif"

ROOT_VARS = ":root{" + ";".join(f"--{k}:{v}" for k, v in PALETTE[DARK].items()) + f";--font:{FONT}" + "}"

_MAIN = (".main", '[data-testid="stMain"]')
MAIN_TEXT_SELECTORS = ", ".join(f"{p} {s}" for p in _MAIN for s in ("p", "li", "label", "h1", "h2", "h3", "h4", "h5"))
BLOCK = '[data-testid="stMainBlockContainer"], .main .block-container'

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Cairo:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, p, label, button, input, textarea, h1, h2, h3, h4, h5,
[data-testid="stSidebar"] { font-family: var(--font) !important; }

.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] * { color: var(--ink) !important; }
[data-testid="stCaptionContainer"], .stCaption, [data-testid="stImageCaption"] { color: var(--muted) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f4a27 0%, #0b3a20 55%, #082c18 100%);
    border-right: 0;
}
[data-testid="stSidebar"] * { color: #e8f5ec; }
.brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .2rem; }
.brand .logo { font-size: 2.1rem; line-height: 1; }
.brand .name { font-size: 1.8rem; font-weight: 800; letter-spacing: -.02em; color: #fff; }
.brand-tag { font-size: .8rem; color: #b6d9c1 !important; margin: 0 0 1.4rem .2rem; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100%; padding: .7rem .9rem; border-radius: 12px; cursor: pointer; transition: background .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(100deg, #1b7a3f, #15803d);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.10);
}
[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 1rem; font-weight: 600; }
.side-sep { height: 1px; background: rgba(255,255,255,.12); margin: 1.2rem 0 .9rem; }
.side-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: #8fc3a0 !important; margin-bottom: .4rem; }
.side-footer { margin-top: 2.2rem; color: #b6d9c1 !important; font-size: .92rem; line-height: 1.55; }

/* ---------- Top bar ---------- */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--card); border: 1px solid var(--line); border-radius: 18px;
    padding: .9rem 1.3rem; margin-bottom: 1rem; box-shadow: 0 4px 16px var(--shadow);
}
.tb-left { display: flex; align-items: center; gap: .9rem; }
.tb-icon { width: 46px; height: 46px; border-radius: 14px; background: var(--acc-soft);
           display: grid; place-items: center; font-size: 1.5rem; }
.tb-title { font-size: 1.3rem; font-weight: 800; color: var(--ink); line-height: 1.25; }
.tb-sub { font-size: .9rem; color: var(--muted); }
.tb-user { display: flex; align-items: center; gap: .55rem; font-weight: 600; color: var(--ink); }
.tb-user .av { width: 38px; height: 38px; border-radius: 50%; background: #166534;
               display: grid; place-items: center; color: #fff; }

/* ---------- Cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card); border: 1px solid var(--line) !important; border-radius: 20px !important;
    box-shadow: 0 6px 22px var(--shadow);
}
.card-head { display: flex; align-items: center; gap: .85rem; margin-bottom: .8rem; }
.card-head .ic { width: 48px; height: 48px; border-radius: 14px; background: var(--acc-soft);
                 display: grid; place-items: center; font-size: 1.4rem; flex-shrink: 0; }
.card-head .t { font-size: 1.4rem; font-weight: 800; color: var(--ink); line-height: 1.2; }
.card-head .s { font-size: .9rem; color: var(--muted); }

/* ---------- Inputs ---------- */
[data-testid="stFileUploader"] section { background: var(--input); border: 1.5px dashed var(--dash); border-radius: 16px; }
[data-testid="stFileUploader"] section * { color: var(--muted) !important; }
[data-testid="stFileUploader"] button { background: #166534 !important; border: 0; border-radius: 10px; }
[data-testid="stFileUploader"] button * { color: #fff !important; }
[data-testid="stImage"] img { border-radius: 16px; }
[data-testid="stAlert"] { background: var(--soft) !important; border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stSlider"] * { color: var(--ink); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; min-height: 2.9rem; font-weight: 700;
    background: var(--card); color: var(--ink); border: 1px solid var(--line);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button p, .stDownloadButton > button p { color: inherit !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #166534, #1a8b46); border: 0; color: #fff;
    box-shadow: 0 6px 16px rgba(22, 101, 52, .25);
}
.stButton > button[kind="primary"] p { color: #fff !important; }
.stButton > button:hover:not(:disabled), .stDownloadButton > button:hover { transform: translateY(-1px); }
.stButton > button:disabled { opacity: .55; }

/* ---------- Prediction ---------- */
.pred-box { display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            background: var(--soft); border: 1px solid var(--line); border-radius: 16px;
            padding: .9rem 1.1rem; margin-top: .4rem; }
.pred-left { display: flex; align-items: center; gap: .9rem; min-width: 0; }
.pred-left img { width: 74px; height: 74px; object-fit: cover; border-radius: 12px; }
.pred-name { font-size: 1.3rem; font-weight: 800; color: var(--ink); overflow-wrap: anywhere; }
.pred-plant { color: var(--muted); font-size: .95rem; margin-bottom: .35rem; }
.badge { display: inline-block; padding: .18rem .7rem; border-radius: 999px; font-size: .78rem;
         font-weight: 700; margin-inline-end: .35rem; }
.badge.warn { background: var(--warn-bg); color: var(--warn-fg); }
.badge.ok { background: var(--acc-soft); color: var(--acc-text); }
.badge.mid { background: var(--mid-bg); color: var(--mid-fg); }
.badge.low { background: var(--bad-bg); color: var(--bad-fg); }
.ring-wrap { text-align: center; flex-shrink: 0; }
.ring { width: 96px; height: 96px; border-radius: 50%; display: grid; place-items: center; }
.ring-in { width: 76px; height: 76px; border-radius: 50%; background: var(--card);
           display: grid; place-items: center; font-size: 1.25rem; font-weight: 800; color: var(--acc-text); }
.ring-cap { font-size: .8rem; color: var(--muted); margin-top: .25rem; }
.sec-title { font-weight: 800; margin: 1rem 0 .5rem; color: var(--ink); }
.other-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; }
.other-item { border: 1px solid var(--line); border-radius: 12px; padding: .65rem .8rem; background: var(--card); }
.other-item .n { font-weight: 650; font-size: .92rem; overflow-wrap: anywhere; color: var(--ink); }
.other-item .p { color: var(--muted); font-size: .88rem; }

/* ---------- Care ---------- */
.info-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; background: var(--soft);
            border: 1px solid var(--line); border-radius: 14px; padding: .9rem 1rem; margin-bottom: 1rem; }
.info-item .l { color: var(--muted); font-size: .8rem; }
.info-item .v { font-weight: 800; font-size: 1rem; color: var(--ink); overflow-wrap: anywhere; }
.care-title { font-size: 1.15rem; font-weight: 800; margin-bottom: .6rem; color: var(--ink); }
.care-item { display: flex; gap: .7rem; align-items: flex-start; margin-bottom: .75rem; line-height: 1.5; color: var(--ink); }
.care-item .ck { width: 22px; height: 22px; border-radius: 50%; background: #15803d; color: #fff;
                 display: grid; place-items: center; font-size: .75rem; flex-shrink: 0; margin-top: .15rem; }
.note-box { background: var(--soft); border: 1px solid var(--line); border-radius: 14px;
            padding: .85rem 1rem; color: var(--muted); font-size: .92rem; margin: 1rem 0; }
.empty-box { background: var(--soft); border: 1px dashed var(--dash); border-radius: 14px;
             padding: 1.4rem 1rem; color: var(--muted); text-align: center; }

/* ---------- Similar ---------- */
.sim-name { font-weight: 700; font-size: .88rem; margin-top: .35rem; overflow-wrap: anywhere; color: var(--ink); }
.sim-score { display: inline-block; margin-top: .25rem; background: var(--acc-soft); color: var(--acc-text);
             border-radius: 999px; padding: .1rem .6rem; font-size: .78rem; font-weight: 600; }

@media (max-width: 900px) {
    .info-row, .other-grid { grid-template-columns: 1fr; }
    .pred-box { flex-direction: column; align-items: flex-start; }
}
"""

DYNAMIC_CSS = f"""
{BLOCK} {{ max-width: 1400px; padding: 1rem 1.8rem 2rem; }}
{MAIN_TEXT_SELECTORS} {{ color: var(--ink); }}
"""

RTL_CSS = f"""
{BLOCK}, [data-testid="stSidebarContent"] {{ direction: rtl; text-align: right; }}
[data-testid="stFileUploader"], [data-testid="stImage"], [data-testid="stSlider"] {{ direction: ltr; text-align: left; }}
[data-testid="stSidebar"] div[role="radiogroup"] label {{ justify-content: flex-start; }}
""" if IS_AR else ""

st.markdown(f"<style>{ROOT_VARS}{STATIC_CSS}{DYNAMIC_CSS}{RTL_CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value))


def image_to_data_uri(img: Image.Image, size: int = 120) -> str:
    thumb = img.copy()
    thumb.thumbnail((size, size))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def card_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="card-head"><div class="ic">{icon}</div>'
        f'<div><div class="t">{esc(title)}</div><div class="s">{esc(subtitle)}</div></div></div>',
        unsafe_allow_html=True,
    )


def open_image(uploaded):
    try:
        return Image.open(uploaded).convert("RGB")
    except Exception:
        st.error(t("img_error"))
        return None


def toggle_view_all():
    st.session_state.show_all_similar = not st.session_state.show_all_similar


def confidence_level(confidence: float):
    if confidence >= 0.85:
        return "ok", t("conf_high")
    if confidence >= 0.60:
        return "mid", t("conf_med")
    return "low", t("conf_low")


def safe_recommendation(prediction):
    try:
        return generate_smart_recommendations(
            class_name=prediction["predicted_class"],
            confidence=float(prediction["confidence"]),
        ), None
    except Exception as error:
        return None, error


def build_report(prediction, recommendation) -> str:
    conf = float(prediction["confidence"])
    rec = recommendation or {}
    lines = [
        t("report_title"),
        "=" * 40,
        f"{t('report_date')}: {datetime.now():%Y-%m-%d %H:%M}",
        f"{t('plant')}: {ar_term(rec.get('plant', t('unknown')))}",
        f"{t('condition')}: {ar_term(rec.get('condition', t('unknown')))}",
        f"{t('report_class')}: {pretty(prediction['predicted_class'])}",
        f"{t('confidence')}: {conf:.1%}",
        "",
        f"{t('what_to_do')}:",
    ]
    lines += [f"- {ar_advice(a)}" for a in rec.get("recommendations", [])]
    lines += ["", t("report_note")]
    return "\n".join(lines)


def render_similar_grid(results, per_row: int = 6) -> None:
    for start in range(0, len(results), per_row):
        row = results[start:start + per_row]
        cols = st.columns(per_row, gap="small")
        for col, item in zip(cols, row):
            with col:
                st.image(item["path"], use_container_width=True)
                score = max(0.0, min(1.0, float(item["similarity"])))
                st.markdown(
                    f'<div class="sim-name">🍃 {esc(ar_term(item["class_name"]))}</div>'
                    f'<div class="sim-score">{score:.0%} {esc(t("similarity"))}</div>',
                    unsafe_allow_html=True,
                )


# --------------------------------------------------
# Sidebar (تم حُذف Settings)
# --------------------------------------------------
NAV_LABELS = {"home": t("nav_home"), "about": t("nav_about")}

with st.sidebar:
    st.markdown(
        f'<div class="brand"><span class="logo">🌿</span><span class="name">GreenLens</span></div>'
        f'<div class="brand-tag">{t("brand_tag")}</div>',
        unsafe_allow_html=True,
    )
    st.radio(
        "Navigation",
        list(NAV_LABELS.keys()),
        key="page",
        format_func=lambda k: NAV_LABELS[k],
        label_visibility="collapsed",
    )
    st.markdown(f'<div class="side-sep"></div><div class="side-label">{t("side_prefs")}</div>', unsafe_allow_html=True)
    st.toggle(t("dark_toggle"), key="dark_mode")
    st.toggle(t("lang_toggle"), key="ar_mode")
    st.markdown(f'<div class="side-footer">{t("side_footer")}</div>', unsafe_allow_html=True)

selected_page = st.session_state.page


# --------------------------------------------------
# Top bar
# --------------------------------------------------
st.markdown(
    f"""
    <div class="topbar">
        <div class="tb-left">
            <div class="tb-icon">🌿</div>
            <div>
                <div class="tb-title">{t("app_title")}</div>
                <div class="tb-sub">{t("app_sub")}</div>
            </div>
        </div>
        <div class="tb-user"><span class="av">🍃</span> GreenLens</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PAGE: Home
# ==================================================
if selected_page == "home":
    left_col, right_col = st.columns([1.15, 1], gap="large")

    # ---------------- Disease Detection card ----------------
    with left_col:
        with st.container(border=True):
            card_header("🔬", t("det_title"), t("det_sub"))

            preview_col, upload_col = st.columns(2, gap="medium")
            with upload_col:
                uploaded_file = st.file_uploader(
                    t("uploader_label"),
                    type=["jpg", "jpeg", "png", "webp"],
                    help=t("uploader_help"),
                    key="home_image_upload",
                    label_visibility="collapsed",
                )
                st.caption(t("supports"))

            image = None
            if uploaded_file is not None:
                sig = (uploaded_file.name, uploaded_file.size)
                if st.session_state.home_sig != sig:
                    st.session_state.home_sig = sig
                    st.session_state.prediction_result = None
                    st.session_state.home_similar = None
                    st.session_state.show_all_similar = False
                image = open_image(uploaded_file)
            else:
                st.session_state.home_sig = None
                st.session_state.prediction_result = None
                st.session_state.home_similar = None

            with preview_col:
                if image is not None:
                    st.image(image, caption=f"{t('uploaded_caption')} · {image.width}×{image.height}", use_container_width=True)
                else:
                    st.markdown(f'<div class="empty-box">🌿<br>{t("no_image")}</div>', unsafe_allow_html=True)

            analyze = st.button(
                t("analyze"),
                key="detect_disease_button",
                type="primary",
                use_container_width=True,
                disabled=image is None,
            )

            if analyze and image is not None:
                try:
                    with st.spinner(t("analyzing")):
                        st.session_state.prediction_result = predict_plant(image)
                except Exception as error:
                    st.session_state.prediction_result = None
                    st.error(f"{t('detect_failed')}: {error}")
                try:
                    with st.spinner(t("searching")):
                        st.session_state.home_similar = search_similar_images(
                            image, top_k=6  # العرض الافتراضي الثابت 6 صور
                        )
                except Exception as error:
                    st.session_state.home_similar = []
                    st.error(f"{t('similar_failed')}: {error}")

            prediction = st.session_state.prediction_result
            st.markdown(f'<div class="sec-title">{t("pred_result")}</div>', unsafe_allow_html=True)
            if prediction is None:
                st.markdown(f'<div class="empty-box">{t("pred_empty")}</div>', unsafe_allow_html=True)
            else:
                predicted_class = prediction["predicted_class"]
                confidence = max(0.0, min(1.0, float(prediction["confidence"])))
                is_healthy = "healthy" in str(predicted_class).lower()

                rec, _ = safe_recommendation(prediction)
                plant_name = ar_term(rec.get("plant", "")) if rec else ""
                if IS_AR and rec and rec.get("condition"):
                    headline = ar_term(rec["condition"])
                else:
                    headline = ar_term(predicted_class)

                thumb = image_to_data_uri(image) if image is not None else ""
                thumb_html = f'<img src="{thumb}">' if thumb else ""
                status_badge = (
                    f'<span class="badge ok">{t("healthy")}</span>'
                    if is_healthy
                    else f'<span class="badge warn">{t("possible_disease")}</span>'
                )
                level_cls, level_txt = confidence_level(confidence)
                degrees = confidence * 360
                st.markdown(
                    f'<div class="pred-box">'
                    f'<div class="pred-left">{thumb_html}<div>'
                    f'<div class="pred-name">{esc(headline)}</div>'
                    f'<div class="pred-plant">{esc(plant_name)}</div>'
                    f'{status_badge}<span class="badge {level_cls}">{esc(level_txt)}</span></div></div>'
                    f'<div class="ring-wrap">'
                    f'<div class="ring" style="background: conic-gradient(#16a34a {degrees:.0f}deg, var(--ring) 0deg);">'
                    f'<div class="ring-in">{confidence:.0%}</div></div>'
                    f'<div class="ring-cap">{t("confidence")}</div></div></div>',
                    unsafe_allow_html=True,
                )
                if level_cls == "low":
                    st.warning(t("low_conf_warn"))

                others = [
                    item for item in prediction.get("top_predictions", [])
                    if item["class_name"] != predicted_class
                ][:3]
                if others:
                    st.markdown(f'<div class="sec-title">{t("other_classes")}</div>', unsafe_allow_html=True)
                    tiles = "".join(
                        f'<div class="other-item"><div class="n">{esc(ar_term(item["class_name"]))}</div>'
                        f'<div class="p">{max(0.0, min(1.0, float(item["confidence"]))):.0%}</div></div>'
                        for item in others
                    )
                    st.markdown(f'<div class="other-grid">{tiles}</div>', unsafe_allow_html=True)

    # ---------------- Care Recommendations card ----------------
    with right_col:
        with st.container(border=True):
            card_header("🌱", t("care_title"), t("care_sub"))
            prediction = st.session_state.prediction_result
            if prediction is None:
                st.markdown(f'<div class="empty-box">{t("care_empty")}</div>', unsafe_allow_html=True)
            else:
                recommendation, rec_error = safe_recommendation(prediction)
                if rec_error is not None:
                    st.error(f"{t('rec_failed')}: {rec_error}")
                else:
                    is_healthy = "healthy" in str(prediction["predicted_class"]).lower()
                    status_text = t("healthy") if is_healthy else t("possible_disease").replace("⚠ ", "")
                    plant_v = ar_term(recommendation.get("plant", t("unknown")))
                    cond_v = ar_term(recommendation.get("condition", t("unknown")))
                    st.markdown(
                        f'<div class="info-row">'
                        f'<div class="info-item"><div class="l">{t("plant")}</div><div class="v">🌿 {esc(plant_v)}</div></div>'
                        f'<div class="info-item"><div class="l">{t("condition")}</div><div class="v">{esc(cond_v)}</div></div>'
                        f'<div class="info-item"><div class="l">{t("status")}</div><div class="v">{esc(status_text)}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if recommendation.get("status"):
                        rec_status = ar_advice(recommendation["status"])
                        st.info(rec_status)

                    st.markdown(f'<div class="care-title">{t("what_to_do")}</div>', unsafe_allow_html=True)
                    advice = recommendation.get("recommendations", [])
                    if advice:
                        st.markdown(
                            "".join(
                                f'<div class="care-item"><div class="ck">✓</div><div>{esc(ar_advice(item))}</div></div>'
                                for item in advice
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(t("no_recs"))

                    disclaimer = recommendation.get("disclaimer") if not IS_AR else None
                    st.markdown(
                        f'<div class="note-box">💡 {esc(disclaimer or t("disclaimer"))}</div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        t("download"),
                        data=build_report(prediction, recommendation),
                        file_name="greenlens_report.txt",
                        mime="text/plain",
                        key="download_report_button",
                        use_container_width=True,
                    )

    # ---------------- Similar Plants card ----------------
    with st.container(border=True):
        home_similar = st.session_state.home_similar
        PREVIEW_COUNT = 6
        head_col, btn_col = st.columns([6, 1])
        with head_col:
            card_header("🖼️", t("sim_title"), t("sim_sub"))
        with btn_col:
            if home_similar and len(home_similar) > PREVIEW_COUNT:
                label = t("show_less") if st.session_state.show_all_similar else t("view_all")
                st.button(label, key="view_all_button", on_click=toggle_view_all, use_container_width=True)

        if home_similar is None:
            st.markdown(f'<div class="empty-box">{t("sim_empty")}</div>', unsafe_allow_html=True)
        elif home_similar:
            visible = home_similar if st.session_state.show_all_similar else home_similar[:PREVIEW_COUNT]
            render_similar_grid(visible, per_row=PREVIEW_COUNT)
        else:
            st.info(t("sim_none"))


# ==================================================
# PAGE: About
# ==================================================
else:
    with st.container(border=True):
        card_header("ℹ️", t("about_title"), t("about_sub"))
        st.write(t("about_text"))
        st.markdown(f"- {t('about_b1')}\n- {t('about_b2')}\n- {t('about_b3')}")
        st.warning(t("about_warn"))


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(t("footer"))