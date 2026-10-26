import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------------------------------------
# 1. PAGE CONFIG & FUTURISTIC CLINIC STYLING
# -------------------------------------------------------------
st.set_page_config(page_title="VetAI Nexus - Image First Clinic", page_icon="🐾", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
.step-badge { background: #111827; border: 1px solid #374151; border-radius: 20px; padding: 6px 18px; font-size: 0.85rem; font-weight: 700; color: #9ca3af; }
.step-badge-active { background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%); color: #ffffff; border: none; box-shadow: 0 0 15px rgba(6, 182, 212, 0.5); }

.scan-container {
position: relative;
display: inline-block;
border-radius: 12px;
overflow: hidden;
border: 2px solid #06b6d4;
box-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
}
.scan-line {
position: absolute;
top: 0; left: 0;
width: 100%; height: 4px;
background: #00f2fe;
box-shadow: 0 0 15px #00f2fe;
animation: scanAnim 2s infinite linear;
}
@keyframes scanAnim { 0% { top: 0%; } 50% { top: 95%; } 100% { top: 0%; } }

.rx-card { background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.7); }
.dose-badge { background: #0284c7; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: bold; font-family: monospace; }
.demo-warning { background: #451a03; border: 1px solid #d97706; color: #fbbf24; padding: 10px 16px; border-radius: 10px; font-size: 0.85rem; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. REAL IMAGE MODEL LOADING (Cow / Buffalo only, for now)
# -------------------------------------------------------------
IMAGE_MODEL_PATH = "cow_disease_model.pth"
CLASSES_PATH = "cow_disease_classes.json"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ছবির ফোল্ডার-ক্লাস নাম -> UI-তে দেখানো নাম ও মেডিসিন ম্যাপিং কী
IMAGE_CLASS_INFO = {
    "lumpy": {
        "display": "LUMPY SKIN DISEASE (LSD)",
        "med_key": "lumpy skin disease",
    },
    "foot-and-mouth": {
        "display": "FOOT AND MOUTH DISEASE (FMD)",
        "med_key": "foot and mouth disease (fmd)",
    },
    "healthy": {
        "display": "সুস্থ প্রাণী (HEALTHY - কোনো রোগ সনাক্ত হয়নি)",
        "med_key": None,
    },
}


@st.cache_resource
def load_cow_image_model():
    """যদি ট্রেইনড মডেল ফাইল পাওয়া যায়, সেটা লোড করে; না পেলে None রিটার্ন করে।"""
    if not (os.path.exists(IMAGE_MODEL_PATH) and os.path.exists(CLASSES_PATH)):
        return None, None

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)  # keys আসে string হিসেবে ("0","1",...)

    num_classes = len(idx_to_class)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(IMAGE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, idx_to_class


def predict_with_real_model(model, idx_to_class, pil_img):
    """আসল ট্রেইনড মডেল দিয়ে ছবি থেকে প্রেডিকশন করে। রিটার্ন করে (folder_class_name, confidence%)."""
    x = infer_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    folder_class = idx_to_class[str(pred_idx.item())]
    return folder_class, conf.item() * 100


# ভ্যালিডেশন সেটে মাপা প্রকৃত accuracy — নতুন করে ট্রেনিং করলে এই সংখ্যাগুলো আপডেট করে দিন
COW_MODEL_VAL_ACCURACY = 94.9        # cow_disease_model.pth (v2 fine-tune, held-out ভ্যালিডেশনে মাপা)
DOG_CAT_MODEL_VAL_ACCURACY = 50.8    # dog_cat_disease_model.pth (v2 fine-tune; Top-3 accuracy ৭০.৪%)

cow_model, cow_idx_to_class = load_cow_image_model()
REAL_MODEL_AVAILABLE = cow_model is not None

# -------------------------------------------------------------
# 2b. REAL IMAGE MODEL LOADING (Dog / Cat, multi-disease)
# -------------------------------------------------------------
DOG_CAT_MODEL_PATH = "dog_cat_disease_model.pth"
DOG_CAT_CLASSES_PATH = "dog_cat_disease_classes.json"


@st.cache_resource
def load_dog_cat_image_model():
    if not (os.path.exists(DOG_CAT_MODEL_PATH) and os.path.exists(DOG_CAT_CLASSES_PATH)):
        return None, None
    with open(DOG_CAT_CLASSES_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(DOG_CAT_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, idx_to_class


dog_cat_model, dog_cat_idx_to_class = load_dog_cat_image_model()
DOG_CAT_MODEL_AVAILABLE = dog_cat_model is not None

# -------------------------------------------------------------
# 2c. TABULAR SYMPTOM MODEL LOADING (Cow / Buffalo / Goat / Sheep)
# -------------------------------------------------------------
TABULAR_MODEL_PATH = "livestock_symptom_model.pkl"


@st.cache_resource
def load_tabular_model():
    if not os.path.exists(TABULAR_MODEL_PATH):
        return None
    import joblib
    return joblib.load(TABULAR_MODEL_PATH)


tabular_data = load_tabular_model()
TABULAR_MODEL_AVAILABLE = tabular_data is not None
if TABULAR_MODEL_AVAILABLE:
    TABULAR_ACCURACY = tabular_data["test_accuracy"] * 100
    SYMPTOM_OPTIONS = tabular_data["symptom_options"]
    # lumpy ও pneumonia-র উপসর্গ ডেটাসেটে হুবহু এক, তাই মডেল ওদের একটি যৌথ
    # ক্লাস হিসেবে শেখে। এই ক্লাসটা পরে ছবি + শ্বাসলক্ষণ দিয়ে ভাঙা হয়।
    MERGED_CLASS = tabular_data.get("merged_class", "lumpy_or_pneumonia")
    PNEUMONIA_SIGNS = tabular_data.get("pneumonia_signs", [])
    PNEUMONIA_THRESHOLD = tabular_data.get("pneumonia_threshold", 4)
else:
    TABULAR_ACCURACY = 0
    SYMPTOM_OPTIONS = []
    MERGED_CLASS = "lumpy_or_pneumonia"
    PNEUMONIA_SIGNS = []
    PNEUMONIA_THRESHOLD = 4


def predict_with_tabular_model(animal, age, temp, sym1, sym2, sym3):
    """Symptom-ভিত্তিক প্রেডিকশন। রিটার্ন করে সব ক্লাসের probability সহ একটা dict।"""
    import pandas as pd
    model = tabular_data["model"]
    encoders = tabular_data["encoders"]
    feature_order = tabular_data["feature_order"]

    row = {
        "Animal": encoders["Animal"].transform([animal])[0],
        "Age": age,
        "Temperature": temp,
        "Symptom 1": encoders["Symptom 1"].transform([sym1])[0],
        "Symptom 2": encoders["Symptom 2"].transform([sym2])[0],
        "Symptom 3": encoders["Symptom 3"].transform([sym3])[0],
    }
    X = pd.DataFrame([row])[feature_order]
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    return dict(zip(classes, proba))


# -------------------------------------------------------------
# 2d. PET TEXT-SYMPTOM MODEL (Dog / Cat) — মালিকের লেখা লক্ষণ থেকে সমস্যার ধরন
# -------------------------------------------------------------
# কুকুর/বিড়ালে ছবির মডেলের accuracy ~৫০%, কিন্তু এই টেক্সট মডেলের ~৭৯% —
# আর বমি/ডায়রিয়া/খোঁড়ানোর মতো অনেক সমস্যা ছবিতে দেখাই যায় না। তাই দুটো
# স্বাধীন প্রমাণ হিসেবে পাশাপাশি দেখানো হয়।
PET_TEXT_MODEL_PATH = "pet_text_symptom_model.pkl"


@st.cache_resource
def load_pet_text_model():
    if not os.path.exists(PET_TEXT_MODEL_PATH):
        return None
    import joblib
    return joblib.load(PET_TEXT_MODEL_PATH)


pet_text_data = load_pet_text_model()
PET_TEXT_AVAILABLE = pet_text_data is not None
PET_TEXT_ACCURACY = pet_text_data["test_accuracy"] * 100 if PET_TEXT_AVAILABLE else 0


def predict_pet_text(description):
    """মালিকের লেখা বর্ণনা থেকে সমস্যার ধরন। রিটার্ন: {ক্লাস: probability}।"""
    pipe = pet_text_data["model"]
    proba = pipe.predict_proba([description])[0]
    return dict(zip(pipe.classes_, proba))


# ছবির ২২টা নির্দিষ্ট রোগ এই ৫টা বিস্তৃত শ্রেণির ভেতরেই পড়ে। দুটো মডেল একমত
# হলে আত্মবিশ্বাস বাড়ে, অমিল হলে ব্যবহারকারীকে সতর্ক করা হয়।
DOG_CAT_DISEASE_TO_CATEGORY = {
    "Mange in Dog": "Skin Irritations",
    "Ringworm in Cat": "Skin Irritations",
    "Scabies in Cat": "Skin Irritations",
    "Skin Allergy in Cat": "Skin Irritations",
    "Skin Allergy in Dog": "Skin Irritations",
    "Hot Spots in Dog": "Skin Irritations",
    "Fungal Infection in Cat": "Skin Irritations",
    "Fungal Infection in Dog": "Skin Irritations",
    "Ear Mites in Cat": "Ear Infections",
    "Worm Infection in Cat": "Parasites",
    "Worm Infection in Dog": "Parasites",
    "Tick Infestation in Dog": "Parasites",
    "Parvovirus in Dog": "Digestive Issues",
    "Feline Panleukopenia": "Digestive Issues",
}

# এই রোগগুলো অত্যন্ত জরুরি ও বিপজ্জনক (কিছু zoonotic/দ্রুত মারাত্মক) —
# বাড়িতে চিকিৎসার বদলে সরাসরি জরুরি ভেট/কর্তৃপক্ষের কাছে যাওয়ার পরামর্শ দেখানো হয়
URGENT_REFERRAL_DISEASES = {
    "anthrax": "Anthrax মানুষের মধ্যেও ছড়াতে পারে (zoonotic) এবং এটি একটি বাধ্যতামূলক-রিপোর্টযোগ্য রোগ। "
               "বাড়িতে কোনো চিকিৎসার চেষ্টা করবেন না — মৃত/অসুস্থ পশু স্পর্শ এড়িয়ে চলুন, "
               "সাথে সাথে নিকটস্থ প্রাণিসম্পদ অফিস ও ভেটেরিনারি ডাক্তারকে জানান।",
    "blackleg": "Blackleg (কালোপা রোগ) কয়েক ঘণ্টার মধ্যেই পশুর মৃত্যু ঘটাতে পারে। "
                "বাড়িতে সময় নষ্ট না করে এখনই নিকটস্থ ভেটেরিনারি ডাক্তারের সাথে যোগাযোগ করুন।",
}

# tabular মডেলের রোগের নাম -> আমাদের medicine_rules-এর key ম্যাপিং
TABULAR_DISEASE_TO_MED_KEY = {
    "foot and mouth": "foot and mouth disease (fmd)",
    "lumpy virus": "lumpy skin disease",
    "pneumonia": "pneumonia / respiratory",
}

# রোগের নাম বাংলায় দেখানোর জন্য
DISEASE_DISPLAY_BN = {
    "anthrax": "ANTHRAX (তড়কা)",
    "blackleg": "BLACKLEG (কালোপা)",
    "foot and mouth": "FOOT AND MOUTH DISEASE (ক্ষুরারোগ)",
    "lumpy virus": "LUMPY SKIN DISEASE (গুটি বসন্ত)",
    "pneumonia": "PNEUMONIA / BRD (নিউমোনিয়া)",
}

# -------------------------------------------------------------
# 3. SPECIES & MEDICINE ROUTING MAP
# -------------------------------------------------------------
species_map = {
    "cow": "গরু (Cow)", "buffalo": "মহিষ (Buffalo)",
    "dog": "কুকুর (Dog)", "cat": "বিড়াল (Cat)",
    "goat": "ছাগল (Goat)", "sheep": "ভেড়া (Sheep)"
}

# রোগভেদে সম্ভাব্য ওষুধের সাজেশন।
#
# এখানে ইচ্ছাকৃতভাবে কোনো ডোজ/মাত্রা রাখা হয়নি। মাত্রা নির্ভর করে পশুর ওজন,
# বয়স, কিডনি-লিভারের অবস্থা, গর্ভাবস্থা, দুধ দেওয়া হচ্ছে কিনা, এবং স্থানীয়ভাবে
# কোন ব্র্যান্ড/ঘনত্ব পাওয়া যাচ্ছে তার ওপর — এগুলো অ্যাপ জানে না। ভুল মাত্রা
# উপকারের বদলে ক্ষতি করে, তাই মাত্রা ঠিক করার দায়িত্ব ভেটেরিনারিয়ানের।
# "role" ক্ষেত্রটি বলে ওষুধটা মূল চিকিৎসা নাকি সহায়ক সেবা।
medicine_rules = {
    "lumpy skin disease": [
        {"name": "Inj. Oxytetracycline (Long Acting)", "purpose": "গুটি ফেটে হওয়া সেকেন্ডারি ব্যাকটেরিয়াল ইনফেকশন ঠেকাতে ও ঘা শুকাতে", "role": "প্রধান"},
        {"name": "Inj. Meloxicam / Paracetamol", "purpose": "ব্যথা, গুটির ফোলা ও জ্বর কমাতে", "role": "সহায়ক"},
        {"name": "Inj. Pheniramine Maleate (Antihistamine)", "purpose": "চুলকানি ও অ্যালার্জি প্রতিক্রিয়া কমাতে", "role": "সহায়ক"},
        {"name": "ক্ষতস্থানে অ্যান্টিসেপটিক ও মাছি-প্রতিরোধী মলম", "purpose": "ঘায়ে মাছি বসা ও পচন ঠেকাতে", "role": "সহায়ক"},
    ],
    "pneumonia / respiratory": [
        {"name": "Inj. Oxytetracycline বা Ceftiofur", "purpose": "ফুসফুসের ব্যাকটেরিয়া (Mannheimia, Pasteurella) দমনে — যত দ্রুত শুরু, তত ভালো ফল", "role": "প্রধান"},
        {"name": "NSAID (Inj. Ketoprofen / Flunixin Meglumine)", "purpose": "জ্বর ও ফুসফুসের প্রদাহ কমাতে", "role": "সহায়ক"},
        {"name": "পর্যাপ্ত পানি ও ইলেক্ট্রোলাইট", "purpose": "পানিশূন্যতা ঠেকাতে", "role": "সহায়ক"},
        {"name": "শুকনো, বাতাস চলাচলকারী ও ধুলামুক্ত আশ্রয়", "purpose": "চিকিৎসার পাশাপাশি পরিবেশ ঠিক না করলে রোগ বারবার ফিরে আসে", "role": "ব্যবস্থাপনা"},
    ],
    "foot and mouth disease (fmd)": [
        {"name": "Potassium Permanganate দ্রবণে ধোয়া", "purpose": "মুখ ও খুরের ঘা পরিষ্কার রাখতে", "role": "প্রধান"},
        {"name": "Inj. Sulpha-Trimethoprim বা Oxytetracycline", "purpose": "খুরের ক্ষতে সেকেন্ডারি সংক্রমণ ঠেকাতে", "role": "সহায়ক"},
        {"name": "Inj. Flunixin Meglumine / Meloxicam", "purpose": "তীব্র ব্যথা ও খোঁড়া ভাব কমাতে", "role": "সহায়ক"},
        {"name": "নরম, তরল খাবার ও আলাদা রাখা", "purpose": "মুখে ঘা থাকায় শক্ত খাবার কষ্টকর; FMD অত্যন্ত সংক্রামক তাই আলাদা রাখা জরুরি", "role": "ব্যবস্থাপনা"},
    ],
    "mastitis": [
        {"name": "Intramammary Antibiotic Infusion (Ceftiofur / Cloxacillin)", "purpose": "ওলানের ভেতরের ব্যাকটেরিয়া দমনে", "role": "প্রধান"},
        {"name": "Inj. Meloxicam", "purpose": "ওলানের ফোলা ও ব্যথা কমাতে", "role": "সহায়ক"},
        {"name": "ঘন ঘন দুধ দোহন ও ওলান পরিষ্কার রাখা", "purpose": "সংক্রমিত দুধ বের করে দিতে", "role": "ব্যবস্থাপনা"},
    ],
    "default_infection": [
        {"name": "ব্রড স্পেকট্রাম অ্যান্টিবায়োটিক", "purpose": "সাধারণ ব্যাকটেরিয়াল ইনফেকশনে", "role": "প্রধান"},
        {"name": "জ্বর ও ব্যথানাশক (NSAID)", "purpose": "জ্বর ও অস্বস্তি কমাতে", "role": "সহায়ক"},
    ]
}

# কুকুর/বিড়ালের রোগগুলোর জন্য এখনো কোনো যাচাইকৃত (verified) ওষুধ/ডোজ তথ্য নেই,
# তাই নির্দিষ্ট ড্রাগ নাম বা মাত্রা না দেখিয়ে সাধারণ যত্ন পরামর্শ + ভেট রেফারেল দেখানো হয়
DOG_CAT_GENERIC_CARE = [
    "আক্রান্ত স্থান পরিষ্কার ও শুকনো রাখুন, নিজে থেকে কোনো ওষুধ প্রয়োগ করবেন না।",
    "প্রাণীকে পর্যাপ্ত পানি ও সহজপাচ্য খাবার দিন, ক্ষুধামান্দ্য থাকলে জোর করে খাওয়াবেন না।",
    "লক্ষণ বাড়তে থাকলে বা ২৪-৪৮ ঘণ্টার মধ্যে উন্নতি না হলে দ্রুত রেজিস্ট্রার্ড ভেটেরিনারিয়ানের কাছে নিয়ে যান।",
    "সংক্রামক হতে পারে এমন রোগে (Mange, Ringworm, Parvovirus ইত্যাদি) অন্য পোষা প্রাণী থেকে আলাদা রাখুন।"
]

# -------------------------------------------------------------
# 4. SESSION STATE MANAGEMENT
# -------------------------------------------------------------
if 'current_step' not in st.session_state: st.session_state.current_step = 1
if 'selected_animal' not in st.session_state: st.session_state.selected_animal = "cow"
if 'detected_disease' not in st.session_state: st.session_state.detected_disease = None
if 'confidence' not in st.session_state: st.session_state.confidence = None
if 'is_real_prediction' not in st.session_state: st.session_state.is_real_prediction = False
if 'is_healthy' not in st.session_state: st.session_state.is_healthy = False
if 'needs_generic_care_only' not in st.session_state: st.session_state.needs_generic_care_only = False
if 'tabular_result' not in st.session_state: st.session_state.tabular_result = None
if 'is_urgent' not in st.session_state: st.session_state.is_urgent = False
if 'urgent_message' not in st.session_state: st.session_state.urgent_message = ""
if 'image_folder_class' not in st.session_state: st.session_state.image_folder_class = None
if 'resp_score' not in st.session_state: st.session_state.resp_score = 0
if 'resp_checked' not in st.session_state: st.session_state.resp_checked = []
if 'fusion_note' not in st.session_state: st.session_state.fusion_note = ""
if 'pet_text_result' not in st.session_state: st.session_state.pet_text_result = None
if 'show_prescription' not in st.session_state: st.session_state.show_prescription = False

s1 = "step-badge-active" if st.session_state.current_step == 1 else "step-badge"
s2 = "step-badge-active" if st.session_state.current_step == 2 else "step-badge"
s3 = "step-badge-active" if st.session_state.current_step == 3 else "step-badge"

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:25px; background:#111827; padding:15px; border-radius:12px;">
<div>
<span class="{s1}">১. ছবি স্ক্যান ও রোগ সনাক্তকরণ (IMAGE DIAGNOSIS)</span> &nbsp;
<span class="{s2}">২. শারীরিক তথ্য ও ওজন (ANIMAL VITALS)</span> &nbsp;
<span class="{s3}">৩. ওষুধ ও প্রদেয় মাত্রা (DOSAGE & RX)</span>
</div>
<span style="color:#06b6d4; font-weight:bold;">⚡ VET-AI CYBER CLINIC</span>
</div>
""", unsafe_allow_html=True)

if not REAL_MODEL_AVAILABLE:
    st.markdown("""
<div class="demo-warning">
⚠️ এখনো কোনো ট্রেইনড ইমেজ মডেল (cow_disease_model.pth) পাওয়া যায়নি — এখন যে রোগ দেখানো হবে তা
শুধু ডেমো লজিক, প্রকৃত ছবি বিশ্লেষণ নয়। <code>train_image_model.py</code> চালিয়ে মডেল তৈরি করুন
এবং এই অ্যাপের পাশে রাখুন।
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# STEP 1: IMAGE SCAN & DISEASE DETECTION
# -------------------------------------------------------------
if st.session_state.current_step == 1:
    st.markdown("<h3 style='color:#38bdf8;'>🐾 প্রাণী ও ছবি দিয়ে সরাসরি রোগ সনাক্তকরণ</h3>", unsafe_allow_html=True)
    
    dataset_species = ["cow", "buffalo", "dog", "cat", "goat", "sheep"]
    animal_display = [species_map.get(a, a.title()) for a in dataset_species]
    selected_idx = st.selectbox("প্রাণীর প্রজাতি বেছে নিন:", range(len(animal_display)), format_func=lambda x: animal_display[x])
    st.session_state.selected_animal = dataset_species[selected_idx]

    is_cow_or_buffalo = st.session_state.selected_animal in ["cow", "buffalo"]

    is_dog_or_cat = st.session_state.selected_animal in ["dog", "cat"]

    if is_cow_or_buffalo and REAL_MODEL_AVAILABLE:
        st.caption("✅ এই প্রজাতির জন্য আসল ট্রেইনড ইমেজ মডেল ব্যবহার হচ্ছে (foot-and-mouth / healthy / lumpy)।")
    elif is_cow_or_buffalo and not REAL_MODEL_AVAILABLE:
        st.caption("⚠️ মডেল ফাইল না পাওয়া যাওয়ায় ডেমো লজিক ব্যবহার হচ্ছে।")
    elif is_dog_or_cat and DOG_CAT_MODEL_AVAILABLE:
        st.caption("✅ এই প্রজাতির জন্য আসল ট্রেইনড ইমেজ মডেল ব্যবহার হচ্ছে। তবে ওষুধের মাত্রা এখনো যাচাই করা নেই, তাই সাধারণ যত্ন পরামর্শ দেখানো হবে।")
    elif is_dog_or_cat and not DOG_CAT_MODEL_AVAILABLE:
        st.caption("⚠️ মডেল ফাইল না পাওয়া যাওয়ায় ডেমো লজিক ব্যবহার হচ্ছে।")
    else:
        st.caption("⚠️ এই প্রজাতির জন্য এখনো কোনো ট্রেইনড ইমেজ মডেল নেই — ডেমো লজিক দেখানো হচ্ছে।")

    is_goat_or_sheep = st.session_state.selected_animal in ["goat", "sheep"]

    if is_goat_or_sheep:
        st.info("ℹ️ ছাগল/ভেড়ার জন্য এখনো কোনো ইমেজ মডেল নেই — সরাসরি পরের ধাপে গিয়ে উপসর্গ (symptoms) দিয়ে রোগ সনাক্ত করা হবে। ছবি আপলোড না করলেও চলবে।")
        st.session_state.detected_disease = None
        st.session_state.detected_med_key = None
        st.session_state.confidence = None
        st.session_state.is_real_prediction = False
        st.session_state.is_healthy = False
        st.session_state.needs_generic_care_only = False
        st.session_state.image_folder_class = None

    st.markdown("<br><h4 style='color:#f3f4f6;'>📷 আক্রান্ত অংশের ছবি আপলোড করুন</h4>", unsafe_allow_html=True)
    up_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if up_file:
        img = Image.open(up_file).convert('RGB')

        if is_cow_or_buffalo and REAL_MODEL_AVAILABLE:
            # ---- আসল মডেল দিয়ে প্রেডিকশন (গরু/মহিষ) ----
            folder_class, confidence = predict_with_real_model(cow_model, cow_idx_to_class, img)
            info = IMAGE_CLASS_INFO.get(folder_class, {"display": folder_class.upper(), "med_key": "default_infection"})
            st.session_state.detected_disease = info["display"]
            st.session_state.detected_med_key = info["med_key"]
            st.session_state.confidence = round(confidence, 1)
            st.session_state.is_real_prediction = True
            st.session_state.is_healthy = (info["med_key"] is None)
            st.session_state.needs_generic_care_only = False
            st.session_state.image_folder_class = folder_class  # lumpy/pneumonia দ্বিধা ভাঙতে পরে কাজে লাগবে
        elif is_dog_or_cat and DOG_CAT_MODEL_AVAILABLE:
            # ---- আসল মডেল দিয়ে প্রেডিকশন (কুকুর/বিড়াল) ----
            folder_class, confidence = predict_with_real_model(dog_cat_model, dog_cat_idx_to_class, img)
            st.session_state.detected_disease = folder_class
            st.session_state.detected_med_key = None  # যাচাইকৃত ডোজ তথ্য নেই
            st.session_state.confidence = round(confidence, 1)
            st.session_state.is_real_prediction = True
            st.session_state.is_healthy = False
            st.session_state.needs_generic_care_only = True
        else:
            # ---- ডেমো লজিক (স্পষ্টভাবে চিহ্নিত, কোনো প্রকৃত মডেল না থাকা প্রজাতির জন্য) ----
            fname = up_file.name.lower()
            if "lumpy" in fname or "lump" in fname:
                st.session_state.detected_disease = "LUMPY SKIN DISEASE (LSD) [DEMO]"
                st.session_state.detected_med_key = "lumpy skin disease"
                st.session_state.confidence = 96.5
            elif "foot" in fname or "fmd" in fname:
                st.session_state.detected_disease = "FOOT AND MOUTH DISEASE (FMD) [DEMO]"
                st.session_state.detected_med_key = "foot and mouth disease (fmd)"
                st.session_state.confidence = 94.2
            elif "pneumonia" in fname:
                st.session_state.detected_disease = "BOVINE / CANINE PNEUMONIA [DEMO]"
                st.session_state.detected_med_key = "pneumonia / respiratory"
                st.session_state.confidence = 92.8
            else:
                st.session_state.detected_disease = "SKIN LESION / DERMATITIS [DEMO]"
                st.session_state.detected_med_key = "default_infection"
                st.session_state.confidence = 89.1
            st.session_state.is_real_prediction = False
            st.session_state.is_healthy = False
            st.session_state.needs_generic_care_only = False

        c_img, c_info = st.columns([1, 2])
        with c_img:
            st.markdown('<div class="scan-container"><div class="scan-line"></div>', unsafe_allow_html=True)
            st.image(img, width=280)
            st.markdown('</div>', unsafe_allow_html=True)
        with c_info:
            st.success("✅ ছবি স্ক্যানিং সম্পন্ন!")
            source_label = "আসল ইমেজ মডেল প্রেডিকশন" if st.session_state.is_real_prediction else "⚠️ ডেমো লজিক (প্রকৃত মডেল না)"
            st.markdown(f"""
<div style="background:#1e1b4b; border:1px solid #6366f1; padding:15px; border-radius:10px; margin-top:10px;">
<small style="color:#a5b4fc;">{source_label}:</small>
<h2 style="margin:5px 0 0 0; color:#38bdf8;">{st.session_state.detected_disease}</h2>
<small style="color:#34d399;">কনফিডেন্স: {st.session_state.confidence}%</small>
</div>
""", unsafe_allow_html=True)
            
    # ছবিতে "healthy" আসা মানে শুধু "ত্বকে কোনো রোগ দেখা যাচ্ছে না" — "পশুটি সুস্থ" নয়।
    # ফুসফুসের রোগ (pneumonia/BRD), পেটের সমস্যা বা জ্বর ছবিতে ধরা পড়ে না, আর
    # মডেলের ট্রেনিং ডেটায় ওই রোগগুলোর কোনো ছবিই নেই। তাই এখানে পথ বন্ধ না করে
    # ব্যবহারকারীকে উপসর্গ দিয়ে এগোনোর সুযোগ দেওয়া হয়।
    if st.session_state.is_healthy:
        st.info(
            "🔍 ছবিতে ত্বকের কোনো রোগ (গুটি/ক্ষত) ধরা পড়েনি। তবে মনে রাখবেন — "
            "নিউমোনিয়া, পেটের অসুখ বা জ্বরের মতো সমস্যা ছবিতে দেখা যায় না। "
            "পশুর মধ্যে অন্য কোনো লক্ষণ থাকলে পরের ধাপে গিয়ে উপসর্গ দিয়ে যাচাই করুন।"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🩺 উপসর্গ দিয়ে যাচাই করে সম্ভাব্য ওষুধ দেখুন ➡️"):
        if not up_file and not is_goat_or_sheep:
            st.warning("⚠️ প্রসেস শুরু করতে অনুগ্রহ করে আক্রান্ত অংশের ছবি আপলোড করুন!")
        else:
            st.session_state.current_step = 2
            st.rerun()

# -------------------------------------------------------------
# STEP 2: VITALS INPUT FOR DOSAGE CALCULATION
# -------------------------------------------------------------
elif st.session_state.current_step == 2:
    selected_bng = species_map.get(st.session_state.selected_animal, st.session_state.selected_animal.title())
    is_livestock = st.session_state.selected_animal in ["cow", "buffalo", "goat", "sheep"]

    if st.session_state.detected_disease:
        st.markdown(f"""
<div style="background:#111827; padding:15px; border-radius:12px; border-left:4px solid #06b6d4; margin-bottom:20px;">
<span style="color:#9ca3af;">● ছবি থেকে সনাক্তকৃত রোগ:</span>
<h3 style="margin:0; color:#38bdf8;">{st.session_state.detected_disease} ({selected_bng})</h3>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="background:#111827; padding:15px; border-radius:12px; border-left:4px solid #06b6d4; margin-bottom:20px;">
<h3 style="margin:0; color:#38bdf8;">{selected_bng} — উপসর্গ (Symptoms) দিয়ে রোগ সনাক্তকরণ</h3>
</div>
""", unsafe_allow_html=True)

    is_dog_or_cat = st.session_state.selected_animal in ["dog", "cat"]

    st.markdown("### 📋 পশুর শারীরিক তথ্য")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("শরীরের তাপমাত্রা / Temperature (°F):", min_value=95.0, max_value=108.0, value=103.5)
    with col2:
        age = st.number_input("পশুর বয়স / Age (Years):", min_value=0.1, max_value=30.0, value=2.5)

    sym1 = sym2 = sym3 = None
    if is_livestock and TABULAR_MODEL_AVAILABLE:
        st.markdown("### 🩺 লক্ষণ নির্বাচন করুন")
        st.caption(f"এই মডেলের পরিমাপকৃত accuracy: {TABULAR_ACCURACY:.1f}%")
        c1, c2, c3 = st.columns(3)
        with c1:
            sym1 = st.selectbox("লক্ষণ ১:", SYMPTOM_OPTIONS, key="sym1_select")
        with c2:
            sym2 = st.selectbox("লক্ষণ ২:", SYMPTOM_OPTIONS, key="sym2_select", index=min(1, len(SYMPTOM_OPTIONS)-1))
        with c3:
            sym3 = st.selectbox("লক্ষণ ৩:", SYMPTOM_OPTIONS, key="sym3_select", index=min(2, len(SYMPTOM_OPTIONS)-1))
    elif is_livestock and not TABULAR_MODEL_AVAILABLE:
        st.caption("⚠️ Symptom-ভিত্তিক মডেল ফাইল (livestock_symptom_model.pkl) পাওয়া যায়নি।")

    # ---- শ্বাসতন্ত্রের লক্ষণ (গবাদি পশু) ----
    # ছবি দিয়ে ফুসফুস দেখা যায় না, তাই নিউমোনিয়া (BRD) নির্ণয়ে এই ক্লিনিক্যাল
    # লক্ষণগুলোই একমাত্র ভরসা। "গায়ে গুটি নেই" কখনোই নিউমোনিয়ার প্রমাণ নয় —
    # সুস্থ পশুর গায়েও গুটি থাকে না। তাই নিউমোনিয়া বলতে হলে এগুলো লাগবেই।
    resp_score = 0
    resp_checked = []
    if is_livestock and PNEUMONIA_SIGNS:
        st.markdown("### 🫁 শ্বাসতন্ত্রের লক্ষণ (যেগুলো আছে টিক দিন)")
        st.caption(
            "নিউমোনিয়া ছবিতে ধরা পড়ে না — এই লক্ষণগুলো দিয়েই সনাক্ত করতে হয়। "
            "কোনোটা না থাকলে ফাঁকা রাখুন।"
        )
        rcols = st.columns(2)
        for i, sign in enumerate(PNEUMONIA_SIGNS):
            with rcols[i % 2]:
                if st.checkbox(sign["bn"], key=f"resp_{sign['key']}"):
                    resp_score += sign["weight"]
                    resp_checked.append(sign["bn"])

    # ---- কুকুর/বিড়ালের জন্য লক্ষণের বর্ণনা (টেক্সট মডেল) ----
    pet_text = ""
    if is_dog_or_cat and PET_TEXT_AVAILABLE:
        st.markdown("### ✍️ লক্ষণ নিজের ভাষায় লিখুন")
        st.caption(
            f"ছবিতে যা দেখা যায় না (বমি, ডায়রিয়া, খোঁড়ানো, কাশি) তা এখানে লিখুন। "
            f"এই মডেলের accuracy {PET_TEXT_ACCURACY:.1f}% — ছবির মডেলের চেয়ে বেশি। ইংরেজিতে লিখলে সবচেয়ে ভালো কাজ করে।"
        )
        pet_text = st.text_area(
            "লক্ষণের বর্ণনা",
            placeholder="যেমন: My dog has been vomiting since yesterday and refuses to eat",
            key="pet_text_input",
            height=90,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        if st.button("⬅️ পেছনে যান (Back)"):
            st.session_state.current_step = 1
            st.rerun()
    with col_b2:
        if st.button("💊 সম্ভাব্য ওষুধ দেখুন ➡️"):
            st.session_state.temp = temp
            st.session_state.age = age
            st.session_state.resp_score = resp_score
            st.session_state.resp_checked = resp_checked
            st.session_state.fusion_note = ""

            # ---- কুকুর/বিড়ালের টেক্সট-উপসর্গ ----
            st.session_state.pet_text_result = None
            if is_dog_or_cat and PET_TEXT_AVAILABLE and pet_text.strip():
                st.session_state.pet_text_result = predict_pet_text(pet_text.strip())

            if is_livestock and TABULAR_MODEL_AVAILABLE and sym1 and sym2 and sym3:
                proba_dict = predict_with_tabular_model(
                    st.session_state.selected_animal, age, temp, sym1, sym2, sym3
                )
                st.session_state.tabular_result = proba_dict
                top_disease = max(proba_dict, key=proba_dict.get)
                st.session_state.is_urgent = False

                if top_disease in URGENT_REFERRAL_DISEASES:
                    st.session_state.is_urgent = True
                    st.session_state.urgent_message = URGENT_REFERRAL_DISEASES[top_disease]
                    st.session_state.detected_disease = DISEASE_DISPLAY_BN.get(top_disease, top_disease.upper())
                    st.session_state.detected_med_key = None
                    st.session_state.needs_generic_care_only = False
                    st.session_state.confidence = round(proba_dict[top_disease] * 100, 1)

                elif top_disease == MERGED_CLASS:
                    # ---- এখানেই ছবি ও শ্বাসলক্ষণ মিলিয়ে সিদ্ধান্ত নেওয়া হয় ----
                    # উপসর্গ নিশ্চিত করেছে রোগটি lumpy অথবা pneumonia, কিন্তু ওদের
                    # উপসর্গ অভিন্ন বলে উপসর্গ দিয়ে আর এগোনো যায় না।
                    img_class = st.session_state.image_folder_class
                    has_resp = resp_score >= PNEUMONIA_THRESHOLD
                    conf = round(proba_dict[top_disease] * 100, 1)

                    if img_class == "lumpy":
                        resolved, med_key = "lumpy virus", "lumpy skin disease"
                        note = ("ছবিতে গায়ে গুটি/ক্ষত সনাক্ত হয়েছে — Lumpy Skin Disease চোখে দেখা যায়, "
                                "তাই ছবির প্রমাণই এখানে নির্ণায়ক।")
                        if has_resp:
                            note += (f" তবে শ্বাসতন্ত্রের লক্ষণও আছে (স্কোর {resp_score}) — "
                                     "একসাথে দুটো রোগ থাকা সম্ভব, ভেটকে দুটোর কথাই জানান।")
                    elif has_resp:
                        resolved, med_key = "pneumonia", "pneumonia / respiratory"
                        note = (f"ছবিতে ত্বকের কোনো রোগ পাওয়া যায়নি, আর শ্বাসতন্ত্রের লক্ষণ স্কোর {resp_score} "
                                f"(সীমা {PNEUMONIA_THRESHOLD}) — তাই নিউমোনিয়া (BRD) সবচেয়ে সম্ভাব্য। "
                                "টিক দেওয়া লক্ষণ: " + ", ".join(resp_checked))
                    elif img_class == "foot-and-mouth":
                        # ছবি আর উপসর্গ পরস্পরবিরোধী — জোর করে কোনো সিদ্ধান্ত চাপানো হয় না
                        resolved, med_key = None, None
                        note = ("⚠️ ছবি বলছে Foot and Mouth Disease, কিন্তু উপসর্গ বলছে Lumpy বা Pneumonia — "
                                "দুটো প্রমাণ মিলছে না। নিশ্চিত হতে ভেটেরিনারি ডাক্তারকে দেখান।")
                    else:
                        # গুরুত্বপূর্ণ: "গুটি নেই" একা নিউমোনিয়ার প্রমাণ নয় — সুস্থ পশুর
                        # গায়েও গুটি থাকে না। শ্বাসলক্ষণ ছাড়া কোনো সিদ্ধান্ত নয়।
                        resolved, med_key = None, None
                        note = ("ছবিতে গুটি পাওয়া যায়নি, তাই Lumpy Skin Disease-এর সম্ভাবনা কম। "
                                "কিন্তু শ্বাসতন্ত্রের লক্ষণও টিক দেওয়া হয়নি, আর গায়ে গুটি না থাকা "
                                "একা নিউমোনিয়ার প্রমাণ নয় — সুস্থ পশুর গায়েও গুটি থাকে না। "
                                "তাই নিশ্চিতভাবে কিছু বলা যাচ্ছে না, ভেটেরিনারি ডাক্তারকে দেখান।")

                    st.session_state.fusion_note = note
                    if resolved:
                        st.session_state.detected_disease = DISEASE_DISPLAY_BN.get(resolved, resolved.upper())
                        st.session_state.detected_med_key = med_key
                        st.session_state.confidence = conf
                        st.session_state.is_real_prediction = True
                        st.session_state.needs_generic_care_only = False
                    else:
                        st.session_state.detected_disease = "নিশ্চিতভাবে সনাক্ত করা যায়নি"
                        st.session_state.detected_med_key = None
                        st.session_state.confidence = conf
                        st.session_state.is_real_prediction = True
                        st.session_state.needs_generic_care_only = True

                else:
                    # foot and mouth — উপসর্গেই ১০০% নিশ্চিত
                    st.session_state.detected_disease = DISEASE_DISPLAY_BN.get(top_disease, top_disease.upper())
                    st.session_state.detected_med_key = TABULAR_DISEASE_TO_MED_KEY.get(top_disease, "default_infection")
                    st.session_state.confidence = round(proba_dict[top_disease] * 100, 1)
                    st.session_state.is_real_prediction = True
                    st.session_state.needs_generic_care_only = False
            else:
                st.session_state.tabular_result = None
                st.session_state.is_urgent = False

            st.session_state.current_step = 3
            st.rerun()

# -------------------------------------------------------------
# STEP 3: AUTOMATED MEDICINE & DOSAGE PRESCRIPTION CARD
# -------------------------------------------------------------
elif st.session_state.current_step == 3:
    selected_bng = species_map.get(st.session_state.selected_animal, st.session_state.selected_animal.title())

    # ---- জরুরি রোগ (Anthrax/Blackleg) হলে সরাসরি urgent referral card দেখানো হয়, ওষুধের তালিকা না ----
    if st.session_state.is_urgent:
        st.markdown(f"""
<div class="rx-card" style="border:2px solid #dc2626;">
<div style="display:flex; align-items:center; gap:12px;">
<span style="font-size:2rem;">🚨</span>
<div>
<h2 style="margin:0; color:#f87171;">জরুরি সতর্কবার্তা — এখনই ব্যবস্থা নিন</h2>
<small style="color:#9ca3af;">{selected_bng} | সম্ভাব্য রোগ: {st.session_state.detected_disease}</small>
</div>
</div>
<hr style="border-color:#374151; margin:15px 0;">
<p style="color:#fca5a5; font-size:1.05rem; line-height:1.6;">{st.session_state.urgent_message}</p>
</div>
""", unsafe_allow_html=True)

        if st.session_state.tabular_result:
            st.markdown("<br><small style='color:#9ca3af;'>উপসর্গ বিশ্লেষণে অন্যান্য সম্ভাবনা:</small>", unsafe_allow_html=True)
            sorted_probs = sorted(st.session_state.tabular_result.items(), key=lambda x: -x[1])
            for disease, prob in sorted_probs[:3]:
                st.progress(float(prob), text=f"{disease} — {prob*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 নতুন ডায়াগনোসিস শুরু করুন (New Diagnosis)"):
            st.session_state.current_step = 1
            st.rerun()
        st.stop()

    med_key = st.session_state.get("detected_med_key") or "default_infection"
    selected_meds = medicine_rules.get(med_key, medicine_rules["default_infection"])

    if st.session_state.is_real_prediction:
        if st.session_state.selected_animal in ["cow", "buffalo"]:
            model_acc = COW_MODEL_VAL_ACCURACY
        elif st.session_state.selected_animal in ["dog", "cat"]:
            model_acc = DOG_CAT_MODEL_VAL_ACCURACY
        else:
            model_acc = TABULAR_ACCURACY
        verified_badge = f"🧠 AI MODEL (পরিমাপকৃত accuracy: {model_acc:.1f}%)"
        badge_color = "#065f46" if model_acc >= 70 else "#7c2d12"
        badge_text_color = "#34d399" if model_acc >= 70 else "#fbbf24"
    else:
        verified_badge = "⚠️ DEMO (প্রকৃত মডেল না)"
        badge_color = "#7c2d12"
        badge_text_color = "#fbbf24"

    st.markdown(f"""
<div class="rx-card">
<div style="display:flex; justify-content:space-between; border-bottom:1px solid #1f2937; padding-bottom:12px;">
<div>
<h3 style="margin:0; color:#38bdf8;">🔬 রোগ নির্ণয়ের ফলাফল</h3>
<small style="color:#9ca3af;">VT-2026-IMAGE-MATCH</small>
</div>
<div style="background:{badge_color}; color:{badge_text_color}; padding:6px 14px; border-radius:20px; font-weight:bold;">
{verified_badge}
</div>
</div>

<div style="display:flex; justify-content:space-between; margin-top:15px; background:#1f2937; padding:12px; border-radius:10px;">
<div>
<small style="color:#9ca3af;">রোগীর প্রজাতি (SPECIES)</small><br>
<b style="color:#ffffff; font-size:1.1rem;">{selected_bng}</b>
</div>
<div>
<small style="color:#9ca3af;">বয়স / তাপমাত্রা</small><br>
<b style="color:#38bdf8; font-size:1.1rem;">{st.session_state.age} বছর · {st.session_state.temp}°F</b>
</div>
<div>
<small style="color:#9ca3af;">কনফিডেন্স</small><br>
<b style="color:#34d399; font-size:1.1rem;">{st.session_state.confidence}%</b>
</div>
</div>

<br>
<small style="color:#9ca3af;">{'ছবি ও উপসর্গ মিলিয়ে সনাক্তকৃত রোগ' if st.session_state.selected_animal in ['cow','buffalo','dog','cat'] else 'উপসর্গ থেকে সনাক্তকৃত রোগ'}</small>
<div style="background:#312e81; border:1px solid #6366f1; padding:15px; border-radius:10px; margin-top:5px;">
<h2 style="margin:0; color:#a5b4fc;">{st.session_state.detected_disease}</h2>
</div>
</div>
""", unsafe_allow_html=True)

    # ---- ছবি ও উপসর্গ কীভাবে মিলিয়ে সিদ্ধান্ত হলো, তার ব্যাখ্যা ----
    if st.session_state.fusion_note:
        st.markdown(f"""
<div class="demo-warning" style="border-color:#0891b2; color:#67e8f9;">
🔗 <b>ছবি + উপসর্গ বিশ্লেষণ:</b> {st.session_state.fusion_note}
</div>
""", unsafe_allow_html=True)

    if not st.session_state.is_real_prediction:
        st.markdown("""
<div class="demo-warning">
⚠️ এই প্রেসক্রিপশনটি ডেমো লজিকের ওপর ভিত্তি করে তৈরি, প্রকৃত ইমেজ মডেল যাচাই করেনি।
সত্যিকারের চিকিৎসার আগে একজন নিবন্ধিত ভেটেরিনারি ডাক্তারের পরামর্শ নিন।
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="demo-warning">
ℹ️ এই ফলাফল একটা AI মডেলের প্রেডিকশন, যেটার সঠিকতা ১০০% নয় (উপরে accuracy % দেখুন)।
এটা প্রাথমিক ধারণা মাত্র — চূড়ান্ত চিকিৎসার আগে একজন নিবন্ধিত ভেটেরিনারি ডাক্তার দিয়ে
নিশ্চিত করিয়ে নিন।
</div>
""", unsafe_allow_html=True)

    if st.session_state.needs_generic_care_only:
        # কুকুর/বিড়ালের রোগের জন্য নির্দিষ্ট ওষুধ/ডোজ যাচাইকৃত নয়,
        # তাই সাধারণ যত্ন পরামর্শ ও ভেট রেফারেল দেখানো হচ্ছে
        st.markdown("""
<div class="demo-warning">
ℹ️ এই রোগের জন্য নির্দিষ্ট ওষুধ ও মাত্রার তথ্য এখনো যাচাইকৃত (verified) নয়, তাই সরাসরি
ওষুধের নাম/ডোজ না দেখিয়ে সাধারণ যত্ন পরামর্শ দেখানো হচ্ছে। সঠিক চিকিৎসার জন্য
নিবন্ধিত ভেটেরিনারিয়ানের পরামর্শ আবশ্যক।
</div>
""", unsafe_allow_html=True)
        for tip in DOG_CAT_GENERIC_CARE:
            st.markdown(f"""
<div style="background:#111827; padding:14px; border-radius:10px; margin-top:10px; border:1px solid #1f2937;">
<span style="color:#f3f4f6;">• {tip}</span>
</div>
""", unsafe_allow_html=True)
    else:
        # ---- সম্ভাব্য ওষুধের সাজেশন (ডোজ ছাড়া) ----
        st.markdown("### 💊 এই রোগে সাধারণত যেসব ওষুধ ব্যবহার হয়")
        role_color = {"প্রধান": "#0284c7", "সহায়ক": "#4b5563", "ব্যবস্থাপনা": "#065f46"}
        for med in selected_meds:
            badge = role_color.get(med.get("role", "সহায়ক"), "#4b5563")
            st.markdown(f"""
<div style="background:#111827; padding:14px; border-radius:10px; margin-top:10px; border:1px solid #1f2937; display:flex; justify-content:space-between; align-items:center; gap:14px;">
<div>
<b style="color:#f3f4f6; font-size:1.05rem;">{med['name']}</b><br>
<small style="color:#9ca3af;">{med['purpose']}</small>
</div>
<span style="background:{badge}; color:#ffffff; padding:5px 12px; border-radius:8px; font-size:0.8rem; font-weight:bold; white-space:nowrap;">{med.get('role','সহায়ক')}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="demo-warning" style="margin-top:14px;">
⚠️ <b>মাত্রা (ডোজ) ইচ্ছাকৃতভাবে দেখানো হয়নি।</b> সঠিক মাত্রা নির্ভর করে পশুর ওজন, বয়স,
কিডনি-লিভারের অবস্থা, গর্ভাবস্থা, দুধ দিচ্ছে কিনা, এবং স্থানীয়ভাবে পাওয়া ব্র্যান্ডের
ঘনত্বের ওপর — যা এই অ্যাপ জানে না। ভুল মাত্রা উপকারের বদলে ক্ষতি করে।
এই তালিকাটা ভেটেরিনারি ডাক্তারের সাথে আলোচনার জন্য, নিজে প্রয়োগ করার জন্য নয়।
</div>
""", unsafe_allow_html=True)

    # ---- উপসর্গ-ভিত্তিক probability (গবাদি পশু) ----
    if st.session_state.tabular_result:
        st.markdown("<br>### 🩺 উপসর্গ-ভিত্তিক বিশ্লেষণ", unsafe_allow_html=True)
        for disease, prob in sorted(st.session_state.tabular_result.items(), key=lambda x: -x[1]):
            label = DISEASE_DISPLAY_BN.get(disease, disease)
            if disease == MERGED_CLASS:
                label = "LUMPY অথবা PNEUMONIA (উপসর্গে আলাদা করা যায় না)"
            st.progress(float(prob), text=f"{label} — {prob*100:.1f}%")

    # ---- কুকুর/বিড়াল: ছবি ও লেখা-উপসর্গ মিলিয়ে দেখা ----
    # দুটো সম্পূর্ণ স্বাধীন প্রমাণ — একমত হলে আত্মবিশ্বাস বাড়ে, অমিল হলে সতর্কতা।
    if st.session_state.pet_text_result:
        st.markdown("<br>### ✍️ লেখা লক্ষণ থেকে বিশ্লেষণ", unsafe_allow_html=True)
        st.caption(f"টেক্সট মডেলের accuracy {PET_TEXT_ACCURACY:.1f}%")
        pet_sorted = sorted(st.session_state.pet_text_result.items(), key=lambda x: -x[1])
        for cat, prob in pet_sorted[:3]:
            st.progress(float(prob), text=f"{cat} — {prob*100:.1f}%")

        img_disease = st.session_state.detected_disease
        img_category = DOG_CAT_DISEASE_TO_CATEGORY.get(img_disease)
        text_category = pet_sorted[0][0]

        if img_category and img_category == text_category:
            st.success(
                f"✅ **দুটো প্রমাণ একমত।** ছবি বলছে \"{img_disease}\", যা \"{img_category}\" শ্রেণির — "
                f"আর আপনার লেখা লক্ষণও একই শ্রেণি বলছে। দুটো স্বাধীন পদ্ধতি মিলে যাওয়ায় "
                f"এই ফলাফলে আস্থা রাখা যায় (যেকোনো একটার চেয়ে বেশি)।"
            )
        elif img_category:
            st.warning(
                f"⚠️ **দুটো প্রমাণ মিলছে না।** ছবি বলছে \"{img_disease}\" (\"{img_category}\" শ্রেণি), "
                f"কিন্তু আপনার লেখা লক্ষণ বলছে \"{text_category}\"। ছবির মডেলের accuracy মাত্র "
                f"{DOG_CAT_MODEL_VAL_ACCURACY:.1f}%, তাই এখানে লেখা লক্ষণকেই বেশি গুরুত্ব দিন — "
                f"এবং ভেটেরিনারি ডাক্তারকে দেখান।"
            )
        else:
            st.info(
                f"ℹ️ ছবি থেকে পাওয়া \"{img_disease}\" রোগটি এই ৫টি বিস্তৃত শ্রেণির কোনোটিতে পড়ে না, "
                f"তাই সরাসরি মেলানো যাচ্ছে না। দুটো ফলাফলই আলাদাভাবে ভেটকে জানান।"
            )

    # ---- ঐচ্ছিক প্রেসক্রিপশন ----
    st.markdown("<br>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        if st.button("📄 প্রেসক্রিপশন হিসেবে দেখুন (ঐচ্ছিক)"):
            st.session_state.show_prescription = not st.session_state.show_prescription
    with col_p2:
        if st.button("🔄 নতুন ডায়াগনোসিস শুরু করুন"):
            st.session_state.current_step = 1
            st.session_state.show_prescription = False
            st.rerun()

    if st.session_state.show_prescription:
        meds_txt = "\n".join(
            f"{i}. {m['name']}  [{m.get('role','সহায়ক')}]\n     {m['purpose']}"
            for i, m in enumerate(selected_meds, 1)
        ) if not st.session_state.needs_generic_care_only else "\n".join(
            f"{i}. {t}" for i, t in enumerate(DOG_CAT_GENERIC_CARE, 1)
        )
        rx_text = f"""ভেটেরিনারি প্রাথমিক মূল্যায়ন (VetAI Nexus)
{'='*58}
প্রজাতি      : {selected_bng}
বয়স         : {st.session_state.age} বছর
তাপমাত্রা    : {st.session_state.temp} °F
সনাক্তকৃত রোগ : {st.session_state.detected_disease}
কনফিডেন্স    : {st.session_state.confidence}%

সম্ভাব্য ওষুধ / পরামর্শ
{'-'*58}
{meds_txt}

{'-'*58}
দ্রষ্টব্য: এটি একটি AI-ভিত্তিক প্রাথমিক মূল্যায়ন, কোনো ভেটেরিনারি
প্রেসক্রিপশন নয়। এখানে ইচ্ছাকৃতভাবে কোনো ওষুধের মাত্রা দেওয়া হয়নি।
চিকিৎসা শুরুর আগে অবশ্যই একজন নিবন্ধিত ভেটেরিনারি ডাক্তারের
পরামর্শ নিন।
"""
        st.code(rx_text, language=None)
        st.download_button(
            "⬇️ ডাউনলোড করুন (.txt)",
            data=rx_text.encode("utf-8"),
            file_name="vet_assessment.txt",
            mime="text/plain",
        )