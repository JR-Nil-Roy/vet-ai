"""
VetAI Nexus — প্রাণীর রোগ নির্ণয়ে AI সহায়ক
============================================
ছবি ও উপসর্গ মিলিয়ে রোগ সনাক্ত করে এবং সেই রোগে সাধারণত ব্যবহৃত
ওষুধের তালিকা দেখায়। কোনো ডোজ দেখানো হয় না — কারণ ব্যাখ্যা
medicine_data.py-তে আছে।

ফ্লো: ছবি আপলোড -> সাথে সাথেই রোগ ও ওষুধ -> (ঐচ্ছিক) উপসর্গ দিয়ে যাচাই
"""

import json
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from medicine_data import (
    GENERIC_PET_CARE,
    LIVESTOCK_MEDICINES,
    PET_MEDICINES,
    URGENT_LIVESTOCK_DISEASES,
    URGENT_PET_DISEASES,
    ZOONOTIC_PET_DISEASES,
)

st.set_page_config(
    page_title="VetAI Nexus",
    page_icon="🐾",
    layout="centered",          # ফোনে পড়ার জন্য centered-ই ভালো
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# স্টাইল — ভেটেরিনারি ক্লিনিকের চেহারা
# ---------------------------------------------------------------
st.markdown("""
<style>
:root {
  --brand:#0d9488; --brand-dk:#0f766e; --brand-lt:#ccfbf1;
  --warm:#f59e0b; --danger:#dc2626;
  --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --surface:#ffffff;
}
.block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:48rem; }

/* ---------- হেডার ---------- */
.vet-hero {
  background:linear-gradient(135deg,#0f766e 0%,#0d9488 55%,#14b8a6 100%);
  border-radius:20px; padding:26px 24px; color:#fff; margin-bottom:22px;
  box-shadow:0 10px 30px rgba(13,148,136,.28); position:relative; overflow:hidden;
}
.vet-hero::after {
  content:"🐾"; position:absolute; right:-10px; bottom:-26px;
  font-size:7rem; opacity:.13; transform:rotate(-18deg);
}
.vet-hero h1 { margin:0; font-size:1.7rem; font-weight:800; letter-spacing:-.02em; color:#fff; }
.vet-hero p  { margin:6px 0 0; opacity:.93; font-size:.95rem; }
.vet-chips { margin-top:14px; display:flex; gap:7px; flex-wrap:wrap; }
.vet-chip {
  background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.3);
  padding:4px 11px; border-radius:999px; font-size:.78rem; font-weight:600;
  backdrop-filter:blur(4px);
}

/* ---------- সেকশন শিরোনাম ---------- */
.vet-sec {
  display:flex; align-items:center; gap:9px; margin:26px 0 12px;
  font-size:1.08rem; font-weight:700; color:var(--ink);
}
.vet-sec::after { content:""; flex:1; height:1px; background:var(--line); }

/* ---------- রোগ নির্ণয়ের কার্ড (প্রেসক্রিপশন ঘরানা) ---------- */
.rx {
  background:var(--surface); border:1px solid var(--line); border-radius:16px;
  overflow:hidden; box-shadow:0 2px 14px rgba(15,23,42,.07); margin-bottom:6px;
}
.rx-top {
  background:linear-gradient(90deg,var(--brand-dk),var(--brand));
  color:#fff; padding:9px 16px; font-size:.76rem; font-weight:700;
  letter-spacing:.08em; text-transform:uppercase;
  display:flex; justify-content:space-between; align-items:center;
}
.rx-body { padding:18px 16px; }
.rx-name { font-size:1.35rem; font-weight:800; color:var(--brand-dk); line-height:1.3; margin:0; }
.rx-meta { display:flex; gap:9px; flex-wrap:wrap; margin-top:12px; }
.rx-stat {
  background:#f1f5f9; border:1px solid var(--line); border-radius:10px;
  padding:7px 13px; min-width:88px;
}
.rx-stat b { display:block; font-size:1.05rem; color:var(--ink); }
.rx-stat span { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }

/* ---------- ওষুধের সারি ---------- */
.med {
  display:flex; gap:12px; align-items:flex-start;
  background:var(--surface); border:1px solid var(--line);
  border-left:4px solid var(--brand); border-radius:12px;
  padding:13px 15px; margin-bottom:9px;
  transition:box-shadow .15s ease;
}
.med:hover { box-shadow:0 3px 12px rgba(15,23,42,.08); }
.med.supp   { border-left-color:#60a5fa; }
.med.manage { border-left-color:#94a3b8; }
.med-ico { font-size:1.25rem; line-height:1.3; }
.med-nm  { font-weight:700; color:var(--ink); font-size:.97rem; }
.med-ds  { color:var(--muted); font-size:.87rem; margin-top:3px; line-height:1.5; }
.med-tag {
  font-size:.68rem; font-weight:700; padding:2px 8px; border-radius:999px;
  background:var(--brand-lt); color:var(--brand-dk); white-space:nowrap;
}
.med-tag.supp   { background:#dbeafe; color:#1d4ed8; }
.med-tag.manage { background:#f1f5f9; color:#475569; }

/* ---------- বোতাম ও আপলোডার ---------- */
.stButton button { min-height:2.9rem; font-weight:700; border-radius:11px; }
[data-testid="stFileUploaderDropzone"] {
  border:2px dashed var(--brand); border-radius:14px; background:#f0fdfa;
}
[data-testid="stExpander"] { border-radius:13px; border:1px solid var(--line); }
[data-testid="stImage"] img { border-radius:13px; border:1px solid var(--line); }

/* ফোনে স্ট্যাট চিপ গুলো যেন ভেঙে না যায় */
@media (max-width:640px){
  .vet-hero h1{ font-size:1.4rem; }
  .rx-stat{ flex:1 1 calc(50% - 9px); min-width:0; }
}
</style>
""", unsafe_allow_html=True)


# ===============================================================
# মডেল লোডিং
# ===============================================================
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@st.cache_resource
def load_image_model(model_path, classes_path):
    """ResNet18 ক্লাসিফায়ার লোড করে। ফাইল না থাকলে (None, None)।"""
    if not (os.path.exists(model_path) and os.path.exists(classes_path)):
        return None, None
    with open(classes_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, idx_to_class


@st.cache_resource
def load_pickle_model(path):
    if not os.path.exists(path):
        return None
    import joblib
    return joblib.load(path)


cow_model, cow_classes = load_image_model("cow_disease_model.pth", "cow_disease_classes.json")
pet_model, pet_classes = load_image_model("dog_cat_disease_model.pth", "dog_cat_disease_classes.json")
goat_model, goat_classes = load_image_model("goat_health_model.pth", "goat_health_classes.json")
symptom_bundle = load_pickle_model("livestock_symptom_model.pkl")
text_bundle = load_pickle_model("pet_text_symptom_model.pkl")
ppr_bundle = load_pickle_model("ppr_model.pkl")

# PPR (ছাগলের প্লেগ) — বাংলাদেশে ছাগলের সবচেয়ে বড় ঘাতক, কিন্তু সাধারণ
# উপসর্গ-ডেটাসেটে এই রোগটাই নেই। তাই আলাদা ক্লিনিক্যাল ডেটা থেকে মডেল।
PPR_ACC = ppr_bundle["accuracy_at_threshold"] * 100 if ppr_bundle else 0
PPR_RECALL = ppr_bundle.get("recall_at_threshold", 0.9) * 100 if ppr_bundle else 0
PPR_THRESHOLD = ppr_bundle.get("threshold", 0.35) if ppr_bundle else 0.35


def predict_ppr(signs, age_adult, is_goat, is_male):
    """signs = {লক্ষণের key: True/False}. রিটার্ন: PPR-এর সম্ভাবনা (0-1)।"""
    import pandas as pd
    row = {f: int(signs.get(f, False)) for f in ppr_bundle["features"]}
    row["age"] = int(age_adult)
    row["animal"] = int(is_goat)
    row["sex"] = int(is_male)
    X = pd.DataFrame([row])[ppr_bundle["features"]]
    return float(ppr_bundle["model"].predict_proba(X)[0][1])

COW_ACC = 94.9          # হোল্ড-আউট ভ্যালিডেশনে মাপা (ফাঁস বাদ দিয়ে ৯৪.৬%)
PET_IMG_ACC = 50.8      # Top-3: 70.4%
GOAT_ACC = 96.4         # সুস্থ বনাম অসুস্থ — দুই শ্রেণির সহজ সমস্যা

# ছাগলের ছবির মডেল শুধু "সুস্থ না অসুস্থ" বলে, নির্দিষ্ট রোগ নয় —
# রোগ জানতে PPR যাচাই ও উপসর্গ অংশ ব্যবহার করতে হয়
GOAT_CLASS_INFO = {
    "healthy_goat":   "দেখে সুস্থ মনে হচ্ছে",
    "unhealthy_goat": "অসুস্থতার লক্ষণ দেখা যাচ্ছে",
}
SYMPTOM_ACC = symptom_bundle["test_accuracy"] * 100 if symptom_bundle else 0
TEXT_ACC = text_bundle["test_accuracy"] * 100 if text_bundle else 0

MERGED_CLASS = symptom_bundle.get("merged_class", "lumpy_or_pneumonia") if symptom_bundle else ""
PNEUMONIA_SIGNS = symptom_bundle.get("pneumonia_signs", []) if symptom_bundle else []
PNEUMONIA_THRESHOLD = symptom_bundle.get("pneumonia_threshold", 4) if symptom_bundle else 4
SYMPTOM_OPTIONS = symptom_bundle["symptom_options"] if symptom_bundle else []


@st.cache_resource
def load_species_verifier():
    """ImageNet-এ ট্রেইন করা ResNet18 — ছবিতে আসলে কোন প্রাণী আছে তা যাচাই করতে।"""
    w = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=w)
    net.eval()
    return net


# ImageNet-1k-এর যে শ্রেণিগুলো আমাদের প্রজাতিগুলোর সাথে মেলে
IMAGENET_GROUPS = {
    "dog": set(range(151, 269)),       # ~১১৮টি কুকুরের জাত
    "cat": set(range(281, 286)),       # গৃহপালিত বিড়াল
    "cattle": {345, 346, 347},         # ox, water buffalo, bison
    "sheep/goat": {348, 349, 350},     # ram, bighorn, ibex
}
SPECIES_GROUP = {
    "cow": "cattle", "buffalo": "cattle",
    "goat": "sheep/goat", "sheep": "sheep/goat",
    "dog": "dog", "cat": "cat",
}
# কোন গ্রুপগুলো একই রোগ-মডেল ব্যবহার করে। কুকুর ও বিড়াল একই মডেলে পড়ে,
# তাই ওদের মধ্যে গুলিয়ে গেলে ক্ষতি নেই — আসল সমস্যা গবাদি পশু বনাম পোষা প্রাণী।
MODEL_FAMILY = {"dog": "pet", "cat": "pet", "cattle": "livestock", "sheep/goat": "livestock"}
FAMILY_BN = {"pet": "কুকুর বা বিড়াল", "livestock": "গরু, মহিষ, ছাগল বা ভেড়া"}


def verify_species(pil_img, expected):
    """ছবিতে নির্বাচিত প্রজাতির প্রাণীই আছে কিনা যাচাই করে।

    রিটার্ন: ("ok" | "mismatch" | "unclear", নিজের স্কোর, অন্য পরিবারের স্কোর)

    "unclear" আসে যখন খুব কাছ থেকে তোলা ছবিতে প্রাণীর আকৃতিই দেখা যায় না
    (চামড়া, চোখ বা দাঁতের ক্লোজআপ) — এটা বৈধ ব্যবহার, তাই আটকানো হয় না।
    """
    net = load_species_verifier()
    x = infer_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(net(x), 1)[0]

    scores = {g: float(sum(p[i] for i in idxs)) for g, idxs in IMAGENET_GROUPS.items()}
    want = MODEL_FAMILY[SPECIES_GROUP[expected]]
    mine = sum(s for g, s in scores.items() if MODEL_FAMILY[g] == want)
    other = sum(s for g, s in scores.items() if MODEL_FAMILY[g] != want)

    if other > 0.55 and mine < 0.10:
        return "mismatch", mine, other
    if mine < 0.15 and other < 0.35:
        return "unclear", mine, other
    return "ok", mine, other


def predict_image(model, idx_to_class, pil_img, topk=3):
    """রিটার্ন: [(ক্লাসের নাম, শতকরা কনফিডেন্স), ...] — সবচেয়ে সম্ভাব্য আগে।
    ছবি ও তার আয়না-প্রতিবিম্ব দুটোর গড় নেওয়া হয় (TTA), এতে সামান্য নির্ভুলতা বাড়ে।"""
    x = infer_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)
        p = (p + torch.softmax(model(torch.flip(x, [3])), 1)) / 2
    p = p[0]
    k = min(topk, len(idx_to_class))
    conf, idx = torch.topk(p, k)
    return [(idx_to_class[str(int(i))], float(c) * 100) for c, i in zip(conf, idx)]


def predict_symptoms(animal, age, temp, s1, s2, s3):
    import pandas as pd
    enc = symptom_bundle["encoders"]
    row = {
        "Animal": enc["Animal"].transform([animal])[0],
        "Age": age,
        "Temperature": temp,
        "Symptom 1": enc["Symptom 1"].transform([s1])[0],
        "Symptom 2": enc["Symptom 2"].transform([s2])[0],
        "Symptom 3": enc["Symptom 3"].transform([s3])[0],
    }
    X = pd.DataFrame([row])[symptom_bundle["feature_order"]]
    model = symptom_bundle["model"]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))

    # ছাগল-ভেড়ায় Lumpy Skin Disease ও Blackleg বাস্তবে হয় না (Lumpy শুধু
    # গরু-মহিষে, Blackleg মূলত গরুতে)। ট্রেনিং ডেটা থেকে ওই সারিগুলো বাদ
    # দেওয়া হয়েছে, তবু বাড়তি সুরক্ষা হিসেবে এখানেও ছেঁকে নেওয়া হয়।
    valid = set(symptom_bundle.get("valid_diseases", {}).get(animal, []))
    if not valid:
        return proba, None

    def is_allowed(disease):
        # যৌথ ক্লাসটি তখনই রাখা হয় যখন তার অন্তত একটি সদস্য ওই প্রজাতিতে হয়
        if disease == MERGED_CLASS:
            return bool(valid & {"lumpy virus", "pneumonia"})
        return disease in valid

    # বৈধ ক্লাস কমে গেলেও softmax একটা না একটা বেছে নিতে বাধ্য — তাই ছাগলে
    # গুটির লক্ষণ দিলে আত্মবিশ্বাসের সাথে "FMD ৯৯%" বলে ফেলত। তাই দেখা হয়,
    # এই উপসর্গ-সমন্বয়টা ওই প্রজাতিতে আদৌ কখনো দেখা গেছে কিনা।
    combo = " + ".join(sorted([s1, s2, s3]))
    known = set(symptom_bundle.get("known_combos", {}).get(animal, []))
    unknown_combo = bool(known) and combo not in known

    # সমন্বয়টা অন্য কোন প্রজাতিতে দেখা যায়, সেটা জানালে ব্যবহারকারী বুঝতে পারেন
    seen_in = [a for a, cs in symptom_bundle.get("known_combos", {}).items()
               if a != animal and combo in set(cs)]
    rejected = (combo, seen_in) if unknown_combo else None

    filtered = {d: p for d, p in proba.items() if is_allowed(d)}
    total = sum(filtered.values())
    norm = {d: p / total for d, p in filtered.items()} if total else proba
    return norm, rejected


def predict_text(description):
    pipe = text_bundle["model"]
    return dict(zip(pipe.classes_, pipe.predict_proba([description])[0]))


# ===============================================================
# রুটিং ম্যাপ
# ===============================================================
SPECIES = {
    "cow": "গরু", "buffalo": "মহিষ", "goat": "ছাগল",
    "sheep": "ভেড়া", "dog": "কুকুর", "cat": "বিড়াল",
}
LIVESTOCK = {"cow", "buffalo", "goat", "sheep"}
PETS = {"dog", "cat"}

COW_CLASS_INFO = {
    "lumpy":          ("LUMPY SKIN DISEASE (গুটি বসন্ত)", "lumpy skin disease"),
    "foot-and-mouth": ("FOOT AND MOUTH DISEASE (ক্ষুরারোগ)", "foot and mouth disease (fmd)"),
    "healthy":        ("ত্বকে কোনো রোগ পাওয়া যায়নি", None),
}

DISEASE_BN = {
    "anthrax": "ANTHRAX (তড়কা)",
    "blackleg": "BLACKLEG (কালোপা)",
    "foot and mouth": "FOOT AND MOUTH DISEASE (ক্ষুরারোগ)",
    "lumpy virus": "LUMPY SKIN DISEASE (গুটি বসন্ত)",
    "pneumonia": "PNEUMONIA / BRD (নিউমোনিয়া)",
}
SYMPTOM_TO_MED = {
    "foot and mouth": "foot and mouth disease (fmd)",
    "lumpy virus": "lumpy skin disease",
    "pneumonia": "pneumonia / respiratory",
}

# ছবির ২২টি রোগ এই ৫টি বিস্তৃত শ্রেণিতে পড়ে — লেখা-উপসর্গের সাথে মেলানোর জন্য
PET_DISEASE_CATEGORY = {
    "Mange in Dog": "Skin Irritations", "Ringworm in Cat": "Skin Irritations",
    "Scabies in Cat": "Skin Irritations", "Skin Allergy in Cat": "Skin Irritations",
    "Skin Allergy in Dog": "Skin Irritations", "Hot Spots in Dog": "Skin Irritations",
    "Fungal Infection in Cat": "Skin Irritations", "Fungal Infection in Dog": "Skin Irritations",
    "Ear Mites in Cat": "Ear Infections",
    "Worm Infection in Cat": "Parasites", "Worm Infection in Dog": "Parasites",
    "Tick Infestation in Dog": "Parasites",
    "Parvovirus in Dog": "Digestive Issues", "Feline Panleukopenia": "Digestive Issues",
}

SPECIES_ICON = {
    "cow": "🐄", "buffalo": "🐃", "goat": "🐐",
    "sheep": "🐑", "dog": "🐕", "cat": "🐈",
}
# ওষুধের ভূমিকা অনুযায়ী আইকন, CSS ক্লাস ও ব্যাজের নাম
ROLE_STYLE = {
    "প্রধান":      ("💊", "",       "প্রধান চিকিৎসা"),
    "সহায়ক":      ("🧴", "supp",   "সহায়ক"),
    "ব্যবস্থাপনা": ("🏠", "manage", "ব্যবস্থাপনা"),
}


# ===============================================================
# পুনঃব্যবহারযোগ্য UI অংশ
# ===============================================================
def section(icon, title):
    st.markdown(f'<div class="vet-sec">{icon} {title}</div>', unsafe_allow_html=True)


def rx_card(kicker, name, stats):
    """প্রেসক্রিপশন-ঘরানার কার্ড। stats = [(লেবেল, মান), ...]"""
    chips = "".join(
        f'<div class="rx-stat"><b>{v}</b><span>{k}</span></div>' for k, v in stats
    )
    st.markdown(
        f'<div class="rx">'
        f'  <div class="rx-top"><span>{kicker}</span><span>Rx</span></div>'
        f'  <div class="rx-body">'
        f'    <p class="rx-name">{name}</p>'
        f'    <div class="rx-meta">{chips}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def show_medicines(meds, heading="এই রোগে সাধারণত যেসব ওষুধ ব্যবহার হয়"):
    section("💊", heading)
    for m in meds:
        icon, cls, tag = ROLE_STYLE.get(m.get("role", "সহায়ক"), ROLE_STYLE["সহায়ক"])
        st.markdown(
            f'<div class="med {cls}">'
            f'  <div class="med-ico">{icon}</div>'
            f'  <div style="flex:1">'
            f'    <div class="med-nm">{m["name"]}</div>'
            f'    <div class="med-ds">{m["purpose"]}</div>'
            f'  </div>'
            f'  <span class="med-tag {cls}">{tag}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "⚠️ **মাত্রা (ডোজ) ইচ্ছাকৃতভাবে দেওয়া হয়নি** — সঠিক মাত্রা নির্ভর করে প্রাণীর ওজন, "
        "বয়স, কিডনি-লিভারের অবস্থা ও গর্ভাবস্থার উপর, যা অ্যাপ জানে না। "
        "এই তালিকা ভেটেরিনারিয়ানের সাথে আলোচনার জন্য, নিজে প্রয়োগের জন্য নয়।"
    )


def show_urgent(title, message):
    st.error(f"### 🚨 জরুরি — এখনই ভেটের কাছে নিন\n\n**{title}**\n\n{message}", icon="🚨")


def confidence_note(pct):
    if pct >= 85:
        return "success", "✅ মডেল এই ফলাফলে বেশ নিশ্চিত"
    if pct >= 60:
        return "info", "🔍 মোটামুটি নিশ্চিত — নিচে উপসর্গ দিয়ে যাচাই করে নিন"
    return "warning", "⚠️ কনফিডেন্স কম — নিচে উপসর্গ দিয়ে যাচাই করুন, বা ভেট দেখান"


# ===============================================================
# হেডার
# ===============================================================
st.markdown(
    '<div class="vet-hero">'
    '  <h1>🩺 VetAI Nexus</h1>'
    '  <p>ছবি ও উপসর্গ মিলিয়ে প্রাণীর রোগ নির্ণয়ে প্রাথমিক সহায়তা</p>'
    '  <div class="vet-chips">'
    f'    <span class="vet-chip">🐄 গরু/মহিষ · {COW_ACC:.0f}%</span>'
    f'    <span class="vet-chip">🩺 উপসর্গ · {SYMPTOM_ACC:.0f}%</span>'
    f'    <span class="vet-chip">🐕 কুকুর/বিড়াল · {PET_IMG_ACC:.0f}%</span>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True,
)

if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None

# ---------------------------------------------------------------
# ধাপ ১ — প্রজাতি ও ছবি
# ---------------------------------------------------------------
section("🐾", "কোন প্রাণী?")
animal = st.segmented_control(
    "প্রাণীর প্রজাতি",
    options=list(SPECIES),
    format_func=lambda a: f"{SPECIES_ICON[a]} {SPECIES[a]}",
    default="cow",
    key="animal_pick",
    label_visibility="collapsed",
)
if not animal:
    animal = "cow"

is_livestock = animal in LIVESTOCK
is_pet = animal in PETS
is_goat_sheep = animal in {"goat", "sheep"}
has_image_model = (
    (animal in {"cow", "buffalo"} and cow_model)
    or (is_pet and pet_model)
    or (is_goat_sheep and goat_model)
)

if is_goat_sheep:
    st.info(
        f"{SPECIES[animal]}-এর জন্য তিনটা পথ আছে: **ছবি** (সুস্থ/অসুস্থ), "
        "**PPR যাচাই**, আর **সাধারণ উপসর্গ**।\n\n"
        "⚠️ তবে জেনে রাখুন: এই মডেলগুলো PPR, Anthrax ও FMD চেনে। "
        f"{SPECIES[animal]}-এর আরও কিছু গুরুত্বপূর্ণ রোগ — **Enterotoxaemia, CCPP, Orf, "
        "goatpox, গোদ (Liver fluke)** — এখনো যোগ করা হয়নি। লক্ষণ এই তালিকার সাথে "
        "না মিললে ভেটেরিনারি ডাক্তার দেখান।",
        icon="ℹ️",
    )
    st.caption(
        f"ছাগল/ভেড়ার ছবির মডেল শুধু **সুস্থ না অসুস্থ** বলতে পারে "
        f"(accuracy **{GOAT_ACC:.1f}%**), নির্দিষ্ট রোগ নয় — সেটা জানতে নিচের "
        "PPR ও উপসর্গ অংশ ব্যবহার করুন।"
    )
else:
    acc = COW_ACC if animal in {"cow", "buffalo"} else PET_IMG_ACC
    st.caption(f"এই প্রজাতির ছবির মডেলের পরিমাপকৃত accuracy: **{acc:.1f}%**")

up_file = None
if has_image_model:
    section("📷", "আক্রান্ত অংশের ছবি দিন")
    up_file = st.file_uploader(
        "ছবি আপলোড",
        type=["jpg", "jpeg", "png"],
        help="ফোনে ক্যামেরা দিয়ে সরাসরি তুলতে পারেন। আক্রান্ত জায়গা যেন স্পষ্ট দেখা যায়।",
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------
# ছবি থেকে ফল — সাথে সাথেই রোগ ও ওষুধ
# ---------------------------------------------------------------
img_class = None          # গরুর ক্ষেত্রে lumpy / healthy / foot-and-mouth
pet_disease = None        # কুকুর/বিড়ালের ২২টি রোগের একটি
goat_state = None         # ছাগল/ভেড়ার ক্ষেত্রে healthy_goat / unhealthy_goat

if up_file:
    img = Image.open(up_file).convert("RGB")

    # ---- প্রজাতি যাচাই ----
    # রোগের মডেলগুলো শুধু নিজেদের ক্লাসই চেনে — অন্য প্রাণীর ছবি দিলেও softmax
    # বাধ্য হয়ে কোনো একটা রোগ বেছে নেয়, এবং কনফিডেন্স ৮০%+ দেখাতে পারে।
    # তাই রোগ নির্ণয়ের আগে ছবিতে আসলে কোন প্রাণী আছে তা দেখে নেওয়া হয়।
    verdict, mine, other = verify_species(img, animal)
    other_family = "livestock" if MODEL_FAMILY[SPECIES_GROUP[animal]] == "pet" else "pet"

    if verdict == "mismatch":
        st.error(
            f"### ⚠️ ছবি ও প্রজাতি মিলছে না\n\n"
            f"আপনি **{SPECIES[animal]}** বেছেছেন, কিন্তু ছবিতে সম্ভবত "
            f"**{FAMILY_BN[other_family]}** আছে (মিল {other*100:.0f}%)।\n\n"
            f"{SPECIES[animal]}-এর মডেল শুধু {SPECIES[animal]}-এর রোগই চেনে। অন্য প্রাণীর ছবি "
            f"দিলে সেটা **বাধ্য হয়ে কোনো একটা রোগ বেছে নেবে** এবং ৮০%+ কনফিডেন্সও "
            f"দেখাতে পারে — অথচ ফলটা পুরোপুরি ভুল।\n\n"
            f"উপরে সঠিক প্রজাতি বেছে নিন।",
            icon="⚠️",
        )
        # যাচাই নিখুঁত নয় (মাঝেমধ্যে সঠিক ছবিকেও ভুল বলতে পারে),
        # তাই ব্যবহারকারীকে জোর করে আটকানো হয় না
        if not st.checkbox("ছবিটা ঠিক আছে, তবুও ফলাফল দেখান", key="force_species"):
            st.stop()
        st.caption("⚠️ প্রজাতি যাচাই উপেক্ষা করা হয়েছে — ফলাফল ভুল হতে পারে।")

    elif verdict == "unclear":
        st.info(
            f"ছবিতে প্রাণীটিকে স্পষ্ট চেনা যাচ্ছে না (সম্ভবত খুব কাছ থেকে তোলা)। "
            f"নিশ্চিত হয়ে নিন এটা **{SPECIES[animal]}**-এরই ছবি — ভুল প্রজাতি বাছলে "
            f"ফলাফল ভুল হবে।",
            icon="ℹ️",
        )

    if animal in {"cow", "buffalo"}:
        preds = predict_image(cow_model, cow_classes, img, topk=3)
        img_class, conf = preds[0]
        display, med_key = COW_CLASS_INFO.get(img_class, (img_class.upper(), "default_infection"))
    elif is_goat_sheep:
        preds = predict_image(goat_model, goat_classes, img, topk=2)
        goat_state, conf = preds[0]
        display, med_key = GOAT_CLASS_INFO.get(goat_state, goat_state), None
    else:
        preds = predict_image(pet_model, pet_classes, img, topk=3)
        pet_disease, conf = preds[0]
        display, med_key = pet_disease, None

    section("🔬", "ছবি থেকে সনাক্ত")
    c1, c2 = st.columns([1, 1.25], vertical_alignment="top")
    with c1:
        st.image(img, use_container_width=True)
    with c2:
        acc_now = (COW_ACC if animal in {"cow", "buffalo"}
                   else GOAT_ACC if is_goat_sheep else PET_IMG_ACC)
        rx_card(
            f"{SPECIES_ICON[animal]} {SPECIES[animal]} · ছবি বিশ্লেষণ",
            display,
            [("কনফিডেন্স", f"{conf:.0f}%"), ("মডেল accuracy", f"{acc_now:.0f}%")],
        )

    level, note = confidence_note(conf)
    getattr(st, level)(note)

    # ২য় ও ৩য় সম্ভাবনা — কুকুর/বিড়ালে Top-1 মাত্র ৫০.৮%, তাই এটা জরুরি
    if len(preds) > 1 and conf < 90:
        with st.expander("অন্যান্য সম্ভাবনা দেখুন"):
            for name, p in preds[1:]:
                label = COW_CLASS_INFO.get(name, (name, None))[0] if animal in {"cow", "buffalo"} else name
                st.progress(min(p / 100, 1.0), text=f"{label} — {p:.1f}%")
            if is_pet:
                st.caption(
                    f"এই মডেলের Top-1 accuracy {PET_IMG_ACC:.1f}%, কিন্তু Top-3 **৭০.৪%** — "
                    "অর্থাৎ সঠিক রোগটি এই তিনটির মধ্যে থাকার সম্ভাবনা অনেক বেশি।"
                )

    st.divider()

    # ---- ওষুধ / জরুরি বার্তা ----
    if animal in {"cow", "buffalo"}:
        if img_class == "healthy":
            st.success(
                "ছবিতে ত্বকের কোনো রোগ (গুটি বা ক্ষত) ধরা পড়েনি।\n\n"
                "তবে মনে রাখবেন — **নিউমোনিয়া, পেটের অসুখ বা জ্বর ছবিতে দেখা যায় না**। "
                "প্রাণীর মধ্যে অন্য কোনো লক্ষণ থাকলে নিচে উপসর্গ দিয়ে যাচাই করুন।",
                icon="✅",
            )
        else:
            show_medicines(LIVESTOCK_MEDICINES[med_key])

    elif is_goat_sheep:
        # ছবি শুধু সুস্থ/অসুস্থ বলে — কোন রোগ তা জানতে উপসর্গ লাগে
        if goat_state == "healthy_goat":
            st.success(
                f"ছবিতে {SPECIES[animal]}টিকে দেখে সুস্থ মনে হচ্ছে।\n\n"
                "তবে ছবি দিয়ে শুধু বাইরের অবস্থা বোঝা যায় — **জ্বর, পেটের অসুখ বা "
                "ফুসফুসের সমস্যা ছবিতে ধরা পড়ে না**। সন্দেহ হলে নিচের PPR যাচাই ও "
                "উপসর্গ অংশ ব্যবহার করুন।",
                icon="✅",
            )
        else:
            st.warning(
                f"ছবিতে {SPECIES[animal]}টির মধ্যে **অসুস্থতার লক্ষণ** দেখা যাচ্ছে।\n\n"
                "কিন্তু ছবি দিয়ে কোন রোগ তা বলা যায় না। **নিচের PPR যাচাই ও উপসর্গ "
                "অংশ পূরণ করুন** — সেখানেই রোগ ও ওষুধ দেখানো হবে।",
                icon="⚠️",
            )
    else:
        if pet_disease in URGENT_PET_DISEASES:
            show_urgent(pet_disease, URGENT_PET_DISEASES[pet_disease])
        else:
            if pet_disease in ZOONOTIC_PET_DISEASES:
                st.warning(
                    "এই রোগটি **মানুষের মধ্যেও ছড়াতে পারে**। প্রাণী ধরার পর সাবান দিয়ে "
                    "হাত ধুয়ে নিন এবং শিশুদের দূরে রাখুন।",
                    icon="⚠️",
                )
            meds = PET_MEDICINES.get(pet_disease)
            if meds:
                show_medicines(meds)
            else:
                section("🏠", "যত্নের পরামর্শ")
                for tip in GENERIC_PET_CARE:
                    st.markdown(f"- {tip}")
                st.caption(
                    "এই রোগের জন্য নির্দিষ্ট ওষুধের তালিকা এখনো যোগ করা হয়নি — "
                    "ভেটেরিনারিয়ানের পরামর্শ নিন।"
                )

        st.info(
            f"কুকুর-বিড়ালের ছবির মডেলের accuracy মাত্র **{PET_IMG_ACC:.1f}%**। "
            "নিচে **লক্ষণ লিখে যাচাই** অংশটা ব্যবহার করলে নির্ভরযোগ্যতা অনেক বাড়বে "
            f"(সেই মডেলের accuracy **{TEXT_ACC:.1f}%**)।",
            icon="💡",
        )

# ---------------------------------------------------------------
# ঐচ্ছিক — উপসর্গ দিয়ে যাচাই
# ---------------------------------------------------------------
st.divider()

# ---- ছাগল/ভেড়া: PPR যাচাই ----
# PPR-এর লক্ষণ সাধারণ উপসর্গ-মডেলের তালিকায় নেই, তাই আলাদা করে নেওয়া হয়
if animal in {"goat", "sheep"} and ppr_bundle:
    with st.expander("🐐 PPR (ছাগলের প্লেগ) যাচাই করুন", expanded=True):
        st.caption(
            f"PPR বাংলাদেশে ছাগলের সবচেয়ে বড় ঘাতক। এই মডেল **আসল ক্লিনিক্যাল ডেটা** "
            f"(২১,১৯৯ রেকর্ড) থেকে তৈরি — accuracy **{PPR_ACC:.0f}%**, "
            f"এবং PPR আক্রান্তদের **{PPR_RECALL:.0f}%** ধরতে পারে।"
        )
        signs = {}
        pc = st.columns(2)
        for i, (key, label) in enumerate(ppr_bundle["sign_labels"].items()):
            signs[key] = pc[i % 2].checkbox(label, key=f"ppr_{key}")

        c1, c2 = st.columns(2)
        age_adult = c1.radio("বয়স", ["বাচ্চা", "প্রাপ্তবয়স্ক"], horizontal=True, key="ppr_age") == "প্রাপ্তবয়স্ক"
        is_male = c2.radio("লিঙ্গ", ["মাদী", "পুরুষ"], horizontal=True, key="ppr_sex") == "পুরুষ"

        if st.button("PPR যাচাই করুন", type="primary", use_container_width=True):
            st.session_state.ppr = predict_ppr(signs, age_adult, animal == "goat", is_male)

    if st.session_state.get("ppr") is not None:
        p = st.session_state.ppr
        section("🐐", "PPR যাচাইয়ের ফল")
        st.progress(min(p, 1.0), text=f"PPR-এর সম্ভাবনা — {p*100:.0f}%")
        if p >= PPR_THRESHOLD:
            st.error(
                f"### ⚠️ PPR-এর সম্ভাবনা আছে ({p*100:.0f}%)\n\n"
                "PPR অত্যন্ত সংক্রামক — একটি আক্রান্ত ছাগল থেকে পুরো পাল আক্রান্ত হতে পারে, "
                "আর মৃত্যুহার ৫০-৮০% পর্যন্ত।\n\n"
                "**এখনই যা করবেন:**\n"
                "- আক্রান্ত প্রাণীকে সাথে সাথে আলাদা করুন\n"
                "- নিকটস্থ ভেটেরিনারি ডাক্তার বা প্রাণিসম্পদ অফিসে জানান\n"
                "- বাকি সুস্থ ছাগল-ভেড়াকে টিকা দেওয়ার ব্যবস্থা করুন\n"
                "- PPR-এর নিজস্ব ওষুধ নেই; অ্যান্টিবায়োটিক শুধু সেকেন্ডারি সংক্রমণ ঠেকায়, "
                "সাথে পানিশূন্যতা রোধে স্যালাইন দিতে হয়",
                icon="⚠️",
            )
        else:
            st.success(
                f"এই লক্ষণগুলোতে PPR-এর সম্ভাবনা কম ({p*100:.0f}%)। "
                "তবে নিচে সাধারণ উপসর্গ দিয়েও যাচাই করে নিন।",
                icon="✅",
            )
            st.caption(
                f"মনে রাখবেন, এই মডেল PPR আক্রান্তদের {PPR_RECALL:.0f}% ধরতে পারে — "
                "অর্থাৎ ১০ জনে ১ জন ধরা নাও পড়তে পারে। লক্ষণ বাড়লে ভেট দেখান।"
            )

if is_livestock and symptom_bundle:
    expanded = animal in {"goat", "sheep"} or not up_file
    with st.expander("🩺 উপসর্গ দিয়ে যাচাই করুন (আরও নিশ্চিত হতে)", expanded=expanded):
        st.caption(f"উপসর্গ-মডেলের পরিমাপকৃত accuracy: **{SYMPTOM_ACC:.1f}%**")

        c1, c2 = st.columns(2)
        age = c1.number_input("বয়স (বছর)", 0.1, 30.0, 2.5, step=0.5)
        temp = c2.slider("তাপমাত্রা (°F)", 95.0, 108.0, 102.5, step=0.1)

        s1 = st.selectbox("লক্ষণ ১", SYMPTOM_OPTIONS, key="s1")
        s2 = st.selectbox("লক্ষণ ২", SYMPTOM_OPTIONS, index=min(1, len(SYMPTOM_OPTIONS) - 1), key="s2")
        s3 = st.selectbox("লক্ষণ ৩", SYMPTOM_OPTIONS, index=min(2, len(SYMPTOM_OPTIONS) - 1), key="s3")

        # নিউমোনিয়া ছবিতে ধরা পড়ে না — এই লক্ষণগুলোই একমাত্র ভরসা
        st.markdown("**শ্বাসতন্ত্রের লক্ষণ** (যেগুলো আছে টিক দিন)")
        st.caption("নিউমোনিয়া ছবিতে দেখা যায় না, তাই এগুলো ছাড়া তা নিশ্চিত করা যায় না।")
        resp_score, resp_checked = 0, []
        rc = st.columns(2)
        for i, sign in enumerate(PNEUMONIA_SIGNS):
            if rc[i % 2].checkbox(sign["bn"], key=f"resp_{sign['key']}"):
                resp_score += sign["weight"]
                resp_checked.append(sign["bn"])

        if st.button("উপসর্গ বিশ্লেষণ করুন", type="primary", use_container_width=True):
            proba, rejected = predict_symptoms(animal, age, temp, s1, s2, s3)
            top = max(proba, key=proba.get)
            st.session_state.diagnosis = {
                "proba": proba, "top": top, "rejected": rejected,
                "resp_score": resp_score, "resp_checked": resp_checked,
            }

    d = st.session_state.diagnosis
    if d:
        section("🩺", "উপসর্গ-ভিত্তিক ফলাফল")

        # লক্ষণ এই প্রজাতিতে হয় না এমন রোগের দিকে ইঙ্গিত করলে, জোর করে
        # বাকি রোগগুলোর একটা বেছে না দিয়ে সেটা স্পষ্ট জানানো হয়
        if d.get("rejected"):
            _, seen_in = d["rejected"]
            extra = ""
            if seen_in:
                others = ", ".join(SPECIES.get(a, a) for a in seen_in)
                extra = (f"\n\nএই লক্ষণগুলো **{others}**-এ দেখা যায়, কিন্তু "
                         f"{SPECIES[animal]}-এ নয়। Lumpy Skin Disease ও Blackleg "
                         f"মূলত গরু-মহিষের রোগ।")
            st.warning(
                f"**এই লক্ষণগুলো {SPECIES[animal]}-এর চেনা কোনো রোগের সাথে মিলছে না।**"
                f"{extra}\n\n"
                f"তাই নিচের ফলাফল নির্ভরযোগ্য নয় — মডেল বাধ্য হয়ে একটা উত্তর বেছেছে। "
                f"হয় লক্ষণ ভুল বাছাই হয়েছে, নয়তো এটি এমন রোগ যা এই মডেল এখনো চেনে না "
                f"(goatpox, Enterotoxaemia, CCPP, Orf)। **ভেটেরিনারি ডাক্তার দেখান।**",
                icon="⚠️",
            )

        for disease, p in sorted(d["proba"].items(), key=lambda x: -x[1]):
            label = ("LUMPY অথবা PNEUMONIA (উপসর্গে আলাদা করা যায় না)"
                     if disease == MERGED_CLASS else DISEASE_BN.get(disease, disease))
            st.progress(min(float(p), 1.0), text=f"{label} — {p*100:.1f}%")

        top = d["top"]

        if top in URGENT_LIVESTOCK_DISEASES:
            show_urgent(DISEASE_BN.get(top, top.upper()), URGENT_LIVESTOCK_DISEASES[top])

        elif top == MERGED_CLASS:
            # উপসর্গ বলছে lumpy বা pneumonia, কিন্তু ওদের উপসর্গ অভিন্ন।
            # ছবি ও শ্বাসলক্ষণ দিয়ে সিদ্ধান্ত নেওয়া হয়।
            has_resp = d["resp_score"] >= PNEUMONIA_THRESHOLD
            if img_class == "lumpy":
                st.success(
                    "**সিদ্ধান্ত: LUMPY SKIN DISEASE** — ছবিতে গায়ে গুটি সনাক্ত হয়েছে, "
                    "আর Lumpy চোখে দেখা যায় বলে ছবির প্রমাণই এখানে নির্ণায়ক।"
                    + (f"\n\nতবে শ্বাসতন্ত্রের লক্ষণও আছে (স্কোর {d['resp_score']}) — "
                       "একসাথে দুটো রোগ থাকা সম্ভব, ভেটকে দুটোর কথাই জানান।" if has_resp else ""),
                    icon="🔗",
                )
                show_medicines(LIVESTOCK_MEDICINES["lumpy skin disease"])
            elif has_resp:
                st.success(
                    f"**সিদ্ধান্ত: PNEUMONIA (BRD)** — ত্বকে কিছু পাওয়া যায়নি, আর "
                    f"শ্বাসতন্ত্রের লক্ষণ স্কোর {d['resp_score']} (সীমা {PNEUMONIA_THRESHOLD})।\n\n"
                    f"টিক দেওয়া লক্ষণ: {', '.join(d['resp_checked'])}",
                    icon="🔗",
                )
                show_medicines(LIVESTOCK_MEDICINES["pneumonia / respiratory"])
            elif img_class == "foot-and-mouth":
                st.warning(
                    "**ছবি ও উপসর্গ পরস্পরবিরোধী** — ছবি বলছে Foot and Mouth Disease, "
                    "কিন্তু উপসর্গ বলছে Lumpy বা Pneumonia। কোনো একটা সিদ্ধান্ত চাপিয়ে "
                    "দেওয়া হচ্ছে না; ভেটেরিনারি ডাক্তারকে দেখিয়ে নিশ্চিত হন।",
                    icon="⚠️",
                )
            else:
                st.warning(
                    "**নিশ্চিতভাবে বলা যাচ্ছে না।** ছবিতে গুটি পাওয়া যায়নি, তাই Lumpy-র "
                    "সম্ভাবনা কম। কিন্তু শ্বাসতন্ত্রের কোনো লক্ষণও টিক দেওয়া হয়নি — আর "
                    "**গায়ে গুটি না থাকা একা নিউমোনিয়ার প্রমাণ নয়**, সুস্থ প্রাণীর গায়েও "
                    "গুটি থাকে না। ভেটেরিনারি ডাক্তারকে দেখান।",
                    icon="⚠️",
                )
        else:
            st.success(f"**সিদ্ধান্ত: {DISEASE_BN.get(top, top.upper())}**", icon="✅")
            show_medicines(LIVESTOCK_MEDICINES[SYMPTOM_TO_MED.get(top, "default_infection")])

elif is_pet and text_bundle:
    with st.expander("✍️ লক্ষণ লিখে যাচাই করুন (ছবির চেয়ে নির্ভরযোগ্য)", expanded=not up_file):
        st.caption(
            f"এই মডেলের accuracy **{TEXT_ACC:.1f}%** — ছবির মডেলের ({PET_IMG_ACC:.1f}%) চেয়ে বেশি। "
            "বমি, ডায়রিয়া, খোঁড়ানো বা কাশির মতো যেসব সমস্যা ছবিতে দেখা যায় না, সেগুলো এখানে লিখুন। "
            "ইংরেজিতে লিখলে সবচেয়ে ভালো কাজ করে।"
        )
        desc = st.text_area(
            "লক্ষণের বর্ণনা",
            placeholder="My dog has been vomiting since yesterday and refuses to eat",
            height=100,
            label_visibility="collapsed",
        )
        if st.button("লক্ষণ বিশ্লেষণ করুন", type="primary", use_container_width=True) and desc.strip():
            st.session_state.diagnosis = {"text": predict_text(desc.strip())}

    d = st.session_state.diagnosis
    if d and "text" in d:
        section("✍️", "লেখা লক্ষণ থেকে বিশ্লেষণ")
        ranked = sorted(d["text"].items(), key=lambda x: -x[1])
        for cat, p in ranked[:3]:
            st.progress(min(float(p), 1.0), text=f"{cat} — {p*100:.1f}%")

        # ছবি ও লেখা — দুটো স্বাধীন প্রমাণ মিলছে কিনা
        if pet_disease:
            img_cat = PET_DISEASE_CATEGORY.get(pet_disease)
            text_cat = ranked[0][0]
            if img_cat and img_cat == text_cat:
                st.success(
                    f"**দুটো প্রমাণ একমত।** ছবি বলছে \"{pet_disease}\" (\"{img_cat}\" শ্রেণি), "
                    "আর আপনার লেখা লক্ষণও একই শ্রেণি বলছে। দুটো আলাদা পদ্ধতি মিলে যাওয়ায় "
                    "এই ফলাফলে যেকোনো একটার চেয়ে বেশি আস্থা রাখা যায়।",
                    icon="✅",
                )
            elif img_cat:
                st.warning(
                    f"**দুটো প্রমাণ মিলছে না।** ছবি বলছে \"{pet_disease}\" (\"{img_cat}\"), "
                    f"কিন্তু লেখা লক্ষণ বলছে \"{text_cat}\"। ছবির মডেলের accuracy মাত্র "
                    f"{PET_IMG_ACC:.1f}%, তাই এখানে **লেখা লক্ষণকে বেশি গুরুত্ব দিন** — "
                    "এবং ভেটেরিনারি ডাক্তারকে দেখান।",
                    icon="⚠️",
                )
            else:
                st.info(
                    f"ছবির ফল \"{pet_disease}\" এই ৫টি বিস্তৃত শ্রেণির কোনোটিতে পড়ে না, "
                    "তাই সরাসরি মেলানো যাচ্ছে না। দুটো ফলাফলই ভেটকে জানান।",
                    icon="ℹ️",
                )

# ---------------------------------------------------------------
# ফুটার
# ---------------------------------------------------------------
st.divider()
st.caption(
    "⚕️ **এটি একটি প্রাথমিক সহায়ক টুল, ভেটেরিনারি ডাক্তারের বিকল্প নয়।** "
    "কোনো ওষুধের মাত্রা এখানে দেওয়া হয় না। চিকিৎসা শুরুর আগে অবশ্যই একজন "
    "নিবন্ধিত ভেটেরিনারি ডাক্তারের পরামর্শ নিন।"
)
