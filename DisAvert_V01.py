# ═══════════════════════════════════════════════════════════════
#  DisAvert  ·  Coopérathon 2026  ·  Défi La Planète
#  Wildfire Risk Intelligence Platform — Canada
#  Version 03  ·  Production MVP
#  Run: py -m streamlit run DisAvert_V01.py
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import requests
from datetime import datetime, timezone
import folium
from streamlit_folium import st_folium
import os, joblib, json, time
from catboost import CatBoostRegressor

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="DisAvert",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
#  COPY  —  EN / FR
# ═══════════════════════════════════════════════════════════════
COPY = {
    "EN": {
        "sb_title":    "DisAvert",
        "sb_sub":      "Wildfire Intelligence · Canada",
        "user_label":  "User profile",
        "user_opts":   ["Firefighters / SOPFEU", "Insurance Company", "Municipality", "Other"],
        "specify":     "Specify your role",
        "loc_label":   "Location",
        "mode_opts":   ["Search address", "Interactive map"],
        "addr_ph":     "e.g. Vancouver, BC",
        "search_btn":  "→ Locate",
        "coords_lbl":  "Active coordinates",
        "sb_meta":     "Model · CatBoost Quantile · 18 vars · Open-Meteo · Coopérathon 2026",
        "tag":         "DisAvert",
        "headline1":   "Every second counts.",
        "headline2":   "Every forest matters.",
        "sub":         "Protecting Canadian communities and ecosystems through AI-powered wildfire risk prediction. Because nature cannot wait.",
        "cta1":        "analyses run",
        "cta2":        "high-risk alerts detected",
        "run_btn":     "⚡  RUN RISK ANALYSIS",
        "map_lbl":     "Select location on map",
        "init":        "Initializing quantile models…",
        "spin":        "Fetching live data · Running inference…",
        "t_result":    "RESULT",
        "t_model":     "MODEL",
        "t_pipeline":  "PIPELINE",
        "prob_lbl":    "Fire Probability",
        "ci_lbl":      "80% Confidence Interval",
        "r_extreme":   "Extreme Risk",
        "r_high":      "High Risk",
        "r_moderate":  "Moderate Risk",
        "live_lbl":    "Live Conditions",
        "live_tag":    "Live",
        "est_tag":     "Estimated",
        "f_temp":      "Temperature",
        "f_hum":       "Humidity",
        "f_wind":      "Wind",
        "f_gust":      "Gusts",
        "f_prec":      "Precipitation",
        "f_dew":       "Dew point",
        "f_surf":      "Surface temp",
        "drivers_lbl": "Primary Risk Drivers",
        "rec_lbl":     "AI Recommendation",
        "fc_lbl":      "7-Day Risk Forecast",
        "loc_lbl2":    "Analysed Location",
        "var_lbl":     "18 model input variables",
        "var_note":    "Static in MVP — will be fed in real-time by Sentinel-2, CDEM and NALCMS in production.",
        "var_note2":   "All values normalized [0, 1] before CatBoost inference.",
        "pip_lbl":     "Data Pipeline — Production",
        "mod_lbl":     "How the Model Works",
        "mod_txt": (
            "DisAvert uses CatBoost gradient boosting trained on 18 meteorological, satellite, and terrain variables. "
            "For each location, it combines live weather conditions with fire weather indices, vegetation health, surface temperature, "
            "historical fire density, and topographic factors to output a wildfire probability score. "
            "Quantile regression produces a confidence interval alongside each prediction,"
            "so decision-makers see not just the expected risk but also the uncertainty around it. "
            "In production, all 18 variables are fed from official Canadian and satellite sources in real time."
        ),
        "footer": "Coopérathon 2026 · Défi La Planète · CatBoost · Open-Meteo · NRCan · NASA FIRMS",
    },
    "FR": {
        "sb_title":    "DisAvert",
        "sb_sub":      "Intelligence Feux · Canada",
        "user_label":  "Profil utilisateur",
        "user_opts":   ["Pompiers / SOPFEU", "Compagnie d'assurance", "Municipalité", "Autre"],
        "specify":     "Précisez votre rôle",
        "loc_label":   "Localisation",
        "mode_opts":   ["Adresse", "Carte interactive"],
        "addr_ph":     "ex. Vancouver, BC",
        "search_btn":  "→ Localiser",
        "coords_lbl":  "Coordonnées actives",
        "sb_meta":     "Modèle · CatBoost Quantile · 18 vars · Open-Meteo · Coopérathon 2026",
        "tag":         "DisAvert",
        "headline1":   "Chaque seconde compte.",
        "headline2":   "Chaque forêt compte.",
        "sub":         "Protéger les communautés et les écosystèmes canadiens grâce à la prédiction IA des feux de forêt. Parce que la nature ne peut pas attendre.",
        "cta1":        "analyses effectuées",
        "cta2":        "alertes à risque élevé",
        "run_btn":     "⚡  LANCER L'ANALYSE",
        "map_lbl":     "Sélectionner la localisation",
        "init":        "Initialisation des modèles quantiles…",
        "spin":        "Collecte des données live · Inférence en cours…",
        "t_result":    "RÉSULTAT",
        "t_model":     "MODÈLE",
        "t_pipeline":  "PIPELINE",
        "prob_lbl":    "Probabilité d'incendie",
        "ci_lbl":      "Intervalle de confiance 80%",
        "r_extreme":   "Risque Extrême",
        "r_high":      "Risque Élevé",
        "r_moderate":  "Risque Modéré",
        "live_lbl":    "Conditions Live",
        "live_tag":    "Live",
        "est_tag":     "Estimé",
        "f_temp":      "Température",
        "f_hum":       "Humidité",
        "f_wind":      "Vent",
        "f_gust":      "Rafales",
        "f_prec":      "Précipitation",
        "f_dew":       "Point de rosée",
        "f_surf":      "Temp. surface",
        "drivers_lbl": "Facteurs de risque principaux",
        "rec_lbl":     "Recommandation IA",
        "fc_lbl":      "Prévision 7 jours",
        "loc_lbl2":    "Localisation analysée",
        "var_lbl":     "18 variables d'entrée du modèle",
        "var_note":    "Variables statiques en MVP — seront alimentées en temps réel par Sentinel-2, CDEM et NALCMS en production.",
        "var_note2":   "Toutes les valeurs sont normalisées [0, 1] avant l'inférence CatBoost.",
        "pip_lbl":     "Pipeline de données — Production",
        "mod_lbl":     "Comment fonctionne le modèle",
        "mod_txt": (
            "DisAvert utilise CatBoost entraîné sur 18 variables météorologiques, satellitaires et topographiques. "
            "Pour chaque localisation, il combine les conditions météo live avec les indices d'incendie, la santé de la végétation, "
            "la température de surface, la densité historique des feux et les facteurs de terrain "
            "pour produire un score de probabilité d'incendie. "
            "La régression quantile produit un intervalle de confiance avec chaque prédiction. "
            "Les décideurs voient à la fois le risque attendu et l'incertitude associée. "
            "En production, les 18 variables sont alimentées en temps réel depuis des sources officielles canadiennes et satellitaires."
        ),
        "footer": "Coopérathon 2026 · Défi La Planète · CatBoost · Open-Meteo · NRCan · NASA FIRMS",
    }
}

RECS = {
    "EN": {
        "extreme": {
            "Firefighters / SOPFEU":
                "Extreme fire risk confirmed. Deploy ground and aerial units immediately. Activate the evacuation plan. Coordinate with civil protection and SOPFEU. Restrict all forest access now.",
            "Insurance Company":
                "Catastrophe protocol activated. Notify policyholders in exposed zones immediately. Mobilize claims assessment teams. Review portfolio exposure in the flagged region. Prepare surge capacity.",
            "Municipality":
                "Trigger municipal emergency plan immediately. Coordinate with SOPFEU and Emergency Management Canada. Issue public alerts. Restrict access to green and forested areas. Mobilize emergency shelter resources.",
            "Other":
                "Extreme risk detected. Immediate action required. Avoid all forested and wildland areas. Follow official SOPFEU communications closely.",
        },
        "high": {
            "Firefighters / SOPFEU":
                "Elevated fire risk. Place units on standby alert. Increase patrol frequency in high-risk sectors. Verify equipment readiness and water reserves. Pre-position aerial assets.",
            "Insurance Company":
                "Elevated risk. Review exposure in flagged zones. Prepare client communications for high-risk areas. Monitor weather evolution over the next 72 hours. Alert claims team.",
            "Municipality":
                "Elevated risk. Restrict all open burns. Advise residents in forest-adjacent areas to remain vigilant. Maintain preventive coordination with SOPFEU. Inform local fire departments.",
            "Other":
                "Elevated risk. Enhanced monitoring recommended. Avoid spark-generating activities in forested areas. Stay informed via official channels.",
        },
        "moderate": {
            "default":
                "Risk within normal parameters. Routine monitoring sufficient. No immediate action required. Continue standard prevention protocols.",
        },
    },
    "FR": {
        "extreme": {
            "Pompiers / SOPFEU":
                "Risque extrême confirmé. Déploiement immédiat des équipes terrestres et aériennes. Activer le plan d'évacuation. Coordonner avec la protection civile et la SOPFEU. Restreindre l'accès aux zones forestières immédiatement.",
            "Compagnie d'assurance":
                "Protocole catastrophe activé. Notifier les assurés dans les zones exposées immédiatement. Mobiliser les équipes d'évaluation des sinistres. Réviser l'exposition du portefeuille dans la région. Préparer la capacité de traitement accrue.",
            "Municipalité":
                "Déclencher le plan d'urgence municipal immédiatement. Coordonner avec la SOPFEU et Urgences Canada. Émettre des alertes publiques. Restreindre l'accès aux espaces verts et forestiers. Mobiliser les ressources d'hébergement d'urgence.",
            "Autre":
                "Risque extrême détecté. Action immédiate requise. Éviter toutes les zones forestières et naturelles. Suivre les communications officielles de la SOPFEU.",
        },
        "high": {
            "Pompiers / SOPFEU":
                "Risque élevé. Mettre les unités en état d'alerte. Augmenter la fréquence des patrouilles dans les secteurs à haut risque. Vérifier l'état de préparation de l'équipement et les réserves en eau.",
            "Compagnie d'assurance":
                "Risque élevé. Réviser l'exposition dans les zones signalées. Préparer les communications clients pour les zones à risque. Surveiller l'évolution des conditions sur 72h. Alerter l'équipe sinistres.",
            "Municipalité":
                "Risque élevé. Restreindre les feux à ciel ouvert. Aviser les résidents des zones adjacentes aux forêts de rester vigilants. Maintenir une coordination préventive avec la SOPFEU.",
            "Autre":
                "Risque élevé. Surveillance accrue recommandée. Éviter les activités générant des étincelles en milieu forestier. Rester informé via les canaux officiels.",
        },
        "moderate": {
            "default":
                "Risque dans les paramètres normaux. Surveillance de routine suffisante. Aucune action immédiate requise. Maintenir les protocoles de prévention standards.",
        },
    }
}

# ═══════════════════════════════════════════════════════════════
#  CSS  —  PURE BLACK, ZERO BLUE
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],.main,.block-container{
  background-color:#000 !important;color:#e0e0e0 !important;
  font-family:'DM Sans',sans-serif !important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="StyledFullScreenButton"],
button[title="View fullscreen"],[data-testid="stBaseButton-headerNoPadding"]{display:none !important;}
/* kill keyboard_double artifact */
[data-baseweb="tab"] span[role="img"],
[data-baseweb="tab"] > div > span:first-child:not([class]),
.stTabs [data-testid="stMarkdownContainer"] > p:empty{display:none !important;}
/* SIDEBAR */
[data-testid="stSidebar"]{background-color:#080808 !important;border-right:1px solid rgba(255,255,255,0.04) !important;}
[data-testid="stSidebar"] *{color:#999 !important;font-family:'DM Sans',sans-serif !important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stTextInput label{
  color:#2a2a2a !important;font-size:0.6rem !important;text-transform:uppercase !important;
  letter-spacing:.12em !important;font-family:'JetBrains Mono',monospace !important;}
[data-testid="stSidebar"] .stSelectbox>div>div,
[data-testid="stSidebar"] .stTextInput>div>div>input,
[data-testid="stSidebar"] .stRadio>div{
  background:#0d0d0d !important;border:1px solid rgba(255,255,255,0.05) !important;
  color:#e0e0e0 !important;border-radius:3px !important;}
[data-testid="stSidebar"] .stButton>button{
  background:transparent !important;color:#e0e0e0 !important;
  border:1px solid rgba(255,255,255,0.08) !important;border-radius:3px !important;
  font-family:'JetBrains Mono',monospace !important;font-size:0.68rem !important;
  letter-spacing:.06em !important;width:100% !important;transition:all .15s !important;}
[data-testid="stSidebar"] .stButton>button:hover{border-color:#e74c3c !important;color:#e74c3c !important;}
/* PRIMARY BUTTON */
.stButton>button[kind="primary"]{
  background:#e74c3c !important;color:#fff !important;border:none !important;
  border-radius:3px !important;font-family:'JetBrains Mono',monospace !important;
  font-size:0.72rem !important;font-weight:500 !important;letter-spacing:.18em !important;
  height:3rem !important;width:100% !important;text-transform:uppercase !important;
  transition:background .15s !important;}
.stButton>button[kind="primary"]:hover{background:#c0392b !important;}
.stButton>button[kind="primary"] *{color:#fff !important;}       
/* TABS */
[data-baseweb="tab-list"]{background:transparent !important;border-bottom:1px solid rgba(255,255,255,0.05) !important;gap:0 !important;}
[data-baseweb="tab"]{
  background:transparent !important;font-family:'JetBrains Mono',monospace !important;
  font-size:0.62rem !important;letter-spacing:.16em !important;text-transform:uppercase !important;
  color:#2a2a2a !important;border-bottom:2px solid transparent !important;padding:.8rem 1.5rem !important;}
[aria-selected="true"][data-baseweb="tab"]{color:#e74c3c !important;border-bottom-color:#e74c3c !important;}
[data-testid="stTabContent"]{background:transparent !important;padding-top:1.5rem !important;}
/* MISC */
.stSpinner>div{border-top-color:#e74c3c !important;}
h1,h2,h3,h4{font-family:'DM Serif Display',serif !important;color:#e0e0e0 !important;}
p,li,span{color:#666;}
hr{border-color:rgba(255,255,255,0.04) !important;}
iframe{border-radius:3px !important;border:1px solid rgba(255,255,255,0.04) !important;}
/* CARDS */
.dg-card{background:#080808;border:1px solid rgba(255,255,255,0.05);border-radius:3px;padding:1.8rem 2rem;height:100%;}
/* LABEL */
.dg-lbl{font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:#1e1e1e;text-transform:uppercase;
  letter-spacing:.18em;margin:1.2rem 0 .8rem 0;display:flex;align-items:center;gap:.8rem;}
.dg-lbl::after{content:'';flex:1;height:1px;background:rgba(255,255,255,0.04);}
/* BADGE */
.badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:600;
  letter-spacing:.16em;text-transform:uppercase;padding:.25rem .7rem;border-radius:2px;margin-top:.5rem;}
.badge-x{background:rgba(231,76,60,0.1);border:1px solid rgba(231,76,60,0.3);color:#e74c3c !important;}
.badge-h{background:rgba(243,156,18,0.08);border:1px solid rgba(243,156,18,0.25);color:#f39c12 !important;}
.badge-m{background:rgba(39,174,96,0.07);border:1px solid rgba(39,174,96,0.2);color:#27ae60 !important;}
/* BAR */
.bar-track{background:rgba(255,255,255,0.04);border-radius:2px;height:3px;width:100%;margin:.9rem 0 .5rem 0;overflow:hidden;}
.bar-fill{height:100%;border-radius:2px;}
/* STAT ROW */
.srow{display:flex;align-items:baseline;justify-content:space-between;
  border-bottom:1px solid rgba(255,255,255,0.03);padding:.55rem 0;}
.srow:last-child{border-bottom:none;}
.sk{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#2a2a2a;text-transform:uppercase;letter-spacing:.07em;}
.sv{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#666;}
/* REC */
.rec{border-left:2px solid #e74c3c;padding:1rem 1.4rem;background:rgba(231,76,60,0.03);
  border-radius:0 3px 3px 0;margin-top:.8rem;}
.rec p{font-family:'DM Sans',sans-serif;font-size:0.85rem;line-height:1.75;color:#666 !important;}
/* LIVE DOT */
.dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:#27ae60;
  margin-right:.35rem;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
/* HERO */
.hero{padding:2.5rem 0 1.5rem 0;}
.hero-tag{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#e74c3c;
  text-transform:uppercase;letter-spacing:.22em;margin-bottom:1.2rem;
  display:flex;align-items:center;gap:.6rem;}
.hero-tag::before{content:'';width:14px;height:1px;background:#e74c3c;}
.hero-h{font-family:'DM Serif Display',serif;font-size:3rem;line-height:1.06;
  letter-spacing:-.02em;color:#e0e0e0;}
.hero-sub{font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#333;
  margin-top:1rem;max-width:500px;line-height:1.7;font-weight:300;}
/* COUNTER */
.ctr-row{display:flex;gap:2.5rem;margin-top:1.8rem;padding-top:1.4rem;border-top:1px solid rgba(255,255,255,0.04);}
.ctr-num{font-family:'DM Serif Display',serif;font-size:1.8rem;color:#e0e0e0;line-height:1;}
.ctr-lbl{font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:#2a2a2a;
  text-transform:uppercase;letter-spacing:.1em;margin-top:.25rem;}
/* DRIVERS */
.drv{display:flex;align-items:center;gap:.8rem;padding:.55rem 0;border-bottom:1px solid rgba(255,255,255,0.03);}
.drv:last-child{border-bottom:none;}
.drv-r{font-family:'DM Serif Display',serif;font-size:1rem;color:rgba(231,76,60,.25);min-width:1.4rem;}
.drv-n{font-family:'JetBrains Mono',monospace;font-size:0.66rem;color:#555;text-transform:uppercase;letter-spacing:.05em;flex:1;}
.drv-v{font-family:'JetBrains Mono',monospace;font-size:0.66rem;color:#e74c3c;}
/* VAR GRID */
.vgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:.45rem;margin-top:.8rem;}
.vitem{background:rgba(255,255,255,0.01);border:1px solid rgba(255,255,255,0.04);border-radius:2px;padding:.6rem .8rem;}
.vname{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#2a2a2a;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.18rem;}
.vval{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555;}
.vdesc{font-size:0.5rem;color:#1a1a1a;font-family:'JetBrains Mono',monospace;line-height:1.5;margin-top:.28rem;}
/* PIPELINE */
.pstep{display:flex;gap:1.2rem;padding:1.1rem 0;border-bottom:1px solid rgba(255,255,255,0.03);}
.pstep:last-child{border-bottom:none;}
.pnum{font-family:'DM Serif Display',serif;font-size:1.4rem;color:rgba(231,76,60,.18);min-width:2.5rem;}
.ptitle{font-family:'DM Sans',sans-serif;font-size:0.83rem;font-weight:500;color:#777;margin-bottom:.25rem;}
.pdesc{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#2a2a2a;line-height:1.85;}
/* SIDEBAR ELEMENTS */
.sb-mark{font-family:'DM Serif Display',serif;font-size:1.35rem;color:#e0e0e0;padding:1.2rem 0 .15rem 0;}
.sb-sub{font-family:'JetBrains Mono',monospace;font-size:0.54rem;color:#1e1e1e;letter-spacing:.14em;text-transform:uppercase;margin-bottom:1.1rem;}
.sb-div{border:none;border-top:1px solid rgba(255,255,255,0.04);margin:.9rem 0;}
.sb-lbl{font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:#1e1e1e;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem;}
/* CI */
.ci-row{display:flex;align-items:baseline;gap:.4rem;margin-top:.4rem;}
.ci-b{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#3a3a3a;}
.ci-dash{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#1a1a1a;}
/* MODEL TEXT */
.mod-p{font-family:'DM Sans',sans-serif;font-size:0.88rem;line-height:1.85;color:#444;max-width:720px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k, v in {
    "lat": 46.8139, "lon": -71.2080,
    "prediction_results": None,
    "addr_label": "Québec, QC",
    "lang": "FR",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # Language toggle
    lang = st.radio("lang", ["FR", "EN"], index=0 if st.session_state.lang == "FR" else 1,
                    horizontal=True, label_visibility="collapsed")
    st.session_state.lang = lang
    C = COPY[lang]

    st.markdown(f"<div class='sb-mark'>{C['sb_title']}</div><div class='sb-sub'>{C['sb_sub']}</div>", unsafe_allow_html=True)
    st.markdown("<hr class='sb-div'>", unsafe_allow_html=True)

    st.markdown(f"<div class='sb-lbl'>{C['user_label']}</div>", unsafe_allow_html=True)
    user_type = st.selectbox("ut", C["user_opts"], label_visibility="collapsed")
    if user_type == C["user_opts"][-1]:  # "Other" / "Autre"
        user_detail = st.text_input("ud", placeholder=C["specify"], label_visibility="collapsed")
        if not user_detail:
            user_detail = user_type
    else:
        user_detail = user_type

    st.markdown("<hr class='sb-div'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sb-lbl'>{C['loc_label']}</div>", unsafe_allow_html=True)
    mode = st.radio("mode", C["mode_opts"], label_visibility="collapsed")

    if mode == C["mode_opts"][0]:
        city_in = st.text_input("city", st.session_state.addr_label,
                                placeholder=C["addr_ph"], label_visibility="collapsed")
        if st.button(C["search_btn"], use_container_width=True):
            with st.spinner("…"):
                try:
                    g = Nominatim(user_agent="disAvert2026", timeout=8)
                    loc = g.geocode(city_in)
                    if loc:
                        st.session_state.lat = loc.latitude
                        st.session_state.lon = loc.longitude
                        st.session_state.addr_label = city_in
                        st.success(f"📍 {loc.address[:40]}…")
                    else:
                        st.error("Not found." if lang == "EN" else "Adresse introuvable.")
                except Exception as e:
                    st.error(str(e))

    st.markdown("<hr class='sb-div'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='sb-lbl'>{C['coords_lbl']}</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#2a2a2a;line-height:1.9;'>
      {st.session_state.lat:.4f}° N<br>
      {abs(st.session_state.lon):.4f}° W<br><br>
      <span style='color:#1a1a1a;font-size:0.54rem;'>{C['sb_meta']}</span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def read_log_stats():
    path = "data/predictions_log.csv"
    if not os.path.exists(path):
        return 0, 0
    try:
        df = pd.read_csv(path)
        total = len(df)
        alerts = len(df[df["risk_level"].isin(["extreme", "high"])]) if "risk_level" in df.columns else 0
        return total, alerts
    except Exception:
        return 0, 0

def generate_synthetic(n=2000):
    np.random.seed(42)
    X = pd.DataFrame({
        'FFMC':              np.random.beta(5, 2, n),
        'DC':                np.random.beta(2, 3, n),
        'FWI':               np.random.beta(2, 4, n),
        'DMC':               np.random.beta(2, 3, n),
        'RH':                np.random.beta(3, 2, n),
        'Temp':              np.random.beta(2, 3, n),
        'Wind':              np.random.beta(2, 4, n),
        'ISI':               np.random.beta(2, 4, n),
        'BUI':               np.random.beta(2, 3, n),
        'NDVI':              np.random.beta(4, 2, n),
        'LST':               np.random.beta(3, 2, n),
        'Hist_Fire_Density': np.random.beta(1, 8, n),
        'NBR':               np.random.beta(3, 2, n),
        'LandCover':         np.random.uniform(0, 1, n),
        'Slope':             np.random.beta(1, 4, n),
        'SPI':               np.random.uniform(0, 1, n),
        'Aspect':            np.random.uniform(0, 1, n),
        'FRP':               np.zeros(n),
    })
    noise = np.random.normal(0, 0.03, n)
    y = np.clip(
        0.35*X['FWI'] + 0.25*(1-X['RH']) + 0.20*X['Wind']
        + 0.10*(1-X['NDVI']) + 0.10*X['Hist_Fire_Density'] + noise,
        0.05, 0.95
    )
    return X, y

@st.cache_resource
def load_models():
    paths = {k: f"models/wildfire_{k}.pkl" for k in ["q10", "q50", "q90"]}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: joblib.load(p) for k, p in paths.items()}
    X, y = generate_synthetic()
    os.makedirs("models", exist_ok=True)
    models = {}
    for key, alpha in [("q10", 0.1), ("q50", 0.5), ("q90", 0.9)]:
        m = CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha}",
            iterations=400, depth=7, learning_rate=0.08,
            random_seed=42, verbose=False
        )
        m.fit(X, y)
        joblib.dump(m, paths[key])
        models[key] = m
    return models

def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"precipitation,surface_temperature,wind_gusts_10m,dew_point_2m"
        f"&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    c = r.json()["current"]
    return {
        "temp":       c.get("temperature_2m",        18.5),
        "rh":         c.get("relative_humidity_2m",  45.0),
        "wind":       c.get("wind_speed_10m",        15.0),
        "precip":     c.get("precipitation",          0.0),
        "surf_temp":  c.get("surface_temperature",   25.0),
        "gusts":      c.get("wind_gusts_10m",        20.0),
        "dew":        c.get("dew_point_2m",           8.0),
    }

def fetch_forecast(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,relative_humidity_2m_mean,"
        f"wind_speed_10m_max,precipitation_sum"
        f"&timezone=auto&forecast_days=7"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = r.json()["daily"]
    return {
        "dates": d["time"],
        "temp":  d["temperature_2m_max"],
        "rh":    d["relative_humidity_2m_mean"],
        "wind":  d["wind_speed_10m_max"],
        "prec":  d["precipitation_sum"],
    }

def build_features(w):
    rh, temp, wind = w["rh"], w["temp"], w["wind"]
    ffmc = max(0, min(101, 82 + wind*0.35 - (rh-40)*0.25 + (temp-15)*0.3))
    fwi  = max(0, min(100, 22 + wind*0.9 + (temp-15)*0.5 - rh*0.2))
    isi  = max(0, min(50,  13 + wind*0.4))
    bui  = max(0, min(200, 68 + (temp-15)*2 - w.get("precip", 0)*5))
    dc   = max(0, min(1000,265 + max(0,temp-15)*10 - w.get("precip", 0)*20))
    dmc  = max(0, min(200, 47 + max(0,temp-10)*5 - w.get("precip", 0)*10))
    spi  = max(0, min(1, 0.5 - w.get("precip", 0)*0.1))
    return {
        'FFMC': ffmc/101, 'DC': dc/1000, 'FWI': fwi/100, 'DMC': dmc/200,
        'RH': rh/100, 'Temp': max(0, min(1,(temp+10)/55)), 'Wind': min(1,wind/100),
        'ISI': isi/50, 'BUI': min(1,bui/200), 'NDVI': 0.43,
        'LST': min(1,(w.get("surf_temp",25)+273)/400),
        'Hist_Fire_Density': 0.0025, 'NBR': 0.30, 'LandCover': 0.5,
        'Slope': 0.09, 'SPI': spi, 'Aspect': 0.55, 'FRP': 0.0,
    }

def get_drivers(feat, lang):
    contrib = {
        ("FWI" if lang=="EN" else "FWI"):                     0.35 * feat["FWI"],
        ("Low Humidity" if lang=="EN" else "Faible humidité"):0.25 * (1-feat["RH"]),
        ("Wind" if lang=="EN" else "Vent"):                   0.20 * feat["Wind"],
        ("Dry Vegetation" if lang=="EN" else "Végétation sèche"):0.10*(1-feat["NDVI"]),
        ("Fire History" if lang=="EN" else "Historique feux"):0.10 * feat["Hist_Fire_Density"],
    }
    return sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]

def get_risk_level(p):
    if p >= 0.75:
        return "extreme"
    elif p >= 0.45:
        return "high"
    else:
        return "moderate"


def get_rec(prob, user_type, lang):
    C_rec = RECS[lang]
    level = get_risk_level(prob)
    bank = C_rec[level]
    return bank.get(user_type, bank.get("default", list(bank.values())[0]))


def clamp(v):
    return max(0.05, min(0.95, v))

# ─── SVG Gauge ────────────────────────────────────────────────
def make_gauge(prob, color):
    # 270° arc, r=72, cx=cy=100
    R = 72
    C_full = 2 * 3.14159 * R          # ≈ 452.4
    arc    = C_full * 0.75             # 270° ≈ 339.3
    filled = arc * prob
    uid = str(int(time.time() * 1000))[-6:]
    return f"""
<div style='display:flex;flex-direction:column;align-items:center;margin:.3rem 0;'>
<svg viewBox="0 0 200 200" width="190" height="190" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      @keyframes gfill{uid}{{
        from{{stroke-dasharray:0 {C_full:.1f};}}
        to{{stroke-dasharray:{filled:.1f} {C_full:.1f};}}
      }}
      .garc{uid}{{
        animation:gfill{uid} 1.1s cubic-bezier(.4,0,.2,1) forwards;
      }}
    </style>
  </defs>
  <!-- track -->
  <circle cx="100" cy="100" r="{R}"
    fill="none" stroke="#111" stroke-width="10"
    stroke-dasharray="{arc:.1f} {C_full:.1f}"
    transform="rotate(135 100 100)" stroke-linecap="round"/>
  <!-- fill -->
  <circle cx="100" cy="100" r="{R}"
    fill="none" stroke="{color}" stroke-width="10"
    class="garc{uid}"
    transform="rotate(135 100 100)" stroke-linecap="round"
    opacity="0.9"/>
  <!-- centre text -->
  <text x="100" y="92" text-anchor="middle"
    font-family="DM Serif Display,serif" font-size="34" fill="#e0e0e0">{prob*100:.0f}</text>
  <text x="100" y="112" text-anchor="middle"
    font-family="JetBrains Mono,monospace" font-size="10" fill="#333">%</text>
</svg>
</div>"""

# ─── 7-day sparkline SVG ──────────────────────────────────────
def make_sparkline(values, color, days):
    n = len(values)
    if n < 2:
        return ""
    W, H = 300, 60
    pad_x = 18
    xs = [pad_x + i * (W - 2*pad_x) / (n-1) for i in range(n)]
    ys = [8 + (1 - v) * (H - 16) for v in values]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    # day labels
    labels = ""
    short_days_en = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for i, (x, day_str) in enumerate(zip(xs, days)):
        try:
            from datetime import datetime as dt
            d = dt.strptime(day_str, "%Y-%m-%d")
            lbl = d.strftime("%a")[:3]
        except Exception:
            lbl = short_days_en[i % 7]
        labels += f'<text x="{x:.1f}" y="78" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="7.5" fill="#3a3a3a">{lbl}</text>'
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" opacity="0.8"/>'
        for x, y in zip(xs, ys)
    )
    return f"""
<svg viewBox="0 0 300 88" width="100%" xmlns="http://www.w3.org/2000/svg" style="margin-top:.6rem;display:block;">
  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="0.6" stroke-linejoin="round"/>
  {circles}
  {labels}
</svg>"""

# ═══════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════
C = COPY[st.session_state.lang]

total_analyses, total_alerts = read_log_stats()

col_h, col_btn = st.columns([3, 1])
with col_h:
    st.markdown(f"""
    <div class='hero'>
      <div class='hero-tag'>{C['tag']}</div>
      <div class='hero-h'>{C['headline1']}<br><em>{C['headline2']}</em></div>
      <div class='hero-sub'>{C['sub']}</div>
      <div class='ctr-row'>
        <div><div class='ctr-num'>{total_analyses}</div><div class='ctr-lbl'>{C['cta1']}</div></div>
        <div><div class='ctr-num'>{total_alerts}</div><div class='ctr-lbl'>{C['cta2']}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_btn:
    st.markdown("<div style='padding-top:3.2rem;'>", unsafe_allow_html=True)
    run = st.button(C["run_btn"], type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"<div class='dg-lbl'>{'Platform' if st.session_state.lang=='EN' else 'Plateforme'}</div>", unsafe_allow_html=True)

# ── INTERACTIVE MAP ────────────────────────────────────────────
if mode == C["mode_opts"][1]:
    st.markdown(f"<div class='dg-lbl'>{C['map_lbl']}</div>", unsafe_allow_html=True)
    m = folium.Map([st.session_state.lat, st.session_state.lon], zoom_start=7, tiles="CartoDB dark_matter")
    folium.Marker([st.session_state.lat, st.session_state.lon],
                  icon=folium.Icon(color="red", icon="fire", prefix="fa")).add_to(m)
    md = st_folium(m, width=None, height=360, returned_objects=["last_clicked"])
    if md and md.get("last_clicked"):
        st.session_state.lat = md["last_clicked"]["lat"]
        st.session_state.lon = md["last_clicked"]["lng"]
        st.rerun()

# ── MODEL LOAD ─────────────────────────────────────────────────
with st.spinner(C["init"]):
    models = load_models()

# ═══════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════
if run:
    with st.spinner(C["spin"]):
        # Live weather
        try:
            w = fetch_weather(st.session_state.lat, st.session_state.lon)
            live = True
        except Exception:
            w = {"temp":18.5,"rh":45.0,"wind":15.0,"precip":0.0,"surf_temp":25.0,"gusts":20.0,"dew":8.0}
            live = False

        # 7-day forecast
        try:
            fc = fetch_forecast(st.session_state.lat, st.session_state.lon)
            fc_probs = []
            for i in range(len(fc["dates"])):
                fw = {"temp":fc["temp"][i],"rh":fc["rh"][i],"wind":fc["wind"][i],
                      "precip":fc["prec"][i],"surf_temp":fc["temp"][i]+2,"gusts":fc["wind"][i]*1.3,"dew":fc["rh"][i]*0.3}
                ff = build_features(fw)
                p = clamp(float(models["q50"].predict(pd.DataFrame([ff]))[0]))
                fc_probs.append(p)
            fc_days = fc["dates"]
        except Exception:
            fc_probs = None
            fc_days  = []

        # Inference
        feat = build_features(w)
        fd   = pd.DataFrame([feat])
        p_lo  = clamp(float(models["q10"].predict(fd)[0]))
        p_mid = clamp(float(models["q50"].predict(fd)[0]))
        p_hi  = clamp(float(models["q90"].predict(fd)[0]))

        level = get_risk_level(p_mid)
        rec   = get_rec(p_mid, user_type, st.session_state.lang)
        top3  = get_drivers(feat, st.session_state.lang)

        result = {
            "timestamp":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lat":           round(st.session_state.lat, 5),
            "lon":           round(st.session_state.lon, 5),
            "address":       st.session_state.addr_label,
            "user_type":     user_type,
            "user_detail":   user_detail,
            "lang":          st.session_state.lang,
            "temp_c":        round(w["temp"], 1),
            "humidity_pct":  round(w["rh"], 1),
            "wind_kmh":      round(w["wind"], 1),
            "precip_mm":     round(w["precip"], 2),
            "surf_temp_c":   round(w["surf_temp"], 1),
            "gusts_kmh":     round(w["gusts"], 1),
            "dew_c":         round(w["dew"], 1),
            "prob_pct":      round(p_mid*100, 2),
            "ci_lo_pct":     round(p_lo*100, 2),
            "ci_hi_pct":     round(p_hi*100, 2),
            "risk_level":    level,
            "recommendation":rec,
            "live_data":     live,
            "fc_probs":      fc_probs,
            "fc_days":       fc_days,
            "top3_drivers":  [(n, round(v,4)) for n,v in top3],
            **{f"feat_{k}":  round(v,6) for k,v in feat.items()},
        }
        st.session_state.prediction_results = result

        os.makedirs("data", exist_ok=True)
        log_row = {k: v for k, v in result.items() if k not in ["fc_probs","fc_days","top3_drivers"]}
        csv_path = "data/predictions_log.csv"
        pd.DataFrame([log_row]).to_csv(csv_path, mode="a",
                                       header=not os.path.exists(csv_path), index=False)
        with open("data/predictions_audit.jsonl", "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

# ═══════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════
if st.session_state.prediction_results:
    r    = st.session_state.prediction_results
    p    = r["prob_pct"] / 100
    p_lo = r["ci_lo_pct"] / 100
    p_hi = r["ci_hi_pct"] / 100
    C    = COPY[st.session_state.lang]

    if p > 0.70:
        badge_cls, risk_lbl, clr, clr2 = "badge-x", C["r_extreme"], "#e74c3c", "#c0392b"
    elif p > 0.40:
        badge_cls, risk_lbl, clr, clr2 = "badge-h", C["r_high"],    "#f39c12", "#e67e22"
    else:
        badge_cls, risk_lbl, clr, clr2 = "badge-m", C["r_moderate"],"#27ae60", "#1e8449"

    live_html = (f"<span class='dot'></span><span style='font-size:0.56rem;color:#27ae60;"
                 f"font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:.1em;'>"
                 f"{C['live_tag']}</span>") if r.get("live_data") else (
                 f"<span style='font-size:0.56rem;color:#2a2a2a;"
                 f"font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:.1em;'>"
                 f"{C['est_tag']}</span>")

    tab1, tab2, tab3 = st.tabs([C["t_result"], C["t_model"], C["t_pipeline"]])

    # ─── TAB 1: RESULT ────────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns([1.1, 1.25, 1.65])

        with c1:
            gauge_svg = make_gauge(p, clr)
            top3_html = ""
            for i, (name, val) in enumerate(r["top3_drivers"], 1):
                    top3_html += f"<div class='drv'><div class='drv-r'>{i}</div><div class='drv-n'>{name}</div><div class='drv-v'>{val*100:.0f}%</div></div>"
            st.markdown(f"""
            <div class='dg-card'>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.56rem;color:#2a2a2a;
                text-transform:uppercase;letter-spacing:.12em;margin-bottom:.6rem;'>{C['prob_lbl']}</div>
              {gauge_svg}
              <div class='{badge_cls} badge' style='display:block;text-align:center;'>{risk_lbl}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.56rem;color:#2a2a2a;
                text-transform:uppercase;letter-spacing:.1em;margin:1.2rem 0 .3rem 0;'>{C['ci_lbl']}</div>
              <div class='ci-row'>
                <span class='ci-b'>{r['ci_lo_pct']:.0f}%</span>
                <span class='ci-dash'>—</span>
                <span class='ci-b'>{r['ci_hi_pct']:.0f}%</span>
              </div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.56rem;color:#2a2a2a;
                text-transform:uppercase;letter-spacing:.1em;margin:1.2rem 0 .3rem 0;'>{C['drivers_lbl']}</div>
              {top3_html}
            </div>
            """, unsafe_allow_html=True)

        with c2:
            fc_html = ""
            if r.get("fc_probs") and r.get("fc_days"):
                sparkline = make_sparkline(r["fc_probs"], clr, r["fc_days"])
                fc_vals = "".join(
                    f"<div style='text-align:center;min-width:0;'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;"
                    f"color:#555;font-weight:500;'>{v*100:.0f}%</div>"
                    f"</div>"
                    for d, v in zip(r["fc_days"], r["fc_probs"])
                )
                fc_html = (
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.56rem;"
                    f"color:#2a2a2a;text-transform:uppercase;letter-spacing:.1em;"
                    f"margin-top:1.2rem;margin-bottom:.3rem;'>{C['fc_lbl']}</div>"
                    f"{sparkline}"
                    f"<div style='display:grid;grid-template-columns:repeat(7,1fr);"
                    f"gap:2px;margin-top:.3rem;'>{fc_vals}</div>"
                )

            st.markdown(f"""
            <div class='dg-card'>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.56rem;color:#2a2a2a;
                text-transform:uppercase;letter-spacing:.12em;margin-bottom:1rem;'>{C['live_lbl']} {live_html}</div>
              <div class='srow'><span class='sk'>{C['f_temp']}</span><span class='sv'>{r['temp_c']} °C</span></div>
              <div class='srow'><span class='sk'>{C['f_hum']}</span><span class='sv'>{r['humidity_pct']} %</span></div>
              <div class='srow'><span class='sk'>{C['f_wind']}</span><span class='sv'>{r['wind_kmh']} km/h</span></div>
              <div class='srow'><span class='sk'>{C['f_gust']}</span><span class='sv'>{r['gusts_kmh']} km/h</span></div>
              <div class='srow'><span class='sk'>{C['f_prec']}</span><span class='sv'>{r['precip_mm']} mm</span></div>
              <div class='srow'><span class='sk'>{C['f_dew']}</span><span class='sv'>{r['dew_c']} °C</span></div>
              <div class='srow'><span class='sk'>{C['f_surf']}</span><span class='sv'>{r['surf_temp_c']} °C</span></div>
              {fc_html}
            </div>
            """, unsafe_allow_html=True)

        with c3:
            bar_w = int(p * 100)
            ts = r["timestamp"]
            st.markdown(f"""
            <div class='dg-card' style='height:100%;'>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.56rem;color:#2a2a2a;
                text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem;'>
                {C['rec_lbl']} · {r['user_detail']}
              </div>
              <div class='rec'><p>{r['recommendation']}</p></div>
              <div class='bar-track' style='margin-top:1.5rem;'>
                <div class='bar-fill' style='width:{bar_w}%;background:linear-gradient(90deg,{clr2},{clr});'></div>
              </div>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#1e1e1e;margin-top:1.2rem;line-height:1.9;'>
                {r['lat']}° N &nbsp; {abs(r['lon'])}° W<br>
                {ts}<br>
                CatBoost Quantile · q10/q50/q90 · Synthetic MVP
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Map
        st.markdown(f"<br><div class='dg-lbl'>{C['loc_lbl2']}</div>", unsafe_allow_html=True)
        m2 = folium.Map([r["lat"], r["lon"]], zoom_start=9, tiles="CartoDB dark_matter")
        folium.CircleMarker([r["lat"], r["lon"]], radius=22,
                            color=clr, fill=True, fill_color=clr, fill_opacity=0.12).add_to(m2)
        folium.Marker([r["lat"], r["lon"]],
                      popup=f"Risk: {r['prob_pct']}% — {risk_lbl}",
                      icon=folium.Icon(color="red" if p>0.70 else "orange" if p>0.40 else "green",
                                       icon="fire", prefix="fa")).add_to(m2)
        st_folium(m2, width=None, height=300, returned_objects=[])

    # ─── TAB 2: MODEL ─────────────────────────────────────────
    with tab2:
        st.markdown(f"<br><div class='dg-lbl'>{C['mod_lbl']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='dg-card'><p class='mod-p'>{C['mod_txt']}</p></div>",
                    unsafe_allow_html=True)

        st.markdown(f"<br><div class='dg-lbl'>{C['var_lbl']}</div>", unsafe_allow_html=True)
        VARS = [
            ("FFMC",              r.get("feat_FFMC",0),              "Fine fuel moisture (0–101)",          "Humidité combustibles fins (0–101)"),
            ("DC",                r.get("feat_DC",0),                "Deep layer drought index",            "Sécheresse couche profonde"),
            ("FWI",               r.get("feat_FWI",0),               "Fire weather index — pivot variable", "Indice météo feux — variable pivot"),
            ("DMC",               r.get("feat_DMC",0),               "Compact organic layer moisture",      "Humidité couche compacte"),
            ("RH",                r.get("feat_RH",0),                "Relative humidity (%)",               "Humidité relative (%)"),
            ("Temp",              r.get("feat_Temp",0),              "Air temperature (°C)",                "Température de l'air (°C)"),
            ("Wind",              r.get("feat_Wind",0),              "Wind speed — pivot variable",         "Vitesse du vent — variable pivot"),
            ("ISI",               r.get("feat_ISI",0),               "Initial spread index",                "Indice de propagation initiale"),
            ("BUI",               r.get("feat_BUI",0),               "Buildup index",                       "Indice de combustible disponible"),
            ("NDVI",              r.get("feat_NDVI",0),              "Vegetation health (Sentinel-2) *",    "Santé végétation (Sentinel-2) *"),
            ("LST",               r.get("feat_LST",0),               "Land surface temperature (MODIS)",    "Température de surface (MODIS)"),
            ("Hist_Fire_Density", r.get("feat_Hist_Fire_Density",0), "Historical fire density (CNFDB) *",   "Densité historique feux (CNFDB) *"),
            ("NBR",               r.get("feat_NBR",0),               "Normalized burn ratio *",             "Rapport brûlé normalisé *"),
            ("LandCover",         r.get("feat_LandCover",0),         "Land cover type (NALCMS) *",          "Type couverture terrestre (NALCMS) *"),
            ("Slope",             r.get("feat_Slope",0),             "Terrain slope (°) *",                 "Pente du terrain (°) *"),
            ("SPI",               r.get("feat_SPI",0),               "Std precipitation index",             "Indice précipitation standardisé"),
            ("Aspect",            r.get("feat_Aspect",0),            "Slope orientation *",                 "Orientation du terrain *"),
            ("FRP",               r.get("feat_FRP",0),               "Fire radiative power (label)",        "Puissance radiative feux (label)"),
        ]
        pivot = {"FWI", "RH", "Wind", "FFMC"}
        cells = ""
        for name, val, desc_en, desc_fr in VARS:
            desc = desc_en if st.session_state.lang == "EN" else desc_fr
            fill_w = int(val * 100)
            col = "#e74c3c" if name in pivot else "#2a2a2a"
            cells += (
                f"<div class='vitem'>"
                f"<div class='vname'>{name}</div>"
                f"<div class='vval'>{val:.4f}</div>"
                f"<div class='bar-track' style='margin:.35rem 0 .25rem 0;height:2px;'>"
                f"<div class='bar-fill' style='width:{fill_w}%;background:{col};'></div></div>"
                f"<div class='vdesc'>{desc}</div>"
                f"</div>"
            )
        st.markdown(f"<div class='vgrid'>{cells}</div>", unsafe_allow_html=True)
        note = C["var_note"]
        note2 = C["var_note2"]
        st.markdown(f"""
        <div style='margin-top:1rem;font-family:JetBrains Mono,monospace;font-size:0.55rem;color:#1e1e1e;line-height:1.9;'>
          * {note}<br>{note2}
        </div>""", unsafe_allow_html=True)

    # ─── TAB 3: PIPELINE ──────────────────────────────────────
    with tab3:
        st.markdown(f"<br><div class='dg-lbl'>{C['pip_lbl']}</div>", unsafe_allow_html=True)
        if st.session_state.lang == "EN":
            steps = [
                ("01","CFWIS Daily (NRCan)","cwfis.cfs.nrcan.gc.ca · Daily · 2000–present","Variables: FFMC, DC, FWI, DMC, ISI, BUI · API JSON/CSV"),
                ("02","Weather ECCC","climate.weather.gc.ca · dd.meteo.gc.ca · Hourly · ~1950–present","Variables: RH, Temperature, Wind · Bulk XML/CSV download"),
                ("03","Sentinel-2 and Landsat (GEE)","Google Earth Engine · 5-day / 16-day composite","Variables: NDVI, NBR · ee.ImageCollection COPERNICUS/S2_SR"),
                ("04","MODIS Terra (NASA)","earthdata.nasa.gov / FIRMS · Daily · 2000–present","Variables: LST (MOD11A1), FRP — active fire training label"),
                ("05","CNFDB (NRCan)","cwfis.cfs.nrcan.gc.ca/ha/nfdb · Annual · 1959–present","Variable: Hist_Fire_Density · KDE on ignition polygons"),
                ("06","SPI Derivation (ECCC)","ECCC precipitation data · Monthly rolling window","Variable: SPI — 3-month rolling Z-score from precip baseline"),
                ("07","CDEM + NALCMS (static)","open.canada.ca (CDEM) · cec.org/nalcms (2020)","Variables: Slope, Aspect, LandCover · GDAL/rasterio processing"),
                ("08","Train · Validate · Deploy","Temporal split 70/15/15 · CatBoost Quantile + Optuna HPO","Explainability: SHAP per prediction · RMSE / AUC-ROC on test set"),
            ]
        else:
            steps = [
                ("01","CFWIS Quotidien (NRCan)","cwfis.cfs.nrcan.gc.ca · Quotidien · 2000–présent","Variables: FFMC, DC, FWI, DMC, ISI, BUI · API JSON/CSV"),
                ("02","Météo ECCC","climate.weather.gc.ca · dd.meteo.gc.ca · Horaire · ~1950–présent","Variables: RH, Température, Vent · Téléchargement bulk XML/CSV"),
                ("03","Sentinel-2 et Landsat (GEE)","Google Earth Engine · Composite 5 jours / 16 jours","Variables: NDVI, NBR · ee.ImageCollection COPERNICUS/S2_SR"),
                ("04","MODIS Terra (NASA)","earthdata.nasa.gov / FIRMS · Quotidien · 2000–présent","Variables: LST (MOD11A1), FRP — label d'incendie actif"),
                ("05","CNFDB (NRCan)","cwfis.cfs.nrcan.gc.ca/ha/nfdb · Annuel · 1959–présent","Variable: Hist_Fire_Density · KDE sur polygones d'ignition"),
                ("06","Dérivation SPI (ECCC)","Données précipitation ECCC · Fenêtre glissante mensuelle","Variable: SPI — Z-score glissant 3 mois sur baseline précip"),
                ("07","CDEM + NALCMS (statiques)","open.canada.ca (CDEM) · cec.org/nalcms (2020)","Variables: Slope, Aspect, LandCover · Traitement GDAL/rasterio"),
                ("08","Entraînement · Validation · Déploiement","Split temporel 70/15/15 · CatBoost Quantile + Optuna HPO","Explicabilité: SHAP par prédiction · RMSE / AUC-ROC sur test set"),
            ]
        steps_html = ""
        for num, title, src, variables in steps:
            steps_html += (
                f"<div class='pstep'>"
                f"<div class='pnum'>{num}</div>"
                f"<div><div class='ptitle'>{title}</div>"
                f"<div class='pdesc'>{src}<br>{variables}</div></div>"
                f"</div>"
            )
        st.markdown(f"<div class='dg-card'>{steps_html}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
C = COPY[st.session_state.lang]
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='border-top:1px solid rgba(255,255,255,0.03);padding-top:1.4rem;
  display:flex;justify-content:space-between;align-items:center;'>
  <div style='font-family:DM Serif Display,serif;font-size:0.95rem;color:#1e1e1e;'>DisAvert</div>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.54rem;color:#1a1a1a;
    letter-spacing:.07em;text-transform:uppercase;'>{C['footer']}</div>
</div>""", unsafe_allow_html=True)
