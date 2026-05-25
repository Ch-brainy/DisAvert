# AlerteClimat – Coopérathon 2026 (Défi La Planète)

Outil d’IA qui prédit en temps réel les incendies de forêt et les inondations pour protéger les populations vulnérables et préserver les écosystèmes du Québec.

## Comment lancer le prototype (en 30 secondes)

```bash
# 1. Cloner le repo
git clone https://github.com/ton-username/alerteclimat.git
cd alerteclimat

# 2. Créer un environnement virtuel (recommandé)
py -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate # Mac/Linux

# 3. Installer toutes les dépendances
py -m pip install -r requirements.txt

# 4. Lancer l’application
py -m streamlit run AlerteClimat_V01.py --server.runOnSave true
# DisAvert
