import numpy as np
import pandas as pd
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Load models
proxy_scaler = joblib.load("models/proxy_scaler.joblib")
proxy_model  = joblib.load("models/proxy_model.joblib")
risk_diab    = joblib.load("models/risk_diab.joblib")
risk_bp      = joblib.load("models/risk_bp.joblib")
risk_lip     = joblib.load("models/risk_lip.joblib")
z_star       = np.load("models/z_star.npy")
Z_KEEP       = joblib.load("models/Z_KEEP.joblib")


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are WellnessZ — an intelligent, calm, premium digital health coach.
You explain physiological results clearly and motivate practical action.
You never mention models or data science.
"""

def predict_clients(df_client):
    df = df_client.copy()

    if "age" not in df.columns:
        df["age"] = 30
    if "sex" not in df.columns:
        df["sex"] = 1

    CLIENT_PROXY_MAP = {
        "bmi": "BMXBMI",
        "hm_visceral_fat": "BMDAVSAD",
        "hm_muscle": "BMXARMC",
        "hm_rm": "MGDCGSZ",
        "age": "RIDAGEYR",
        "sex": "RIAGENDR"
    }

    X = df[list(CLIENT_PROXY_MAP.keys())].rename(columns=CLIENT_PROXY_MAP)

    for col in X.columns:
        X[col] = (
            X[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )

    X = X.apply(pd.to_numeric, errors="coerce")
    X["MGDCGSZ"] = np.log1p(X["MGDCGSZ"].clip(lower=0))

    valid = X.notna().all(axis=1)
    X_clean = X.loc[valid]

    Z_masked = proxy_model.predict(proxy_scaler.transform(X_clean))

    Z_CONF = np.array([1.0, 0.8, 0.8, 0.5])
    health_distance = np.linalg.norm((Z_masked - z_star[Z_KEEP]) * Z_CONF, axis=1)

    Z_full = np.zeros((Z_masked.shape[0], 8))
    Z_full[:, Z_KEEP] = Z_masked

    pred_diab = risk_diab.predict_proba(Z_full)[:,1]
    pred_bp   = risk_bp.predict_proba(Z_full)[:,1]
    pred_lip  = risk_lip.predict_proba(Z_full)[:,1]

    out = pd.DataFrame({
        "client_id": df.loc[valid, "client_id"],
        "health_distance": health_distance,
        "pred_diab": pred_diab,
        "pred_bp": pred_bp,
        "pred_lip": pred_lip
    })

    def focus(i):
        if pred_diab[i] > 0.4: return "metabolic_reset"
        if pred_lip[i] > 0.4:  return "lipid_optimization"
        if health_distance[i] > 2.5: return "fat_loss"
        return "muscle_building"

    def triage(i):
        if pred_diab[i] > 0.7 or pred_bp[i] > 0.7 or pred_lip[i] > 0.8:
            return "COACH_REQUIRED"
        if health_distance[i] < 1.5:
            return "AUTO"
        return "HYBRID_MONITOR"

    out["control_focus"] = [focus(i) for i in range(len(out))]
    out["triage"] = [triage(i) for i in range(len(out))]

    return out.reset_index(drop=True)


def generate_explanation(row):
    prompt = f"""
Client profile:

Health distance: {row.health_distance:.2f}
Diabetes risk: {row.pred_diab:.3f}
BP risk: {row.pred_bp:.3f}
Lipid risk: {row.pred_lip:.3f}
Primary focus: {row.control_focus}
Coaching level: {row.triage}
"""

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return resp.choices[0].message.content
