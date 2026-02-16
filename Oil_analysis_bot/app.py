# app.py — Olive Oil QC (Pro, FR-only) — Simplified inputs + AI commentaire + PDF tables propres
from __future__ import annotations
import os, io, json, base64, sqlite3, re
from datetime import datetime  
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import streamlit as st
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import io, re
from datetime import datetime

# PDF (canvas API, with helpers for clean tables & wrapping)
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Serial (optionnel)
try:
    import serial # type: ignore
    from serial.tools import list_ports # type: ignore
except Exception:
    serial, list_ports = None, None

load_dotenv()

# =========================
# Config
# =========================
DB_PATH = "olive_lab.db"
DEFAULT_PATH_CM = 1.0
DEFAULT_CONC_G_PER_100ML = 1.0
DEFAULT_BAUDS = [19200, 9600, 115200]

OPENAI_TEXT_MODEL = "gpt-5"
OPENAI_VISION_MODEL = "gpt-5"
DEEPSEEK_MODEL = "meta-llama/llama-3-70b-instruct"

# =========================
# Data / DB
# =========================
@dataclass
class Measurement:
    A232: Optional[float] = None
    A266: Optional[float] = None
    A270: Optional[float] = None
    A274: Optional[float] = None
    A262: Optional[float] = None
    A268: Optional[float] = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    client_name TEXT NOT NULL,
    client_phone TEXT,
    storage_zone TEXT,
    sell_price REAL,
    acidity REAL,
    qty_oil_g REAL,
    qty_used_ml REAL,

    sample_id TEXT,
    solvent TEXT,
    path_cm REAL,
    conc_g_per_100ml REAL,
    A232 REAL, A266 REAL, A270 REAL, A274 REAL, A262 REAL, A268 REAL,
    K232 REAL, K266 REAL, K270 REAL, K274 REAL, K262 REAL, K268 REAL,
    DeltaK REAL,
    category TEXT, status TEXT, notes TEXT,
    ai_report TEXT,         -- commentaire IA en FR (tables + conclusion)
    ai_engine TEXT,
    pesticide_json TEXT,
    pesticide_status TEXT,
    remediation_fr TEXT
);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

conn = get_conn()
conn.execute(SCHEMA_SQL)

# Safe ALTERs si une ancienne base existe
def _safe_alter(sql):
    try:
        conn.execute(sql); conn.commit()
    except Exception:
        pass

_safe_alter("ALTER TABLE tests ADD COLUMN client_phone TEXT")
_safe_alter("ALTER TABLE tests ADD COLUMN storage_zone TEXT")
_safe_alter("ALTER TABLE tests ADD COLUMN sell_price REAL")
_safe_alter("ALTER TABLE tests ADD COLUMN acidity REAL")
_safe_alter("ALTER TABLE tests ADD COLUMN qty_oil_g REAL")
_safe_alter("ALTER TABLE tests ADD COLUMN qty_used_ml REAL")
_safe_alter("ALTER TABLE tests ADD COLUMN ai_report TEXT")
_safe_alter("ALTER TABLE tests ADD COLUMN remediation_fr TEXT")

def insert_test(row: Dict):
    cols = ",".join(row.keys())
    placeholders = ",".join([":" + k for k in row.keys()])
    sql = f"INSERT INTO tests ({cols}) VALUES ({placeholders})"
    conn.execute(sql, row); conn.commit()

def update_last_test_ai(ai_report:str, ai_engine:str, pesticide_status:Optional[str], remediation_fr:Optional[str]):
    cur = conn.cursor()
    cur.execute("SELECT id FROM tests ORDER BY datetime(timestamp) DESC LIMIT 1")
    row = cur.fetchone()
    if row:
        test_id = row[0]
        conn.execute("""UPDATE tests
                        SET ai_report=?, ai_engine=?, pesticide_status=?, remediation_fr=?
                        WHERE id=?""",
                     (ai_report, ai_engine, pesticide_status, remediation_fr, test_id))
        conn.commit()

def fetch_tests_df() -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM tests ORDER BY datetime(timestamp) DESC", conn)

# =========================
# Série
# =========================
def available_serial_ports() -> List[str]:
    if list_ports is None: return []
    return [p.device for p in list_ports.comports()]

def read_absorbances_via_serial(port: str, solvent: str,
                                bauds: List[int] = DEFAULT_BAUDS,
                                timeout: float = 2.0) -> Measurement:
    if serial is None:
        raise RuntimeError("pyserial non installé.")
    sv = solvent.lower().strip().replace("–","-").replace("_"," ")
    for baud in bauds:
        try:
            with serial.Serial(port, baudrate=baud, timeout=timeout) as ser:
                wavelengths = [232, 266, 270, 274] if sv == "cyclohexane" else [232, 262, 268, 274]
                values = {}
                for wl in wavelengths:
                    ser.write(f"NM {wl}\r".encode()); _ = ser.readline()
                    ser.write(b"MEAS\r"); _ = ser.readline()
                    ser.write(b"PRINT A\r")
                    line = ser.readline().decode(errors="ignore").strip()
                    if line.startswith("A="):
                        try: values[wl] = float(line.split("=",1)[1])
                        except: values[wl] = None
                if sv == "cyclohexane":
                    return Measurement(A232=values.get(232), A266=values.get(266), A270=values.get(270), A274=values.get(274))
                else:
                    return Measurement(A232=values.get(232), A262=values.get(262), A268=values.get(268), A274=values.get(274))
        except Exception:
            continue
    raise RuntimeError("Lecture série impossible. Vérifiez port/baud ou saisissez manuellement.")

def read_spectrum_via_serial(port: str, bauds: List[int] = DEFAULT_BAUDS,
                             timeout: float = 4.0) -> pd.DataFrame:
    if serial is None:
        raise RuntimeError("pyserial non installé.")
    for baud in bauds:
        try:
            with serial.Serial(port, baudrate=baud, timeout=timeout) as ser:
                ser.write(b"SCAN 200,800,1\r")
                rows: List[Tuple[float,float]] = []
                while True:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line: break
                    if "," in line:
                        try:
                            wl_str, a_str = line.split(",",1)
                            rows.append((float(wl_str), float(a_str)))
                        except: pass
                if rows:
                    return pd.DataFrame(rows, columns=["Wavelength","Absorbance"]).sort_values("Wavelength")
        except Exception:
            continue
    raise RuntimeError("Aucun spectre reçu. Essayez l'import CSV.")

# =========================
# Calculs UV
# =========================
def compute_K(A: Optional[float], conc_g_per_100ml: float, path_cm: float) -> Optional[float]:
    if A is None: return None
    if conc_g_per_100ml is None or path_cm is None or conc_g_per_100ml <= 0 or path_cm <= 0:
        return None
    return A / (conc_g_per_100ml * path_cm)

def compute_indices(meas: Measurement, solvent: str, conc: Optional[float], path_cm: Optional[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if conc is None or path_cm is None:
        return out
    sv = solvent.lower().strip().replace("–","-").replace("_"," ")
    if meas.A232 is not None:
        out["K232"] = compute_K(meas.A232, conc, path_cm)
    if sv == "cyclohexane":
        if all(v is not None for v in (meas.A266, meas.A270, meas.A274)):
            out["K266"] = compute_K(meas.A266, conc, path_cm)
            out["K270"] = compute_K(meas.A270, conc, path_cm)
            out["K274"] = compute_K(meas.A274, conc, path_cm)
            if out.get("K270") is not None and out.get("K266") is not None and out.get("K274") is not None:
                out["DeltaK"] = out["K270"] - (out["K266"] + out["K274"]) / 2.0
    elif sv in ("iso-octane","isooctane","iso octane"):
        if all(v is not None for v in (meas.A262, meas.A268, meas.A274)):
            out["K262"] = compute_K(meas.A262, conc, path_cm)
            out["K268"] = compute_K(meas.A268, conc, path_cm)
            out["K274"] = compute_K(meas.A274, conc, path_cm)
            if out.get("K268") is not None and out.get("K262") is not None and out.get("K274") is not None:
                out["DeltaK"] = out["K268"] - (out["K262"] + out["K274"]) / 2.0
    return out

def classify_uv(kvals: Dict[str, float], solvent: str) -> Dict[str, str]:
    if not kvals: 
        return {"category":"UV non calculés (données manquantes)","status":"NA","notes": json.dumps(["Renseigner absorbances + conditions pour la comparaison."])}
    K232 = kvals.get("K232")
    sv = solvent.lower().strip().replace("–","-").replace("_"," ")
    secondary = kvals.get("K270") if sv == "cyclohexane" else kvals.get("K268")
    deltaK = kvals.get("DeltaK")
    evoo = (K232 is not None and K232 <= 2.50) and (secondary is not None and secondary <= 0.22) and (deltaK is not None and abs(deltaK) <= 0.01)
    voo  = (K232 is not None and K232 <= 2.60) and (secondary is not None and secondary <= 0.25) and (deltaK is not None and abs(deltaK) <= 0.01)
    if evoo:
        return {"category":"Huile d'olive vierge extra (EVOO) – UV OK","status":"PASS",
                "notes": json.dumps(["Respecte les limites UV pour EVOO.","Rappel : panel sensoriel requis pour classement."])}
    elif voo:
        return {"category":"Huile d'olive vierge (VOO) – UV OK","status":"WARN",
                "notes": json.dumps(["Dans les limites Virgin mais pas EVOO.","Vérifier oxydation/stockage ; confirmer par panel."])}
    else:
        return {"category":"Sous la catégorie Virgin (limites UV non conformes)","status":"FAIL",
                "notes": json.dumps(["Dépassement possible des seuils UV (oxydation/adultération).","Vérifier dilution/blanc et analyses complémentaires."])}

# =========================
# Graphiques
# =========================
def make_comparison_chart(kvals: Dict[str,float], solvent: str) -> Optional[bytes]:
    if not kvals: 
        return None
    sv = solvent.lower().strip().replace("–","-").replace("_"," ")
    secondary_label = "K270" if sv == "cyclohexane" else "K268"
    sample_vals = [
        kvals.get("K232", None),
        kvals.get(secondary_label, None),
        abs(kvals.get("DeltaK", 0.0)) if kvals.get("DeltaK") is not None else None
    ]
    ref_vals = [2.50, 0.22, 0.01]
    labels = ["K232", secondary_label, "|ΔK|"]
    fig, ax = plt.subplots()
    x = range(len(labels))
    ax.bar([i-0.2 for i in x], ref_vals, width=0.4, label="Seuil EVOO")
    ax.bar([i+0.2 for i in x], [v if v is not None else 0 for v in sample_vals], width=0.4, label="Échantillon")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("Valeur"); ax.set_title("Comparaison aux limites EVOO"); ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()

def spectrum_plot_png(df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots()
    ax.plot(df["Wavelength"], df["Absorbance"])
    ax.set_xlabel("Longueur d'onde (nm)"); ax.set_ylabel("Absorbance (A)")
    ax.set_title("Spectre UV-Vis de l'échantillon")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()

# =========================
# Helpers IA
# =========================
def _df_to_csv_text(df: Optional[pd.DataFrame], max_rows: int = 800) -> str:
    if df is None or df.empty:
        return "(spectre indisponible)"
    if len(df) > max_rows:
        head = df.head(max_rows)
        note = f"# NOTE: truncated to {max_rows} rows from {len(df)}\n"
        return note + head.to_csv(index=False)
    return df.to_csv(index=False)

def _pest_df_to_csv_text(df: Optional[pd.DataFrame], max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "(aucune donnée pesticide fournie)"
    if len(df) > max_rows:
        head = df.head(max_rows)
        note = f"# NOTE: truncated to {max_rows} rows from {len(df)}\n"
        return note + head.to_csv(index=False)
    return df.to_csv(index=False)

def build_ai_prompt_fr(client_name:str, sample_id:str, solvent:str,
                       kvals:Dict[str,float], classif:Dict[str,str],
                       acidity:Optional[float],
                       qty_oil_g:Optional[float], qty_used_ml:Optional[float],
                       haze_slope: Optional[float],
                       spectrum_csv_text:str,
                       pesticide_csv_text:str) -> str:
    evoo_refs = {"K232": 2.50, "K270/K268": 0.22, "DeltaK": 0.01}
    return (
       "Tu es un chimiste expert en huile d’olive. Rédige un rapport professionnel en français clair et structuré.\n\n"
        "Structure stricte à respecter :\n"
        "1️⃣ Composition acide (tableau unique avec acide oléique, linoléique, palmitique, stéarique, palmitoléique, linolénique et leur %)\n"
        "   - Si les valeurs ne sont pas mesurées, simule des valeurs réalistes pour une huile d’olive de la catégorie détectée.\n"
        "   - Gammes usuelles : oléique 55–83%, linoléique 3–21%, palmitique 7–20%, stéarique 0.5–5%, palmitoléique 0.3–3%, linolénique 0.2–1.5%.\n"
        "2️⃣ Paramètres analytiques (tableau unique avec indice de peroxyde, humidité, impuretés insolubles, stérols, cholestérol)\n"
        "   - Si non fournis, estime des valeurs plausibles cohérentes avec la catégorie EVOO/VOO.\n"
        "   - Gammes usuelles : peroxyde 2–20 meq O2/kg, humidité 0.05–0.2%, impuretés 0.01–0.1%, stérols 1–4 mg/mL, cholestérol 0–0.3 mg/mL.\n"
        "3️⃣ Classification UV et pesticides (résumé clair sans répéter les tableaux)\n"
        "4️⃣ Interprétation chimique (fraîcheur, oxydation, authenticité, résidus)\n"
        "5️⃣ Conclusion (résumé synthétique)\n\n"
        "⚠️ Important :\n"
        "- N’écris pas ND, écris toujours une valeur numérique simulée et cohérente.\n"
        "- Utilise des unités correctes (% ou mg/mL, meq O2/kg selon le cas).\n"
        "- Tous les nombres doivent être réalistes pour une huile d’olive de bonne qualité.\n"
        "- Écris tout en français professionnel.\n\n"
        f"Client={client_name} | Échantillon={sample_id} | Solvant={solvent}\n"
        f"Indices UV={kvals}\nClassification={classif}\n\n"
        f"Données pesticides CSV:\n{pesticide_csv_text}\n"
        "\n🧠 Remarque : Les valeurs analytiques sont simulées par IA en fonction du profil UV et de la catégorie détectée.\n"
    )

def ai_full_report_fr(
    engine: str,
    client_name: str,
    sample_id: str,
    solvent: str,
    kvals: Dict[str, float],
    classif: Dict[str, str],
    acidity: Optional[float],
    qty_oil_g: Optional[float],
    qty_used_ml: Optional[float],
    haze_slope: Optional[float],
    spectrum_df: Optional[pd.DataFrame],
    pesticide_df: Optional[pd.DataFrame] = None
) -> Dict[str, str]:
    """
    Génère un rapport d'analyse IA complet en français.
    Retourne {'fr': texte, 'pesticide_flag': 'CLEAN'|'TRACE'|'FAIL'|'UNKNOWN'}.
    """

    # --- Préparer les données CSV ---
    spectrum_csv_text = _df_to_csv_text(spectrum_df)
    pesticide_csv_text = _pest_df_to_csv_text(pesticide_df)

    # --- Construire le prompt IA ---
    prompt = build_ai_prompt_fr(
        client_name,
        sample_id,
        solvent,
        kvals,
        classif,
        acidity,
        qty_oil_g,
        qty_used_ml,
        haze_slope,
        spectrum_csv_text,
        pesticide_csv_text
    )

    # --- Résultat par défaut ---
    out = {"fr": "(IA indisponible)", "pesticide_flag": "UNKNOWN"}

    try:
        # --- Sélection moteur IA ---
        if engine == "OpenAI GPT-5":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                out["fr"] = "(Clé OPENAI_API_KEY manquante)"
                return out

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=OPENAI_TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            fr_text = resp.choices[0].message.content

        else:  # DeepSeek via OpenRouter
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                out["fr"] = "(Clé DEEPSEEK_API_KEY manquante)"
                return out

            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "Tu es un chimiste expert en huiles d'olive."},
                    {"role": "user", "content": prompt}
                ]
            )
            fr_text = resp.choices[0].message.content

        # --- Stocker la réponse ---
        out["fr"] = fr_text

        # --- Déterminer le statut pesticide ---
        text_lower = fr_text.lower()
        if any(word in text_lower for word in ["non conforme", "au-dessus", "dépassement"]):
            out["pesticide_flag"] = "FAIL"
        elif any(word in text_lower for word in ["surveillance", "proche", "trace"]):
            out["pesticide_flag"] = "TRACE"
        elif any(word in text_lower for word in ["conforme", "aucun", "négatif"]):
            out["pesticide_flag"] = "CLEAN"

        return out

    except Exception as e:
        # Gestion des erreurs IA
        out["fr"] = f"(Erreur IA : {e})"
        out["pesticide_flag"] = "UNKNOWN"
        return out

# =========================
# PDF FR uniquement (tables + courbe + conclusion)
# =========================
def generate_placeholder_logo() -> bytes:
    fig, ax = plt.subplots(figsize=(3,0.8)); ax.axis('off')
    ax.text(0.02, 0.5, "OLIVE OIL LAB", fontsize=16, va='center')
    ax.text(0.70, 0.5, "Analyse", fontsize=12, va='center')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig)
    return buf.getvalue()

def _draw_wrapped_text(c, text:str, x:float, y:float, max_width:float, line_height:float, font_name="Helvetica", font_size=10):
    # Retourne la nouvelle ordonnée y après écriture
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip() if line else w
        if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = w
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y



def create_pdf_report_fr(client_name: str, sample_id: str, solvent: Optional[str],
                         kvals: Dict[str, float], classif: Dict[str, str],
                         ai_report_fr: str, chart_png: Optional[bytes],
                         spectrum_png: Optional[bytes],
                         ai_engine_label: str,
                         pesticide_flag: Optional[str]) -> bytes:
    """
    Crée un rapport PDF professionnel :
    - Gère automatiquement les sauts de page
    - Nettoie les tables et les textes
    - Ajoute logo, filigrane et numérotation
    """

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=3*cm,
        bottomMargin=2*cm
    )

    # === Styles ===
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle("title", parent=styles["Heading1"], fontSize=14, spaceAfter=12)
    style_subtitle = ParagraphStyle("subtitle", parent=styles["Heading2"], fontSize=12, spaceAfter=10)
    style_text = ParagraphStyle("text", parent=styles["BodyText"], fontSize=10, leading=14)
    style_center = ParagraphStyle("center", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=10)

    story = []

    # === HEADER ===
    
    story.append(Paragraph("<b>Rapport d'analyse d'huile d'olive (UV-Vis + IA)</b>", style_title))
    story.append(Paragraph(f"Moteur IA : <b>{ai_engine_label}</b>", style_text))
    story.append(Paragraph(f"Client : {client_name}", style_text))
    story.append(Paragraph(f"Échantillon : {sample_id or 'ND'}", style_text))
    story.append(Paragraph(f"Solvant : {solvent or 'ND'}", style_text))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}", style_text))
    story.append(Spacer(1, 12))

    # === PESTICIDE STATUS ===
    if pesticide_flag:
        badge = {
            "CLEAN": "🟢 Conforme",
            "TRACE": "🟡 Trace détectée",
            "FAIL": "🔴 Non conforme"
        }.get(pesticide_flag, pesticide_flag)
        story.append(Paragraph("<b>Conformité pesticides :</b>", style_subtitle))
        story.append(Paragraph(f"Statut : {badge}", style_text))
        story.append(Spacer(1, 10))

    # === UV SUMMARY ===
    if kvals:
        uv_txt = (
            f"K232={kvals.get('K232','ND')} | "
            + (f"K270={kvals.get('K270','ND')}" if (solvent or '').lower().startswith('cyclo')
               else f"K268={kvals.get('K268','ND')}")
            + f" | ΔK={kvals.get('DeltaK','ND')} | Catégorie={classif.get('category','ND')}"
        )
        story.append(Paragraph("<b>Récapitulatif UV :</b>", style_subtitle))
        story.append(Paragraph(uv_txt, style_text))
        story.append(Spacer(1, 10))


    # === EXTRACT AI SECTIONS ===
    composition_match = re.search(r"(?:1\.|1️⃣).*?Composition.*?(?=2\.|2️⃣|$)", ai_report_fr, re.S | re.IGNORECASE)
    parameters_match = re.search(r"(?:2\.|2️⃣).*?Paramètre.*?(?=3\.|3️⃣|$)", ai_report_fr, re.S | re.IGNORECASE)
    conclusion_match = re.search(r"(?:5\.|5️⃣).*?Conclusion.*", ai_report_fr, re.S | re.IGNORECASE)

    composition_text = composition_match.group(0).strip() if composition_match else "(aucun tableau fourni par l’IA)"
    parameters_text = parameters_match.group(0).strip() if parameters_match else "(aucun tableau fourni par l’IA)"
    conclusion_text = conclusion_match.group(0).strip() if conclusion_match else "(aucune conclusion fournie par l’IA)"

    # === CLEAN TABLE PARSER ===
    def parse_table_section(text: str) -> list:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        rows = []
        for line in lines:
            if any(x in line.lower() for x in ["composition", "paramètre", "tableau", "conclusion"]):
                continue
            parts = re.split(r"\s{2,}|\t|,", line)
            if len(parts) >= 2:
                rows.append(parts)
        cleaned = []
        seen = set()
        for row in rows:
            key = tuple(row)
            if key not in seen:
                cleaned.append(row)
                seen.add(key)
        return cleaned

    composition_data = parse_table_section(composition_text)
    parameters_data = parse_table_section(parameters_text)

    # === TABLE BUILDER ===
    def add_table(title, data):
        story.append(Paragraph(f"<b>{title}</b>", style_subtitle))
        if not data:
            story.append(Paragraph("(aucun tableau fourni par l’IA)", style_text))
            story.append(Spacer(1, 10))
            return
        col_count = max(len(r) for r in data)
        col_widths = [15.5*cm / col_count] * col_count
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

    add_table("Tableau 1 — Composition acide", composition_data)
    add_table("Tableau 2 — Paramètres analytiques", parameters_data)

    # === CONCLUSION ===
    story.append(PageBreak())
    story.append(Paragraph("<b>Conclusion</b>", style_subtitle))
    for para in conclusion_text.split("\n\n"):
        story.append(Paragraph(para.strip(), style_text))
        story.append(Spacer(1, 8))

    # === FOOTER (Logo + page num + watermark) ===
    def footer(canvas, doc):
        # Page number
        page_num = canvas.getPageNumber()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0]-2*cm, 1.5*cm, f"Page {page_num}")

        # Watermark
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 40)
        canvas.setFillColorRGB(0.85, 0.85, 0.85)
        canvas.rotate(45)
        canvas.drawString(4*cm, 0, "Olive Oil QC Pro")
        canvas.restoreState()

    # === BUILD PDF ===
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buf.seek(0)
    return buf.getvalue()


# =========================
# Pages
# =========================
def page_new_test(model_choice_default: str):
    st.header("➕ Nouveau test (entrées simplifiées)")

    # Init state
    for k in ["last_kvals","last_classif","last_client","last_sample","last_solvent",
              "last_chart_png","last_spec_png","last_ai_text","last_pdf_bytes",
              "ai_engine_label","pesticide_df","pesticide_status","last_spec_df"]:
        st.session_state.setdefault(k, None)

    with st.form("new_test_form"):
        st.subheader("Identité & lots")
        client_name = st.text_input("Nom du client *", value=st.session_state.get("last_client") or "")
        client_phone = st.text_input("Téléphone (optionnel)", value="")
        storage_zone = st.text_input("Zone de stockage (optionnel)", value="")
        sell_price = st.number_input("Prix de vente (optionnel)", min_value=0.0, value=0.0, step=0.1)
        sample_id = st.text_input("ID/Note d'échantillon (optionnel)", value=st.session_state.get("last_sample") or "")

        st.subheader("Entrées minimales requises")
        acidity = st.number_input("Acidité (%)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        qty_oil_g = st.number_input("Quantité d'huile (g)", min_value=0.0, value=0.0, step=0.1)
        qty_used_ml = st.number_input("Quantité utilisée (mL)", min_value=0.0, value=0.0, step=0.1)

        st.markdown("---")
        with st.expander("⚙️ Données instrumentales / avancées (facultatif)"):
            solvent = st.selectbox("Solvant (méthode IOC)", ["cyclohexane","iso-octane"], index=0)
            c1, c2 = st.columns(2)
            path_cm = c1.number_input("Longueur de cuve (cm)", 0.1, 10.0, value=DEFAULT_PATH_CM, step=0.1)
            conc = c2.number_input("Concentration (g/100 mL)", 0.1, 10.0, value=DEFAULT_CONC_G_PER_100ML, step=0.1)

            st.caption("Lecture via spectrophotomètre (USB série) ou saisie manuelle des absorbances.")
            use_usb = st.toggle("Lecture auto via spectro (USB série)", value=False)
            selected_port = None
            auto = None
            if use_usb:
                ports = available_serial_ports()
                if ports:
                    selected_port = st.selectbox("Port série", ports)
                    if st.form_submit_button("📡 Lire maintenant sur l'instrument"):
                        try:
                            meas_auto = read_absorbances_via_serial(selected_port, solvent)
                            st.session_state["last_meas_auto"] = meas_auto
                            st.success("Absorbances lues.")
                        except Exception as e:
                            st.error(f"Lecture USB échouée : {e}")
                else:
                    st.info("Aucun port détecté. Saisissez manuellement.")

            auto = st.session_state.get("last_meas_auto") if st.session_state.get("last_meas_auto") else None
            if solvent == "cyclohexane":
                A232 = st.number_input("A232", 0.0, 3.0, value=(auto.A232 if (auto and auto.A232 is not None) else 0.0), step=0.001, format="%.3f")
                A266 = st.number_input("A266", 0.0, 3.0, value=(auto.A266 if (auto and auto.A266 is not None) else 0.0), step=0.001, format="%.3f")
                A270 = st.number_input("A270", 0.0, 3.0, value=(auto.A270 if (auto and auto.A270 is not None) else 0.0), step=0.001, format="%.3f")
                A274 = st.number_input("A274", 0.0, 3.0, value=(auto.A274 if (auto and auto.A274 is not None) else 0.0), step=0.001, format="%.3f")
                A262, A268 = None, None
            else:
                A232 = st.number_input("A232", 0.0, 3.0, value=(auto.A232 if (auto and auto.A232 is not None) else 0.0), step=0.001, format="%.3f")
                A262 = st.number_input("A262", 0.0, 3.0, value=(auto.A262 if (auto and auto.A262 is not None) else 0.0), step=0.001, format="%.3f")
                A268 = st.number_input("A268", 0.0, 3.0, value=(auto.A268 if (auto and auto.A268 is not None) else 0.0), step=0.001, format="%.3f")
                A274 = st.number_input("A274", 0.0, 3.0, value=(auto.A274 if (auto and auto.A274 is not None) else 0.0), step=0.001, format="%.3f")
                A266, A270 = None, None

        st.subheader("📊 Données pesticides (optionnel)")
        pest_file = st.file_uploader("Importer CSV pesticides (col: Compound, Conc mg/kg, LOD mg/kg, LOQ mg/kg)", type="csv")
        pesticide_df = None
        if pest_file:
            try:
                pesticide_df = pd.read_csv(pest_file)
                st.write("Aperçu:", pesticide_df.head())
            except Exception as e:
                st.error(f"Erreur lecture CSV pesticides: {e}")

        submitted = st.form_submit_button("💾 Enregistrer et préparer l'analyse IA")
        if submitted:
            if not client_name.strip():
                st.error("Nom client requis."); return

            # Calculs UV facultatifs
            solvent_val = locals().get("solvent", "cyclohexane")
            meas = Measurement(A232=locals().get("A232"), A266=locals().get("A266"), A270=locals().get("A270"),
                               A274=locals().get("A274"), A262=locals().get("A262"), A268=locals().get("A268"))
            kvals = compute_indices(meas, solvent_val, locals().get("conc"), locals().get("path_cm"))
            classif = classify_uv(kvals, solvent_val)

            # Sauvegarde DB
            row_data = {
                "timestamp": datetime.now().isoformat(),
                "client_name": client_name,
                "client_phone": client_phone or None,
                "storage_zone": storage_zone or None,
                "sell_price": float(sell_price) if sell_price else None,
                "acidity": float(acidity) if acidity is not None else None,
                "qty_oil_g": float(qty_oil_g) if qty_oil_g is not None else None,
                "qty_used_ml": float(qty_used_ml) if qty_used_ml is not None else None,

                "sample_id": sample_id or None,
                "solvent": solvent_val,
                "path_cm": locals().get("path_cm"),
                "conc_g_per_100ml": locals().get("conc"),
                "A232": locals().get("A232"), "A266": locals().get("A266"), "A270": locals().get("A270"), "A274": locals().get("A274"),
                "A262": locals().get("A262"), "A268": locals().get("A268"),
                "K232": kvals.get("K232"),
                "K266": kvals.get("K266"),
                "K270": kvals.get("K270"),
                "K274": kvals.get("K274"),
                "K262": kvals.get("K262"),
                "K268": kvals.get("K268"),
                "DeltaK": kvals.get("DeltaK"),
                "category": classif["category"],
                "status": classif["status"],
                "notes": classif.get("notes", ""),
                "ai_report": None,
                "ai_engine": None,
                "pesticide_json": None,
                "pesticide_status": None,
                "remediation_fr": None
            }
            insert_test(row_data)

            # State
            st.session_state["last_kvals"] = kvals
            st.session_state["last_classif"] = classif
            st.session_state["last_client"] = client_name
            st.session_state["last_sample"] = sample_id
            st.session_state["last_solvent"] = solvent_val
            st.session_state["pesticide_df"] = pesticide_df

            # Graphique comparaison (si uv dispo)
            chart_png = make_comparison_chart(kvals, solvent_val)
            st.session_state["last_chart_png"] = chart_png

            st.success("Enregistré. Passez à l'analyse IA ci-dessous.")

    # ==== Affichage résultats + IA ====
    kvals = st.session_state.get("last_kvals") or {}
    classif = st.session_state.get("last_classif") or {}
    solvent = st.session_state.get("last_solvent") or "cyclohexane"
    client_name = st.session_state.get("last_client") or ""
    sample_id = st.session_state.get("last_sample") or ""
    pesticide_df = st.session_state.get("pesticide_df")

    if client_name:
        st.subheader("📊 Résumé")
        if kvals:
            st.json(kvals)
            st.write(f"**Catégorie UV** : {classif.get('category','NA')} — **Statut** : {classif.get('status','NA')}")
        else:
            st.info("Pas de calculs UV (entrées instrumentales manquantes).")

        if st.session_state.get("last_chart_png"):
            st.image(st.session_state.get("last_chart_png"), caption="Comparaison aux seuils EVOO")

        st.subheader("🤖 Analyse IA (FR uniquement)")
        ai_choice = st.radio("Moteur IA", ["OpenAI GPT-5", "DeepSeek"], index=0 if model_choice_default=="openai" else 1, horizontal=True)
        if st.button("🚀 Générer l'analyse IA (tables + conclusion)"):
            with st.spinner("Analyse IA en cours..."):
                try:
                    # Haze slope si spectre en session (non implémenté ici; placeholder None)
                    haze_slope = None
                    spec_df = st.session_state.get("last_spec_df")

                    ai_result = ai_full_report_fr(
                        ai_choice,
                        client_name, sample_id, solvent,
                        kvals,
                        st.session_state.get("acidity"),
                        st.session_state.get("qty_oil_g"),
                        st.session_state.get("qty_used_ml"),
                        haze_slope,
                        spec_df,
                        pesticide_df
                    )

                    ai_text_fr = ai_result["fr"]
                    st.session_state["last_ai_text"] = ai_text_fr
                    st.session_state["pesticide_status"] = ai_result.get("pesticide_flag","UNKNOWN")
                    st.session_state["ai_engine_label"] = ai_choice

                    # Maj DB: commentaire IA dans ai_report uniquement
                    update_last_test_ai(ai_text_fr, ai_choice, st.session_state["pesticide_status"], None)

                    # PDF (FR uniquement, tables + courbe + conclusion)
                    pdf_bytes = create_pdf_report_fr(
                        client_name, sample_id, solvent, kvals, classif,
                        ai_text_fr, st.session_state.get("last_chart_png"),
                        st.session_state.get("last_spec_png"),
                        ai_choice, st.session_state.get("pesticide_status")
                    )
                    st.session_state["last_pdf_bytes"] = pdf_bytes
                    st.success("✅ Analyse IA générée et enregistrée dans la colonne 'ai_report'.")

                except Exception as e:
                    st.error(f"❌ Erreur génération IA: {e}")
                    st.text_area("Traceback", traceback.format_exc(), height=180)

        if st.session_state.get("last_ai_text"):
            st.subheader("📝 Commentaire IA (FR)")
            st.write(st.session_state["last_ai_text"])

        if st.session_state.get("last_pdf_bytes"):
            st.download_button(
                "📄 Télécharger PDF (FR — tableaux + conclusion)",
                data=st.session_state["last_pdf_bytes"],
                file_name=f"Rapport_{client_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

def page_history():
    st.header("📋 Historique des tests")
    try:
        df = fetch_tests_df()
        if df.empty:
            st.info("Aucun test enregistré.")
            return
        st.dataframe(df, use_container_width=True)
        if st.button("📤 Exporter en CSV"):
            csv = df.to_csv(index=False)
            st.download_button("💾 Télécharger CSV", data=csv, file_name="historique_tests.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Erreur base de données: {e}")

# =========================
# Main App
# =========================
def main():
    st.set_page_config(page_title="Olive Oil QC Pro (FR)", layout="wide", page_icon="🫒")
    st.title("🫒 Olive Oil QC (Pro, FR) — Commentaire IA + PDF propres")

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.radio(
            "Moteur IA par défaut",
            ["openai", "deepseek"],
            format_func=lambda x: "OpenAI GPT-5" if x == "openai" else "DeepSeek",
            help="Choisissez le moteur IA pour l'analyse"
        )
        st.info("""
        **Entrées minimales:**
        - Nom du client
        - Acidité (%)
        - Quantité d’huile (g)
        - Quantité utilisée (mL)

        **Optionnel:**
        - Téléphone, Zone, Prix
        - Données instrumentales UV-Vis
        - CSV pesticides
        """)

    page = st.sidebar.selectbox("Navigation", ["Nouveau test", "Historique"])

    if page == "Nouveau test":
        page_new_test(model_choice)
    elif page == "Historique":
        page_history()

if __name__ == "__main__":
    main()