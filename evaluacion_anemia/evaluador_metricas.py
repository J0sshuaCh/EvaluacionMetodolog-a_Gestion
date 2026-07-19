"""
evaluador_metricas.py - Groundedness + Concordancia con Directrices
====================================================================
Implementa dos metricas del framework:

1. GROUNDEDNESS (Faithfulness):
   Evaluacion LLM-as-Judge usando Gemma 4 31B sobre la respuesta ConRAG
   vs su contexto recuperado.

2. CONCORDANCIA CON DIRECTRICES:
   Compuesta = 70% Similitud Semantica (LLM-as-Judge con Gemma 4 31B)
              + 30% Coincidencia de Entidades (NLP regex)
   - ConRAG: evaluacion completa (semantica + entidades)
   - SinRAG: solo entidades (componente NLP, sin gastar RPD en evaluacion)
"""

import re
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from google import genai

from config import (
    GEMINI_API_KEY,
    EVALUATION_MODEL,
    RETRY_DELAY,
    MAX_RETRIES,
    MAX_RETRY_BACKOFF,
    OUTPUT_DIR,
    CHECKPOINT_PATH,
)
from rate_limiter import (
    get_handler_evaluacion,
    RPDAgotadoError,
    rpd_status,
)


# ── Cliente Gemini ──────────────────────────────────────────────
_cliente = None


def get_cliente():
    global _cliente
    if _cliente is None:
        _cliente = genai.Client(api_key=GEMINI_API_KEY)
    return _cliente


def _call_gemini(prompt: str, desc: str = "") -> Optional[str]:
    """LLM call con Gemma 4 31B, rate limiting y RPD check."""
    client = get_cliente()
    handler = get_handler_evaluacion()

    for intento in range(1, MAX_RETRIES + 1):
        try:
            handler.limiter.wait()
            response = client.models.generate_content(
                model=EVALUATION_MODEL,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 2048},
            )
            if response.text:
                return response.text.strip()
            return ""
        except RPDAgotadoError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                espera = min(RETRY_DELAY * (2 ** intento), MAX_RETRY_BACKOFF)
                if intento == MAX_RETRIES:
                    print(f"  [ERROR] Cuota agotada ({desc}): {e}")
                    return None
                print(f"  [RATE] Cuota evaluacion (intento {intento}, espera {espera}s): {desc}")
                time.sleep(espera)
            else:
                if intento == MAX_RETRIES:
                    print(f"  [ERROR] ({desc}): {e}")
                    return None
                if "404" in error_str or "not found" in error_str:
                    print(f"  [ERROR] Recurso no encontrado ({desc}): {e}")
                    return None
                espera = min(RETRY_DELAY * (2 ** intento), MAX_RETRY_BACKOFF)
                print(f"  [RETRY] Error (intento {intento}, espera {espera}s): {desc}")
                time.sleep(espera)
    return None


# ── 1. GROUNDEDNESS (Faithfulness) ──────────────────────────────
PROMPT_GROUNDEDNESS = """Eres un evaluador de fidelidad medica. Determina que tan
fiel (grounded) es una respuesta al contexto oficial.

CONTEXTO OFICIAL:
```
{contexto}
```

RESPUESTA GENERADA:
```
{respuesta}
```

INSTRUCCIONES:
1. Enumera cada AFIRMACION CLINICA en la respuesta.
2. Clasifica cada una como:
   - "RESPALDADA": Se sustenta en el contexto.
   - "CONTRADICTA": El contexto dice algo opuesto.
   - "NO_MENCIONADA": No aparece en el contexto.
3. FAITHFULNESS = respaldadas / total

RESPONDE EN JSON:
{{"faithfulness": 0.XX, "total_afirmaciones": N, "respaldadas": N,
  "contradictas": N, "no_mencionadas": N,
  "detalle_afirmaciones": [{{"afirmacion": "...", "clasificacion": "..."}}]}}
"""


def evaluar_groundedness(respuesta: str, contexto: str) -> Dict:
    """Evalua faithfulness de una respuesta vs su contexto."""
    if not respuesta or not contexto:
        return {"faithfulness": 0.0, "total_afirmaciones": 0,
                "respaldadas": 0, "contradictas": 0, "no_mencionadas": 0,
                "detalle_afirmaciones": [], "error": "vacio"}

    prompt = PROMPT_GROUNDEDNESS.format(
        contexto=contexto[:8000],
        respuesta=respuesta[:4000],
    )
    result_text = _call_gemini(prompt, desc="Groundedness")

    if not result_text:
        return {"faithfulness": 0.0, "total_afirmaciones": 0, "error": "fallo"}

    try:
        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else json.loads(result_text)
        result.setdefault("faithfulness", 0.0)
        result.setdefault("total_afirmaciones", 0)
        result.setdefault("respaldadas", 0)
        result.setdefault("contradictas", 0)
        result.setdefault("no_mencionadas", 0)
        result.setdefault("detalle_afirmaciones", [])
        return result
    except (json.JSONDecodeError, ValueError) as e:
        return {"faithfulness": 0.0, "total_afirmaciones": 0,
                "error": f"parse: {e}"}


# ── 2. CONCORDANCIA ─────────────────────────────────────────────
PATRONES_ENTIDADES = {
    "DOSIS_MG_KG": re.compile(
        r"(\d+\.?\d*)\s*(?:mg|miligramos?)\s*(?:/|por|al dia por)\s*(?:kg|kilo|kilogramo)",
        re.IGNORECASE,
    ),
    "DOSIS_GOTAS": re.compile(
        r"(\d+)\s*(?:gotas?)(?:\s*de\s*(\w+\s*\w*\s*(?:ferroso|hierro|polimaltosado)))?",
        re.IGNORECASE,
    ),
    "DOSIS_ML": re.compile(
        r"(\d+\.?\d*)\s*(?:ml|mililitros?|mililitro)",
        re.IGNORECASE,
    ),
    "EDAD_MESES": re.compile(
        r"(\d+)\s*(?:mes(?:es)?)",
        re.IGNORECASE,
    ),
    "EDAD_ANIOS": re.compile(
        r"(\d+)\s*(?:anios?|anitos?)",
        re.IGNORECASE,
    ),
    "EDAD_DIAS": re.compile(
        r"(\d+)\s*(?:dias?|dias? de vida)",
        re.IGNORECASE,
    ),
    "FRECUENCIA_DIARIA": re.compile(
        r"(\d+)\s*veces?\s*(?:al|por)\s*dia",
        re.IGNORECASE,
    ),
    "FRECUENCIA_SEMANAL": re.compile(
        r"(\d+)\s*veces?\s*(?:por|a la|a la)\s*semana",
        re.IGNORECASE,
    ),
    "FRECUENCIA_DIARIO": re.compile(
        r"\bdiari[oa]\b",
        re.IGNORECASE,
    ),
    "VALOR_HB": re.compile(
        r"(\d+\.?\d*)\s*(?:g/dL|gr/dl|g/dl|gramos por decilitro)",
        re.IGNORECASE,
    ),
    "PESO_KG": re.compile(
        r"(\d+\.?\d*)\s*(?:kg|kilos?|kilogramos?)",
        re.IGNORECASE,
    ),
}

ALIMENTOS_CLAVE = [
    "sangrecita", "bazo", "higado", "higado", "bofe", "pulmon", "pulmon",
    "bonito", "jurel", "caballa", "pescado oscuro",
    "lentejas", "frijoles", "quinua", "kiwicha", "espinaca", "betarraga",
    "sulfato ferroso", "hierro polimaltosado", "micronutrientes",
    "chispitas", "hierro heminico", "heminico", "hierro no heminico",
    "vitamina c", "acido ascorbico", "acido ascorbico", "limon", "naranja",
    "leche materna", "alimentacion complementaria",
]


def extraer_entidades(texto: str) -> Dict[str, List[str]]:
    """Extrae entidades clinicas con regex."""
    entidades = {}
    for nombre, patron in PATRONES_ENTIDADES.items():
        matches = patron.findall(texto)
        if matches:
            if isinstance(matches[0], tuple):
                entidades[nombre] = [m[0] for m in matches if m[0]]
            else:
                entidades[nombre] = list(set(matches))

    alimentos_encontrados = []
    texto_lower = texto.lower()
    for alimento in ALIMENTOS_CLAVE:
        if alimento in texto_lower:
            alimentos_encontrados.append(alimento)
    if alimentos_encontrados:
        entidades["ALIMENTOS"] = alimentos_encontrados
    return entidades


def calcular_coincidencia_entidades(
    ent_respuesta: Dict[str, List[str]],
    ent_referencia: Dict[str, List[str]],
) -> float:
    """Proporcion de entidades que coinciden con la referencia."""
    if not ent_referencia:
        return 0.0
    if not ent_respuesta:
        return 0.0

    total_coincidencias = 0
    total_entidades_ref = 0

    for tipo, valores_ref in ent_referencia.items():
        if tipo not in ent_respuesta:
            total_entidades_ref += len(valores_ref)
            continue
        valores_resp = ent_respuesta[tipo]
        for v_ref in valores_ref:
            total_entidades_ref += 1
            for v_resp in valores_resp:
                if v_ref.lower() in v_resp.lower() or v_resp.lower() in v_ref.lower():
                    total_coincidencias += 1
                    break

    if total_entidades_ref == 0:
        return 1.0
    return total_coincidencias / total_entidades_ref


# ── 2b. Answer Correctness via LLM (solo ConRAG) ────────────────
PROMPT_CORRECTNESS = """Eres un evaluador de precision medica. Compara una respuesta
con la respuesta de referencia oficial (ground truth).

PREGUNTA:
```
{pregunta}
```

RESPUESTA DE REFERENCIA:
```
{ground_truth}
```

RESPUESTA GENERADA:
```
{respuesta}
```

Evalua (0.0 a 1.0):
1. PRECISION_FACTUAL: Datos clinicos coinciden con la referencia?
2. RELEVANCIA: Responde directamente sin divagar?
3. SEGURIDAD: Informacion medicamente segura?

RESPONDE EN JSON:
{{"precision_factual": 0.XX, "relevancia": 0.XX, "seguridad": 0.XX,
  "answer_correctness": 0.XX, "errores_identificados": [], "aciertos_identificados": []}}
"""


def evaluar_answer_correctness(pregunta: str, respuesta: str, ground_truth: str) -> Dict:
    """Evalua que tan correcta es una respuesta vs la referencia."""
    if not respuesta or not ground_truth:
        return {"precision_factual": 0.0, "relevancia": 0.0, "seguridad": 0.0,
                "answer_correctness": 0.0, "error": "vacio"}

    prompt = PROMPT_CORRECTNESS.format(
        pregunta=pregunta[:1000],
        ground_truth=ground_truth[:4000],
        respuesta=respuesta[:4000],
    )
    result_text = _call_gemini(prompt, desc="AnswerCorrectness")

    if not result_text:
        return {"answer_correctness": 0.0, "error": "fallo"}

    try:
        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else json.loads(result_text)
        result.setdefault("precision_factual", 0.0)
        result.setdefault("relevancia", 0.0)
        result.setdefault("seguridad", 0.0)
        result.setdefault("answer_correctness", 0.0)
        result.setdefault("errores_identificados", [])
        result.setdefault("aciertos_identificados", [])
        return result
    except (json.JSONDecodeError, ValueError) as e:
        return {"answer_correctness": 0.0, "error": f"parse: {e}"}


def calcular_concordancia(
    pregunta: str,
    respuesta: str,
    ground_truth: str,
    usar_llm: bool = True,
    peso_semantica: float = 0.7,
    peso_entidades: float = 0.3,
) -> Dict:
    """Calcula Concordancia.

    Si usar_llm=False, solo calcula entidades (ahorra llamadas API).
    Esto se usa para SinRAG.
    """
    if usar_llm:
        correctness = evaluar_answer_correctness(pregunta, respuesta, ground_truth)
        score_semantica = correctness.get("answer_correctness", 0.0)
    else:
        correctness = {}
        score_semantica = 0.0

    ent_respuesta = extraer_entidades(respuesta)
    ent_referencia = extraer_entidades(ground_truth)
    score_entidades = calcular_coincidencia_entidades(ent_respuesta, ent_referencia)

    if usar_llm:
        concordancia = peso_semantica * score_semantica + peso_entidades * score_entidades
    else:
        concordancia = score_entidades  # Solo entidades

    return {
        "concordancia": round(concordancia, 4),
        "score_semantico": round(score_semantica, 4),
        "score_entidades": round(score_entidades, 4),
        "precision_factual": correctness.get("precision_factual", 0.0),
        "relevancia": correctness.get("relevancia", 0.0),
        "seguridad": correctness.get("seguridad", 0.0),
        "errores": correctness.get("errores_identificados", []),
        "aciertos": correctness.get("aciertos_identificados", []),
        "entidades_encontradas": ent_respuesta,
        "entidades_referencia": ent_referencia,
    }


# ── Pipeline de evaluacion ──────────────────────────────────────
def evaluar_todo(df: pd.DataFrame, archivo_salida: str = None) -> pd.DataFrame:
    """Ejecuta metricas sobre el DataFrame completo.

    ConRAG: evaluacion completa (Groundedness + Concordancia semantica + entidades)
    SinRAG: solo entidades (gasta 0 RPD de evaluacion)
    """
    print("=" * 60)
    print("EVALUACION DE METRICAS")
    print("=" * 60)
    eval_status = rpd_status()["evaluacion"]
    print(f"  Modelo evaluacion: {eval_status['modelo']}")
    print(f"  RPD disponibles: {eval_status['restantes']}/{eval_status['limite']}")

    req_cols = ["Pregunta", "Respuesta_Referencia_Ground_Truth",
                 "Respuesta_Sin_RAG", "Respuesta_Con_RAG"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"  [ERROR] Faltan columnas: {missing}")
        return df

    score_cols = [
        "Groundedness_ConRAG", "Faithfulness_Afirmaciones",
        "Faithfulness_Respaldadas",
        "Concordancia_SinRAG", "Concordancia_ConRAG",
        "Concordancia_Semantica_SinRAG", "Concordancia_Semantica_ConRAG",
        "Concordancia_Entidades_SinRAG", "Concordancia_Entidades_ConRAG",
        "Precision_Factual_SinRAG", "Precision_Factual_ConRAG",
        "Seguridad_SinRAG", "Seguridad_ConRAG",
        "Errores_SinRAG", "Errores_ConRAG",
    ]
    for col in score_cols:
        if col not in df.columns:
            df[col] = np.nan
            if col in ["Errores_SinRAG", "Errores_ConRAG"]:
                df[col] = df[col].astype(object)

    total = len(df)
    print(f"  Evaluando {total} preguntas...")

    try:
        # ── Groundedness (solo ConRAG) ──
        print("\n  1. GROUNDEDNESS (Faithfulness) - ConRAG")
        for idx in tqdm(range(total), desc="Groundedness"):
            # Saltar si ya evaluado (reanudacion)
            if not pd.isna(df.at[idx, "Groundedness_ConRAG"]):
                continue
            respuesta = str(df.at[idx, "Respuesta_Con_RAG"] or "")
            contexto = str(df.at[idx, "Contexto_Recuperado"] or "")

            if not respuesta or not contexto:
                df.at[idx, "Groundedness_ConRAG"] = 0.0
                df.at[idx, "Faithfulness_Afirmaciones"] = 0
                df.at[idx, "Faithfulness_Respaldadas"] = 0
                continue

            result = evaluar_groundedness(respuesta, contexto)
            df.at[idx, "Groundedness_ConRAG"] = result.get("faithfulness", 0.0)
            df.at[idx, "Faithfulness_Afirmaciones"] = result.get("total_afirmaciones", 0)
            df.at[idx, "Faithfulness_Respaldadas"] = result.get("respaldadas", 0)

        # ── Concordancia ──
        print("\n  2. CONCORDANCIA CON DIRECTRICES")
        for idx in tqdm(range(total), desc="Concordancia"):
            # Saltar si ConRAG ya evaluado (reanudacion)
            if not pd.isna(df.at[idx, "Concordancia_ConRAG"]):
                continue
            pregunta = str(df.at[idx, "Pregunta"] or "")
            ground_truth = str(df.at[idx, "Respuesta_Referencia_Ground_Truth"] or "")
            resp_sin = str(df.at[idx, "Respuesta_Sin_RAG"] or "")
            resp_con = str(df.at[idx, "Respuesta_Con_RAG"] or "")

            if pregunta and ground_truth:
                # Sin RAG: solo entidades (sin gastar RPD en LLM)
                if resp_sin:
                    conc_sin = calcular_concordancia(
                        pregunta, resp_sin, ground_truth, usar_llm=False
                    )
                    df.at[idx, "Concordancia_SinRAG"] = conc_sin["concordancia"]
                    df.at[idx, "Concordancia_Semantica_SinRAG"] = conc_sin["score_semantico"]
                    df.at[idx, "Concordancia_Entidades_SinRAG"] = conc_sin["score_entidades"]
                    df.at[idx, "Precision_Factual_SinRAG"] = conc_sin["precision_factual"]
                    df.at[idx, "Seguridad_SinRAG"] = conc_sin["seguridad"]
                    df.at[idx, "Errores_SinRAG"] = "; ".join(conc_sin.get("errores", []))

                # Con RAG: evaluacion completa con LLM
                if resp_con:
                    conc_con = calcular_concordancia(
                        pregunta, resp_con, ground_truth, usar_llm=True
                    )
                    df.at[idx, "Concordancia_ConRAG"] = conc_con["concordancia"]
                    df.at[idx, "Concordancia_Semantica_ConRAG"] = conc_con["score_semantico"]
                    df.at[idx, "Concordancia_Entidades_ConRAG"] = conc_con["score_entidades"]
                    df.at[idx, "Precision_Factual_ConRAG"] = conc_con["precision_factual"]
                    df.at[idx, "Seguridad_ConRAG"] = conc_con["seguridad"]
                    df.at[idx, "Errores_ConRAG"] = "; ".join(conc_con.get("errores", []))

            if archivo_salida and (idx + 1) % 20 == 0:
                _guardar(df, archivo_salida)

    except RPDAgotadoError:
        print(f"\n  [RPD] Cuota diaria de evaluacion agotada.")
        if archivo_salida:
            _guardar(df, archivo_salida)
            print(f"  Checkpoint guardado en: {archivo_salida}")
        print("  Reanuda manana con: python main.py --saltar-generacion")
        raise

    if archivo_salida:
        _guardar(df, archivo_salida)
        print(f"\n  Resultados guardados en: {archivo_salida}")

    print("\n  RESUMEN DE METRICAS:")
    print(f"    Groundedness (ConRAG):    {df['Groundedness_ConRAG'].mean():.3f}")
    print(f"    Concordancia (SinRAG):     {df['Concordancia_SinRAG'].mean():.3f}")
    print(f"    Concordancia (ConRAG):     {df['Concordancia_ConRAG'].mean():.3f}")

    return df


def _guardar(df, ruta):
    import os
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    df.to_excel(ruta, index=False, engine="openpyxl")
