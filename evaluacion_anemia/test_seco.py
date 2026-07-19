"""
test_seco.py - Prueba sin API calls
=====================================
Verifica imports, estructura de datos, y generacion de graficos
sin consumir RPD. Si falla algo, reporta antes de gastar recursos.
"""

import os
import sys
import time
import numpy as np
import pandas as pd


PASS = 0
FAIL = 0

LOG_PATH = os.path.join(os.path.dirname(__file__), "resultados", "test_seco.log")

def log(msg):
    print(msg)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def test(nombre: str, condicion: bool, detalle: str = ""):
    global PASS, FAIL
    if condicion:
        PASS += 1
    else:
        FAIL += 1
    icono = "OK" if condicion else "FAIL"
    msg = f"  [{icono}] {nombre}"
    log(msg)
    if not condicion and detalle:
        log(f"       {detalle}")


# ── 1. Verificar entorno ─────────────────────────────────────────
log("=" * 60)
log("TEST SECO - Verificacion de entorno")
log("=" * 60)

test("Python >= 3.10", sys.version_info >= (3, 10), sys.version)
test("Directorio existe", os.path.isdir("evaluacion_anemia"))


# ── 2. Config ────────────────────────────────────────────────────
print("\n[2/8] Config...")
try:
    from config import (
        GEMINI_API_KEY, GENERATION_MODEL, EVALUATION_MODEL,
        EMBEDDING_MODEL, DATASET_PATH, OUTPUT_DIR,
        RPM_GENERACION, RPD_GENERACION,
        RPM_EVALUACION, RPD_EVALUACION,
        CHECKPOINT_PATH, DELAY_ENTRE_GENERACION, DELAY_ENTRE_EVALUACION,
    )
    test("Config importado", True)
    test("API Key presente", GEMINI_API_KEY != "AQUI_TU_API_KEY",
         "(usando placeholder o key real)")
    test("Generation model", "flash" in GENERATION_MODEL.lower() or "gemini" in GENERATION_MODEL.lower(),
         GENERATION_MODEL)
    test("Evaluation model", "gemma" in EVALUATION_MODEL.lower() or "gemini" in EVALUATION_MODEL.lower(),
         EVALUATION_MODEL)
    test("Embedding model", "embedding" in EMBEDDING_MODEL.lower(), EMBEDDING_MODEL)
    test("RPM generacion 1-15", 1 <= RPM_GENERACION <= 15)
    test("RPD generacion <= 500", RPD_GENERACION <= 500)
    test("RPM evaluacion 1-15", 1 <= RPM_EVALUACION <= 15)
    test("RPD evaluacion <= 1500", RPD_EVALUACION <= 1500)
    test("Delay generacion ~4.3s", 3.0 <= DELAY_ENTRE_GENERACION <= 5.0)
    test("Delay evaluacion ~4.3s", 3.0 <= DELAY_ENTRE_EVALUACION <= 5.0)
    test("Output dir", OUTPUT_DIR)
    test("Checkpoint path", CHECKPOINT_PATH)
except Exception as e:
    test("Config importado", False, str(e))


# ── 3. Rate Limiter ──────────────────────────────────────────────
print("\n[3/8] Rate Limiter...")
try:
    from rate_limiter import (
        RateLimiter, get_handler_generacion, get_handler_evaluacion,
        get_handler_embedding, RPDAgotadoError, rpd_status,
    )

    lim = RateLimiter(max_calls=100, window=60, delay_between=0.01)
    test("Limitador creado", True)
    test("Limite max_calls", lim.max_calls == 100)

    t0 = time.time()
    lim.wait()  # primera vez no deberia bloquear mas de ~0.01s
    test("wait() rapido en frio", time.time() - t0 < 1.0)
    test("total_calls incrementa", lim.total_calls > 0)

    h_gen = get_handler_generacion()
    h_eval = get_handler_evaluacion()
    h_emb = get_handler_embedding()
    test("Handler generacion", h_gen is not None)
    test("Handler evaluacion", h_eval is not None)
    test("Handler embeddings", h_emb is not None)
    test("Handlers separados", (id(h_gen) != id(h_eval)) and (id(h_eval) != id(h_emb)))

    status = rpd_status()
    test("RPD status dict", isinstance(status, dict))
    test("RPD tiene generacion", "generacion" in status)
    test("RPD tiene evaluacion", "evaluacion" in status)
    test("RPD tiene embedding", "embedding" in status)

    try:
        raise RPDAgotadoError("test")
    except RPDAgotadoError:
        test("RPDAgotadoError atrapable", True)

except Exception as e:
    import traceback
    test("Rate limiter completo", False, traceback.format_exc())


# ── 4. Dataset ───────────────────────────────────────────────────
print("\n[4/8] Dataset...")
try:
    from generar_dataset import main as gen_dataset

    # Verificar que existe (o generarlo en seco)
    ruta_dataset = os.path.join(os.path.dirname(__file__), "..", "dataset_anemia_100.csv")
    if os.path.exists(ruta_dataset):
        df = pd.read_csv(ruta_dataset, encoding="utf-8")
        test(f"Dataset cargado: {len(df)} filas", len(df) >= 50)
        test("Tiene columna Pregunta", "Pregunta" in df.columns)
        test("Tiene columna Categoria", "Categoria" in df.columns)
        test("Tiene columna ID", "ID" in df.columns)
        if "Categoria" in df.columns:
            cats = df["Categoria"].unique()
            test(f"Categorias: {len(cats)}", len(cats) >= 2)
            cats_largas = [c for c in cats if len(str(c)) > 60]
            test("Sin categorias sospechosas Q029", len(cats_largas) == 0,
                 f"Sospechosas: {cats_largas}")
        if "Pregunta" in df.columns:
            test("Preguntas no vacias", df["Pregunta"].notna().all())
    else:
        test("Dataset existe", False, f"No encontrado en {ruta_dataset}")
except Exception as e:
    import traceback
    test("Dataset", False, traceback.format_exc())


# ── 5. Procesar documentos + retrieval ───────────────────────────
print("\n[5/8] Procesar documentos y retrieval...")
try:
    import procesar_documentos
    import retrieval_local
    test("procesar_documentos importado", True)
    test("retrieval_local importado", True)

    # Verificar que RecuperadorLocal existe
    rec = retrieval_local.RecuperadorLocal()
    test("RecuperadorLocal instanciado", True)
    test("Tiene metodo cargar", hasattr(rec, "cargar"))
    test("Tiene metodo recuperar", hasattr(rec, "recuperar"))

    # cache_existe
    existe = procesar_documentos.cache_existe()
    test("cache_existe() ejecuta sin error", True)

except Exception as e:
    import traceback
    test("Documentos", False, traceback.format_exc())


# ── 6. Generacion de respuestas (sin API) ────────────────────────
print("\n[6/8] Generador de respuestas...")
try:
    import generador_respuestas
    test("generador_respuestas importado", True)
    test("Tiene RPDAgotadoError", hasattr(generador_respuestas, "RPDAgotadoError"))
    test("Tiene generar_todas_respuestas",
         hasattr(generador_respuestas, "generar_todas_respuestas"))
except Exception as e:
    test("Generador", False, str(e))


# ── 7. Evaluadores (sin API) ─────────────────────────────────────
print("\n[7/8] Evaluadores...")
try:
    import evaluador_metricas
    test("evaluador_metricas importado", True)
    test("Tiene evaluar_todo", hasattr(evaluador_metricas, "evaluar_todo"))

    import evaluador_opacidad
    test("evaluador_opacidad importado", True)
    test("Tiene evaluar_opacidad", hasattr(evaluador_opacidad, "evaluar_opacidad"))
    # Probar NLP puro sin API
    texto_prueba = "Tal vez el bebe podria tener anemia. Generalmente se recomienda hierro."
    res = evaluador_opacidad.analizar_opacidad(texto_prueba)
    test("analizar_opacidad funciona", res["total_marcadores"] > 0)
    test("opacidad_score es float", isinstance(res["opacidad_score"], float))

    import evaluador_pod
    test("evaluador_pod importado", True)
    test("Tiene evaluar_pod", hasattr(evaluador_pod, "evaluar_pod"))

except Exception as e:
    import traceback
    test("Evaluadores", False, traceback.format_exc())


# ── 8. Graficos con datos sinteticos ─────────────────────────────
print("\n[8/8] Graficos con datos sinteticos...")
try:
    from generador_graficos import generar_graficos, tabla_medias_por_categoria, tabla_delta_individual

    # Crear DataFrame sintetico con todas las columnas esperadas
    np.random.seed(42)
    n = 10
    df_sintetico = pd.DataFrame({
        "ID": range(n),
        "Categoria": ["Prevencion"] * 4 + ["Diagnostico"] * 3 + ["Tratamiento"] * 3,
        "Pregunta": [f"Pregunta {i}" for i in range(n)],
        "Respuesta_Sin_RAG": [f"Respuesta sin RAG {i}" for i in range(n)],
        "Respuesta_Con_RAG": [f"Respuesta con RAG {i}" for i in range(n)],
        "Groundedness_ConRAG": np.random.uniform(0.3, 1.0, n),
        "Faithfulness_Afirmaciones": np.random.randint(1, 10, n),
        "Faithfulness_Respaldadas": np.random.randint(1, 8, n),
        "Concordancia_SinRAG": np.random.uniform(0.2, 0.9, n),
        "Concordancia_ConRAG": np.random.uniform(0.3, 1.0, n),
        "Concordancia_Semantica_SinRAG": np.random.uniform(0.2, 0.9, n),
        "Concordancia_Semantica_ConRAG": np.random.uniform(0.3, 1.0, n),
        "Concordancia_Entidades_SinRAG": np.random.uniform(0.1, 0.8, n),
        "Concordancia_Entidades_ConRAG": np.random.uniform(0.2, 0.9, n),
        "Precision_Factual_SinRAG": np.random.uniform(0.2, 0.9, n),
        "Precision_Factual_ConRAG": np.random.uniform(0.3, 1.0, n),
        "Seguridad_SinRAG": np.random.uniform(0.3, 1.0, n),
        "Seguridad_ConRAG": np.random.uniform(0.4, 1.0, n),
        "Opacidad_Score_SinRAG": np.random.uniform(0, 3, n),
        "Opacidad_Score_ConRAG": np.random.uniform(0, 2.5, n),
        "Opacidad_Clasificacion_SinRAG": ["BAJA"] * n,
        "Opacidad_Clasificacion_ConRAG": ["BAJA"] * n,
        "Opacidad_Densidad_SinRAG": np.random.uniform(0, 1.5, n),
        "Opacidad_Densidad_ConRAG": np.random.uniform(0, 1.2, n),
        "Opacidad_Marcadores_SinRAG": np.random.randint(0, 10, n),
        "Opacidad_Marcadores_ConRAG": np.random.randint(0, 8, n),
        "Opacidad_Detalle_SinRAG": [""] * n,
        "Opacidad_Detalle_ConRAG": [""] * n,
        "PoD_Score_SinRAG": np.random.uniform(0, 0.5, n),
        "PoD_Score_ConRAG": np.random.uniform(0, 0.4, n),
        "PoD_Ajustado_SinRAG": np.random.uniform(0, 0.5, n),
        "PoD_Ajustado_ConRAG": np.random.uniform(0, 0.4, n),
        "PoD_Clasificacion_SinRAG": ["BAJO"] * n,
        "PoD_Clasificacion_ConRAG": ["BAJO"] * n,
        "PoD_Persuasividad_SinRAG": np.random.uniform(0, 0.5, n),
        "PoD_Persuasividad_ConRAG": np.random.uniform(0, 0.4, n),
        "PoD_Falsedad_SinRAG": np.random.uniform(0, 0.3, n),
        "PoD_Falsedad_ConRAG": np.random.uniform(0, 0.2, n),
        "PoD_Senales_Engano_SinRAG": np.random.uniform(0, 0.4, n),
        "PoD_Senales_Engano_ConRAG": np.random.uniform(0, 0.3, n),
    })

    test("DataFrame sintetico creado", len(df_sintetico) == n)

    ruta = generar_graficos(df_sintetico)
    test("generar_graficos() ejecuta sin error", os.path.isdir(ruta))

    # Verificar archivos generados
    archivos = os.listdir(ruta)
    test(f"Archivos en {ruta}: {len(archivos)}", len(archivos) >= 3)
    for req in ["tabla_medias_categoria.csv", "tabla_deltas_individual.csv"]:
        test(f"Contiene {req}", req in archivos)

    # Tablas individuales
    t1 = tabla_medias_por_categoria(df_sintetico)
    test("tabla_medias_por_categoria() OK", len(t1) > 0)
    t2 = tabla_delta_individual(df_sintetico)
    test("tabla_delta_individual() OK", len(t2) > 0)

except Exception as e:
    import traceback
    test("Graficos", False, traceback.format_exc())


# ── 9. Consolidacion (Excel sin API) ─────────────────────────────
print("\n[9/8] Consolidacion Excel...")
try:
    from consolidacion import exportar_excel
    ruta_excel = exportar_excel(df_sintetico, nombre_archivo="test_seco.xlsx")
    test("Excel generado", os.path.exists(ruta_excel), ruta_excel)
    if os.path.exists(ruta_excel):
        tam = os.path.getsize(ruta_excel)
        test(f"Tamano: {tam/1024:.1f} KB", tam > 0)
        # Leer de vuelta
        df_check = pd.read_excel(ruta_excel, engine="openpyxl")
        test("Excel legible", f"{len(df_check)} filas")
        # Limpiar
        os.remove(ruta_excel)

except Exception as e:
    import traceback
    test("Excel", False, traceback.format_exc())


# ── 10. Main.py parseo de args ───────────────────────────────────
print("\n[10/8] main.py argumentos...")
try:
    import main as main_module
    test("main.py importado", True)
    test("Tiene main()", callable(main_module.main))
    test("Tiene mostrar_status", hasattr(main_module, "mostrar_status"))
    test("Tiene --status", "--status" in open(
        os.path.join(os.path.dirname(__file__), "main.py"), encoding="utf-8"
    ).read())
except Exception as e:
    test("main.py", False, str(e))


# ── Resumen ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESUMEN: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print("=" * 60)

if FAIL > 0:
    print("\n⚠  Algunas pruebas fallaron. Revisa los detalles arriba.")
    sys.exit(1)
else:
    print("\nTODO OK - Pipeline listo para ejecucion real.")
    print("  Comando: cd evaluacion_anemia && python main.py")
    print("  O con pasos separados:")
    print("    python main.py --forzar-dataset")
    print("    python main.py (generacion + evaluacion completa)")
    sys.exit(0)
