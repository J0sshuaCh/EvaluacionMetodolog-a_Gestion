"""
generar_dataset.py - Genera dataset_anemia_100.csv desde el notebook
=====================================================================
Extrae la lista de preguntas del notebook de construcción del dataset
y la exporta a CSV limpio para el pipeline de evaluación.

Uso:  python generar_dataset.py
"""

import json
import csv
import os
import re
import sys
import io

# Forzar UTF-8 en terminal Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Rutas ────────────────────────────────────────────────────────
NOTEBOOK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "construccion_del_dataset.ipynb")
)
CSV_OUTPUT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "dataset_anemia_100.csv")
)


def extraer_lista_desde_notebook(ruta_notebook):
    """Lee el .ipynb, encuentra la variable dataset_prevencion_y_nutricion
    y la evalúa para obtener la lista de diccionarios."""
    with open(ruta_notebook, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Buscar el código fuente de la celda que define la lista
    codigo_completo = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "dataset_prevencion_y_nutricion" in src:
                codigo_completo = src
                break

    if not codigo_completo:
        raise ValueError(
            "No se encontró la variable 'dataset_prevencion_y_nutricion' en el notebook."
        )

    # Extraer SÓLO la asignación de la lista (hasta el ']' de cierre)
    # Usamos exec en un namespace limpio
    namespace = {}
    exec(codigo_completo, namespace)
    datos = namespace.get("dataset_prevencion_y_nutricion")

    if datos is None:
        raise ValueError("La variable 'dataset_prevencion_y_nutricion' no se definió correctamente.")

    return datos


def generar_csv(datos, ruta_csv):
    """Escribe la lista de diccionarios a CSV."""
    columnas = [
        "ID",
        "Categoria",
        "Pregunta",
        "Respuesta_Referencia_Ground_Truth",
        "Respuesta_Sin_RAG",
        "Respuesta_Con_RAG",
    ]

    os.makedirs(os.path.dirname(ruta_csv) or ".", exist_ok=True)

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for row in datos:
            # Asegurar columna Respuesta_con_RAG (puede no existir)
            if "Respuesta_Con_RAG" not in row:
                row["Respuesta_Con_RAG"] = ""
            writer.writerow(row)

    print(f"✅ Dataset generado: {ruta_csv}")
    print(f"   └─ {len(datos)} filas | Columnas: {', '.join(columnas)}")


def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"❌ No se encontró el notebook en:\n   {NOTEBOOK_PATH}")
        print("   Asegúrate de que 'construccion_del_dataset.ipynb' esté en la carpeta padre.")
        sys.exit(1)

    print("📖 Extrayendo datos desde el notebook...")
    datos = extraer_lista_desde_notebook(NOTEBOOK_PATH)
    print(f"   └─ {len(datos)} preguntas encontradas.")

    # ── Estadísticas ──
    categorias = {}
    for d in datos:
        cat = d.get("Categoria", "Sin categoría")
        categorias[cat] = categorias.get(cat, 0) + 1
    print("   └─ Distribución por categoría:")
    for cat, count in sorted(categorias.items()):
        print(f"       • {cat}: {count} preguntas")

    generar_csv(datos, CSV_OUTPUT)


if __name__ == "__main__":
    main()
