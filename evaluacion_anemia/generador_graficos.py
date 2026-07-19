"""
generador_graficos.py - Figuras exportables para articulo
=========================================================
Genera:
  1. Tablas de medias por categoria (LaTeX + CSV)
  2. Grafico de barras agrupadas SinRAG vs ConRAG
  3. Scatter plot Concordancia vs PoD
  4. Mapa de calor de correlaciones entre metricas
  5. Radar chart comparativo por categoria
  6. Distribucion de scores (violin/boxplot)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from config import OUTPUT_DIR

# Intentar importar matplotlib; fallar gracefulmente si no esta
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "figure.dpi": 150,
    })

    MATPLOTLIB_DISPONIBLE = True
except ImportError:
    MATPLOTLIB_DISPONIBLE = False

try:
    import seaborn as sns
    SEABORN_DISPONIBLE = True
except ImportError:
    SEABORN_DISPONIBLE = False


# ── Paleta para articulo cientifico ─────────────────────────────
COLOR_SINRAG = "#4C72B0"      # Azul
COLOR_CONRAG = "#DD8452"      # Naranja
COLOR_CATEGORIAS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#937860"]
COLOR_CORRELACION = "vlag"

DIR_GRAFICOS = os.path.join(OUTPUT_DIR, "graficos_articulo")


def _asegurar_dir_graficos():
    if not os.path.exists(DIR_GRAFICOS):
        os.makedirs(DIR_GRAFICOS, exist_ok=True)
    return DIR_GRAFICOS


# ====================================================================
# 1. Tablas de medias por categoria
# ====================================================================

METRICAS_TABLA = [
    ("Concordancia_SinRAG", "Concordancia_ConRAG", "Concordancia con Directrices"),
    ("Concordancia_Semantica_SinRAG", "Concordancia_Semantica_ConRAG", "  Componente Semantico"),
    ("Concordancia_Entidades_SinRAG", "Concordancia_Entidades_ConRAG", "  Coincidencia Entidades"),
    ("Precision_Factual_SinRAG", "Precision_Factual_ConRAG", "Precision Factual"),
    ("Seguridad_SinRAG", "Seguridad_ConRAG", "Seguridad"),
    ("Opacidad_Score_SinRAG", "Opacidad_Score_ConRAG", "Opacidad Epistemica (menor=mejor)"),
    ("PoD_Ajustado_SinRAG", "PoD_Ajustado_ConRAG", "Potencial de Engano (menor=mejor)"),
    ("PoD_Persuasividad_SinRAG", "PoD_Persuasividad_ConRAG", "Persuasividad (menor=mejor)"),
]

METRICAS_PUNTUALES = [
    "Groundedness_ConRAG",
]


def tabla_medias_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla plana: filas=metricas, cols=categorias+global."""
    if "Categoria" not in df.columns:
        return pd.DataFrame({"error": ["Columna Categoria no encontrada"]})

    categorias = sorted(df["Categoria"].dropna().unique())
    filas = []

    # Metricas pareadas (SinRAG / ConRAG)
    for col_sin, col_con, nombre in METRICAS_TABLA:
        fila = {"Metrica": nombre}
        for cat in categorias:
            mask = df["Categoria"] == cat
            val_sin = df.loc[mask, col_sin].mean() if col_sin in df.columns else np.nan
            val_con = df.loc[mask, col_con].mean() if col_con in df.columns else np.nan
            fila[f"{cat}_SinRAG"] = round(val_sin, 4) if not pd.isna(val_sin) else ""
            fila[f"{cat}_ConRAG"] = round(val_con, 4) if not pd.isna(val_con) else ""
        # Global
        g_sin = df[col_sin].mean() if col_sin in df.columns else np.nan
        g_con = df[col_con].mean() if col_con in df.columns else np.nan
        fila["Global_SinRAG"] = round(g_sin, 4) if not pd.isna(g_sin) else ""
        fila["Global_ConRAG"] = round(g_con, 4) if not pd.isna(g_con) else ""
        fila["Delta_(Con-Sin)"] = round(g_con - g_sin, 4) if (not pd.isna(g_sin) and not pd.isna(g_con)) else ""
        filas.append(fila)

    # Metricas puntuales
    for col in METRICAS_PUNTUALES:
        if col not in df.columns:
            continue
        nombre = col.replace("_", " ")
        fila = {"Metrica": nombre}
        for cat in categorias:
            mask = df["Categoria"] == cat
            val = df.loc[mask, col].mean()
            fila[cat] = round(val, 4) if not pd.isna(val) else ""
        fila["Global"] = round(df[col].mean(), 4)
        fila["Delta_(Con-Sin)"] = ""
        filas.append(fila)

    tabla = pd.DataFrame(filas)

    # Exportar CSV
    ruta_csv = os.path.join(_asegurar_dir_graficos(), "tabla_medias_categoria.csv")
    tabla.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"  Tabla CSV: {ruta_csv}")

    # Exportar LaTeX
    ruta_tex = os.path.join(DIR_GRAFICOS, "tabla_medias_categoria.tex")
    try:
        with open(ruta_tex, "w", encoding="utf-8") as f:
            f.write(tabla.to_latex(index=False, float_format="%.4f"))
        print(f"  Tabla LaTeX: {ruta_tex}")
    except Exception as e:
        print(f"  [AVISO] No se pudo exportar LaTeX: {e}")

    return tabla


# ====================================================================
# 2. Grafico de barras agrupadas
# ====================================================================

def _grafico_barras_agrupadas(df: pd.DataFrame):
    """Bar chart: SinRAG vs ConRAG para cada metrica."""
    if not MATPLOTLIB_DISPONIBLE:
        print("  [AVISO] matplotlib no disponible, saltando barras agrupadas")
        return

    metricas = [(s, c, n) for s, c, n in METRICAS_TABLA
                if s in df.columns and c in df.columns]

    if not metricas:
        return

    n = len(metricas)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    medias_sin = [df[s].mean() for s, _, _ in metricas]
    medias_con = [df[c].mean() for _, c, _ in metricas]

    bars1 = ax.bar(x - width / 2, medias_sin, width, label="Sin RAG",
                   color=COLOR_SINRAG, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, medias_con, width, label="Con RAG",
                   color=COLOR_CONRAG, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Metrica")
    ax.set_ylabel("Score promedio")
    ax.set_title("Comparacion Sin RAG vs Con RAG", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, _, n in metricas], rotation=25, ha="right", fontsize=13)
    ax.legend(fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # Valores sobre barras (sobre la mas alta del par)
    for i in range(n):
        max_val = max(medias_sin[i], medias_con[i])
        ax.text(i, max_val + 0.02, f"{max_val:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "barras_sinrag_vs_conrag.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico: {ruta}")


# ====================================================================
# 3. Scatter plot: Concordancia vs PoD
# ====================================================================

def _grafico_scatter_concordancia_pod(df: pd.DataFrame):
    """Scatter: Concordancia (x) vs PoD (y). Punto=1 pregunta."""
    if not MATPLOTLIB_DISPONIBLE:
        return

    for scenario, label, color in [
        ("SinRAG", "Sin RAG", COLOR_SINRAG),
        ("ConRAG", "Con RAG", COLOR_CONRAG),
    ]:
        col_conc = f"Concordancia_{scenario}"
        col_pod = f"PoD_Ajustado_{scenario}"

        if col_conc not in df.columns or col_pod not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        # Colorear por categoria si existe
        if "Categoria" in df.columns:
            categorias = df["Categoria"].dropna().unique()
            for i, cat in enumerate(sorted(categorias)):
                mask = df["Categoria"] == cat
                ax.scatter(
                    df.loc[mask, col_conc], df.loc[mask, col_pod],
                    c=COLOR_CATEGORIAS[i % len(COLOR_CATEGORIAS)],
                    label=cat, alpha=0.7, edgecolors="white", linewidth=0.5, s=80,
                )
            ax.legend(fontsize=13)
        else:
            ax.scatter(df[col_conc], df[col_pod],
                       c=color, alpha=0.6, edgecolors="white", linewidth=0.5, s=60)

        # Linea de tendencia
        try:
            mask_valid = df[col_conc].notna() & df[col_pod].notna()
            x_vals = df.loc[mask_valid, col_conc].values.astype(float)
            y_vals = df.loc[mask_valid, col_pod].values.astype(float)
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.6, linewidth=1.5)
            # Correlacion
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=15, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        except Exception:
            pass

        ax.set_xlabel("Concordancia con Directrices")
        ax.set_ylabel("Potencial de Engano (PoD)")
        ax.set_title(f"Concordancia vs PoD - {label}", fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, linestyle=":")

        # Cuadrantes
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.2)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.2)

        plt.tight_layout()
        ruta = os.path.join(DIR_GRAFICOS, f"scatter_concordancia_pod_{scenario}.png")
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grafico: {ruta}")


# ====================================================================
# 4. Mapa de calor de correlaciones
# ====================================================================

def _grafico_heatmap_correlaciones(df: pd.DataFrame):
    """Heatmap de correlaciones entre todas las metricas."""
    if not MATPLOTLIB_DISPONIBLE or not SEABORN_DISPONIBLE:
        print("  [AVISO] seaborn no disponible, saltando heatmap")
        return

    cols_correlacion = [
        "Groundedness_ConRAG",
        "Concordancia_SinRAG", "Concordancia_ConRAG",
        "Concordancia_Semantica_SinRAG", "Concordancia_Semantica_ConRAG",
        "Concordancia_Entidades_SinRAG", "Concordancia_Entidades_ConRAG",
        "Precision_Factual_SinRAG", "Precision_Factual_ConRAG",
        "Seguridad_SinRAG", "Seguridad_ConRAG",
        "Opacidad_Score_SinRAG", "Opacidad_Score_ConRAG",
        "PoD_Ajustado_SinRAG", "PoD_Ajustado_ConRAG",
        "PoD_Persuasividad_SinRAG", "PoD_Persuasividad_ConRAG",
    ]
    cols_disponibles = [c for c in cols_correlacion if c in df.columns]
    if len(cols_disponibles) < 3:
        return

    corr_df = df[cols_disponibles].dropna().corr()

    # Nombres cortos para el eje
    labels_cortos = {
        "Groundedness_ConRAG": "Groundedness",
        "Concordancia_SinRAG": "Concord_Sin",
        "Concordancia_ConRAG": "Concord_Con",
        "Concordancia_Semantica_SinRAG": "Sem_Sin",
        "Concordancia_Semantica_ConRAG": "Sem_Con",
        "Concordancia_Entidades_SinRAG": "Ent_Sin",
        "Concordancia_Entidades_ConRAG": "Ent_Con",
        "Precision_Factual_SinRAG": "PrecFact_Sin",
        "Precision_Factual_ConRAG": "PrecFact_Con",
        "Seguridad_SinRAG": "Seg_Sin",
        "Seguridad_ConRAG": "Seg_Con",
        "Opacidad_Score_SinRAG": "Opac_Sin",
        "Opacidad_Score_ConRAG": "Opac_Con",
        "PoD_Ajustado_SinRAG": "PoD_Sin",
        "PoD_Ajustado_ConRAG": "PoD_Con",
        "PoD_Persuasividad_SinRAG": "Persuas_Sin",
        "PoD_Persuasividad_ConRAG": "Persuas_Con",
    }
    corr_df = corr_df.rename(index=labels_cortos, columns=labels_cortos)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f",
        cmap=COLOR_CORRELACION, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "r de Pearson"},
        ax=ax,
    )
    ax.set_title("Correlacion entre Metricas de Evaluacion", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "heatmap_correlaciones.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico: {ruta}")

    # Exportar matriz de correlacion a CSV
    ruta_csv = os.path.join(DIR_GRAFICOS, "matriz_correlacion.csv")
    corr_df.to_csv(ruta_csv, encoding="utf-8")
    print(f"  Matriz CSV: {ruta_csv}")


# ====================================================================
# 5. Radar chart comparativo por categoria
# ====================================================================

def _grafico_radar(df: pd.DataFrame):
    """Radar chart: perfil de metricas por categoria."""
    if not MATPLOTLIB_DISPONIBLE or "Categoria" not in df.columns:
        return

    categorias = sorted(df["Categoria"].dropna().unique())
    if len(categorias) < 1:
        return

    # Seleccionar metricas clave para el radar
    cols_radar = [
        "Concordancia_ConRAG",
        "Precision_Factual_ConRAG",
        "Seguridad_ConRAG",
        "Groundedness_ConRAG",
    ]
    cols_radar_invert = [
        "Opacidad_Score_ConRAG",
        "PoD_Ajustado_ConRAG",
    ]

    cols_disponibles = [c for c in cols_radar if c in df.columns]
    cols_inv = [c for c in cols_radar_invert if c in df.columns]
    todas_cols = cols_disponibles + cols_inv

    if len(todas_cols) < 3:
        return

    n_cols = len(todas_cols)
    angles = np.linspace(0, 2 * np.pi, n_cols, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar circulo

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))

    for i, cat in enumerate(categorias):
        mask = df["Categoria"] == cat
        values = []
        for c in cols_disponibles:
            v = df.loc[mask, c].mean()
            values.append(v if not pd.isna(v) else 0)
        for c in cols_inv:
            v = df.loc[mask, c].mean()
            values.append(1 - v if not pd.isna(v) else 0)  # Invertir para radar
        values += values[:1]

        color = COLOR_CATEGORIAS[i % len(COLOR_CATEGORIAS)]
        ax.plot(angles, values, "o-", linewidth=1.5, label=cat, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Etiquetas
    labels = [c.replace("_ConRAG", "").replace("_", "\n") for c in cols_disponibles]
    labels += [c.replace("_ConRAG", "").replace("_", "\n") + "\n(inv)" for c in cols_inv]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_title("Perfil de Metricas por Categoria (Con RAG)",
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=13)

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "radar_por_categoria.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico: {ruta}")


# ====================================================================
# 6. Distribuciones (violin/boxplot)
# ====================================================================

def _grafico_distribuciones(df: pd.DataFrame):
    """Boxplot lado a lado de metricas clave."""
    if not MATPLOTLIB_DISPONIBLE or not SEABORN_DISPONIBLE:
        return

    pares_box = [
        ("Concordancia_SinRAG", "Concordancia_ConRAG", "Concordancia"),
        ("Opacidad_Score_SinRAG", "Opacidad_Score_ConRAG", "Opacidad Epistemica"),
        ("PoD_Ajustado_SinRAG", "PoD_Ajustado_ConRAG", "PoD Ajustado"),
    ]

    for col_sin, col_con, titulo in pares_box:
        if col_sin not in df.columns or col_con not in df.columns:
            continue

        datos = pd.DataFrame({
            "Sin RAG": df[col_sin].dropna(),
            "Con RAG": df[col_con].dropna(),
        })

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(data=datos, inner="box", palette=[COLOR_SINRAG, COLOR_CONRAG], ax=ax)
        ax.set_title(f"Distribucion de {titulo}", fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        plt.tight_layout()
        ruta = os.path.join(DIR_GRAFICOS, f"distribucion_{col_sin.replace('_SinRAG','')}.png")
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grafico: {ruta}")


# ====================================================================
# 7. Tabla de delta (ConRAG - SinRAG) por pregunta
# ====================================================================

def tabla_delta_individual(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con el delta individual por pregunta para cada metrica."""
    filas = []
    for idx in df.index:
        fila = {"ID": df.at[idx, "ID"] if "ID" in df.columns else idx,
                "Categoria": df.at[idx, "Categoria"] if "Categoria" in df.columns else ""}
        for col_sin, col_con, nombre in METRICAS_TABLA:
            if col_sin in df.columns and col_con in df.columns:
                v_sin = df.at[idx, col_sin]
                v_con = df.at[idx, col_con]
                if pd.notna(v_sin) and pd.notna(v_con):
                    fila[f"Delta_{nombre.split()[0]}"] = round(v_con - v_sin, 4)
                else:
                    fila[f"Delta_{nombre.split()[0]}"] = ""
        if "Groundedness_ConRAG" in df.columns:
            fila["Groundedness"] = round(df.at[idx, "Groundedness_ConRAG"], 4) \
                if pd.notna(df.at[idx, "Groundedness_ConRAG"]) else ""
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    ruta_csv = os.path.join(_asegurar_dir_graficos(), "tabla_deltas_individual.csv")
    tabla.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"  Deltas CSV: {ruta_csv}")
    return tabla


# ====================================================================
# Pipeline principal
# ====================================================================

def generar_graficos(df: pd.DataFrame) -> str:
    """Genera todas las figuras y tablas.

    Args:
        df: DataFrame completo con todas las metricas.

    Returns:
        Ruta al directorio con los archivos generados.
    """
    print("\n" + "=" * 60)
    print("📊 GENERACION DE GRAFICOS Y TABLAS PARA ARTICULO")
    print("=" * 60)

    _asegurar_dir_graficos()

    n_graf = 0

    # 1. Tabla de medias
    print("\n[1/7] Tabla de medias por categoria...")
    tabla_medias_por_categoria(df)
    n_graf += 1

    # 2. Tabla de deltas individuales
    print("[2/7] Tabla de deltas individuales...")
    tabla_delta_individual(df)
    n_graf += 1

    # 3. Barras agrupadas
    print("[3/7] Barras agrupadas SinRAG vs ConRAG...")
    _grafico_barras_agrupadas(df)
    n_graf += 1

    # 4. Scatter Concordancia vs PoD
    print("[4/7] Scatter Concordancia vs PoD...")
    _grafico_scatter_concordancia_pod(df)
    n_graf += 1

    # 5. Heatmap de correlaciones
    print("[5/7] Mapa de calor de correlaciones...")
    _grafico_heatmap_correlaciones(df)
    n_graf += 1

    # 6. Radar por categoria
    print("[6/7] Radar chart por categoria...")
    _grafico_radar(df)
    n_graf += 1

    # 7. Distribuciones
    print("[7/7] Distribuciones (violin/boxplot)...")
    _grafico_distribuciones(df)
    n_graf += 1

    print(f"\n✅ Todos los graficos en: {DIR_GRAFICOS}")
    archivos = os.listdir(DIR_GRAFICOS)
    for a in sorted(archivos):
        ruta_completa = os.path.join(DIR_GRAFICOS, a)
        tam = os.path.getsize(ruta_completa)
        print(f"   {a:45s} {tam/1024:.1f} KB")

    return DIR_GRAFICOS
