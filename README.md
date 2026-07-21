# EvaluacionMetodología_Gestion

Evaluación de metodologías de Retrieval-Augmented Generation (RAG) aplicadas a un asistente nutricional materno-infantil (ANMI) enfocado en la prevención, diagnóstico y tratamiento de la anemia infantil.

## Objetivo

Comparar sistemáticamente la calidad de respuestas generadas por un LLM **con RAG** vs **sin RAG**, midiendo si la recuperación de documentos oficiales (MINSA, OMS) reduce alucinaciones y mejora la fiabilidad en un dominio clínico de alto riesgo.

## Dimensiones evaluadas

| Métrica | Descripción |
|---|---|
| **Groundedness** | Fidelidad de la respuesta al contexto recuperado |
| **Concordancia con Directrices** | Similitud semántica y entidades clínicas con la respuesta oficial |
| **Opacidad Epistémica** | Lenguaje evasivo, vago, pseudocientífico o de mitos médicos |
| **Potencial de Engaño (PoD)** | `(1 - Concordancia) × Persuasividad` — riesgo de una respuesta convincente pero incorrecta |
| **Precisión Factual** | Exactitud de datos clínicos (dosis, valores, edades) |
| **Seguridad** | Si la información es médicamente segura |

## Arquitectura del pipeline

```
Dataset (100 preguntas) → Documentos PDF → Embeddings + Chunks → Generación SinRAG/ConRAG
                           ↓
    Evaluación (LLM-as-Judge + NLP) → Consolidación → Reporte Excel + Gráficos
```

- **Generación**: `gemini-3.1-flash-lite` (200 llamadas)
- **Evaluación LLM-as-Judge**: `gemma-4-31b-it` (~300 llamadas)
- **Embeddings**: `gemini-embedding-2`
- **Métricas NLP puras** (0 llamadas API): opacidad epistémica, extracción de entidades clínicas

## Requisitos

```bash
pip install -r evaluacion_anemia/requirements.txt
```

Configurar `GEMINI_API_KEY` en `evaluacion_anemia/config.py`.

## Uso

```bash
cd evaluacion_anemia

# Pipeline completo
python main.py

# Ver estado / presupuesto RPD restante
python main.py --status

# Reanudar sin regenerar respuestas
python main.py --saltar-generacion

# Solo exportar Excel + gráficos desde checkpoint
python main.py --solo-exportar

# Forzar reprocesamiento de documentos
python main.py --forzar-documentos
```

## Salida

- `resultados/evaluacion_resultados_final.xlsx` — Reporte con 3 hojas (detalle, resumen por categoría, comparativa RAG)
- `resultados/graficos_articulo/` — Gráficos en PNG y CSV para publicación
- `dataset_anemia_100.csv` — Dataset de 100 preguntas con ground truth

## Estructura del proyecto

```
evaluacion_anemia/
├── main.py                    # Orquestador
├── config.py                  # Configuración central
├── generador_respuestas.py    # Generación Sin RAG / Con RAG
├── retrieval_local.py         # Búsqueda vectorial (cosine similarity)
├── procesar_documentos.py     # Ingesta PDF → chunks → embeddings
├── evaluador_metricas.py      # Groundedness + Concordancia
├── evaluador_opacidad.py      # Opacidad epistémica (NLP)
├── evaluador_pod.py           # Potencial de engaño
├── evaluador_ragas.py         # Evaluación complementaria RAGAS
├── consolidacion.py           # Exportación Excel final
├── generador_graficos.py      # Gráficos matplotlib/seaborn
├── generar_dataset.py         # Extracción de dataset desde notebook
├── rate_limiter.py            # Control de rate limiting + RPD
├── test_seco.py               # Prueba simulada
├── requirements.txt
├── cache_documentos/          # [gitignored] Caché de chunks/embeddings
└── resultados/                # Reportes, gráficos, logs
```
