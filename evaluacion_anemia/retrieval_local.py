"""
retrieval_local.py - Busqueda vectorial local (cosine similarity)
==================================================================
Dada una consulta, la embeda con Gemini Embedding 1 y encuentra
los top-k chunks mas similares por similitud coseno.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from google import genai

from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    TOP_K,
)
from procesar_documentos import cargar_cache
from rate_limiter import get_handler_embed_query


# ── Cliente Gemini ──────────────────────────────────────────────
_cliente = None


def get_cliente():
    global _cliente
    if _cliente is None:
        _cliente = genai.Client(api_key=GEMINI_API_KEY)
    return _cliente


def embed_consulta(texto: str) -> Optional[np.ndarray]:
    """Genera embedding para una consulta con rate limiting."""
    handler = get_handler_embed_query()
    client = get_cliente()

    def _do_embed():
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texto[:2000],
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

    return handler.call(_do_embed, desc="embed consulta")


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similitud coseno entre dos vectores."""
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b, axis=1)
    if norma_a == 0 or (norma_b == 0).any():
        return 0.0
    return float(np.dot(b, a) / (norma_b * norma_a))


class RecuperadorLocal:
    """Motor de busqueda vectorial local."""

    def __init__(self):
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._cargado = False

    def cargar(self) -> bool:
        """Carga los chunks y embeddings desde la cache."""
        try:
            self.chunks, self.embeddings, metadata = cargar_cache()
            self._cargado = True
            print(f"  Recuperador listo: {len(self.chunks)} chunks indexados")
            return True
        except Exception as e:
            print(f"  [ERROR] Al cargar cache en recuperador: {e}")
            print("  Ejecuta primero 'procesar_documentos.py' para generar la cache.")
            return False

    @property
    def esta_cargado(self) -> bool:
        return self._cargado and self.embeddings is not None and len(self.chunks) > 0

    def recuperar(self, consulta: str, k: int = None) -> List[Dict]:
        """Recupera los top-k chunks mas similares a la consulta."""
        if not self.esta_cargado:
            print("  [AVISO] Recuperador no cargado. Ejecuta .cargar() primero.")
            return []

        k = k or TOP_K

        emb_consulta = embed_consulta(consulta)
        if emb_consulta is None:
            return []

        emb_consulta_norm = emb_consulta / (np.linalg.norm(emb_consulta) + 1e-10)
        emb_chunks_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )
        scores = np.dot(emb_chunks_norm, emb_consulta_norm)
        indices_top = np.argsort(scores)[::-1][:k]

        resultados = []
        for idx in indices_top:
            score = float(scores[idx])
            if score < 0.3:
                continue
            chunk = self.chunks[idx]
            resultados.append({
                "texto": chunk["texto"],
                "fuente": chunk.get("fuente", "desconocido"),
                "pagina": chunk.get("pagina", 0),
                "score": round(score, 4),
            })
        return resultados

    def recuperar_y_formatear(self, consulta: str, k: int = None) -> Tuple[str, List[Dict]]:
        """Recupera contexto y lo formatea como texto para el prompt."""
        resultados = self.recuperar(consulta, k=k)
        if not resultados:
            return "", []

        partes = []
        for i, r in enumerate(resultados, 1):
            fuente = r["fuente"]
            pag = r["pagina"]
            partes.append(
                f"[Documento {i}: {fuente} (pag. {pag}) - "
                f"relevancia: {r['score']:.2%}]\n{r['texto']}"
            )

        contexto = "\n\n---\n\n".join(partes)
        return contexto, resultados


# ── Funcion helper ──────────────────────────────────────────────
def recuperar_contexto(consulta: str) -> Tuple[str, List[Dict]]:
    """Helper: carga cache, recupera contexto, lo devuelve formateado."""
    rec = RecuperadorLocal()
    if not rec.cargar():
        return "", []
    return rec.recuperar_y_formatear(consulta)
