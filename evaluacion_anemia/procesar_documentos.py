"""
procesar_documentos.py - Procesa PDFs -> chunks -> embeddings
===============================================================
Lee los PDFs de normas MINSA/OMS, extrae texto, divide en chunks,
genera embeddings con Gemini Embedding 1 (batching) y los cachea en disco.
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple
from google import genai

from config import (
    GEMINI_API_KEY,
    DOCUMENTS_DIR,
    CACHE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)
from rate_limiter import get_handler_embedding, RPDAgotadoError


# ── Cliente Gemini (singleton) ─────────────────────────────────
_cliente = None


def get_cliente():
    global _cliente
    if _cliente is None:
        _cliente = genai.Client(api_key=GEMINI_API_KEY)
    return _cliente


# ── Lectura de PDFs ─────────────────────────────────────────────
def obtener_pdfs(directorio: str) -> List[str]:
    """Lista todos los PDFs en el directorio de documentos."""
    if not os.path.isdir(directorio):
        print(f"  [AVISO] Directorio no encontrado: {directorio}")
        return []
    pdfs = [
        os.path.join(directorio, f)
        for f in sorted(os.listdir(directorio))
        if f.lower().endswith(".pdf")
    ]
    return pdfs


def extraer_texto_pdf(ruta_pdf: str) -> List[Dict]:
    """Extrae texto pagina por pagina de un PDF."""
    import fitz  # PyMuPDF
    paginas = []
    try:
        doc = fitz.open(ruta_pdf)
        nombre_archivo = os.path.basename(ruta_pdf)
        for num_pag in range(len(doc)):
            pagina = doc[num_pag]
            texto = pagina.get_text("text").strip()
            if texto:
                paginas.append({
                    "fuente": nombre_archivo,
                    "pagina": num_pag + 1,
                    "texto": texto,
                })
        doc.close()
    except Exception as e:
        print(f"  [ERROR] Al leer {ruta_pdf}: {e}")
    return paginas


# ── Chunking ────────────────────────────────────────────────────
def chunkear_texto(
    paginas: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """Divide paginas en chunks con solapamiento."""
    chunks = []
    buffer = ""
    buffer_meta = None

    for pag in paginas:
        texto = pag["texto"]
        meta = {"fuente": pag["fuente"], "pagina": pag["pagina"]}

        if buffer:
            corte = max(
                texto[:overlap].rfind(". "),
                texto[:overlap].rfind("\n\n"),
                texto[:overlap].rfind(".\n"),
                0,
            )
            if corte > 0:
                solapamiento = texto[: corte + 1]
                resto = texto[corte + 1:]
            else:
                solapamiento = ""
                resto = texto

            texto_completo = buffer + " " + solapamiento
            if len(texto_completo) > chunk_size * 0.5:
                chunks.append({
                    "texto": texto_completo.strip(),
                    **buffer_meta,
                })
            buffer = resto
            buffer_meta = meta
        else:
            buffer = texto
            buffer_meta = meta

        while len(buffer) > chunk_size:
            corte = buffer[:chunk_size].rfind(". ")
            if corte < chunk_size * 0.3:
                corte = buffer[:chunk_size].rfind(" ")
            if corte < 20:
                corte = chunk_size

            chunk_texto = buffer[: corte + 1].strip()
            if chunk_texto:
                chunks.append({"texto": chunk_texto, **meta})
            buffer = buffer[corte + 1:]

    if buffer.strip():
        chunks.append({"texto": buffer.strip(), **buffer_meta})

    return chunks


# ── Embeddings con Batching ─────────────────────────────────────
def generar_embedding_batch(textos: List[str]) -> List[Optional[np.ndarray]]:
    """Genera embeddings para MULTIPLES textos en UNA sola llamada API."""
    if not textos:
        return []
    try:
        client = get_cliente()
        # Truncar cada texto
        textos_trunc = [t[:2000] for t in textos]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=textos_trunc,
        )
        return [
            np.array(emb.values, dtype=np.float32)
            for emb in result.embeddings
        ]
    except Exception as e:
        print(f"  [ERROR] Generando embedding batch: {e}")
        return [None] * len(textos)


def generar_embeddings(
    chunks: List[Dict], batch_size: int = 1
) -> Tuple[List[Dict], np.ndarray]:
    """Genera embeddings para todos los chunks con batching.

    Args:
        chunks: Lista de chunks con key 'texto'.
        batch_size: Textos por llamada API (max recomendado: 20).

    Returns:
        (chunks_validos, embeddings_array)
    """
    handler = get_handler_embedding()
    total = len(chunks)
    chunks_validos = []
    embeddings_lista = []

    for i in range(0, total, batch_size):
        lote = chunks[i: i + batch_size]
        textos = [c["texto"][:2000] for c in lote]

        print(f"  Embeddings lote {i // batch_size + 1}/{(total - 1) // batch_size + 1} "
              f"({i + 1}-{min(i + batch_size, total)}/{total})")

        # Llamada con rate limiting
        try:
            resultado = handler.call(
                generar_embedding_batch, textos,
                desc=f"embeddings lote {i // batch_size + 1}"
            )
        except RPDAgotadoError:
            print("  [RPD] Embeddings pausado por limite diario.")
            print(f"  Procesados {len(chunks_validos)}/{total} chunks antes de pausa.")
            return chunks_validos, np.array(embeddings_lista) if embeddings_lista else np.array([])

        if resultado:
            for chunk, emb in zip(lote, resultado):
                if emb is not None:
                    chunks_validos.append(chunk)
                    embeddings_lista.append(emb)

    if not chunks_validos:
        print("  [ERROR] No se generaron embeddings validos.")
        return [], np.array([])

    embeddings_array = np.array(embeddings_lista)
    print(f"  OK {len(chunks_validos)} chunks con embeddings "
          f"(dimension: {embeddings_array.shape[1]})")
    return chunks_validos, embeddings_array


# ── Cache ───────────────────────────────────────────────────────
def _hash_documentos(directorio: str) -> str:
    """Genera un hash del contenido de los PDFs."""
    pdfs = obtener_pdfs(directorio)
    hasher = hashlib.md5()
    for pdf in sorted(pdfs):
        hasher.update(pdf.encode())
        try:
            with open(pdf, "rb") as f:
                hasher.update(f.read(1024 * 100))
        except Exception:
            pass
    return hasher.hexdigest()[:12]


def cache_existe() -> bool:
    """Verifica si hay cache valido."""
    return (
        os.path.exists(os.path.join(CACHE_DIR, "chunks.json"))
        and os.path.exists(os.path.join(CACHE_DIR, "embeddings.npy"))
        and os.path.exists(os.path.join(CACHE_DIR, "metadata.json"))
    )


def cargar_cache() -> Tuple[List[Dict], np.ndarray, Dict]:
    """Carga chunks, embeddings y metadata desde cache."""
    with open(os.path.join(CACHE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(os.path.join(CACHE_DIR, "embeddings.npy"))
    with open(os.path.join(CACHE_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Cache cargada: {len(chunks)} chunks, {len(embeddings)} embeddings")
    return chunks, embeddings, metadata


def guardar_cache(chunks, embeddings, metadata):
    """Guarda chunks, embeddings y metadata en cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(CACHE_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(CACHE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Cache guardada: {len(chunks)} chunks en {CACHE_DIR}")


# ── Pipeline principal ──────────────────────────────────────────
def procesar_documentos(forzar_reprocesar: bool = False):
    """Pipeline completo: detecta PDFs -> extrae -> chunkea -> embeddings -> cachea."""
    print("=" * 60)
    print("PROCESAMIENTO DE DOCUMENTOS")
    print("=" * 60)

    if not GEMINI_API_KEY or GEMINI_API_KEY == "AQUI_TU_API_KEY":
        print("[ERROR] Configura GEMINI_API_KEY en config.py")
        return None, None, None

    if not os.path.isdir(DOCUMENTS_DIR):
        print(f"  [AVISO] Directorio de documentos no encontrado: {DOCUMENTS_DIR}")
        return None, None, None

    if not forzar_reprocesar and cache_existe():
        try:
            metadata_cached = json.load(
                open(os.path.join(CACHE_DIR, "metadata.json"), "r", encoding="utf-8")
            )
            print(f"  Cache encontrada ({metadata_cached.get('hash', '?')}).")
            print("  Usar --forzar para reprocesar documentos.")
            return cargar_cache()
        except Exception as e:
            print(f"  [ERROR] Al leer cache: {e}. Reprocesando...")

    pdfs = obtener_pdfs(DOCUMENTS_DIR)
    if not pdfs:
        print(f"  [AVISO] No se encontraron PDFs en: {DOCUMENTS_DIR}")
        return None, None, None

    print(f"  PDFs encontrados: {len(pdfs)}")
    for p in pdfs:
        print(f"    - {os.path.basename(p)}")

    todas_paginas = []
    for pdf in pdfs:
        print(f"\n  Leyendo: {os.path.basename(pdf)}")
        paginas = extraer_texto_pdf(pdf)
        print(f"    -> {len(paginas)} paginas con texto")
        todas_paginas.extend(paginas)

    if not todas_paginas:
        print("  [AVISO] No se pudo extraer texto de ningun PDF.")
        return None, None, None

    print(f"\n  Chunking ({CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP})...")
    chunks = chunkear_texto(todas_paginas)
    print(f"    -> {len(chunks)} chunks generados")

    print(f"\n  Generando embeddings (modelo: {EMBEDDING_MODEL}, batching 20)...")
    print(f"    {len(chunks)} chunks en ~{(len(chunks) + 19) // 20} lotes...")
    chunks_validos, embeddings_array = generar_embeddings(chunks)

    if len(chunks_validos) == 0:
        print("  [ERROR] No se generaron embeddings validos.")
        return None, None, None

    doc_hash = _hash_documentos(DOCUMENTS_DIR)
    metadata = {
        "hash": doc_hash,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "total_chunks": len(chunks_validos),
        "fuentes": list(set(c["fuente"] for c in chunks_validos)),
    }
    guardar_cache(chunks_validos, embeddings_array, metadata)
    return chunks_validos, embeddings_array, metadata


if __name__ == "__main__":
    import sys
    forzar = "--forzar" in sys.argv
    procesar_documentos(forzar_reprocesar=forzar)
