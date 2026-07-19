"""
rate_limiter.py - Control de tasa para API Gemini
===================================================
Soporta rate limiters independientes para cada modelo:
  - Generacion (gemini-3.1-flash-lite):  ~14 RPM,  480 RPD
  - Evaluacion (gemma-4-31b):            ~14 RPM, 1480 RPD
  - Embedding (gemini-embedding-001):    ~90 RPM,  990 RPD

Ademas incluye un contador RPD persistente que evita exceder
la cuota diaria del modelo, deteniendo el pipeline si es necesario.
"""

import time
import json
import os
import threading
from datetime import date
from typing import Optional, Dict

from config import (
    RPM_GENERACION,
    RPD_GENERACION,
    RPM_EVALUACION,
    RPD_EVALUACION,
    RATE_LIMIT_WINDOW_SECONDS,
    DELAY_ENTRE_GENERACION,
    DELAY_ENTRE_EVALUACION,
    DELAY_ENTRE_EMBEDDINGS,
    DELAY_ENTRE_GROUPS,
    GROUP_SIZE,
    RETRY_DELAY,
    MAX_RETRIES,
    MAX_RETRY_BACKOFF,
    RPD_COUNTER_PATH,
)


# ====================================================================
# RPD Counter Persistente
# ====================================================================

class RPDCounter:
    """Contador de Requests Por Dia, persistente en disco.

    Al llegar al limite, mark_exhausted() evita mas llamadas.
    Se resetea automaticamente cada dia.
    """

    def __init__(self, name: str, daily_limit: int, path: str = RPD_COUNTER_PATH):
        self.name = name
        self.daily_limit = daily_limit
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> Dict:
        """Carga el contador desde disco."""
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Reset si es otro dia
                if data.get("date") != str(date.today()):
                    data = self._fresh()
            else:
                data = self._fresh()
        except Exception:
            data = self._fresh()
        return data

    def _fresh(self) -> Dict:
        return {
            "date": str(date.today()),
            "counters": {},
        }

    def _save(self):
        """Guarda el contador a disco."""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except Exception:
            pass

    def increment(self) -> int:
        """Incrementa el contador y retorna el nuevo total."""
        with self._lock:
            # Verificar si es nuevo dia
            if self._data.get("date") != str(date.today()):
                self._data = self._fresh()

            counter = self._data["counters"].get(self.name, 0)
            counter += 1
            self._data["counters"][self.name] = counter
            self._save()
            return counter

    @property
    def count(self) -> int:
        with self._lock:
            if self._data.get("date") != str(date.today()):
                self._data = self._fresh()
                self._save()
            return self._data["counters"].get(self.name, 0)

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self.count)

    @property
    def is_exhausted(self) -> bool:
        return self.count >= self.daily_limit

    def reset(self):
        with self._lock:
            self._data = self._fresh()
            self._save()


# ====================================================================
# Rate Limiter (token bucket)
# ====================================================================

class RateLimiter:
    """Token bucket rate limiter.

    Uso:
        limiter = RateLimiter(max_calls=14, window=60)
        limiter.wait()  # Bloquea hasta slot disponible
    """

    def __init__(self, max_calls: int, window: int = 60,
                 delay_between: float = 4.3,
                 rpd_counter: Optional[RPDCounter] = None):
        self.max_calls = max_calls
        self.window = window
        self.delay_between = delay_between
        self.rpd_counter = rpd_counter
        self._timestamps: list[float] = []
        self._lock = threading.Lock()
        self._call_count = 0

    def wait(self):
        """Espera hasta slot disponible. Si RPD agotado, levanta excepcion."""
        # Verificar RPD primero
        if self.rpd_counter and self.rpd_counter.is_exhausted:
            raise RPDAgotadoError(
                f"RPD agotado para {self.rpd_counter.name}: "
                f"{self.rpd_counter.count}/{self.rpd_counter.daily_limit}"
            )

        with self._lock:
            now = time.time()
            cutoff = now - self.window
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) >= self.max_calls:
                sleep_time = self._timestamps[0] + self.window - now + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self._timestamps.append(time.time())
            self._call_count += 1

        # Pausa estandar entre llamadas
        if self.delay_between > 0:
            time.sleep(self.delay_between)

        # Pausa entre grupos
        if self._call_count > 0 and self._call_count % GROUP_SIZE == 0:
            time.sleep(DELAY_ENTRE_GROUPS)

        # Incrementar contador RPD
        if self.rpd_counter:
            self.rpd_counter.increment()

    @property
    def total_calls(self) -> int:
        return self._call_count


# ====================================================================
# Error personalizado
# ====================================================================

class RPDAgotadoError(Exception):
    """Se lanza cuando se agota la cuota diaria del modelo."""
    pass


# ====================================================================
# Handler con retry + backoff
# ====================================================================

class APICallHandler:
    """Manejador de llamadas API con rate limiting y retry con backoff."""

    def __init__(self, rate_limiter: RateLimiter):
        self.limiter = rate_limiter

    def call(self, fn, *args, desc: str = "", **kwargs):
        """Ejecuta una funcion con rate limiting y reintentos."""
        for intento in range(1, MAX_RETRIES + 1):
            try:
                self.limiter.wait()
                return fn(*args, **kwargs)
            except RPDAgotadoError:
                # RPD agotado - propagar hacia arriba para que main.py maneje
                raise
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    espera = min(RETRY_DELAY * (2 ** intento), MAX_RETRY_BACKOFF)
                    if intento == MAX_RETRIES:
                        print(f"   [ERROR] Cuota agotada tras {MAX_RETRIES} intentos ({desc}): {e}")
                        return None
                    print(f"   [RATE] Cuota excedida (intento {intento}/{MAX_RETRIES}, espera {espera}s): {desc[:50]}")
                    time.sleep(espera)
                else:
                    if intento == MAX_RETRIES:
                        print(f"   [ERROR] Fallo tras {MAX_RETRIES} intentos ({desc}): {e}")
                        return None
                    if "404" in error_str or "not found" in error_str:
                        print(f"   [ERROR] Recurso no encontrado ({desc}): {e}")
                        return None
                    espera = min(RETRY_DELAY * (2 ** intento), MAX_RETRY_BACKOFF)
                    print(f"   [RETRY] Intento {intento}/{MAX_RETRIES} ({desc[:50]}), espera {espera}s: {e}")
                    time.sleep(espera)
        return None


# ====================================================================
# Singletons globales
# ====================================================================

_rpd_gen: Optional[RPDCounter] = None
_rpd_eval: Optional[RPDCounter] = None
_rpd_emb: Optional[RPDCounter] = None
_limiter_gen: Optional[RateLimiter] = None
_limiter_eval: Optional[RateLimiter] = None
_limiter_emb: Optional[RateLimiter] = None
_limiter_emb_query: Optional[RateLimiter] = None
_handler_gen: Optional[APICallHandler] = None
_handler_eval: Optional[APICallHandler] = None
_handler_emb: Optional[APICallHandler] = None
_handler_emb_query: Optional[APICallHandler] = None


def _get_rpd_gen() -> RPDCounter:
    global _rpd_gen
    if _rpd_gen is None:
        _rpd_gen = RPDCounter("generacion", RPD_GENERACION)
    return _rpd_gen


def _get_rpd_eval() -> RPDCounter:
    global _rpd_eval
    if _rpd_eval is None:
        _rpd_eval = RPDCounter("evaluacion", RPD_EVALUACION)
    return _rpd_eval


def _get_rpd_emb() -> RPDCounter:
    global _rpd_emb
    if _rpd_emb is None:
        _rpd_emb = RPDCounter("embedding", 990)  # 990 de 1000
    return _rpd_emb


def get_limiter_generacion() -> RateLimiter:
    global _limiter_gen
    if _limiter_gen is None:
        _limiter_gen = RateLimiter(
            max_calls=RPM_GENERACION,
            window=RATE_LIMIT_WINDOW_SECONDS,
            delay_between=DELAY_ENTRE_GENERACION,
            rpd_counter=_get_rpd_gen(),
        )
    return _limiter_gen


def get_limiter_evaluacion() -> RateLimiter:
    global _limiter_eval
    if _limiter_eval is None:
        _limiter_eval = RateLimiter(
            max_calls=RPM_EVALUACION,
            window=RATE_LIMIT_WINDOW_SECONDS,
            delay_between=DELAY_ENTRE_EVALUACION,
            rpd_counter=_get_rpd_eval(),
        )
    return _limiter_eval


def get_limiter_embedding() -> RateLimiter:
    global _limiter_emb
    if _limiter_emb is None:
        _limiter_emb = RateLimiter(
            max_calls=90,
            window=RATE_LIMIT_WINDOW_SECONDS,
            delay_between=DELAY_ENTRE_EMBEDDINGS,
            rpd_counter=_get_rpd_emb(),
        )
    return _limiter_emb


def get_limiter_embed_query() -> RateLimiter:
    global _limiter_emb_query
    if _limiter_emb_query is None:
        _limiter_emb_query = RateLimiter(
            max_calls=30,
            window=RATE_LIMIT_WINDOW_SECONDS,
            delay_between=2.0,
            rpd_counter=_get_rpd_emb(),
        )
    return _limiter_emb_query


def get_handler_generacion() -> APICallHandler:
    global _handler_gen
    if _handler_gen is None:
        _handler_gen = APICallHandler(get_limiter_generacion())
    return _handler_gen


def get_handler_evaluacion() -> APICallHandler:
    global _handler_eval
    if _handler_eval is None:
        _handler_eval = APICallHandler(get_limiter_evaluacion())
    return _handler_eval


def get_handler_embedding() -> APICallHandler:
    global _handler_emb
    if _handler_emb is None:
        _handler_emb = APICallHandler(get_limiter_embedding())
    return _handler_emb


def get_handler_embed_query() -> APICallHandler:
    global _handler_emb_query
    if _handler_emb_query is None:
        _handler_emb_query = APICallHandler(get_limiter_embed_query())
    return _handler_emb_query


def rpd_status() -> Dict[str, Dict]:
    """Retorna estado de todos los contadores RPD para mostrar al usuario."""
    return {
        "generacion": {
            "modelo": "gemini-3.1-flash-lite",
            "usadas": _get_rpd_gen().count,
            "limite": RPD_GENERACION,
            "restantes": _get_rpd_gen().remaining,
            "agotado": _get_rpd_gen().is_exhausted,
        },
        "evaluacion": {
            "modelo": "gemma-4-31b-it",
            "usadas": _get_rpd_eval().count,
            "limite": RPD_EVALUACION,
            "restantes": _get_rpd_eval().remaining,
            "agotado": _get_rpd_eval().is_exhausted,
        },
        "embedding": {
            "modelo": "gemini-embedding-001",
            "usadas": _get_rpd_emb().count,
            "limite": 990,
            "restantes": _get_rpd_emb().remaining,
            "agotado": _get_rpd_emb().is_exhausted,
        },
    }
