#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nim_probe_models.py
-------------------
Diagnostica NVIDIA NIM: recupera, da DUE fonti indipendenti, i metadati dei
modelli disponibili sull'account, per capire cosa l'API/SDK espone realmente
(id, context_window, parametri di sampling, chat_template_kwargs, ecc.).

PRIVACY: la chiave API viene letta SOLO da variabili d'ambiente.
         Non viene mai stampata, loggata né scritta su file.

Usage:
  python nim_probe_models.py [--out-dir OUTPUT_DIR]

Output files:
  - nim_raw_dump.json        : full GET /v1/models response
  - nim_langchain_dump.json  : model list + all attributes from ChatNVIDIA
  - nim_probe.log            : readable text log
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
import argparse
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent  # extension root

_args_parser = argparse.ArgumentParser(description="Probe NVIDIA NIM models via API and LangChain SDK.")
_args_parser.add_argument("--out-dir", default=str(_ROOT), help="Output directory for dumps (default: extension root)")
_args = _args_parser.parse_args()

OUT_DIR = Path(_args.out_dir)
RAW_OUT = OUT_DIR / "nim_raw_dump.json"
LANG_OUT = OUT_DIR / "nim_langchain_dump.json"
LOG_OUT = OUT_DIR / "nim_probe.log"

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODELS_URL = f"{BASE_URL}/models"
TIMEOUT_SEC = 30

ENV_NAMES = ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
class Tee:
    """Scrive contemporaneamente su console e su file di log (UTF-8)."""

    def __init__(self, log_path: Path) -> None:
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._console = sys.stdout

    def write(self, data: str) -> None:
        try:
            self._console.write(data)
        except Exception:
            pass
        self._file.write(data)

    def flush(self) -> None:
        try:
            self._console.flush()
        except Exception:
            pass
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def get_api_key() -> str:
    for name in ENV_NAMES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    raise SystemExit(
        "Nessuna chiave trovata. Esporta la variabile d'ambiente "
        "NVIDIA_API_KEY (o NVIDIA_NIM_API_KEY) prima di lanciare lo script.\n"
        "  PowerShell:  $env:NVIDIA_API_KEY = \"nvapi-...\"\n"
        "  Bash:        export NVIDIA_API_KEY=\"nvapi-...\""
    )


def _redact(s: str) -> str:
    import re
    return re.sub(r"nvapi-[A-Za-z0-9._-]+", "nvapi-[REDACTED]", s)


def safe_dump(obj: Any, path: Path) -> None:
    def default(o: Any) -> Any:
        d = {}
        # intuitivo attrs (__dict__) se presente
        try:
            d.update(vars(o))
        except Exception:
            pass
        # attributi via dir(), scartando私有 dunder
        for k in dir(o):
            if k.startswith("__"):
                continue
            if k in d:
                continue
            try:
                v = getattr(o, k)
            except Exception:
                continue
            if callable(v):
                continue
            d[k] = v
        return d

    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fonte A — Endpoint API raw (nessuna dipendenza esterna)
# ---------------------------------------------------------------------------
def probe_raw(api_key: str, out: Tee) -> dict:
    out.write("\n" + "=" * 70 + "\n")
    out.write("[A] Endpoint raw: GET /v1/models\n")
    out.write("=" * 70 + "\n")

    # urllib: zero dipendenze
    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = "(corpo non leggibile)"
    except Exception as e:
        out.write(f"[A] ERRORE di rete: {type(e).__name__}: {_redact(str(e))}\n")
        return {"ok": False, "error": _redact(str(e)), "trace": _redact(traceback.format_exc())}

    elapsed = time.time() - t0
    out.write(f"[A] HTTP status: {status}  ({elapsed:.2f}s)\n")

    try:
        parsed = json.loads(body)
    except Exception:
        out.write("[A] Risposta NON JSON (prime 300 char):\n")
        out.write(body[:300] + "\n")
        return {"ok": False, "http_status": status, "raw_body_head": body[:500]}

    data = parsed.get("data") if isinstance(parsed, dict) else parsed
    n = len(data) if isinstance(data, list) else "N/A"
    out.write(f"[A] Modelli in 'data': {n}\n")

    # ispeziona la STRUTTURA completa del primo record (per vedere quali campi
    # espone NVIDIA via API: id, object, owned_by, context_window, ecc.)
    if isinstance(data, list) and data:
        first = data[0]
        out.write("[A] Primo record (chiavi complete):\n")
        out.write(json.dumps(first, indent=2, ensure_ascii=False) + "\n")

        keys: set[str] = set()
        for item in data:
            if isinstance(item, dict):
                keys.update(item.keys())
        out.write(f"[A] Insieme delle chiavi viste in tutti i record: {sorted(keys)}\n")

    safe_dump(parsed, RAW_OUT)
    out.write(f"[A] Dump integro salvato in: {RAW_OUT}\n")
    return {"ok": status == 200, "http_status": status, "count": n, "parsed": parsed}


# ---------------------------------------------------------------------------
# Fonte B — SDK LangChain (richiede langchain_nvidia_ai_endpoints)
# ---------------------------------------------------------------------------
def probe_langchain(api_key: str, out: Tee) -> dict:
    out.write("\n" + "=" * 70 + "\n")
    out.write("[B] SDK: ChatNVIDIA.get_available_models()\n")
    out.write("=" * 70 + "\n")

    try:
        import langchain_nvidia_ai_endpoints as lc
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except Exception as e:
        out.write(
            f"[B] SDK NON installato ({type(e).__name__}). "
            f"Installa con:\n    python -m pip install langchain-nvidia-ai-endpoints\n"
        )
        return {"ok": False, "reason": "sdk_not_installed", "error": _redact(str(e))}

    out.write(f"[B] Versione SDK: {getattr(lc, '__version__', '(sconosciuta)')}\n")

    # L'SDK legge NVIDIA_API_KEY da env; se assente gliela passiamo noi
    os.environ.setdefault("NVIDIA_API_KEY", api_key)
    try:
        client = ChatNVIDIA()
        models = client.get_available_models()
    except Exception as e:
        out.write(f"[B] ERRORE get_available_models: {type(e).__name__}: {_redact(str(e))}\n")
        return {"ok": False, "error": _redact(str(e)), "trace": _redact(traceback.format_exc())}

    models_list = list(models) if not isinstance(models, list) else models
    out.write(f"[B] Modelli restituiti: {len(models_list)}\n")

    # Campi di configurazione che ci interessano (sottoinsieme pulito).
    # Tutto il resto (client, _abc_impl, model_fields*, endpoint, thinking_prefix,
    # model_config, model_extra, ...) viene scartato perch non serializzabile o inutile.
    KEEP_FIELDS = (
        "id", "model_type", "supports_thinking",
        "thinking_param_enable", "thinking_param_disable",
        "supports_tools", "supports_structured_output", "deprecated",
        "aliases", "base_model",
    )

    models_clean: list[dict] = []
    thinking_ids: list[str] = []
    deprecated_ids: list[str] = []
    vlm_ids: list[str] = []
    skipped_on_err = 0

    for m in models_list:
        try:
            mid = getattr(m, "id", None) or getattr(m, "model", None)
            if not mid:
                skipped_on_err += 1
                continue
            mid = str(mid)

            record: dict[str, object] = {"id": mid}
            for k in KEEP_FIELDS:
                if k == "id":
                    continue
                if not hasattr(m, k):
                    continue
                try:
                    v = getattr(m, k)
                except Exception:
                    continue
                record[k] = v

            # tipo modello normalizzato a stringa
            mt = record.get("model_type")
            record["model_type"] = str(mt) if mt is not None else ""

            models_clean.append(record)

            if record.get("supports_thinking"):
                thinking_ids.append(mid)
            if record.get("deprecated"):
                deprecated_ids.append(mid)
            if str(record.get("model_type")).lower() == "vlm":
                vlm_ids.append(mid)
        except Exception:
            skipped_on_err += 1

    out.write(f"[B] Modelli puliti estratti: {len(models_clean)}"

              f"   | thinking: {len(thinking_ids)}"

              f"   | vlm: {len(vlm_ids)}"

              f"   | deprecated: {len(deprecated_ids)}")

    if skipped_on_err:
        out.write(f"[B] Modelli saltati per errore: {skipped_on_err}\n")

    safe_dump(
        {"sdk_version": getattr(lc, "__version__", None),
         "count": len(models_clean), "models": models_clean},
        LANG_OUT,
    )
    out.write(f"[B] Dump pulito salvato in: {LANG_OUT}\n")
    return {"ok": True, "count": len(models_clean), "models": models_clean}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Tee(LOG_OUT)
    sys.stdout = out

    out.write("=" * 70 + "\n")
    out.write("nim_probe_models.py - diagnostica NVIDIA NIM\n")
    out.write(f"Python: {sys.version.splitlines()[0]}\n")
    out.write(f"Out dir: {OUT_DIR}\n")
    out.write("=" * 70 + "\n\n")

    api_key = get_api_key()
    out.write("Chiave API: PRESENTE (valore NON mostrato)\n")
    out.write(f"Lunghezza chiave: {len(api_key)} caratteri\n")
    out.write(f"Prefisso chiave: {api_key[:6]}{'...' if len(api_key) > 6 else ''}\n")

    res_raw = probe_raw(api_key, out)
    res_lc = probe_langchain(api_key, out)

    out.write("\n" + "=" * 70 + "\n")
    out.write("RIEPILOGO\n")
    out.write("=" * 70 + "\n")
    out.write(f"[A] raw  -> ok={res_raw.get('ok')} count={res_raw.get('count')} status={res_raw.get('http_status')}\n")
    out.write(f"[B] SDK  -> ok={res_lc.get('ok')} count={res_lc.get('count')} reason={res_lc.get('reason','')}\n")
    out.write("\nFile generati:\n")
    out.write(f"  - {RAW_OUT.name}:  {'OK' if RAW_OUT.exists() else 'NO'}\n")
    out.write(f"  - {LANG_OUT.name}: {'OK' if LANG_OUT.exists() else 'NO'}\n")
    out.write(f"  - {LOG_OUT.name}:   OK\n")

    out.close()


if __name__ == "__main__":
    main()