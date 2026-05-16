"""
server.py — SMC Scalping API v1.0
Despliega en Render. Recibe velas desde MT5, corre el engine SMC,
valida la señal con IA (Groq) y retorna la decisión final.
"""

import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx

import smc_scalping_engine as smc

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL    = "llama3-70b-8192"
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
API_SECRET    = os.environ.get("API_SECRET", "")  # Header de seguridad opcional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("server")

app = FastAPI(title="SMC Scalping API", version="1.0")


# ═══════════════════════════════════════════════════════════════
# MODELOS
# ═══════════════════════════════════════════════════════════════

class CandleIn(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: str                 # "M1" o "M5"
    candles: List[CandleIn]
    current_price: float
    spread: float = 0.0
    session: str = ""


class SignalOut(BaseModel):
    symbol: str
    timeframe: str
    valid: bool
    direction: str
    score_smc: int
    score_ai: int                  # 1-10 dado por Groq
    ai_comment: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr1: float
    rr2: float
    atr: float
    trend: str
    zone: str
    zone_pct: float
    liquidity_swept: str
    momentum_dir: str
    momentum_score: float
    confluencias: List[str]
    reason: str


# ═══════════════════════════════════════════════════════════════
# PROMPT GROQ
# ═══════════════════════════════════════════════════════════════

def build_groq_prompt(result: smc.SMCAnalysis, symbol: str) -> str:
    ob_info = "ninguno"
    if result.nearest_ob:
        ob = result.nearest_ob
        ob_info = f"{ob.direction} [{ob.mitigation_state}] {ob.bottom:.5f}-{ob.top:.5f} strength={ob.strength:.2f}"

    fvg_info = "ninguno"
    if result.nearest_fvg:
        fvg = result.nearest_fvg
        fvg_info = f"{fvg.direction} [{fvg.fill_state}] {fvg.bottom:.5f}-{fvg.top:.5f} size={fvg.size:.5f}"

    patron = result.candle_pattern.kind if result.candle_pattern else "ninguno"
    patron_str = f"{patron} (fuerza {result.candle_pattern.strength:.0%})" if result.candle_pattern else "ninguno"

    confluencias_str = "\n".join(f"  - {c}" for c in result.confluencias) or "  (ninguna)"

    return f"""Eres un trader institucional experto en SMC (Smart Money Concepts) y scalping.
Analiza esta señal y da tu veredicto.

SÍMBOLO: {symbol}
TEMPORALIDAD: {result.timeframe}
DIRECCIÓN PROPUESTA: {result.direction}

ESTRUCTURA:
  Tendencia interna : {result.trend}
  BOS               : {result.last_bos}
  CHoCH             : {result.last_choch}
  Zona              : {result.zone} ({result.zone_pct:.0f}%)

ZONAS INSTITUCIONALES:
  Order Block : {ob_info}
  FVG         : {fvg_info}

LIQUIDEZ:
  BSL         : {result.buyside_liquidity}
  SSL         : {result.sellside_liquidity}
  Swept       : {result.liquidity_swept}

MOMENTUM:
  Dirección   : {result.momentum_dir}
  Fuerza      : {result.momentum_score:.0%}

PATRÓN DE VELA: {patron_str}
ENTRY TRIGGER : {"CONFIRMADO" if result.entry_triggered else "NO CONFIRMADO"}
ATR           : {result.atr}

NIVELES:
  Entry : {result.entry}
  SL    : {result.sl}
  TP1   : {result.tp1}  (RR {result.rr1:.1f})
  TP2   : {result.tp2}  (RR {result.rr2:.1f})

CONFLUENCIAS SMC ({len(result.confluencias)}):
{confluencias_str}

SCORE SMC: {result.score}/10

INSTRUCCIONES:
Responde ÚNICAMENTE con un JSON válido, sin texto extra, sin markdown, sin explicaciones fuera del JSON.
El JSON debe tener exactamente estas dos claves:
  "score": número entero del 1 al 10
  "comment": string de máximo 2 líneas con tu veredicto

Criterios para el score:
  8-10 : señal institucional clara, confluencias sólidas, buen RR, momentum alineado
  5-7  : señal decente pero con alguna debilidad
  3-4  : señal débil, pocas confluencias o RR marginal
  1-2  : no operar, condiciones pobres o contradictorias

JSON:"""


# ═══════════════════════════════════════════════════════════════
# LLAMADA A GROQ
# ═══════════════════════════════════════════════════════════════

async def call_groq(prompt: str) -> tuple[int, str]:
    """
    Llama a Groq y retorna (score_ai, comment).
    Si falla, retorna (0, "IA no disponible").
    """
    if not GROQ_API_KEY:
        return 0, "GROQ_API_KEY no configurada"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(GROQ_URL, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Limpiar posibles backticks
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        score = int(data.get("score", 0))
        comment = str(data.get("comment", ""))
        score = max(1, min(10, score))
        return score, comment

    except json.JSONDecodeError as e:
        log.warning(f"Groq JSON parse error: {e} | raw: {raw[:200]}")
        return 0, "Error al parsear respuesta IA"
    except Exception as e:
        log.warning(f"Groq error: {e}")
        return 0, f"Error IA: {str(e)[:80]}"


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}


@app.post("/analyze", response_model=SignalOut)
async def analyze(
    req: AnalyzeRequest,
    x_api_secret: Optional[str] = Header(default=None),
):
    # Seguridad opcional
    if API_SECRET and x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not req.candles:
        raise HTTPException(status_code=400, detail="Sin velas")

    log.info(f"Analizando {req.symbol} [{req.timeframe}] — {len(req.candles)} velas | precio={req.current_price}")

    # ── SMC Engine ───────────────────────────────────────────────
    candles_raw = [c.dict() for c in req.candles]

    result = smc.analyze(
        candles_raw=candles_raw,
        current_price=req.current_price,
        timeframe=req.timeframe,
        spread=req.spread,
        session=req.session,
    )

    log.info(
        f"[{req.timeframe}] SMC → Dir:{result.direction} Score:{result.score} "
        f"Valid:{result.valid} | {result.reason or 'OK'}"
    )

    # ── Groq AI ──────────────────────────────────────────────────
    score_ai = 0
    ai_comment = ""

    # Solo llamar a Groq si hay señal con dirección real y score mínimo
    if result.direction != "NEUTRAL" and result.score >= 4:
        prompt = build_groq_prompt(result, req.symbol)
        score_ai, ai_comment = await call_groq(prompt)
        log.info(f"[{req.timeframe}] AI Score: {score_ai} | {ai_comment[:80]}")

    # ── Decisión final ───────────────────────────────────────────
    # valid final = SMC válido + AI score >= 5 (o AI no disponible y SMC válido)
    final_valid = result.valid and (score_ai >= 5 or score_ai == 0)

    return SignalOut(
        symbol=req.symbol,
        timeframe=req.timeframe,
        valid=final_valid,
        direction=result.direction,
        score_smc=result.score,
        score_ai=score_ai,
        ai_comment=ai_comment,
        entry=result.entry,
        sl=result.sl,
        tp1=result.tp1,
        tp2=result.tp2,
        rr1=result.rr1,
        rr2=result.rr2,
        atr=result.atr,
        trend=result.trend,
        zone=result.zone,
        zone_pct=result.zone_pct,
        liquidity_swept=result.liquidity_swept,
        momentum_dir=result.momentum_dir,
        momentum_score=result.momentum_score,
        confluencias=result.confluencias,
        reason=result.reason or "OK",
    )
