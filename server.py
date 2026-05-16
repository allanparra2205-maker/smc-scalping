"""
server.py — SMC Scalping API v1.0
Despliega en Render. Recibe velas desde MT5, corre el engine SMC,
valida la señal con IA (Groq) y retorna la decisión final.
"""

import os
import json
import logging
from typing import List, Optional, Tuple
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

def _describir_vela_srv(c: dict, idx: int) -> str:
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
    cuerpo    = abs(cl - o)
    rng       = max(h - l, 1e-9)
    meca_sup  = h - max(o, cl)
    meca_inf  = min(o, cl) - l
    tipo      = "ALCISTA" if cl > o else "BAJISTA" if cl < o else "DOJI"
    fuerza    = "FUERTE" if cuerpo / rng > 0.6 else "DÉBIL"
    mechas    = ""
    if meca_sup > cuerpo * 1.5:
        mechas = " [MECHA_SUP_LARGA]"
    elif meca_inf > cuerpo * 1.5:
        mechas = " [MECHA_INF_LARGA]"
    return f"  V{idx:02d}: {tipo} {fuerza} O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{cl:.2f}{mechas}"


def _patron_srv(candles: list) -> str:
    if len(candles) < 3:
        return "sin datos"
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if all(c["close"] > c["open"] for c in [c1, c2, c3]):
        return "3 velas alcistas — impulso comprador"
    if all(c["close"] < c["open"] for c in [c1, c2, c3]):
        return "3 velas bajistas — impulso vendedor"
    rng  = max(c3["high"] - c3["low"], 1e-9)
    body = abs(c3["close"] - c3["open"])
    ms   = c3["high"] - max(c3["close"], c3["open"])
    mi   = min(c3["close"], c3["open"]) - c3["low"]
    if ms > body * 1.5 and ms / rng > 0.45:
        return "pin bar bajista — rechazo de altos"
    if mi > body * 1.5 and mi / rng > 0.45:
        return "pin bar alcista — rechazo de bajos"
    if body / rng < 0.2:
        return "doji — indecisión"
    return "sin patrón claro"


def build_groq_prompt(result: smc.SMCAnalysis, symbol: str, candles: list = None) -> str:
    ob_info = "ninguno"
    if result.nearest_ob:
        ob = result.nearest_ob
        ob_info = f"{ob.direction} {ob.bottom:.2f}-{ob.top:.2f} [{ob.mitigation_state}] fuerza:{ob.strength:.2f}"

    fvg_info = "ninguno"
    if result.nearest_fvg:
        fvg = result.nearest_fvg
        fvg_info = f"{fvg.direction} {fvg.bottom:.2f}-{fvg.top:.2f} [{fvg.fill_state}] size:{fvg.size:.2f}"

    confluencias_str = "\n".join(f"  - {c}" for c in result.confluencias) or "  (ninguna)"

    # Velas reales si están disponibles
    velas_txt = "no disponibles"
    patron_txt = "no disponible"
    if candles and len(candles) >= 3:
        recientes = candles[-12:]
        velas_txt = "\n".join(_describir_vela_srv(c, i+1) for i, c in enumerate(recientes))
        patron_txt = _patron_srv(candles)

    return f"""Eres un trader institucional experto en {symbol} {result.timeframe} con enfoque SMC.
Analiza las velas reales y el contexto SMC para decidir si esta señal es válida.

════ VELAS RECIENTES (más antigua → más reciente) ════
{velas_txt}

Patrón en últimas 3 velas: {patron_txt}

════ CONTEXTO SMC ════
Dirección   : {result.direction}
Tendencia   : {result.trend}
Zona        : {result.zone} ({result.zone_pct:.0f}%)
BOS/CHoCH   : {result.last_bos} / {result.last_choch}
Sweep       : {result.liquidity_swept}
Momentum    : {result.momentum_dir} ({result.momentum_score:.0%})
OB          : {ob_info}
FVG         : {fvg_info}
Score SMC   : {result.score}/10
ATR         : {result.atr}

════ NIVELES ════
Entry:{result.entry} SL:{result.sl} TP1:{result.tp1}(RR{result.rr1}) TP2:{result.tp2}(RR{result.rr2})

════ CONFLUENCIAS ({len(result.confluencias)}) ════
{confluencias_str}

════ RAZONA Y DECIDE ════
1. ¿Las velas muestran rechazo real en la dirección propuesta?
2. ¿El patrón confirma o contradice la señal?
3. ¿El precio está en zona institucional válida o ya la superó?
4. ¿Momentum apoya la entrada?
5. ¿El SL tiene sentido?

DESCARTAR si: últimas 3 velas van en dirección contraria sin rechazo, precio superó la zona, doji sin confirmación, momentum fuertemente contrario sin sweep, score <= 5 sin confluencias de peso.

Responde SOLO con JSON válido:
{{"score": 1-10, "comment": "razonamiento breve en 1 línea"}}

JSON:"""


# ═══════════════════════════════════════════════════════════════
# LLAMADA A GROQ
# ═══════════════════════════════════════════════════════════════

async def call_groq(prompt: str) -> Tuple[int, str]:
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
        prompt = build_groq_prompt(result, req.symbol, candles_raw)
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
