"""
SMC Scalping Engine v1.0
Motor SMC mono-temporalidad para scalping en M1 y M5.
Cada temporalidad se analiza de forma completamente independiente.

Correcciones vs v3:
  - Una sola temporalidad por llamada (no multi-TF)
  - Sweep no es obligatorio para señal válida
  - Score recalibrado para scalping
  - Lookbacks dinámicos por TF
  - Filtro de rango más estricto
  - Filtro de spread
  - Momentum como confluencia real
  - Entry trigger: cierre confirmado, no solo "precio cerca"
  - Invalidación de OB en 1 vela fuerte (fast_invalidation)
  - min_gap de FVG relativo al ATR real del TF
  - Razón de descarte detallada por cada condición fallida
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import statistics


# ═══════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Candle:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return max(0.0, self.high - max(self.close, self.open))

    @property
    def lower_wick(self) -> float:
        return max(0.0, min(self.close, self.open) - self.low)

    @property
    def range(self) -> float:
        return max(0.0, self.high - self.low)

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass
class SwingPoint:
    index: int
    price: float
    time: str
    kind: str  # 'SH' | 'SL'


@dataclass
class OrderBlock:
    index: int
    top: float
    bottom: float
    time: str
    direction: str  # 'bullish' | 'bearish'
    strength: float = 0.0
    tested: bool = False
    mitigated: bool = False
    invalidated: bool = False
    mitigation_state: str = "UNTESTED"
    fill_pct: float = 0.0
    touches: int = 0
    source_index: int = 0
    impulse_index: int = 0


@dataclass
class FVG:
    index: int
    top: float
    bottom: float
    time: str
    direction: str  # 'bullish' | 'bearish'
    filled: bool = False
    partial_fill: bool = False
    fill_state: str = "UNFILLED"
    size: float = 0.0
    fill_pct: float = 0.0
    first_touch_index: int = -1
    full_fill_index: int = -1


@dataclass
class CandlePattern:
    kind: str
    strength: float
    index: int


@dataclass
class ScalpingParams:
    """Parámetros dinámicos según la temporalidad."""
    timeframe: str
    swing_lookback: int        # Velas para confirmar swing
    ob_lookback: int           # Velas hacia atrás para buscar OBs
    fvg_lookback: int          # Velas hacia atrás para buscar FVGs
    ranging_atr_mult: float    # ATR * x para detectar rango
    min_candles: int           # Mínimo de velas para operar
    atr_period: int
    ob_strength_min: float     # Fuerza mínima de OB válido
    fvg_gap_atr: float         # Gap mínimo de FVG en múltiplos de ATR
    sweep_lookback: int        # Velas para detectar sweep
    choch_lookback: int        # Velas para CHoCH en retest
    momentum_lookback: int     # Velas para medir momentum
    spread_max_atr: float      # Spread máximo en ATR
    score_threshold: int       # Score mínimo para señal válida
    rr_min: float              # RR mínimo TP1
    zone_proximity_atr: float  # Distancia máxima a zona en ATR


@dataclass
class SMCAnalysis:
    # Identificación
    timeframe: str = ""

    # Estructura interna
    trend: str = "NEUTRAL"
    last_bos: str = "NONE"
    last_choch: str = "NONE"

    # Zona
    zone: str = "EQUILIBRIO"
    zone_pct: float = 50.0

    # Calidad de datos
    parse_errors: int = 0

    # Zonas institucionales
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FVG] = field(default_factory=list)
    nearest_ob: Optional[OrderBlock] = None
    nearest_fvg: Optional[FVG] = None
    candle_pattern: Optional[CandlePattern] = None

    # Liquidez
    equal_highs: List[float] = field(default_factory=list)
    equal_lows: List[float] = field(default_factory=list)
    buyside_liquidity: float = 0.0
    sellside_liquidity: float = 0.0
    liquidity_swept: str = "NONE"

    # Momentum (nuevo en v1.0)
    momentum_score: float = 0.0
    momentum_dir: str = "NEUTRAL"

    # Confluencias
    confluencias: List[str] = field(default_factory=list)
    score: int = 0
    structural_count: int = 0
    zone_count: int = 0
    confirmation_count: int = 0

    # Señal
    direction: str = "NEUTRAL"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    rr1: float = 0.0
    rr2: float = 0.0
    valid: bool = False
    reason: str = ""
    atr: float = 0.0

    # Filtros
    spread_filtered: bool = False
    ranging_market: bool = False
    entry_triggered: bool = False  # Cierre confirmó la entrada


# ═══════════════════════════════════════════════════════════════
# PARÁMETROS POR TEMPORALIDAD
# ═══════════════════════════════════════════════════════════════

def get_params(timeframe: str) -> ScalpingParams:
    """
    Parámetros optimizados para scalping según el TF.
    M1: más agresivo, lookbacks más largos en velas, umbrales más bajos.
    M5: más conservador, señales de más calidad.
    """
    tf = timeframe.upper()

    if tf == "M1":
        return ScalpingParams(
            timeframe="M1",
            swing_lookback=2,
            ob_lookback=80,
            fvg_lookback=60,
            ranging_atr_mult=4.0,
            min_candles=60,
            atr_period=14,
            ob_strength_min=0.28,
            fvg_gap_atr=0.20,
            sweep_lookback=5,
            choch_lookback=8,
            momentum_lookback=5,
            spread_max_atr=0.20,
            score_threshold=5,
            rr_min=1.5,
            zone_proximity_atr=1.0,
        )

    # M5 por defecto
    return ScalpingParams(
        timeframe="M5",
        swing_lookback=3,
        ob_lookback=60,
        fvg_lookback=50,
        ranging_atr_mult=5.0,
        min_candles=40,
        atr_period=14,
        ob_strength_min=0.22,      # era 0.33
        fvg_gap_atr=0.20,          # era 0.25
        sweep_lookback=4,
        choch_lookback=10,
        momentum_lookback=6,
        spread_max_atr=0.28,
        score_threshold=4,         # era 5
        rr_min=1.5,
        zone_proximity_atr=1.8,    # era 1.2
    )


# ═══════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════

def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _mean_volume(candles: List[Candle]) -> float:
    vols = [c.volume for c in candles if c.volume > 0]
    return statistics.mean(vols) if vols else 0.0


def _zone_distance_price(price: float, bottom: float, top: float) -> float:
    lo, hi = min(bottom, top), max(bottom, top)
    if price < lo:
        return lo - price
    if price > hi:
        return price - hi
    return 0.0


# ═══════════════════════════════════════════════════════════════
# PARSEO
# ═══════════════════════════════════════════════════════════════

def parse_candles(raw: list, stats: Optional[dict] = None) -> List[Candle]:
    result: List[Candle] = []
    if stats is None:
        stats = {"errors": 0}
    for c in raw or []:
        try:
            result.append(Candle(
                time=str(c.get("time", "")),
                open=float(c.get("open", 0)),
                high=float(c.get("high", 0)),
                low=float(c.get("low", 0)),
                close=float(c.get("close", 0)),
                volume=int(c.get("volume", 0)),
            ))
        except Exception:
            stats["errors"] = stats.get("errors", 0) + 1
    return result


# ═══════════════════════════════════════════════════════════════
# ATR
# ═══════════════════════════════════════════════════════════════

def calculate_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    window = candles[-(period + 1):] if len(candles) > period else candles
    trs = []
    for i in range(1, len(window)):
        c, prev = window[i], window[i - 1]
        tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
        trs.append(tr)
    return round(statistics.mean(trs), 5) if trs else 0.0


# ═══════════════════════════════════════════════════════════════
# SWINGS
# ═══════════════════════════════════════════════════════════════

def find_swings(
    candles: List[Candle],
    lookback: int = 3,
    min_size_atr: float = 0.0,
) -> List[SwingPoint]:
    """
    Detecta swing highs y lows.
    min_size_atr: filtra swings muy pequeños para reducir ruido en M1.
    """
    swings: List[SwingPoint] = []
    if len(candles) < (lookback * 2) + 1:
        return swings

    for i in range(lookback, len(candles) - lookback):
        c = candles[i]
        eps = max(abs(c.close) * 1e-9, 1e-9)
        future = candles[i + 1: i + lookback + 1]

        left_highs = [candles[i - j].high for j in range(1, lookback + 1)]
        right_highs = [candles[i + j].high for j in range(1, lookback + 1)]
        left_lows = [candles[i - j].low for j in range(1, lookback + 1)]
        right_lows = [candles[i + j].low for j in range(1, lookback + 1)]

        higher = all(c.high >= h - eps for h in left_highs + right_highs)
        lower = all(c.low <= l + eps for l in left_lows + right_lows)
        strict_high = any(c.high > h + eps for h in left_highs + right_highs)
        strict_low = any(c.low < l - eps for l in left_lows + right_lows)

        plateau_high = (
            not strict_high and higher and
            any(abs(c.high - h) <= eps for h in left_highs + right_highs) and
            any(fc.close < fc.open or fc.close < c.close - eps for fc in future)
        )
        plateau_low = (
            not strict_low and lower and
            any(abs(c.low - l) <= eps for l in left_lows + right_lows) and
            any(fc.close > fc.open or fc.close > c.close + eps for fc in future)
        )

        if (higher and strict_high) or plateau_high:
            prev_low = min(candles[i - j].low for j in range(1, lookback + 1))
            if min_size_atr <= 0 or (c.high - prev_low) >= min_size_atr:
                swings.append(SwingPoint(i, c.high, c.time, "SH"))

        if (lower and strict_low) or plateau_low:
            prev_high = max(candles[i - j].high for j in range(1, lookback + 1))
            if min_size_atr <= 0 or (prev_high - c.low) >= min_size_atr:
                swings.append(SwingPoint(i, c.low, c.time, "SL"))

    return swings


def _compress_swings(swings: List[SwingPoint]) -> List[SwingPoint]:
    """Elimina swings consecutivos del mismo tipo conservando el más extremo."""
    ordered = sorted(swings, key=lambda s: s.index)
    compressed: List[SwingPoint] = []
    for s in ordered:
        if not compressed:
            compressed.append(s)
            continue
        last = compressed[-1]
        if s.kind == last.kind:
            if (s.kind == "SH" and s.price >= last.price) or \
               (s.kind == "SL" and s.price <= last.price):
                compressed[-1] = s
        else:
            compressed.append(s)
    return compressed


def _last_structural_leg(swings: List[SwingPoint]) -> Optional[Tuple[float, float, str]]:
    seq = _compress_swings(swings)
    for i in range(len(seq) - 1, 0, -1):
        a, b = seq[i - 1], seq[i]
        if a.kind == b.kind:
            continue
        low = min(a.price, b.price)
        high = max(a.price, b.price)
        hint = "BULLISH" if a.kind == "SL" else "BEARISH"
        return low, high, hint
    return None


# ═══════════════════════════════════════════════════════════════
# TENDENCIA — Una sola temporalidad
# ═══════════════════════════════════════════════════════════════

def classify_trend(
    swings: List[SwingPoint],
    candles: Optional[List[Candle]] = None,
) -> Tuple[str, str, str]:
    """
    Clasifica la tendencia desde los swings de la misma TF.
    Retorna (trend, last_bos, last_choch).
    """
    seq = _compress_swings(swings)
    if len(seq) < 4:
        return "NEUTRAL", "NONE", "NONE"

    tail = seq[-4:]
    kinds = [s.kind for s in tail]
    trend = "NEUTRAL"
    last_bos = "NONE"
    last_choch = "NONE"

    bull = (
        kinds == ["SH", "SL", "SH", "SL"] and
        tail[2].price > tail[0].price and
        tail[3].price > tail[1].price
    )
    bear = (
        kinds == ["SL", "SH", "SL", "SH"] and
        tail[2].price < tail[0].price and
        tail[3].price < tail[1].price
    )

    if bull:
        trend = "BULLISH"
    elif bear:
        trend = "BEARISH"
    else:
        highs = [s for s in seq if s.kind == "SH"]
        lows = [s for s in seq if s.kind == "SL"]
        if len(highs) >= 2 and len(lows) >= 2:
            h1, h2 = highs[-2], highs[-1]
            l1, l2 = lows[-2], lows[-1]
            if h2.price > h1.price and l2.price > l1.price:
                trend = "BULLISH"
            elif h2.price < h1.price and l2.price < l1.price:
                trend = "BEARISH"

    if candles and trend != "NEUTRAL":
        last_close = candles[-1].close
        highs = [s for s in seq if s.kind == "SH"]
        lows = [s for s in seq if s.kind == "SL"]
        if highs and lows:
            lh = highs[-1]
            ll = lows[-1]
            if trend == "BULLISH":
                if last_close > lh.price:
                    last_bos = "BULLISH"
                elif last_close < ll.price:
                    last_choch = "BEARISH"
            elif trend == "BEARISH":
                if last_close < ll.price:
                    last_bos = "BEARISH"
                elif last_close > lh.price:
                    last_choch = "BULLISH"

    return trend, last_bos, last_choch


# ═══════════════════════════════════════════════════════════════
# ZONA PREMIUM / DISCOUNT
# ═══════════════════════════════════════════════════════════════

def calculate_zone(
    candles: List[Candle],
    current_price: float,
    swings: Optional[List[SwingPoint]] = None,
) -> Tuple[str, float]:
    if not candles:
        return "EQUILIBRIO", 50.0

    highest = lowest = None

    if swings:
        leg = _last_structural_leg(swings)
        if leg:
            lowest, highest, _ = leg

    if highest is None or lowest is None:
        highest = max(c.high for c in candles)
        lowest = min(c.low for c in candles)

    rng = highest - lowest
    if rng <= 0:
        return "EQUILIBRIO", 50.0

    pct = ((current_price - lowest) / rng) * 100.0

    if pct > 55.0:
        zone = "PREMIUM"
    elif pct < 45.0:
        zone = "DISCOUNT"
    else:
        zone = "EQUILIBRIO"

    return zone, round(pct, 1)


# ═══════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════

def find_order_blocks(
    candles: List[Candle],
    atr: float,
    lookback: int = 60,
    fast_invalidation: bool = True,
) -> List[OrderBlock]:
    """
    Detecta OBs con displacement real.
    fast_invalidation=True (scalping): invalida con 1 cierre fuerte fuera de la zona.
    """
    obs: List[OrderBlock] = []
    if len(candles) < 4:
        return obs

    window = candles[-lookback:] if len(candles) >= lookback else candles
    base_index = len(candles) - len(window)
    avg_volume = _mean_volume(window)
    min_impulse = max(atr * 0.7, 0.0)
    bos_buffer = max(atr * 0.08, 0.0)
    inv_buffer = max(atr * 0.12, 0.0)

    for i in range(1, len(window) - 2):
        prev = window[i - 1]
        curr = window[i]

        local_start = max(0, i - 3)
        local_high = max(c.high for c in window[local_start:i]) if i > 0 else curr.high
        local_low = min(c.low for c in window[local_start:i]) if i > 0 else curr.low

        body_ratio = curr.body / max(prev.body, 1e-9)
        vol_ratio = (curr.volume / avg_volume) if avg_volume > 0 and curr.volume > 0 else 1.0
        vol_boost = _clamp(0.5 + min(vol_ratio, 3.0) * 0.25, 0.5, 1.25)

        for direction in ("bearish", "bullish"):
            if direction == "bearish":
                displacement = (
                    prev.is_bullish and curr.is_bearish and
                    curr.body >= min_impulse and
                    curr.close < local_low - bos_buffer and
                    curr.body >= prev.body * 0.85
                )
            else:
                displacement = (
                    prev.is_bearish and curr.is_bullish and
                    curr.body >= min_impulse and
                    curr.close > local_high + bos_buffer and
                    curr.body >= prev.body * 0.85
                )

            if not displacement:
                continue

            ob_top = max(prev.high, prev.open, prev.close)
            ob_bottom = min(prev.low, prev.open, prev.close)
            zone_size = max(ob_top - ob_bottom, 1e-9)

            fill_pct = 0.0
            touches = 0
            tested = mitigated = invalidated = False
            mitigation_state = "UNTESTED"
            running_edge = ob_bottom if direction == "bearish" else ob_top
            outside_count = 0

            for k in range(i + 1, len(window)):
                c = window[k]
                intersects = c.high >= ob_bottom and c.low <= ob_top

                if intersects:
                    tested = True
                    touches += 1
                    if direction == "bearish":
                        running_edge = max(running_edge, min(c.high, ob_top))
                        fill_pct = _clamp((running_edge - ob_bottom) / zone_size, 0.0, 1.0)
                    else:
                        running_edge = min(running_edge, max(c.low, ob_bottom))
                        fill_pct = _clamp((ob_top - running_edge) / zone_size, 0.0, 1.0)
                    if fill_pct > 0:
                        mitigated = True
                        mitigation_state = "PARTIAL"

                beyond = (
                    c.close > ob_top + inv_buffer if direction == "bearish"
                    else c.close < ob_bottom - inv_buffer
                )

                if beyond:
                    outside_count += 1
                    # Scalping: 1 vela fuerte cierra fuera = invalidado
                    limit = 1 if (fast_invalidation and c.body >= atr * 0.8) else 2
                    if outside_count >= limit:
                        invalidated = True
                        mitigation_state = "INVALIDATED"
                        break
                else:
                    outside_count = 0

            if fill_pct >= 0.999:
                mitigation_state = "FULL"
                mitigated = True

            if direction == "bearish":
                strength = _clamp(
                    0.35 * _clamp(curr.body / max(atr, 1e-9), 0.0, 2.0) / 2.0 +
                    0.25 * _clamp(body_ratio / 2.0, 0.0, 1.0) +
                    0.20 * _clamp(vol_boost, 0.5, 1.25) +
                    0.20 * _clamp((local_low - curr.close) / max(curr.range, 1e-9), 0.0, 1.0),
                    0.0, 1.0,
                )
            else:
                strength = _clamp(
                    0.35 * _clamp(curr.body / max(atr, 1e-9), 0.0, 2.0) / 2.0 +
                    0.25 * _clamp(body_ratio / 2.0, 0.0, 1.0) +
                    0.20 * _clamp(vol_boost, 0.5, 1.25) +
                    0.20 * _clamp((curr.close - local_high) / max(curr.range, 1e-9), 0.0, 1.0),
                    0.0, 1.0,
                )

            obs.append(OrderBlock(
                index=base_index + i - 1,
                top=round(ob_top, 5),
                bottom=round(ob_bottom, 5),
                time=prev.time,
                direction=direction,
                strength=round(strength, 3),
                tested=tested,
                mitigated=mitigated,
                invalidated=invalidated,
                mitigation_state=mitigation_state,
                fill_pct=round(fill_pct, 3),
                touches=touches,
                source_index=base_index + i - 1,
                impulse_index=base_index + i,
            ))

    obs.sort(key=lambda x: (x.invalidated, x.fill_pct, -x.strength, -x.index))
    return obs[-10:]


# ═══════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ═══════════════════════════════════════════════════════════════

def find_fvgs(
    candles: List[Candle],
    atr: float,
    lookback: int = 50,
    min_gap_atr: float = 0.25,
) -> List[FVG]:
    """
    Detecta FVGs con gap mínimo relativo al ATR del TF.
    Evita micro-gaps irrelevantes en M1.
    """
    fvgs: List[FVG] = []
    if len(candles) < 3:
        return fvgs

    window = candles[-lookback:] if len(candles) >= lookback else candles
    base_index = len(candles) - len(window)
    min_gap = max(atr * min_gap_atr, 0.0)
    bos_buffer = max(atr * 0.08, 0.0)

    for i in range(1, len(window) - 1):
        prev, curr, nxt = window[i - 1], window[i], window[i + 1]

        local_start = max(0, i - 3)
        local_high = max(c.high for c in window[local_start:i]) if i > 0 else curr.high
        local_low = min(c.low for c in window[local_start:i]) if i > 0 else curr.low

        for direction in ("bullish", "bearish"):
            if direction == "bullish":
                gap = nxt.low - prev.high
                condition = (
                    gap > min_gap and curr.is_bullish and
                    curr.close > local_high + bos_buffer and
                    curr.body >= prev.body * 0.75
                )
                if not condition:
                    continue
                bottom, top = prev.high, nxt.low
            else:
                gap = prev.low - nxt.high
                condition = (
                    gap > min_gap and curr.is_bearish and
                    curr.close < local_low - bos_buffer and
                    curr.body >= prev.body * 0.75
                )
                if not condition:
                    continue
                top, bottom = prev.low, nxt.high

            gap_size = max(top - bottom, 1e-9)
            deepest = top if direction == "bullish" else bottom
            first_touch_index = full_fill_index = -1

            for k in range(i + 2, len(window)):
                c = window[k]
                if direction == "bullish":
                    if c.low <= top:
                        if first_touch_index == -1:
                            first_touch_index = base_index + k
                        deepest = min(deepest, c.low)
                        if deepest <= bottom:
                            full_fill_index = base_index + k
                            break
                else:
                    if c.high >= bottom:
                        if first_touch_index == -1:
                            first_touch_index = base_index + k
                        deepest = max(deepest, c.high)
                        if deepest >= top:
                            full_fill_index = base_index + k
                            break

            if direction == "bullish":
                fill_pct = _clamp((top - max(deepest, bottom)) / gap_size, 0.0, 1.0) if deepest < top else 0.0
            else:
                fill_pct = _clamp((min(deepest, top) - bottom) / gap_size, 0.0, 1.0) if deepest > bottom else 0.0

            if fill_pct >= 0.999:
                fill_state, filled, partial_fill = "FULL", True, True
            elif fill_pct > 0:
                fill_state, filled, partial_fill = "PARTIAL", False, True
            else:
                fill_state, filled, partial_fill = "UNFILLED", False, False

            fvgs.append(FVG(
                index=base_index + i,
                top=round(top, 5),
                bottom=round(bottom, 5),
                time=curr.time,
                direction=direction,
                filled=filled,
                partial_fill=partial_fill,
                fill_state=fill_state,
                size=round(gap_size, 5),
                fill_pct=round(fill_pct, 3),
                first_touch_index=first_touch_index,
                full_fill_index=full_fill_index,
            ))

    fvgs.sort(key=lambda x: (x.fill_state == "FULL", x.fill_pct, -x.index))
    return fvgs[-10:]


# ═══════════════════════════════════════════════════════════════
# LIQUIDEZ
# ═══════════════════════════════════════════════════════════════

def find_liquidity(
    candles: List[Candle],
    atr: float,
    swing_lookback: int = 3,
) -> Tuple[List[float], List[float], float, float]:
    if len(candles) < 3:
        return [], [], 0.0, 0.0

    tolerance = max(atr * 0.12, 0.0)
    swings = find_swings(candles, lookback=swing_lookback)
    sh = [s for s in swings if s.kind == "SH"]
    sl = [s for s in swings if s.kind == "SL"]

    def build_clusters(points: List[SwingPoint]) -> Tuple[List[float], float]:
        if not points:
            return [], 0.0
        clusters = []
        used: set = set()
        for i, item in enumerate(points):
            if i in used:
                continue
            group = [item]
            for j in range(i + 1, len(points)):
                if j not in used and abs(item.price - points[j].price) <= tolerance:
                    group.append(points[j])
                    used.add(j)
            if len(group) >= 2:
                center = round(sum(x.price for x in group) / len(group), 5)
                span = max(x.price for x in group) - min(x.price for x in group)
                latest = max(x.index for x in group)
                clusters.append((center, len(group), span, latest))
        clusters.sort(key=lambda x: (x[1], -x[2], x[3]), reverse=True)
        return [c[0] for c in clusters], (clusters[0][0] if clusters else 0.0)

    eq_highs, bsl = build_clusters(sh)
    eq_lows, ssl = build_clusters(sl)
    return eq_highs, eq_lows, round(bsl, 5), round(ssl, 5)


def detect_liquidity_sweep(
    candles: List[Candle],
    bsl: float,
    ssl: float,
    atr: float,
    lookback: int = 5,
) -> str:
    if not candles or (bsl <= 0 and ssl <= 0):
        return "NONE"

    recent = candles[-lookback:]
    if len(recent) < 2:
        return "NONE"

    for idx in range(max(0, len(recent) - 2), len(recent)):
        c = recent[idx]
        prior = recent[:idx]
        mecha_sup = max(0.0, c.high - max(c.close, c.open))
        mecha_inf = max(0.0, min(c.close, c.open) - c.low)

        if bsl > 0 and prior:
            approached = any(p.high >= bsl - atr * 0.15 for p in prior)
            if approached and c.high > bsl and c.close < bsl and mecha_sup > atr * 0.25:
                return "BUYSIDE"

        if ssl > 0 and prior:
            approached = any(p.low <= ssl + atr * 0.15 for p in prior)
            if approached and c.low < ssl and c.close > ssl and mecha_inf > atr * 0.25:
                return "SELLSIDE"

    return "NONE"


# ═══════════════════════════════════════════════════════════════
# MOMENTUM — Nuevo en v1.0
# ═══════════════════════════════════════════════════════════════

def calculate_momentum(
    candles: List[Candle],
    lookback: int = 5,
) -> Tuple[float, str]:
    """
    Mide fuerza y dirección del impulso reciente.
    Combina ratio de cuerpos direccionales con velocidad (cuerpos crecientes).
    Retorna (score 0-1, 'BULLISH' | 'BEARISH' | 'NEUTRAL').
    """
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    if not recent:
        return 0.0, "NEUTRAL"

    bull_body = sum(c.body for c in recent if c.is_bullish)
    bear_body = sum(c.body for c in recent if c.is_bearish)
    total = bull_body + bear_body

    if total == 0:
        return 0.0, "NEUTRAL"

    bull_ratio = bull_body / total
    bear_ratio = bear_body / total

    # Velocidad: proporción de velas con cuerpo creciente
    bodies = [c.body for c in recent]
    velocity = 0.0
    if len(bodies) >= 2:
        inc = sum(1 for i in range(1, len(bodies)) if bodies[i] > bodies[i - 1])
        velocity = inc / (len(bodies) - 1)

    dominant = bull_ratio if bull_ratio > bear_ratio else bear_ratio
    direction = "BULLISH" if bull_ratio > bear_ratio else "BEARISH"

    if dominant <= 0.55:
        return round(dominant, 3), "NEUTRAL"

    score = _clamp(dominant * (1.0 + velocity * 0.3), 0.0, 1.0)
    return round(score, 3), direction


# ═══════════════════════════════════════════════════════════════
# PATRONES DE VELA
# ═══════════════════════════════════════════════════════════════

def detect_candle_patterns(
    candles: List[Candle],
    atr: float,
    direction: str,
) -> Optional[CandlePattern]:
    if len(candles) < 3:
        return None

    lookback = min(20, len(candles))
    avg_body = sum(c.body for c in candles[-lookback:]) / lookback
    avg_volume = _mean_volume(candles[-lookback:])

    for offset in range(1, 4):
        if len(candles) < offset + 1:
            break

        c = candles[-offset]
        prev = candles[-offset - 1]
        prev_body = max(prev.body, 1e-9)
        c_range = max(c.range, 1e-9)
        vol_boost = 1.0
        if avg_volume > 0 and c.volume > 0:
            vol_boost = 1.0 + min(c.volume / avg_volume, 3.0) * 0.08

        if direction == "SELL":
            sup = c.upper_wick
            if sup > c.body * 2.0 and sup > atr * 0.25 and c.body < avg_body * 1.3:
                return CandlePattern("pin_bar_bear", round(_clamp(sup / c_range * vol_boost, 0.0, 1.0), 3), -offset)
            if c.is_bearish and c.body > prev_body * 1.05 and c.close < prev.low and c.body > avg_body * 0.7:
                return CandlePattern("engulf_bear", round(_clamp(c.body / prev_body * vol_boost, 0.0, 1.0), 3), -offset)
            if c.is_bearish and c.body > atr * 1.1 and c.lower_wick < c.body * 0.25:
                return CandlePattern("inst_bear", round(_clamp(c.body / max(atr, 1e-9) * vol_boost, 0.0, 1.0), 3), -offset)

        elif direction == "BUY":
            inf = c.lower_wick
            if inf > c.body * 2.0 and inf > atr * 0.25 and c.body < avg_body * 1.3:
                return CandlePattern("pin_bar_bull", round(_clamp(inf / c_range * vol_boost, 0.0, 1.0), 3), -offset)
            if c.is_bullish and c.body > prev_body * 1.05 and c.close > prev.high and c.body > avg_body * 0.7:
                return CandlePattern("engulf_bull", round(_clamp(c.body / prev_body * vol_boost, 0.0, 1.0), 3), -offset)
            if c.is_bullish and c.body > atr * 1.1 and c.upper_wick < c.body * 0.25:
                return CandlePattern("inst_bull", round(_clamp(c.body / max(atr, 1e-9) * vol_boost, 0.0, 1.0), 3), -offset)

    return None


# ═══════════════════════════════════════════════════════════════
# CHoCH EN RETEST
# ═══════════════════════════════════════════════════════════════

def detect_choch_on_retest(
    candles: List[Candle],
    direction: str,
    atr: float,
    lookback: int = 10,
) -> bool:
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    if len(recent) < 5:
        return False

    internal = _compress_swings(find_swings(recent, lookback=2))
    if len(internal) < 3:
        return False

    last_index = len(recent) - 1
    break_thr = max(atr * 0.18, recent[-1].range * 0.10, 1e-9)

    if direction == "SELL":
        ph = next((s for s in reversed(internal) if s.kind == "SH" and s.index < last_index), None)
        if not ph:
            return False
        post_lows = [s for s in internal if s.kind == "SL" and s.index > ph.index]
        if not post_lows:
            return False
        bc = recent[-1]
        return (bc.close < post_lows[-1].price - break_thr and
                bc.low < post_lows[-1].price and bc.close < bc.open)

    elif direction == "BUY":
        pl = next((s for s in reversed(internal) if s.kind == "SL" and s.index < last_index), None)
        if not pl:
            return False
        post_highs = [s for s in internal if s.kind == "SH" and s.index > pl.index]
        if not post_highs:
            return False
        bc = recent[-1]
        return (bc.close > post_highs[-1].price + break_thr and
                bc.high > post_highs[-1].price and bc.close > bc.open)

    return False


# ═══════════════════════════════════════════════════════════════
# FILTRO DE RANGO
# ═══════════════════════════════════════════════════════════════

def is_ranging(
    candles: List[Candle],
    atr: float,
    lookback: int = 20,
    threshold: float = 4.0,
) -> bool:
    if len(candles) < lookback or atr <= 0:
        return False
    recent = candles[-lookback:]
    rng = max(c.high for c in recent) - min(c.low for c in recent)
    return rng < atr * threshold


# ═══════════════════════════════════════════════════════════════
# ENTRY TRIGGER — Cierre confirmado (crítico para scalping)
# ═══════════════════════════════════════════════════════════════

def check_entry_trigger(
    candles: List[Candle],
    direction: str,
    nearest_ob: Optional[OrderBlock],
    nearest_fvg: Optional[FVG],
    atr: float,
) -> bool:
    """
    Verifica que la última vela cerró confirmando la entrada.
    BUY: cierre alcista con precio dentro o cerca de zona.
    SELL: cierre bajista con precio dentro o cerca de zona.
    Evita entrar cuando el precio solo "pasó por" la zona sin confirmar dirección.
    """
    if len(candles) < 2:
        return False

    last = candles[-1]
    tolerance = atr * 0.5

    def in_zone() -> bool:
        if nearest_ob:
            if _zone_distance_price(last.close, nearest_ob.bottom, nearest_ob.top) <= tolerance:
                return True
        if nearest_fvg:
            if _zone_distance_price(last.close, nearest_fvg.bottom, nearest_fvg.top) <= tolerance:
                return True
        return False

    if direction == "BUY":
        return last.is_bullish and in_zone()
    elif direction == "SELL":
        return last.is_bearish and in_zone()

    return False


# ═══════════════════════════════════════════════════════════════
# SL / TP
# ═══════════════════════════════════════════════════════════════

def _select_sl_anchor(
    direction: str,
    ob: Optional[OrderBlock],
    fvg: Optional[FVG],
) -> Optional[float]:
    anchors: List[float] = []
    if direction == "BUY":
        if ob and ob.direction == "bullish":
            anchors.append(ob.bottom)
        if fvg and fvg.direction == "bullish":
            anchors.append(fvg.bottom)
    elif direction == "SELL":
        if ob and ob.direction == "bearish":
            anchors.append(ob.top)
        if fvg and fvg.direction == "bearish":
            anchors.append(fvg.top)
    if not anchors:
        return None
    return min(anchors) if direction == "BUY" else max(anchors)


def calculate_sl_tp(
    direction: str,
    entry: float,
    ob: Optional[OrderBlock],
    fvg: Optional[FVG],
    atr: float,
) -> Tuple[float, float, float]:
    buffer = max(atr * 0.4, 0.0)
    anchor = _select_sl_anchor(direction, ob, fvg)

    if direction == "BUY":
        sl = (anchor - buffer) if anchor is not None else (entry - atr * 1.5)
        if sl >= entry:
            sl = entry - atr * 1.5
        sl_dist = entry - sl
        tp1 = entry + sl_dist * 1.5
        tp2 = entry + sl_dist * 3.0

    elif direction == "SELL":
        sl = (anchor + buffer) if anchor is not None else (entry + atr * 1.5)
        if sl <= entry:
            sl = entry + atr * 1.5
        sl_dist = sl - entry
        tp1 = entry - sl_dist * 1.5
        tp2 = entry - sl_dist * 3.0

    else:
        return 0.0, 0.0, 0.0

    return round(sl, 5), round(tp1, 5), round(tp2, 5)


# ═══════════════════════════════════════════════════════════════
# SCORE — Recalibrado para scalping
# ═══════════════════════════════════════════════════════════════

def count_score(confluencias: List[str]) -> int:
    """
    Score 1-10 para scalping.
    - Sweep suma pero NO es requerido para valid=True
    - Momentum y CHoCH de confirmación tienen más peso
    - Umbral más bajo (5 vs 6)
    """
    unique = list(dict.fromkeys(confluencias))

    weights = [
        ("Barrido de SSL", 2.0),
        ("Barrido de BSL", 2.0),
        ("CHoCH de confirmacion", 2.0),
        ("CHoCH ", 1.5),
        ("BOS ", 1.5),
        ("OB ", 1.8),
        ("FVG ", 1.5),
        ("Momentum BULLISH", 1.5),
        ("Momentum BEARISH", 1.5),
        ("Vela institucional", 1.3),
        ("Pin bar", 1.0),
        ("Engulfing", 1.1),
        ("Tendencia interna", 1.2),
        ("Precio en zona", 1.2),
        ("Sesion de alta liquidez", 0.4),
    ]

    total = 0.0
    for item in unique:
        matched = False
        for key, w in weights:
            if key in item:
                total += w
                matched = True
                break
        if not matched:
            total += 0.3

    normalized = _clamp(total / 11.0, 0.0, 1.0)
    return max(1, min(10, int(round(1 + normalized * 9))))


def _count_confluence_categories(confluencias: List[str]) -> Tuple[int, int, int]:
    """Cuenta por categoría: (structural, zone, confirmation)."""
    structural = zone = confirmation = 0
    for c in confluencias:
        if any(k in c for k in ["Tendencia interna", "BOS ", "CHoCH "]):
            structural += 1
        elif any(k in c for k in ["OB ", "FVG ", "Precio en zona"]):
            zone += 1
        elif any(k in c for k in ["Pin bar", "Engulfing", "Vela institucional",
                                    "Barrido", "Momentum", "CHoCH de confirmacion"]):
            confirmation += 1
    return structural, zone, confirmation


# ═══════════════════════════════════════════════════════════════
# HELPERS DE SELECCIÓN Y ENTRADA
# ═══════════════════════════════════════════════════════════════

def _select_nearest_ob(
    obs: List[OrderBlock],
    direction: str,
    current_price: float,
    min_strength: float = 0.28,
) -> Optional[OrderBlock]:
    ob_dir = "bullish" if direction == "BUY" else "bearish"
    candidates = [
        ob for ob in obs
        if not ob.invalidated
        and ob.direction == ob_dir
        and ob.strength >= min_strength
        and ob.mitigation_state != "FULL"
    ]
    if not candidates:
        return None

    state_bonus = {
        "UNTESTED": -0.10, "TESTED": -0.03,
        "PARTIAL": 0.04, "FULL": 0.15, "INVALIDATED": 10.0,
    }

    def key(ob: OrderBlock):
        dist = _zone_distance_price(current_price, ob.bottom, ob.top)
        return (dist, state_bonus.get(ob.mitigation_state, 0.0), -ob.strength, -ob.index)

    return min(candidates, key=key)


def _select_nearest_fvg(
    fvgs: List[FVG],
    direction: str,
    current_price: float,
) -> Optional[FVG]:
    fvg_dir = "bullish" if direction == "BUY" else "bearish"
    candidates = [f for f in fvgs if f.fill_state != "FULL" and f.direction == fvg_dir]
    if not candidates:
        return None

    fill_bonus = {"UNFILLED": -0.10, "PARTIAL": 0.04, "FULL": 0.15}

    def key(fvg: FVG):
        dist = _zone_distance_price(current_price, fvg.bottom, fvg.top)
        return (dist, fill_bonus.get(fvg.fill_state, 0.0), -fvg.size, -fvg.index)

    return min(candidates, key=key)


def _is_price_near_zone(
    current_price: float,
    nearest_ob: Optional[OrderBlock],
    nearest_fvg: Optional[FVG],
    atr: float,
    proximity_atr: float = 1.0,
) -> bool:
    tolerance = atr * proximity_atr
    if nearest_ob and _zone_distance_price(current_price, nearest_ob.bottom, nearest_ob.top) <= tolerance:
        return True
    if nearest_fvg and _zone_distance_price(current_price, nearest_fvg.bottom, nearest_fvg.top) <= tolerance:
        return True
    return False


def _zone_entry_from_ob_fvg(
    direction: str,
    nearest_ob: Optional[OrderBlock],
    nearest_fvg: Optional[FVG],
    current_price: float,
) -> float:
    ob_zone = (min(nearest_ob.bottom, nearest_ob.top), max(nearest_ob.bottom, nearest_ob.top)) if nearest_ob else None
    fvg_zone = (min(nearest_fvg.bottom, nearest_fvg.top), max(nearest_fvg.bottom, nearest_fvg.top)) if nearest_fvg else None

    primary = fvg_zone or ob_zone
    secondary = ob_zone if fvg_zone else None

    if primary is None:
        return round(current_price, 5)

    bottom, top = primary

    # Si hay solapamiento OB + FVG, usar el centro del solapamiento
    if secondary is not None:
        ov_bottom = max(bottom, secondary[0])
        ov_top = min(top, secondary[1])
        if ov_top > ov_bottom:
            return round((ov_bottom + ov_top) / 2.0, 5)

    mid = (bottom + top) / 2.0

    if direction == "BUY":
        if current_price > top:
            return round(top - (top - bottom) * 0.2, 5)
        if current_price < bottom:
            return round(bottom + (top - bottom) * 0.35, 5)
    elif direction == "SELL":
        if current_price < bottom:
            return round(bottom + (top - bottom) * 0.2, 5)
        if current_price > top:
            return round(top - (top - bottom) * 0.35, 5)

    return round(mid, 5)


# ═══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL — analyze()
# ═══════════════════════════════════════════════════════════════

def analyze(
    candles_raw: list,
    current_price: float,
    timeframe: str = "M5",
    spread: float = 0.0,
    session: str = "",
) -> SMCAnalysis:
    """
    Análisis SMC mono-temporalidad para scalping.

    Uso para M1 y M5 de forma independiente:
        result_m1 = analyze(candles_m1, price, timeframe="M1", spread=spread)
        result_m5 = analyze(candles_m5, price, timeframe="M5", spread=spread)

    Args:
        candles_raw:   List[dict] con keys: time, open, high, low, close, volume
        current_price: Precio actual del mercado
        timeframe:     'M1' o 'M5'
        spread:        Spread actual del broker en precio (no pips)
        session:       Sesión actual (opcional, suma al score)

    Returns:
        SMCAnalysis — result.valid=True cuando hay señal operativa
    """
    result = SMCAnalysis(timeframe=timeframe.upper())
    params = get_params(timeframe)

    # ── PASO 1: Parsear ──────────────────────────────────────────────
    stats: dict = {"errors": 0}
    candles = parse_candles(candles_raw, stats)
    result.parse_errors = stats["errors"]

    if len(candles) < params.min_candles:
        result.reason = f"Datos insuficientes: {len(candles)} velas (mín {params.min_candles})"
        return result

    # ── PASO 2: ATR ──────────────────────────────────────────────────
    atr = calculate_atr(candles, period=params.atr_period)
    result.atr = atr

    if atr <= 0:
        result.reason = "ATR inválido"
        return result

    # ── PASO 3: Filtro de spread ─────────────────────────────────────
    if spread > 0 and spread > atr * params.spread_max_atr:
        result.spread_filtered = True
        result.reason = f"Spread {spread:.5f} excede límite {atr * params.spread_max_atr:.5f}"
        return result

    # ── PASO 4: Filtro de rango ──────────────────────────────────────
    if is_ranging(candles, atr, lookback=20, threshold=params.ranging_atr_mult):
        result.ranging_market = True
        result.reason = "Mercado en consolidación"
        result.score = 1
        return result

    # ── PASO 5: Tendencia interna ────────────────────────────────────
    min_swing_size = atr * 0.8
    swings = find_swings(candles, lookback=params.swing_lookback, min_size_atr=min_swing_size)
    trend, last_bos, last_choch = classify_trend(swings, candles)
    result.trend = trend
    result.last_bos = last_bos
    result.last_choch = last_choch

    if trend == "NEUTRAL":
        result.reason = "Sin tendencia interna clara"
        result.score = 1
        return result

    # ── PASO 6: Zona Premium / Discount ─────────────────────────────
    zone, zone_pct = calculate_zone(candles, current_price, swings)
    result.zone = zone
    result.zone_pct = zone_pct

    # ── PASO 7: Order Blocks ─────────────────────────────────────────
    obs = find_order_blocks(candles, atr, lookback=params.ob_lookback, fast_invalidation=True)
    result.order_blocks = obs

    # ── PASO 8: FVGs ─────────────────────────────────────────────────
    fvgs = find_fvgs(candles, atr, lookback=params.fvg_lookback, min_gap_atr=params.fvg_gap_atr)
    result.fvgs = fvgs

    # ── PASO 9: Liquidez ─────────────────────────────────────────────
    eq_highs, eq_lows, bsl, ssl = find_liquidity(candles, atr, swing_lookback=params.swing_lookback)
    result.equal_highs = eq_highs[:3]
    result.equal_lows = eq_lows[:3]
    result.buyside_liquidity = bsl
    result.sellside_liquidity = ssl
    result.liquidity_swept = detect_liquidity_sweep(
        candles, bsl, ssl, atr, lookback=params.sweep_lookback
    )

    # ── PASO 10: Momentum ────────────────────────────────────────────
    mom_score, mom_dir = calculate_momentum(candles, lookback=params.momentum_lookback)
    result.momentum_score = mom_score
    result.momentum_dir = mom_dir

    # ── PASO 11: Dirección ───────────────────────────────────────────
    if trend == "BULLISH" and zone == "DISCOUNT":
        direction = "BUY"
    elif trend == "BEARISH" and zone == "PREMIUM":
        direction = "SELL"
    else:
        result.direction = "NEUTRAL"
        result.reason = f"Tendencia {trend} pero zona {zone} no alineada"
        result.score = 2
        return result

    result.direction = direction

    # ── PASO 12: Zonas institucionales más cercanas ──────────────────
    nearest_ob = _select_nearest_ob(obs, direction, current_price, min_strength=params.ob_strength_min)
    nearest_fvg = _select_nearest_fvg(fvgs, direction, current_price)
    result.nearest_ob = nearest_ob
    result.nearest_fvg = nearest_fvg

    # ── PASO 13: Precio en zona ──────────────────────────────────────
    price_in_zone = _is_price_near_zone(
        current_price, nearest_ob, nearest_fvg, atr,
        proximity_atr=params.zone_proximity_atr,
    )

    # ── PASO 14: CHoCH en retest ─────────────────────────────────────
    choch_retest = detect_choch_on_retest(candles, direction, atr, lookback=params.choch_lookback)

    # ── PASO 15: Patrón de vela ──────────────────────────────────────
    candle_pattern = detect_candle_patterns(candles, atr, direction)
    result.candle_pattern = candle_pattern

    # ── PASO 16: Entry trigger — cierre confirmado ───────────────────
    entry_triggered = check_entry_trigger(candles, direction, nearest_ob, nearest_fvg, atr)
    result.entry_triggered = entry_triggered

    # ── PASO 17: Confluencias ────────────────────────────────────────
    confluencias: List[str] = []

    confluencias.append(f"Tendencia interna {trend}")

    if last_bos != "NONE":
        confluencias.append(f"BOS {last_bos} confirmado")

    if last_choch != "NONE":
        confluencias.append(f"CHoCH {last_choch} confirmado")

    if (direction == "BUY" and zone == "DISCOUNT") or (direction == "SELL" and zone == "PREMIUM"):
        confluencias.append(f"Precio en zona {zone} ({zone_pct:.0f}%)")

    if nearest_ob:
        tipo = "alcista" if nearest_ob.direction == "bullish" else "bajista"
        confluencias.append(f"OB {tipo} [{nearest_ob.mitigation_state}] {nearest_ob.bottom:.5f}-{nearest_ob.top:.5f}")

    if nearest_fvg:
        tipo = "alcista" if nearest_fvg.direction == "bullish" else "bajista"
        confluencias.append(f"FVG {tipo} [{nearest_fvg.fill_state}] {nearest_fvg.bottom:.5f}-{nearest_fvg.top:.5f}")

    if result.liquidity_swept == "SELLSIDE" and direction == "BUY":
        confluencias.append("Barrido de SSL — trampa bajista completada")
    elif result.liquidity_swept == "BUYSIDE" and direction == "SELL":
        confluencias.append("Barrido de BSL — trampa alcista completada")

    if choch_retest:
        confluencias.append("CHoCH de confirmacion en retest")

    mom_expected = "BULLISH" if direction == "BUY" else "BEARISH"
    if mom_dir == mom_expected:
        confluencias.append(f"Momentum {mom_dir} (fuerza {mom_score:.0%})")

    if candle_pattern:
        nombres = {
            "pin_bar_bear":  "Pin bar bajista",
            "pin_bar_bull":  "Pin bar alcista",
            "engulf_bear":   "Engulfing bajista",
            "engulf_bull":   "Engulfing alcista",
            "inst_bear":     "Vela institucional bajista",
            "inst_bull":     "Vela institucional alcista",
        }
        confluencias.append(
            f"{nombres.get(candle_pattern.kind, candle_pattern.kind)} (fuerza {candle_pattern.strength:.0%})"
        )

    sesiones_validas = {"Londres", "Nueva York", "Overlap Londres-NY", "Tokio", "Sydney"}
    if session in sesiones_validas:
        confluencias.append(f"Sesion de alta liquidez: {session}")

    result.confluencias = confluencias
    sc, zc, cc = _count_confluence_categories(confluencias)
    result.structural_count = sc
    result.zone_count = zc
    result.confirmation_count = cc

    # ── PASO 18: Score ───────────────────────────────────────────────
    result.score = count_score(confluencias)

    # ── PASO 19: Entry / SL / TP ─────────────────────────────────────
    entry = _zone_entry_from_ob_fvg(direction, nearest_ob, nearest_fvg, current_price)
    sl, tp1, tp2 = calculate_sl_tp(direction, entry, nearest_ob, nearest_fvg, atr)
    result.entry = entry
    result.sl = sl
    result.tp1 = tp1
    result.tp2 = tp2

    if sl != 0 and sl != entry:
        sl_dist = abs(entry - sl)
        result.rr1 = round(abs(tp1 - entry) / sl_dist, 2)
        result.rr2 = round(abs(tp2 - entry) / sl_dist, 2)

    # ── PASO 20: Validación final ────────────────────────────────────
    # Sweep ya NO es obligatorio — solo suma score.
    # Entry trigger sí es obligatorio en scalping.
    checks = {
        "score_suficiente":   result.score >= params.score_threshold,
        "estructura":         sc >= 1,
        "zona_institucional": zc >= 1,
        "confirmacion":       cc >= 1,
        "sl_valido":          result.sl != 0,
        "tp_valido":          result.tp1 != 0,
        "rr_suficiente":      result.rr1 >= params.rr_min,
        "precio_en_zona":     price_in_zone,
        "entry_trigger":      entry_triggered or result.score >= 7,
    }

    result.valid = all(checks.values())

    if not result.valid:
        detalle = {
            "score_suficiente":   f"score {result.score} (mín {params.score_threshold})",
            "estructura":         "sin BOS/CHoCH/tendencia",
            "zona_institucional": "sin OB o FVG válido",
            "confirmacion":       "sin patrón/momentum/sweep/CHoCH retest",
            "sl_valido":          "SL no calculado",
            "tp_valido":          "TP no calculado",
            "rr_suficiente":      f"RR {result.rr1:.2f} (mín {params.rr_min})",
            "precio_en_zona":     "precio fuera de zona institucional",
            "entry_trigger":      "cierre no confirmó la entrada",
        }
        fallos = [detalle[k] for k, ok in checks.items() if not ok]
        result.reason = "Sin señal: " + " | ".join(fallos)

    return result
