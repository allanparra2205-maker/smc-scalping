# SMC Scalping Engine

Motor de análisis SMC mono-temporalidad con validación IA (Groq).

## Arquitectura

```
[MT5 local]  →  HTTP POST (velas)  →  [Render server]
                                            ↓
                                       SMC Engine
                                            ↓
                                        Groq API
                                            ↓
              ←  JSON (señal + score IA)  ←
```

## Archivos

| Archivo | Dónde va |
|---|---|
| `smc_scalping_engine.py` | GitHub + Render |
| `server.py` | GitHub + Render |
| `requirements.txt` | GitHub + Render |
| `mt5_connector.py` | Solo local (no subir) |
| `.env` | Solo local (no subir) |

## Deploy en Render

1. Conecta este repo en [render.com](https://render.com)
2. Nuevo servicio → **Web Service**
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Variables de entorno en Render:
   - `GROQ_API_KEY` → tu key de [console.groq.com](https://console.groq.com)
   - `API_SECRET` → string secreto que también pones en tu `.env` local (opcional)

## Setup local (MT5)

1. Crea un archivo `.env` en la misma carpeta que `mt5_connector.py`:

```
SERVER_URL=https://tu-app.onrender.com
API_SECRET=tu_secret_opcional
SYMBOL=XAUUSD
```

2. Instala dependencias:

```bash
pip install MetaTrader5 requests python-dotenv
```

3. Abre MT5, luego corre:

```bash
python mt5_connector.py
```

## Temporalidades

M1 y M5 se analizan de forma **completamente independiente**.
Cada una tiene sus propios parámetros de lookback, umbrales y score.

## Señal válida

Una señal se imprime en consola cuando:
- Score SMC >= 5
- Score IA >= 5
- Entry trigger confirmado (cierre de vela)
- Precio dentro de zona institucional (OB o FVG)
- RR >= 1.5
