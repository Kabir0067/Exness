# Системаи Скальпинг Portfolio (XAUUSDm + BTCUSDm)

Ин лоиҳа системаи савдои алгоритмист, ки ба MetaTrader 5 пайваст мешавад ва ду стратегияро ҳамзамон иҷро мекунад:
яке барои тилло (XAUUSDm) ва дигаре барои биткоин (BTCUSDm). Ҳар ду pipeline ҳамзамон кор мекунанд,
сигнал месозанд, risk‑қоидаҳоро мегузаранд ва ордерҳоро ба MT5 мефиристанд.

README пурра мувофиқи коди воқеӣ навишта шудааст.

## Система чӣ мекунад

- Ду pipeline (XAU + BTC) ҳамзамон таҳлил мешавад.
- 3 таймфрейм истифода мекунад: M1, M5, M15. Агар M5/M15 дастрас набошад → ба M1 меафтад.
- Сигнал + нақшаи SL/TP/lot месозад, баъд order‑ро дар MT5 иҷро мекунад.
- Telegram‑бот барои назорат ва идоракунии система дорад.
- Логҳои health/diagnostic ба `Logs/` менависад.

## Архитектура (модулҳои асосӣ)

### Portfolio engine
Файл: `Bot/portfolio_engine.py`

Масъулиятҳо:
- Пайвастшавӣ ба MT5 ва сохтани pipeline‑ҳо.
- Санҷиши “freshness”‑и маълумот.
- Гирифтани кандидатҳо аз ҳар сигнал engine.
- Иҷрои order‑ҳо тавассути worker.
- Лог/диагностикаи health.

### Strategy pipeline‑ҳо
XAU: `StrategiesXau/`  
BTC: `StrategiesBtc/`

Ҳар pipeline:
- `indicators.py` (feature engine)
- `signal_engine.py` (logic барои сигнал)
- `risk_management.py` (risk + sizing)

### MT5 пайвастшавӣ
Файл: `mt5_client.py`

Функсияҳо:
- Initialize/login бо retry ва recovery.
- Single‑instance lock (як Python процесс).

### Order execution
Файл: `ExnessAPI/order_execution.py`

Функсияҳо:
- Market order бо SL/TP.
- Retry барои баъзе retcode‑ҳо.
- Metrics (latency/slippage) ба лог.

### Telegram бот
Файл: `Bot/bot.py`

Функсияҳо:
- Start/Stop engine
- Status/Balance/Open positions/Profit‑Loss

## Индикаторҳо

Индикаторҳои классикӣ (XAU ва BTC):
- EMA (short/mid/long/vlong)
- ATR
- RSI
- ADX
- MACD (signal + histogram)
- Bollinger Bands
- Z‑Volume (z‑score)

Илова барои BTC (derived features):
- VWAP
- ATR percent
- BB width
- EMA/RSI/MACD slope
- Return features (ret1, ret3)

Сигналҳои структурӣ (pattern/filter):
- FVG (fair value gap)
- Liquidity sweep
- Order block
- Divergence (RSI/MACD)
- Near round‑number

## Runtime талабот

- Windows + MetaTrader 5 (логиншуда).
- Python 3.10+.
- MT5 AutoTrading фаъол.

## Танзим (Setup)

### 1) Dependencies
```powershell
pip install -r requirements.txt
```

### 2) .env (environment variables)
```ini
EXNESS_LOGIN=12345678
EXNESS_PASSWORD=your_password
EXNESS_SERVER=Exness-MT5Real
BOT_TOKEN=your_telegram_bot_token
ADMIN_ID=your_telegram_user_id
```

Опсионалӣ (MT5 control):
```ini
MT5_PATH=C:\Path\To\terminal64.exe
MT5_PORTABLE=0
MT5_AUTOSTART=1
MT5_TIMEOUT_MS=90000
```

## Запуск
```powershell
py main.py
```

Флагҳо:
- `--headless` → бе Telegram
- `--engine-only` → танҳо engine

## Логҳо ва мониторинг

Логҳо дар `Logs/` навишта мешаванд.

Муҳим:
- `Logs/portfolio_engine_health.log`
- `Logs/order_execution_health.log`
- `Logs/telegram.log`
- `Logs/mt5.log`

Майдонҳои маъмул:
- `xau_ok` / `btc_ok` → data fresh?
- `xau_reason` / `btc_reason` → сабаби блок
- `spread_pts` → spread дар points
- `q` → queue size

## Risk ва safety control

Дар `StrategiesXau/risk_management.py` ва `StrategiesBtc/risk_management.py`:
- daily loss / drawdown limits
- signal throttling (hour/day)
- latency + spread breakers
- execution quality monitor
- optional session filtering
- cooldown баъди fill ё latency

## Маҳдудиятҳо

- Агар MT5 барҳои нав надиҳад → trade block мешавад.
- Агар spread баланд бошад → trade block мешавад.
- Агар M5/M15 дастрас набошад → система M1‑ро истифода мебарад.

## Project structure

```text
Exness/
├── Bot/
│   ├── portfolio_engine.py
│   ├── bot.py
│   └── engine.py
├── DataFeed/
│   ├── market_feed.py
│   └── btc_feed.py
├── ExnessAPI/
│   ├── order_execution.py
│   ├── orders.py
│   └── history.py
├── StrategiesXau/
│   ├── indicators.py
│   ├── risk_management.py
│   └── signal_engine.py
├── StrategiesBtc/
│   ├── indicators.py
│   ├── risk_management.py
│   └── signal_engine.py
├── Logs/
├── config_xau.py
├── config_btc.py
└── main.py
```
