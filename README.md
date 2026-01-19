# Системаи Скальпинг Portfolio (XAUUSDm + BTCUSDm)

Ин лоиҳа системаи савдои алгоритмист, ки ба MetaTrader 5 пайваст мешавад ва ду стратегияро ҳамзамон иҷро мекунад:
XAUUSDm (тилло) ва BTCUSDm (биткоин). Ҳар ду pipeline ҳамзамон кор мекунанд, сигнал месозанд,
risk‑қоидаҳоро мегузаранд ва ордерҳоро ба MT5 мефиристанд.

README бо код мувофиқ аст. ENV танҳо барои 5 тағйирёбандаи махфӣ истифода мешавад.

## Система чӣ мекунад

- Ду pipeline (XAU + BTC) ҳамзамон таҳлил мешавад; ҳар ду метавонад ордер кушояд.
- Таймфреймҳо: M1 + M5 + M15 (агар M5/M15 дастрас набошад → ба M1 меафтад).
- Сигнал + нақшаи SL/TP/lot месозад, баъд order‑ро дар MT5 иҷро мекунад.
- Telegram‑бот барои назорат ва идоракунии система дорад (паём барои ҳар ордери кушодашуда).
- Логҳои health/diagnostic ба `Logs/` менависад (health ба Telegram фиристода намешавад).
- Аз рӯи дақиқии сигнал 1–3 ордер мекушояд (ҳамзамон).
- Ҳатто бо ордерҳои кушода ҳам анализ идома меёбад.
- BTC 24/7 кор мекунад; XAU 24/5 (market_open_24_5 назорат мекунад).

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
- Танҳо админ: `/tek_prof` ва `/stop_ls`
- Ҳар ордери кушодашуда ба админ SMS мефиристад (бо формати тоҷикӣ).

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
Танҳо ҳамин 5 тағйирёбанда аз ENV хонда мешавад. Дигар ҳама танзимот дар `config_xau.py` ва `config_btc.py` муайян шудааст.

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
- `Logs/portfolio_engine_diag.jsonl`
- `Logs/order_execution.log` (health log ғайрифаъол аст)
- `Logs/telegram.log`
- `Logs/mt5.log`
- `Logs/main.log`

Майдонҳои маъмул:
- `xau_ok` / `btc_ok` → data fresh?
- `xau_reason` / `btc_reason` → сабаби блок
- `spread_pts` → spread дар points
- `q` → queue size

## Risk ва safety control

Дар `StrategiesXau/risk_management.py` ва `StrategiesBtc/risk_management.py`:
- режимҳои A/B/C (рӯзи нав — reset)
- signal throttling (hour/day)
- latency + spread breakers
- execution quality monitor
- optional session filtering
- cooldown баъди fill ё latency

## Маҳдудиятҳо

- Агар MT5 барҳои нав надиҳад → trade block мешавад.
- Агар spread баланд бошад → trade block мешавад.
- Агар M5/M15 дастрас набошад → система M1‑ро истифода мебарад.

## Танзимоти касбӣ (config)

Танзимоти асосӣ дар ин файлҳо ҷойгир аст:
- `config_xau.py` (тилло)
- `config_btc.py` (биткоин)

Муҳим:
- `daily_target_pct=0.10` → ҳадафи рӯзона 10% → режим B.
- `daily_loss_b_pct=0.02` ва `daily_loss_c_pct=0.03` → режими B/C.
- Дар C: **hard‑stop** фаъол аст (engine барои он рӯз қатъ мешавад).
- `enforce_daily_limits=True` → режимҳои A/B/C фаъол.
- `ignore_daily_stop_for_trading=False` → hard‑stop воқеан engine‑ро қатъ мекунад.
- `enforce_drawdown_limits=False` → drawdown ордерро қатъ намекунад.
- `multi_order_confidence_tiers` ва `multi_order_max_orders` → 1–3 ордер аз рӯи дақиқии сигнал.
- `max_signals_per_day=0` → лимит нест (ҳар сигнал → ордер).
- `ignore_external_positions=True` → ордерҳои дастӣ ба ҳисобҳои risk/phase таъсир намерасонанд.
- `magic=777001` → magic number барои фарқ кардани ордерҳои бот.
- `ignore_microstructure=True` → BTC аз micro‑филтрҳо блок намешавад.

Барои скалпинг‑суръат:
- `poll_seconds_fast=0.05`
- `decision_debounce_ms=50`

## Маълумоти муҳим (MT5)

- Агар `IPC timeout` бинед, ин аз MT5 аст (на аз код).
- MT5 бояд кушода, login шуда ва AutoTrading фаъол бошад.
- Агар лозим шавад, `mt5_path`‑ро дар `config_xau.py`/`config_btc.py` муайян кунед.

## Тавзеҳ: /tek_prof ва /stop_ls

Ин ду команда **TP/SL‑и ҳамаи позицияҳои кушода**‑ро бо **USD** ҳисоб мекунанд (на бо пункт).

Формулаи умумӣ (барои ҳар позиция):
- `profit_per_tick = trade_tick_value * volume`
- `ticks_needed = usd / profit_per_tick`
- `price_delta = ticks_needed * trade_tick_size`

Барои TP:
- BUY → `TP = open_price + price_delta`
- SELL → `TP = open_price - price_delta`

Барои SL:
- BUY → `SL = open_price - price_delta`
- SELL → `SL = open_price + price_delta`

Агар broker `trade_tick_value` ё `trade_tick_size` = 0 диҳад, ҳисоб имконнопазир мешавад ва позиция **skip** мешавад.

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