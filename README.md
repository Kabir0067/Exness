<div align="center">

# QuantCore Pro

### Автоматизированная торговая система для XAU и BTC на MetaTrader 5

*39 000 строк Python. Детерминированный конвейер. ML-валидация. Контроль риска на каждом уровне.*

<br/>

<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/MetaTrader_5-Exness-0078D4?style=flat-square" />
<img src="https://img.shields.io/badge/ML-CatBoost-FFCC00?style=flat-square" />
<img src="https://img.shields.io/badge/Telegram-Bot_Control-26A5E4?style=flat-square&logo=telegram&logoColor=white" />
<img src="https://img.shields.io/badge/Tests-116_passing-00C853?style=flat-square" />

<br/><br/>

| XAUUSDm (Gold) | BTCUSDm (Bitcoin) | Архитектура | Риск |
|:-:|:-:|:-:|:-:|
| 24/5 с rollover blackout | 24/7 | Deterministic FSM | Kelly/4 + hard cap |
| ATR×2.5 SL | ATR×3.0 SL | Multi-worker execution | Max DD 6%, daily 4% |
| Spread gate 0.025% | Spread gate 0.05% | Fill recovery + sync | Kill switch (auto) |

</div>

---

## Что это за система

QuantCore Pro — полноценная автоматизированная торговая система, построенная как инженерный продукт, а не как скрипт-советник.

Система принимает рыночные данные с MetaTrader 5, вычисляет более 40 технических и структурных признаков, пропускает их через обученную ML-модель и каскад валидационных фильтров, рассчитывает позицию по формуле Келли с жёстким ограничением, отправляет ордер с контролем исполнения и верифицирует результат — полный цикл от тика до проверенной сделки.

Каждый этап цепочки спроектирован с учётом отказоустойчивости: что произойдёт, если MT5 отключится, данные устареют, модель вернёт мусор, брокер отклонит ордер, баланс обнулится. На каждый из этих сценариев есть конкретный обработчик.

---

## Почему эта система сложная

Это не один файл с `if RSI > 70: sell`. Это **65 модулей**, связанных в единый конвейер:

- **Модульная архитектура** — каждый компонент изолирован и тестируем: данные, признаки, сигналы, риск, исполнение, контроль
- **Детерминированный FSM** — конечный автомат с 7 состояниями управляет циклом торговли, исключая недетерминированное поведение
- **CatBoost ML-пайплайн** — обучение, walk-forward анализ, Monte Carlo стресс-тест, gate-система качества — всё до первого live-ордера
- **Многослойная защита капитала** — kill switch, circuit breaker по волатильности, breaker по latency/slippage, daily loss limit, peak drawdown guard, hard lot cap
- **Контроль исполнения** — дедупликация ордеров, верификация fill, пост-fill прикрепление SL/TP с retry, ghost fill detection, order sync manager
- **Полная наблюдаемость** — JSONL диагностика, CSV метрик исполнения, watchdog snapshot, rejection stats, Wilson CI для win rate
- **Telegram-контроль** — start/stop, статус, ручной override, дебаунс от двойных нажатий

---

## Архитектура системы

```
                          ┌─────────────────────────────────────────────────────────┐
                          │                     main.py                             │
                          │          preflight → auto-train → gate → supervisor     │
                          └────────────────────────────┬────────────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
              ┌─────▼──────┐                  ┌────────▼────────┐                 ┌───────▼───────┐
              │  Telegram   │                  │  MultiAsset     │                 │   Health      │
              │  Bot        │◄────────────────►│  TradingEngine  │────────────────►│   Monitor     │
              │  (control)  │   start/stop     │  (FSM core)     │   heartbeat     │   (watchdog)  │
              └─────────────┘                  └────────┬────────┘                 └───────────────┘
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          │                            │                            │
                  ┌───────▼───────┐           ┌────────▼────────┐          ┌────────▼────────┐
                  │  DataFeed     │           │  Pipeline       │          │  Execution      │
                  │  (XAU / BTC)  │           │  (per asset)    │          │  Manager        │
                  │  MT5 + ticks  │           │                 │          │  + Workers      │
                  └───────┬───────┘           └────────┬────────┘          └────────┬────────┘
                          │                            │                            │
                          ▼                            ▼                            ▼
               ┌────────────────┐         ┌──────────────────┐          ┌────────────────────┐
               │ FeatureEngine  │────────►│  SignalEngine     │────────►│  RiskManager       │
               │ 40+ indicators │         │  100-pt scoring   │         │  Kelly/4 sizing    │
               │ smart-money    │         │  ML bridge        │         │  ATR SL/TP         │
               │ MTF analysis   │         │  25+ filters      │         │  kill switch       │
               └────────────────┘         └──────────────────┘          └────────┬───────────┘
                                                                                 │
                                                                                 ▼
                                                                      ┌────────────────────┐
                                                                      │  MT5 order_send    │
                                                                      │  fill verify       │
                                                                      │  SL/TP attach      │
                                                                      │  order sync        │
                                                                      └────────────────────┘
```

### Полная цепочка принятия решения

```
DATA ──► FEATURES ──► MODEL ──► SIGNALS ──► FILTERS ──► RISK ──► ORDER ──► FILL ──► VERIFY ──► LOG
 │          │           │          │           │          │         │         │          │        │
 MT5      40+ ind    CatBoost   100-pt     25+ gates   Kelly    MT5 API   retcode    sync    JSONL
 ticks    ATR/BB     predict    scoring    MTF/D1/     /4 cap   dedup     SL/TP     broker   CSV
 bars     patterns   threshold  confidence spread/vol  phase    idempot   rebase    recon    TG
```

Если любой этап возвращает отказ — торговля не происходит. Система fail-safe по дизайну.

---

## Основные модули

### Ядро (`core/`)

| Модуль | LOC | Назначение |
|--------|----:|------------|
| `config.py` | 713 | Единый конфиг с asset-specific override (XAU/BTC) |
| `feature_engine.py` | 982 | 40+ индикаторов, smart-money паттерны, MTF анализ |
| `signal_engine.py` | 1 968 | 100-балльная система скоринга, ML-bridge, 25+ фильтров |
| `risk_engine.py` | 2 074 | Позиционный сайзинг, SL/TP, kill switch, фазы, breakers |
| `model_gate.py` | 783 | Multi-layer gate: Sharpe ≥ 0.5, WR ≥ 52%, DD ≤ 25% |
| `portfolio_risk.py` | — | Кросс-активный контроль, корреляция, портфельный DD |

### Исполнение (`Bot/Motor/`)

| Модуль | LOC | Назначение |
|--------|----:|------------|
| `engine.py` | 2 436 | Главный оркестратор, FSM, watchdog, recovery |
| `pipeline.py` | — | Per-asset signal pipeline с кэшированием |
| `execution.py` | — | ExecutionWorker threads, bounded queue |
| `execution_manager.py` | — | Enqueue, dedup, cooldown, verification |
| `order_sync_manager.py` | — | Broker reconciliation, deferred fill resolution |
| `inference_engine.py` | 932 | CatBoost inference, bar-level cache, threshold |
| `scheduler.py` | — | Asset scheduling, signal density control |
| `fsm_runtime.py` | — | FSM dispatch loop, state transitions |

### Обучение и валидация (`Backtest/`)

| Модуль | LOC | Назначение |
|--------|----:|------------|
| `model_train.py` | 2 317 | CatBoost training, TSCV, feature pipeline, window creation |
| `engine.py` | 1 852 | Backtest simulation, WFA, Monte Carlo (10K paths), stress |
| `metrics.py` | — | Sharpe, profit factor, expectancy, risk-of-ruin |

### Данные и исполнение

| Модуль | LOC | Назначение |
|--------|----:|------------|
| `mt5_client.py` | 1 398 | MT5 connection management, async dispatch, reconnect |
| `DataFeed/xau_market_feed.py` | 914 | XAU real-time data, tick stats, microstructure |
| `DataFeed/btc_market_feed.py` | 1 189 | BTC real-time data, 24/7, regime estimation |
| `ExnessAPI/order_execution.py` | 1 799 | Market order, SL/TP robust attach, fill recovery |
| `ExnessAPI/functions.py` | 2 093 | Position management, close, adaptive risk |

---

## Возможности

### Анализ рынка
- **Multi-timeframe**: M1, M5, M15, H1, H4, D1 — данные синхронизированы, признаки shift-aware
- **40+ индикаторов**: EMA (8/21/50/200), RSI, MACD, ADX, Bollinger Bands, CCI, Stochastic, OBV, ATR, KER, RVI
- **Smart-money паттерны**: Fair Value Gaps, Liquidity Sweeps, Order Blocks, Stop-Hunt detection
- **Микроструктура**: tick momentum, cumulative delta, OFI proxy, spread dynamics

### ML-модель
- **CatBoost** gradient boosting с 2000 итераций
- **Time-series cross-validation** (4 fold, хронологический порядок)
- **Walk-forward анализ** с адаптивным масштабированием окон
- **Monte Carlo стресс-тест**: 10 000 случайных путей с bootstrap, tail shocks, execution drag
- **Автоматический retrain** при устаревании модели или провале gate

### Генерация сигналов
- **100-балльная система**: trend (24), momentum (15), volatility (10), structure (12), flow (15), tick momentum (12), volume delta (8), mean reversion (4)
- **Confidence mapping** с cap по силе сигнала
- **Sniper floor**: сигналы ниже 75% confidence отклоняются
- **D1 confluence**: daily trend усиливает или ослабляет сигнал
- **MTF penalties**: конфликт M5/M15 с основным сигналом снижает confidence

### Управление риском
- **Kelly/4 sizing**: четверть от формулы Келли, с абсолютным потолком 2% на сделку
- **ATR-based SL/TP**: adaptive с fractal volatility stop (KER + RVI weighted)
- **Hard lot cap**: XAU ≤ 1.0 lot, BTC ≤ 0.50 lot — не обходится ни при каких условиях
- **Daily loss limit**: 4% — при достижении торговля останавливается до нового UTC-дня
- **Peak drawdown guard**: 3% от пика — soft stop; 6% — hard stop
- **Kill switch**: expectancy < -0.5 И winrate < 30% → KILLED; expectancy < -0.2 → COOLING
- **Phase escalation**: A → B → C с уменьшением размера позиции (1.0x → 0.75x → стоп)

### Исполнение
- **Bounded queue** (maxsize=50) с multi-worker dispatch
- **Idempotency**: edge-trigger dedup + seen-index + cooldown
- **Fill verification**: immediate retcode check + broker-side history probe
- **SL/TP robust attach**: 4 retry с расширением дистанции, fail-safe close
- **Order sync manager**: deferred resolution для ambiguous fills, 8-секундный timeout
- **Ghost fill detection**: проверка `history_deals_get` после transport failure

### Защита и восстановление
- **Volatility circuit breaker**: ATR/SMA(ATR,50) > 3.0 → cooldown 30 мин
- **Execution breaker**: p95 latency > 550ms или slippage > 20pts → cooldown 2 мин
- **Spread spike blocker**: > 3σ от исторического spread → блокировка
- **Flash crash guard**: > 3.5σ price move → cooldown 5 мин
- **FSM auto-restart**: 3 попытки с exponential backoff (10s / 20s / 30s)
- **STOP.lock**: физический файл-выключатель — мгновенная остановка и flatten
- **MT5 identity check**: верификация account login при каждом reconnect

---

## Наблюдаемость

Система генерирует данные для мониторинга на каждом уровне:

| Метрика | Источник | Назначение |
|---------|----------|------------|
| `signals_24h` | watchdog snapshot | Количество сигналов за 24ч |
| `FILTER_REJECTION_STATS` | pipeline health log | signal_rate, top rejection reasons |
| `wr_ci_95_lo / wr_ci_95_hi` | watchdog (Wilson CI) | Статистическая достоверность win rate |
| `latency_p95_ms` | watchdog | Качество исполнения |
| `slippage_p95_points` | watchdog | Проскальзывание |
| `dd_pct` | watchdog | Drawdown от пика |
| `kill_switch` | risk engine | ACTIVE / COOLING / KILLED |
| Execution CSV | `Logs/exec_metrics_*.csv` | Per-trade latency, slippage, spread |
| Health heartbeat | `portfolio_engine_health.log` | MT5 status, module status, error counts |
| Diagnostics | `portfolio_engine_diag.jsonl` | Full engine state per cycle |

---

## Model Quality Gate

Ни один актив не допускается к торговле без прохождения gate:

```
                   ┌─────────────────────────────┐
                   │       MODEL GATE CHECK       │
                   │                               │
                   │  state file exists?     ──────► NO → blocked
                   │  status = VERIFIED?     ──────► NO → blocked
                   │  unsafe = false?        ──────► NO → blocked
                   │  real_backtest = true?   ──────► NO → blocked
                   │  WFA passed?            ──────► NO → blocked
                   │  stress test passed?    ──────► NO → blocked
                   │  Sharpe ≥ 0.5?          ──────► NO → blocked
                   │  Win Rate ≥ 52%?        ──────► NO → blocked
                   │  Max Drawdown ≤ 25%?    ──────► NO → blocked
                   │  model .pkl exists?     ──────► NO → blocked
                   │  metadata .json exists? ──────► NO → blocked
                   │                               │
                   │  ALL PASS → trading allowed   │
                   └───────────────────────────────┘
```

Если gate не пройден — система может работать, но ордера не отправляются. Это ключевое отличие от систем, которые «просто торгуют».

---

## Текущий статус

> **Архитектура: сильная. Live edge: не доказан.**

Система прошла полный forensic audit (апрель 2026) с исправлением 7 материальных багов и устранением lookahead bias в feature engine. 116 автоматических тестов проходят, включая proof-of-no-lookahead тесты.

| Аспект | Статус |
|--------|--------|
| Архитектура и code quality | Проверена, баги исправлены |
| Lock discipline и concurrency | Проверена, deadlock-free |
| Risk management logic | Проверена, hard caps enforced |
| Execution safety | Multi-layer dedup, fill verify, sync |
| Backtest credibility | Требует re-run после fix lookahead |
| Live statistical edge | **Не доказан** — требует demo validation |
| Model freshness | Требует переобучения после lookahead fix |
| Multi-day stability | **Не проверена** — требует soak test |

**Что это значит:** код готов к запуску в demo-режиме для сбора реальной runtime статистики. Переход на live с реальными деньгами обоснован только после подтверждения edge на demo данных (≥30 сделок, Wilson CI lower bound ≥ 0.50).

---

## Для кого

- **Algo-трейдеры** — как reference architecture для MT5 автоматизации
- **Python-разработчики** — как пример сложной многопоточной системы с FSM, bounded queues, lock discipline
- **Quant-инженеры** — CatBoost pipeline, walk-forward validation, Monte Carlo, Kelly sizing
- **Исследователи MT5 automation** — полный цикл от тика до verified fill с error recovery
- **Трейдеры** — для понимания, как строится серьёзная торговая инфраструктура vs. простой индикаторный бот

---

## Технологический стек

| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.12 |
| Брокер | MetaTrader 5 (Exness) |
| ML | CatBoost 1.2, scikit-learn 1.8 |
| Индикаторы | TA-Lib, pandas-ta |
| Данные | pandas 2.3, NumPy 2.2 |
| Статистика | SciPy 1.17 |
| Контроль | pyTelegramBotAPI 4.29 |
| Concurrency | threading, queue.Queue, RLock |
| Тесты | pytest 9.0 (116 тестов) |
| Платформа | Windows 10/11 (MT5 requirement) |

---

## Установка и запуск

### 1. Зависимости

```bash
pip install -r requirements.txt
```

> TA-Lib требует предварительной установки C-библиотеки. На Windows — скачайте `.whl` с [неофициального репозитория](https://github.com/cgohlke/talib-build/releases).

### 2. Конфигурация

Создайте файл `.env` в корне проекта:

```ini
EXNESS_LOGIN=12345678
EXNESS_PASSWORD=your_password
EXNESS_SERVER=Exness-MT5Real

TG_TOKEN=your_telegram_bot_token
TG_ADMIN_ID=your_telegram_user_id
```

При первом запуске без `.env` система автоматически создаст шаблон.

### 3. Запуск

```bash
# Production mode (требует .env с реальными credentials)
python main.py

# Dry-run mode (без MT5, без реальных ордеров)
DRY_RUN=1 python main.py
```

### 4. Тесты

```bash
python -m pytest tests/ -v
```

### Ключевые переменные окружения

| Переменная | По умолчанию | Описание |
|------------|:------------:|----------|
| `DRY_RUN` | `0` | Симуляция без MT5 |
| `PARTIAL_GATE_MODE` | `1` | Торговля только по assets, прошедшим gate |
| `AUTO_TRAIN_ASSETS` | `XAU,BTC` | Активы для auto-train при старте |
| `BACKTEST_FORCE_RETRAIN` | `0` | Принудительное переобучение |
| `GATE_RETRAIN_COOLDOWN_SEC` | `300` | Cooldown между retrain попытками |

---

## Структура проекта

```
QuantCore/
│
├── main.py                         # Entry point, supervisor, lifecycle
├── mt5_client.py                   # MT5 connection, lock discipline, async dispatch
├── log_config.py                   # Logging setup
│
├── core/                           # Ядро торговой логики
│   ├── config.py                   #   Unified config (XAU/BTC override)
│   ├── feature_engine.py           #   40+ indicators, patterns, shift-aware
│   ├── signal_engine.py            #   100-pt scoring, ML bridge, filters
│   ├── risk_engine.py              #   Kelly sizing, SL/TP, kill switch
│   ├── model_gate.py               #   Quality gate (Sharpe/WR/DD)
│   ├── model_manager.py            #   Model load/save/verify
│   ├── model_retrainer.py          #   Runtime retraining trigger
│   ├── portfolio_risk.py           #   Cross-asset risk
│   └── transaction_costs.py        #   Realistic cost model
│
├── Backtest/                       # Обучение и валидация
│   ├── model_train.py              #   CatBoost training pipeline
│   ├── engine.py                   #   Backtest + WFA + Monte Carlo
│   └── metrics.py                  #   Performance metrics
│
├── Bot/                            # Runtime и контроль
│   ├── bot.py                      #   Telegram handlers
│   ├── bot_utils.py                #   TG utilities, notifications
│   ├── portfolio_engine.py         #   Portfolio orchestrator
│   └── Motor/                      #   Execution engine
│       ├── engine.py               #     FSM orchestrator (2400 LOC)
│       ├── pipeline.py             #     Per-asset signal pipeline
│       ├── execution.py            #     Worker threads
│       ├── execution_manager.py    #     Order lifecycle
│       ├── order_sync_manager.py   #     Broker reconciliation
│       ├── inference_engine.py     #     CatBoost inference
│       ├── scheduler.py            #     Asset scheduling
│       ├── fsm_runtime.py          #     FSM dispatch
│       ├── fsm_services.py         #     FSM step handlers
│       └── fsm_types.py            #     State definitions
│
├── DataFeed/                       # Рыночные данные
│   ├── xau_market_feed.py          #   XAU data + microstructure
│   └── btc_market_feed.py          #   BTC data + regime estimation
│
├── ExnessAPI/                      # Broker API layer
│   ├── order_execution.py          #   Market orders, SL/TP attach
│   ├── functions.py                #   Position management
│   ├── history.py                  #   Trade history
│   └── daily_balance.py            #   Balance tracking
│
├── strategies/                     # Asset stack wrappers
├── tests/                          # 116 automated tests
├── Logs/                           # Runtime logs + exec metrics
└── Artifacts/                      # Models + backtest reports
```

---

## Disclaimer

Эта система — инженерный исследовательский проект. Она **не гарантирует прибыль** и не является финансовым советом.

- Алгоритмическая торговля сопряжена с существенным финансовым риском
- Прошлые результаты бэктеста не гарантируют будущих результатов
- Live deployment требует постоянного мониторинга и понимания рисков
- Система спроектирована с защитой капитала, но ни одна защита не является абсолютной
- Используйте систему на свой страх и риск, начиная с demo-счёта

---

<div align="center">

*Разработано Кабиром Гафуровым*

*Вопросы архитектуры, предложения и замечания — через Issues*

</div>
