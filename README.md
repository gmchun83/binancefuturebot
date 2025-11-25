**🚀 Binance Futures Breakout Trading Bot**

Automated TP/SL Management • Multi-R Take Profit System • Python 3.11+


A high-precision Binance USDT-M Futures trading bot that uses a breakout strategy with automated:

✔ Entry detection

✔ Stop-loss placement

✔ TP1/TP2/TP3 targets (R-Multiple system)

✔ Dynamic SL trailing logic


**📌 Features**

**🔥 Breakout Trading Logic**

Entry triggered by confirmed price breakout (15m/1h recommended)

Based on reliable swing-structure support & resistance breakouts

🎯 Multi-Target R-Multiple Take Profit System

| Target  | R Multiple | Purpose                                   |
| ------- | ---------- | ----------------------------------------- |
| **TP1** | **1.5R**   | Lock in early profit + reduce risk        |
| **TP2** | **2R**     | Strong reward with balanced safety        |
| **TP3** | **3R**     | Max profit on major breakout continuation |

🛡️ Advanced Stop-Loss Management

SL placed at correct structural support/resistance

After TP1 hit → SL moves to Break-Even (BE)

After TP2 hit → SL moves to TP1 level

After TP3 hit → Position fully closed

This protects gains while maximizing further profit.


**🧠 Trading Flow Diagram**

flowchart TD

    %% ==== STAGES ====
    Start([🟢 Start Bot])
    Breakout{Breakout\nDetected?}
    Entry[📌 Place Entry Order\n— Market or Limit —]
    SL[🛡️ Set Initial Stop-Loss\n(below/above structure)]
    Monitor[📊 Monitor Price Action]

    TP1Hit{TP1\nReached?}
    MoveToBE[🔒 Move SL to\nBreak-Even]

    TP2Hit{TP2\nReached?}
    MoveToTP1[🔒 Move SL to\nTP1 Level]

    TP3Hit{TP3\nReached?}
    ClosePos[🚀 Close\nRemaining Position]

    SLHit{SL\nHit?}
    StopLossExit[❌ Stop-Loss Triggered]

    End([🏁 Trade Completed])

    %% ==== FLOW ====
    Start --> Breakout
    Breakout -->|Yes| Entry
    Breakout -->|No| Breakout

    Entry --> SL
    SL --> Monitor

    %% TP1 Branch
    Monitor -->|Yes| TP1Hit
    TP1Hit --> MoveToBE --> Monitor

    %% TP2 Branch
    Monitor -->|Yes| TP2Hit
    TP2Hit --> MoveToTP1 --> Monitor

    %% TP3 Branch
    Monitor -->|Yes| TP3Hit
    TP3Hit --> ClosePos --> End

    %% Stop Loss
    Monitor -->|SL Hit| SLHit --> StopLossExit --> End


**🧩 Installation**

✔ Requirements

Python 3.11+

Install required packages:

pip install pydantic aiohttp python-dotenv numpy


**🔧 Install Binance USDS-M Futures SDK**

⚠ pip install binance-connector-python DOES NOT include the required derivatives USDS-M futures modules.
You must install it manually:

1. Clone Binance connector repo:
git clone https://github.com/binance/binance-connector-python.git

2. cd binance-connector-python

3. Add USDS Futures module path:
export PYTHONPATH=$PYTHONPATH:$(pwd)/clients/derivatives_trading_usds_futures/src

4. Locate binance_common
find . -type d -name "binance_common"


Usually found at:

clients/binance_common/src/binance_common


Add it to PYTHONPATH:

export PYTHONPATH=$PYTHONPATH:$(pwd)/common/src:$(pwd)/clients/derivatives_trading_usds_futures/src

4. Install the futures package

cd clients/derivatives_trading_usds_futures

pip install .

5. Verify installation
   
python3.11 -c "from binance_sdk_derivatives_trading_usds_futures import derivatives_trading_usds_futures; print('OK')"


**🔑 Environment Setup**

Rename .env.example:

mv .env.example .env


Fill in:

BINANCE_API_KEY=

BINANCE_API_SECRET=

BASE_PATH=https://demo-fapi.binance.com

TELEGRAM_BOT_TOKEN=

TELEGRAM_CHAT_ID=

Note:Use BASE PATH https://fapi.binance.com for live

**▶ How to Run the Bot**

🔁 Loop Mode (Recommended – runs continuously)

python3.11 binancefuturebot.py --loop

🎯 One-Time Mode (Run once & exit)

python3.11 binancefuturebot.py

**🧪 Testing**

To test derivatives futures import:

python3.11 -c "from binance_sdk_derivatives_trading_usds_futures import derivatives_trading_usds_futures; print('OK')"

**📚 Documentation**

Binance Official API Docs

https://developers.binance.com/
