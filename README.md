**Binance future bot**

Using outbreak strategic for tp and sl

[Start Bot]

- Entry on confirmed breakout (e.g., 15m or 1h chart)
- Risk defined: (Entry - SL) = R

[Create SL]

- Stop-loss = below/above breakout base (support or resistance)

[Create TP]

Target R-Multiple Purpose

- TP1 1.5R Lock in partial profit, reduce risk

- TP2 2R Solid return with good risk-reward

- TP3 3R Maximize profit on strong breakout moves


[SL Management]

- After TP1, move SL to break-even

- After TP2, move SL to TP1

- After TP3,close position 


**Documentation**

https://developers.binance.com/

**Requirement**
- Python 3.11 above 
- pip install pydantic 
- pip install aiohttp 
- pip install python-dotenv 
- pip install numpy

**Install manually to include derivatives usds margined futures library**

- git clone https://github.com/binance/binance-connector-python.git 
- cd binance-connector-python 
- export PYTHONPATH=$PYTHONPATH:$(pwd)/clients/derivatives_trading_usds_futures/src
- 
Note:pip install binance-connector-python wont include derivatives usds margined futures library

**Find Binance Common**

- find . -type d -name "binance_common" 
- Expected output (based on Binance’s repo structure): 
- clients/binance_common/src/binance_common 
- export 
PYTHONPATH=$PYTHONPATH:$(pwd)/common/src:$(pwd)/clients/derivatives_trading_usd
 s_futures/src 

**Then go back folder to binance-connector-python**

- cd clients/derivatives_trading_usds_futures 
- pip install .

**Testing derivatives trading usds futures**

- python3.11 -c "from binance_sdk_derivatives_trading_usds_futures import derivatives_trading_usds_futures; print('OK')"

**How to run**

Rename .env.example to .ev then keyin binance api key and telegam api token

  For Looping
  - python3.11 binancefuturebot.py --loop
 
  For Run Once
  - python3.11 binancefuturebot.py
