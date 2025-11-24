**Binance future bot**
**Documentation**
https://developers.binance.com/

**Requirement  **
- Python 3.11 above 
- pip install pydantic 
- pip install aiohttp 
- pip install python-dotenv 
- pip install numpy

**Install manuall to include derivatives usds margined futures libray**

- git clone https://github.com/binance/binance-connector-python.git 
- cd binance-connector-python 
- export PYTHONPATH=$PYTHONPATH:$(pwd)/clients/derivatives_trading_usds_futures/src

**Find Binance Common **

- find . -type d -name "binance_common" 
- Expected output (based on Binance’s repo structure): 
- clients/binance_common/src/binance_common 
- export 
PYTHONPATH=$PYTHONPATH:$(pwd)/common/src:$(pwd)/clients/derivatives_trading_usd
 s_futures/src 

Then go back folder to binance-connector-python

- cd clients/derivatives_trading_usds_futures 
- pip install .

**Testing**

- python3.11 -c "from binance_sdk_derivatives_trading_usds_futures import derivatives_trading_usds_futures; print('OK')"
