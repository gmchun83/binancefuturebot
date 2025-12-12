# binancefuturev26.py
# Fully corrected trading bot with robust monitoring and clean loop exits.
# TODO: set .env with API_KEY, API_SECRET, BASE_PATH, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

import os, time, requests, logging, numpy as np
import hmac, hashlib, urllib.parse

from dotenv import load_dotenv
load_dotenv()
import argparse

from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
    ConfigurationRestAPI,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    KlineCandlestickDataIntervalEnum,
    NewOrderSideEnum,
)

# ---------- CONFIG ----------
BASE_URL = os.getenv("BASE_PATH")  # TODO: ensure this points to binance futures URL
if not BASE_URL:
    raise ValueError("Missing BASE_PATH in .env file!")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # TODO
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # TODO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/bot.log", mode="a"),
        logging.StreamHandler()
    ]
)

configuration_rest_api = ConfigurationRestAPI(
    api_key=os.getenv("API_KEY", ""),   # TODO
    api_secret=os.getenv("API_SECRET", ""),  # TODO
    base_path=BASE_URL,
)
client = DerivativesTradingUsdsFutures(config_rest_api=configuration_rest_api)

time_offset = 0

# ---------- TELEGRAM ----------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials not set. Skipping alert.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        logging.error(f"Failed to send Telegram: {e}")

# ---------- TIME ----------
def sync_server_time():
    try:
        r = requests.get(f"{BASE_URL}/fapi/v1/time", timeout=5)
        r.raise_for_status()
        server_time = r.json()["serverTime"]
        local_time = int(time.time() * 1000)
        offset = server_time - local_time
        global time_offset
        time_offset = offset
        logging.info(f"🕒 Server time offset: {offset} ms (Server: {server_time}, Local: {local_time})")
        return offset
    except Exception as e:
        logging.error(f"sync_server_time error: {e}")
        return 0

def now_ms():
    base_time = int(time.time() * 1000)
    return base_time + (time_offset if 'time_offset' in globals() else 0)

# ---------- MARKET HELPERS ----------

# ---------- SIGNALS ----------
def get_breakout_signal(symbol="BTCUSDT", interval="1m", lookback=20, atr_period=14, atr_mult_filter=0.6):
    """
    Donchian breakout with simple volatility gate.
    Returns: ("BUY"/"SELL"/None, last_close, atr)
    """
    try:
        resp = client.rest_api.kline_candlestick_data(
            symbol=symbol,
            interval=KlineCandlestickDataIntervalEnum["INTERVAL_1m"].value,
            limit=max(lookback + 50, 80)
        )
        data = resp.data()
        highs = np.array([float(x[2]) for x in data])
        lows  = np.array([float(x[3]) for x in data])
        closes= np.array([float(x[4]) for x in data])

        tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
        atr = float(np.mean(tr[-atr_period:]))
        last_close = float(closes[-1])

        dc_high = float(np.max(highs[-(lookback+1):-1]))
        dc_low  = float(np.min(lows [-(lookback+1):-1]))

        ch_width = dc_high - dc_low
        if ch_width < atr_mult_filter * atr:
            return None, last_close, atr

        if last_close > dc_high:
            return "BUY", last_close, atr
        elif last_close < dc_low:
            return "SELL", last_close, atr
        else:
            return None, last_close, atr
    except Exception as e:
        logging.error(f"get_breakout_signal() error: {e}")
        return None, None, None

# ---------- TARGETS / RISK ----------
def compute_targets_enhanced(entry_price, atr, side, strength=0.5):
    """
    SL = entry ± k*ATR ; TP1/TP2/TP3 = dynamic R
    """
    k = 4.0 if atr and atr > 100 else 3.0
    sl = entry_price - k * atr if side == "BUY" else entry_price + k * atr
    R = abs(entry_price - sl)

    def lerp(a, b, t): return a + (b - a) * t
    m1 = lerp(1.3, 1.7, strength)
    m2 = lerp(2.0, 2.8, strength)
    m3 = lerp(3.0, 4.2, strength)

    if side == "BUY":
        tp1 = entry_price + m1 * R
        tp2 = entry_price + m2 * R
        tp3 = entry_price + m3 * R
    else:
        tp1 = entry_price - m1 * R
        tp2 = entry_price - m2 * R
        tp3 = entry_price - m3 * R

    return tp1, tp2, tp3, sl, R

def calculate_position_size(risk_usdt, entry_price, sl_price):
    risk_per_contract = abs(entry_price - sl_price)
    if risk_per_contract == 0:
        return 0
        
    position_size = risk_usdt / risk_per_contract
    
    # Calculate notional value
    notional_value = position_size * entry_price
    
    # Ensure minimum notional of $100
    min_notional = 100
    if notional_value < min_notional:
        # Calculate required position size for $100 notional
        required_size = min_notional / entry_price
        logging.info(f"📈 Position too small: ${notional_value:.2f}. Increasing to minimum ${min_notional}")
        return required_size
    
    return position_size

def check_balance_details():
    """Correct Testnet balance using RAW /fapi/v2/account"""
    try:
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")

        ts = int(time.time() * 1000)
        query = f"timestamp={ts}"
        signature = hmac.new(
            api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

        url = f"{BASE_URL}/fapi/v2/account?{query}&signature={signature}"
        r = requests.get(url, headers={"X-MBX-APIKEY": api_key}, timeout=10)
        data = r.json()

        # Find USDT asset
        usdt = next((a for a in data.get("assets", []) if a["asset"] == "USDT"), None)

        if not usdt:
            logging.error("❌ No USDT asset found in account response")
            return None

        wallet = float(usdt["walletBalance"])
        available = float(usdt["availableBalance"])
        margin = float(usdt["marginBalance"])

        logging.info(
            f"💰 Testnet USDT — Wallet:{wallet}, Available:{available}, Margin:{margin}"
        )

        return {
            "wallet_balance": wallet,
            "available_balance": available,
            "margin_balance": margin,
            "total_wallet": float(data.get("totalWalletBalance", wallet)),
            "total_margin": float(data.get("totalMarginBalance", margin)),
            "unrealized_pnl": float(data.get("totalUnrealizedProfit", 0)),
        }

    except Exception as e:
        logging.error(f"check_balance_details error: {e}")
        return None


def set_leverage(symbol="BTCUSDT", leverage=20):
    """
    Set leverage for a symbol
    """
    try:
        response = client.rest_api.change_initial_leverage(
            symbol=symbol,
            leverage=leverage
        )
        data = response.data() if callable(getattr(response, "data", None)) else response
        logging.info(f"✅ Leverage set to {leverage}x for {symbol}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to set leverage: {e}")
        send_telegram(f"❌ <b>Leverage Setting Failed</b>\nError: {e}")
        return False

def pre_trade_checks(symbol="BTCUSDT"):
    """
    Perform pre-trade validation checks
    Returns: (bool_success, error_message)
    """
    try:
        # Check if position already exists
        position, amt, _ = get_current_position(symbol)  # Updated to handle 3 values
        if abs(amt) > 0.0001:
            return False, f"Position already exists: {amt}"
        
        # For testnet, be more lenient with balance checks
        balance_info = check_balance_details()
        if balance_info:
            available_balance = balance_info.get('available_balance', 0)
            total_wallet = balance_info.get('total_wallet', 0)
            
            # Very lenient check for testnet
            if total_wallet < 0.1:  # Only fail if less than 10 cents
                return False, f"Very low testnet balance: {total_wallet} USDT"
                
            logging.info(f"✅ Testnet balance sufficient: {available_balance} available, {total_wallet} total")
        else:
            logging.warning("⚠️ Could not fetch balance details, proceeding anyway for testnet")
        
        # Check if symbol is trading
        try:
            r = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=5)
            r.raise_for_status()
            symbols = r.json()["symbols"]
            symbol_info = next((s for s in symbols if s["symbol"] == symbol), None)
            if not symbol_info:
                return False, f"Symbol {symbol} not found"
            if symbol_info.get("status") != "TRADING":
                return False, f"Symbol {symbol} not trading: {symbol_info.get('status')}"
        except Exception as e:
            logging.warning(f"Symbol status check warning: {e}")
        
        return True, "All checks passed"
        
    except Exception as e:
        logging.error(f"pre_trade_checks error: {e}")
        return False, f"Pre-trade check error: {e}"
    

# ---------- ORDERS / STATUS ----------
def get_all_orders(symbol="BTCUSDT", start_time=None, limit=100):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            sync_server_time()
            response = client.rest_api.all_orders(
                symbol=symbol, limit=limit, start_time=start_time, recv_window=10000
            )
            rate_limits = getattr(response, "rate_limits", [])
            logging.info(f"📥 API Rate Limits: {rate_limits}")

            raw = response.data() if hasattr(response, "data") and callable(response.data) else response
            orders = raw if isinstance(raw, list) else []
            formatted = []
            for order in orders:
                if isinstance(order, dict):
                    formatted.append(order)
                elif hasattr(order, "model_dump"):
                    formatted.append(order.model_dump())
                elif hasattr(order, "__dict__"):
                    formatted.append(vars(order))
                else:
                    try:
                        formatted.append(dict(order))
                    except:
                        continue
            logging.info(f"✅ Retrieved {len(formatted)} orders from Binance.")
            return formatted
        except Exception as e:
            logging.warning(f"get_all_orders attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return []

def get_order_status(symbol, order_id=None, client_order_id=None, start_time=None):
    try:
        orders = get_all_orders(symbol, start_time=start_time)
        if not orders:
            logging.warning(f"No orders found for {symbol}")
            return None
        
        target = None
        for o in orders:
            # Handle different response formats
            o_id = o.get("orderId") or o.get("order_id")
            c_id = o.get("clientOrderId") or o.get("client_order_id")
            
            if order_id and (str(o_id) == str(order_id)):
                target = o
                break
            if client_order_id and (c_id == client_order_id):
                target = o
                break
        
        if target:
            status = target.get("status", "")
            cid = target.get("clientOrderId") or target.get("client_order_id")
            logging.info(f"Order status: {status} for {client_order_id or order_id} (ClientOrderId: {cid})")
            return target
        else:
            logging.info(f"Order not found for {client_order_id or order_id}")
            return None
    except Exception as e:
        logging.error(f"get_order_status error: {e}")
        return None

def check_tp_filled(symbol, client_order_id, start_time=None):
    od = get_order_status(symbol=symbol, client_order_id=client_order_id, start_time=start_time)
    if not od:
        return False
    status = od.get("status")
    executed_qty = float(od.get("executedQty", od.get("cumQty", 0) or 0))
    orig_qty = float(od.get("origQty", 0) or od.get("orig_qty", 0) or 0)
    filled = (status == "FILLED" and executed_qty >= orig_qty and orig_qty > 0)
    if filled:
        logging.info(f"✅ TP filled: {client_order_id}")
    return filled

def check_tp_filled_with_retry(symbol, client_order_id, max_retries=3, start_time=None):
    for _ in range(max_retries):
        try:
            return check_tp_filled(symbol, client_order_id, start_time)
        except Exception as e:
            logging.warning(f"check_tp retry error: {e}")
        time.sleep(1)
    return False

def cancel_all_open_tps(symbol="BTCUSDT"):
    try:
        response = client.rest_api.cancel_all_open_orders(symbol=symbol)
        rate_limits = getattr(response, "rate_limits", [])
        logging.info(f"cancel_all_open_orders rate limits: {rate_limits}")
        data = response.data() if callable(getattr(response, "data", None)) else response
        logging.info(f"🛑 cancel_all_open_orders response: {data}")
    except Exception as e:
        logging.error(f"cancel_all_open_tps error: {e}")

def get_current_position(symbol="BTCUSDT", max_retries=5):
    for attempt in range(max_retries):
        try:
            # Use direct API call for better reliability
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            
            ts = int(time.time() * 1000)
            query = f"timestamp={ts}"
            signature = hmac.new(
                api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()
            
            url = f"{BASE_URL}/fapi/v2/positionRisk?{query}&signature={signature}"
            r = requests.get(url, headers={"X-MBX-APIKEY": api_key}, timeout=10)
            positions = r.json()
            
            position = next((p for p in positions if p.get("symbol") == symbol), None)
            if position:
                pos_amt = float(position.get("positionAmt", 0))
                entry_price = float(position.get("entryPrice", 0))
                logging.info(f"📊 {symbol} posAmt={pos_amt}, entry={entry_price}")
                # Return the position dictionary and both values
                return position, pos_amt, entry_price
            else:
                logging.info(f"📊 {symbol} posAmt=0.0, entry=0.0")
                return None, 0.0, 0.0
                
        except Exception as e:
            logging.error(f"get_current_position attempt {attempt+1} error: {e}")
            time.sleep(1)
    
    # Fallback to SDK method
    try:
        resp = client.rest_api.position_information_v2()
        positions = resp.data()
        position = next((p for p in positions if getattr(p, "symbol", None) == symbol), None)
        if position:
            pos_amt = float(getattr(position, "positionAmt", 0))
            entry_price = float(getattr(position, "entryPrice", 0))
            logging.info(f"📊 {symbol} posAmt={pos_amt}, entry={entry_price} (SDK fallback)")
            return position, pos_amt, entry_price
    except Exception as e:
        logging.error(f"SDK fallback also failed: {e}")
    
    return None, 0.0, 0.0

def close_open_position(symbol="BTCUSDT", side=None, ts=None):
    try:
        position, pos_amt, _ = get_current_position(symbol)  # Updated to handle 3 values
        if not position:
            logging.info("No open position found.")
            return
        if pos_amt == 0:
            logging.info("No open position to close.")
            return
        entry_side = "BUY" if pos_amt > 0 else "SELL"
        close_side = "SELL" if entry_side == "BUY" else "BUY"
        qty_to_close = abs(pos_amt)
        logging.warning(f"⚠️ Forcing close: {symbol} {qty_to_close} {close_side}")
        r = client.rest_api.new_order(
            symbol=symbol,
            side=close_side,
            type="MARKET",
            quantity=qty_to_close,
            new_client_order_id=f"{symbol}_CLOSE_{ts}",
            reduce_only=True,
            recv_window=60000
        )
        _ = r.data() if callable(getattr(r,"data",None)) else r
    except Exception as e:
        logging.error(f"close_open_position error: {e}")


# ---------- TARGETS / RISK ----------

# ---------- SL / TPs ----------
def place_stop_loss(symbol, side, qty, stop_price, ts):
    """
    Place a STOP_MARKET order for USDⓈ-M Futures with improved error handling.
    """
    try:
        # Validate quantity
        if qty <= 0:
            logging.error(f"❌ Invalid quantity for SL: {qty}")
            return None
            
        sl_side = "SELL" if side == "BUY" else "BUY"

        try:
            response = requests.get(
                f"{BASE_URL}/fapi/v1/premiumIndex",
                params={"symbol": symbol},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            current_price = float(data.get("markPrice", 0))
        except Exception as e:
            logging.error(f"⚠️ Failed to fetch mark price: {e}")
            current_price = 0.0

        logging.info(f"📈 Current mark price for {symbol}: {current_price}")

        # Enhanced price validation with buffer
        buffer_pct = 0.001  # 0.1% buffer
        buffer_amount = current_price * buffer_pct
        
        if side == "BUY" and stop_price >= (current_price - buffer_amount):
            logging.warning(f"⛔ Skipping SL placement: stop_price {stop_price} too close to current_price {current_price}")
            # Adjust SL to be safely below current price
            stop_price = current_price - buffer_amount
            logging.info(f"🔄 Adjusted SL to: {stop_price:.2f}")
            
        if side == "SELL" and stop_price <= (current_price + buffer_amount):
            logging.warning(f"⛔ Skipping SL placement: stop_price {stop_price} too close to current_price {current_price}")
            # Adjust SL to be safely above current price
            stop_price = current_price + buffer_amount
            logging.info(f"🔄 Adjusted SL to: {stop_price:.2f}")

        # ✅ --- Safe to place stop order ---
        # Use SDK with proper parameter formatting
        response = client.rest_api.new_order(
            symbol=symbol,
            side=sl_side,
            type="STOP_MARKET",
            stop_price=str(round(stop_price, 1)),  # Round to 1 decimal for price
            quantity=round(qty, 3),  # Round to 3 decimals for BTC quantity
            close_position=False,
            reduce_only=True,
            working_type="CONTRACT_PRICE",
            new_client_order_id=f"{symbol}_SL_{ts}",
            recv_window=10000,
            time_in_force="GTC"
        )

        # Process response
        data_method = getattr(response, "data", None)
        data = data_method() if callable(data_method) else response
        
        # Convert to dict if needed
        if not isinstance(data, dict):
            if hasattr(data, "__dict__"):
                data = data.__dict__
            elif hasattr(data, "model_dump"):
                data = data.model_dump()
            else:
                # Try to convert to dict
                try:
                    data = dict(data)
                except:
                    data = {}
        
        logging.info(f"✅ Stop loss placed @ {stop_price}")
        order_id = data.get("order_id") or data.get("orderId", "N/A")
        send_telegram(f"🛑 <b>Stop Loss Placed</b>\n"
              f"Symbol: {symbol}\n"
              f"Client Order ID: {symbol}_SL_{ts}\n"
              f"Price: {stop_price}\n"
              f"Qty: {qty}\n"
              f"Order ID: {order_id}")

        return data

    except Exception as e:
        logging.error(f"place_stop_loss() exception: {e}")
        
        # Add fallback to direct API if SDK fails
        try:
            logging.warning("Trying direct API fallback for stop loss...")
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            
            params = {
                "symbol": symbol,
                "side": sl_side,
                "type": "STOP_MARKET",
                "quantity": round(qty, 3),
                "stopPrice": str(round(stop_price, 1)),
                "reduceOnly": "true",
                "workingType": "CONTRACT_PRICE",
                "newClientOrderId": f"{symbol}_SL_{ts}",
                "recvWindow": 10000,
                "timeInForce": "GTC",
                "timestamp": int(time.time() * 1000)
            }
            
            query_string = urllib.parse.urlencode(params)
            signature = hmac.new(
                api_secret.encode(), query_string.encode(), hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            
            url = f"{BASE_URL}/fapi/v1/order"
            headers = {"X-MBX-APIKEY": api_key}
            response = requests.post(url, headers=headers, data=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logging.info(f"✅ Stop loss placed via direct API @ {stop_price}")
            return data
            
        except Exception as api_error:
            logging.error(f"Direct API fallback also failed: {api_error}")
            send_telegram(f"❌ <b>Stop Loss Failed</b>\nBoth SDK and direct API failed\nError: {str(e)[:100]}")
            return None

# Update the fetch_symbol_filters function to get precision info
def fetch_symbol_filters(symbol: str):
    try:
        r = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=10)
        r.raise_for_status()
        data = r.json()
        info = next(s for s in data["symbols"] if s["symbol"] == symbol)
        
        price_filter = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
        lot_filter = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
        
        tick = float(price_filter["tickSize"])
        step = float(lot_filter["stepSize"])
        
        # Get price precision from tick size
        tick_str = price_filter["tickSize"]
        if 'e-' in tick_str:
            price_precision = int(tick_str.split('e-')[1])
        else:
            price_precision = len(tick_str.split('.')[1]) if '.' in tick_str else 0
        
        # Get quantity precision from step size
        step_str = lot_filter["stepSize"]
        if 'e-' in step_str:
            qty_precision = int(step_str.split('e-')[1])
        else:
            qty_precision = len(step_str.split('.')[1]) if '.' in step_str else 0
        
        return tick, step, price_precision, qty_precision
    except Exception as e:
        logging.warning(f"fetch_symbol_filters fallback, error: {e}")
        return 0.1, 0.001, 1, 3  # BTCUSDT-safe fallback

# Update round_to_step and round_price to use proper formatting
def round_to_step(value: float, step: float, qty_precision: int) -> str:
    """Round quantity to step size and format with correct precision"""
    rounded = (int(value / step)) * step
    # Format with proper precision, removing trailing zeros
    return format(rounded, f".{qty_precision}f").rstrip('0').rstrip('.')

def round_price(value: float, tick: float, price_precision: int) -> str:
    """Round price to tick size and format with correct precision"""
    rounded = (int(value / tick)) * tick
    # Format with proper precision
    return format(rounded, f".{price_precision}f")

# Update the place_take_profits_enhanced function
def place_take_profits_enhanced(symbol, side, qty_total, entry_price, atr, ts, strength=0.5):
    try:
        tp_side = "SELL" if side == "BUY" else "BUY"
        tick, step, price_precision, qty_precision = fetch_symbol_filters(symbol)
        tp1, tp2, tp3, _, _ = compute_targets_enhanced(entry_price, atr, side, strength)

        base = np.array([0.40, 0.30, 0.30])
        strong = np.array([0.30, 0.25, 0.45])
        weights = base + (strong - base) * strength
        w1, w2, w3 = (weights / weights.sum())
        
        # Calculate quantities with proper rounding
        q1_raw = qty_total * w1
        q2_raw = qty_total * w2
        q3_raw = qty_total * w3
        
        # Round quantities with proper precision
        q1 = round_to_step(q1_raw, step, qty_precision)
        q2 = round_to_step(q2_raw, step, qty_precision)
        q3 = round_to_step(q3_raw, step, qty_precision)
        
        # Ensure minimum tradable per TP
        min_qty = step
        q1_float = float(q1)
        q2_float = float(q2)
        q3_float = float(q3)
        
        if q1_float < min_qty:
            q1 = round_to_step(min_qty, step, qty_precision)
        if q2_float < min_qty:
            q2 = round_to_step(min_qty, step, qty_precision)
        if q3_float < min_qty:
            q3 = round_to_step(min_qty, step, qty_precision)

        # Round prices with proper precision
        p1 = round_price(tp1, tick, price_precision)
        p2 = round_price(tp2, tick, price_precision)
        p3 = round_price(tp3, tick, price_precision)
        
        logging.info(f"TP1: price={p1}, qty={q1}")
        logging.info(f"TP2: price={p2}, qty={q2}")
        logging.info(f"TP3: price={p3}, qty={q3}")

        def post_only_limit(price, qty, label):
            try:
                # Convert strings to float for the API (Binance expects string inputs)
                price_float = float(price)
                qty_float = float(qty)
                
                logging.info(f"Placing {label}: price={price_float}, qty={qty_float}")
                
                r = client.rest_api.new_order(
                    symbol=symbol,
                    side=tp_side,
                    type="LIMIT",
                    time_in_force="GTX",  # post-only
                    quantity=qty_float,
                    price=str(price_float),  # Ensure string format
                    reduce_only=True,
                    new_client_order_id=f"{symbol}_{label}_{ts}",
                    recv_window=10000
                )
                return r.data() if callable(getattr(r,"data",None)) else r
            except Exception as e:
                logging.warning(f"{label} GTX rejected ({e}), retry GTC")
                try:
                    r = client.rest_api.new_order(
                        symbol=symbol,
                        side=tp_side,
                        type="LIMIT",
                        time_in_force="GTC",
                        quantity=qty_float,
                        price=str(price_float),
                        reduce_only=True,
                        new_client_order_id=f"{symbol}_{label}_{ts}",
                        recv_window=10000
                    )
                    return r.data() if callable(getattr(r,"data",None)) else r
                except Exception as e2:
                    logging.error(f"{label} GTC also failed: {e2}")
                    raise e2

        placed = []
        orders_to_place = [
            (p1, q1, "TP1"),
            (p2, q2, "TP2"), 
            (p3, q3, "TP3")
        ]
        
        for price, qty, label in orders_to_place:
            if float(qty) <= 0: 
                continue
            try:
                od = post_only_limit(price, qty, label)
                oid = getattr(od, "orderId", None) or getattr(od, "order_id", "N/A")
                logging.info(f"✅ {label} placed @ {price} qty={qty} (orderId={oid})")
                send_telegram(f"🎯 <b>{label}</b>\n{symbol}\nPrice: {price}\nQty: {qty}\nCID: {symbol}_{label}_{ts}")
                placed.append((label, f"{symbol}_{label}_{ts}", price, qty))
            except Exception as e:
                logging.error(f"Failed to place {label}: {e}")
                continue
                
        return placed
    except Exception as e:
        logging.error(f"place_take_profits_enhanced error: {e}")
        return []


# ---------- TRADE MONITOR ----------
def position_is_closed(symbol):
    """
    Return True only if API confirms position size is 0.
    """
    try:
        _, amt, _ = get_current_position(symbol)  # Updated to handle 3 values
        return float(amt) == 0.0
    except Exception as e:
        logging.error(f"position_is_closed error: {e}")
        return False  # Do NOT assume closed on error

# Replace the existing move_stop_loss function with this working version:
def move_stop_loss(symbol, old_order_id, side, qty, new_stop_price, ts):
    """
    Move stop-loss with enhanced fallback mechanisms
    """
    try:
        # Validate quantity first
        if qty <= 0:
            logging.error(f"❌ Cannot move SL: Invalid quantity: {qty}")
            return None

        # Always try to get current position first for safety
        position, pos_amt, _ = get_current_position(symbol, max_retries=3)
        
        # If API returns a valid position, use that quantity instead
        if position and abs(pos_amt) > 0.0001:
            actual_qty = abs(pos_amt)
            logging.info(f"🔄 Using API position quantity: {actual_qty} (instead of passed: {qty})")
            qty = actual_qty
        else:
            logging.warning(f"⚠️ No valid position from API, using passed quantity: {qty}")

        # Keep using /fapi/v1/premiumIndex for mark price
        try:
            response = requests.get(
                f"{BASE_URL}/fapi/v1/premiumIndex",
                params={"symbol": symbol},
                timeout=5
            )
            response.raise_for_status()
            current_price = float(response.json().get("markPrice", 0))
            logging.info(f"📈 Current mark price for {symbol}: {current_price}")
        except Exception as e:
            logging.error(f"⚠️ Failed to fetch mark price: {e}")
            current_price = 0.0

        # Enhanced validation with proper side logic
        buffer_pct = 0.002  # 0.2% buffer
        buffer_amount = current_price * buffer_pct
        
        # FIXED: Correct SL validation logic for SELL positions
        if side == "BUY":  # Long position
            if new_stop_price >= (current_price - buffer_amount):
                logging.warning(f"⛔ Invalid SL (BUY): {new_stop_price} ≥ mark {current_price} (with buffer)")
                adjusted_sl = current_price - buffer_amount
                logging.info(f"🔄 Adjusting SL to: {adjusted_sl:.2f}")
                new_stop_price = adjusted_sl
        else:  # SELL position (Short)
            if new_stop_price <= (current_price + buffer_amount):
                logging.warning(f"⛔ Invalid SL (SELL): {new_stop_price} ≤ mark {current_price} (with buffer)")
                adjusted_sl = current_price + buffer_amount
                logging.info(f"🔄 Adjusting SL to: {adjusted_sl:.2f}")
                new_stop_price = adjusted_sl

        # Generate unique client order ID using current timestamp
        unique_ts = now_ms()
        new_client_order_id = f"{symbol}_SL_{unique_ts}_replacement"

        # Place new stop-loss first
        new_sl = place_stop_loss(symbol, side, qty, new_stop_price, unique_ts)
        
        if new_sl:
            logging.info("✅ New SL placed successfully. Now canceling old SL.")
            try:
                # Cancel old SL order
                cancel_response = client.rest_api.cancel_order(
                    symbol=symbol, 
                    order_id=old_order_id
                )
                logging.info(f"✅ Old SL order {old_order_id} canceled.")
                
                # Get new SL order ID for tracking
                new_sl_order_id = new_sl.get("orderId") or new_sl.get("order_id")
                
                # Send Telegram notification
                send_telegram(
                    f"🛡️ <b>Stop Loss Moved</b>\n"
                    f"Symbol: {symbol}\n"
                    f"From Order: {old_order_id}\n"
                    f"To New SL: {new_stop_price:.2f}\n"
                    f"New Order ID: {new_sl_order_id}\n"
                    f"Position: {qty}\n"
                    f"Reason: TP hit"
                )
                
                return new_sl
            except Exception as ce:
                logging.error(f"⚠️ Error canceling old SL: {ce}")
                # Don't return None here - the new SL is already placed
                send_telegram(f"⚠️ <b>SL Move Warning</b>\n"
                          f"Symbol: {symbol}\n"
                          f"New SL placed but failed to cancel old SL: {ce}")
                return new_sl
        else:
            logging.error("❌ Failed to place new SL, old SL not canceled.")
            send_telegram(f"❌ <b>SL Move Failed</b>\n"
                      f"Symbol: {symbol}\n"
                      f"Failed to place new stop loss")
            return None

    except Exception as e:
        logging.error(f"move_stop_loss() exception: {e}")
        send_telegram(f"❌ <b>SL Move Error</b>\n"
                  f"Symbol: {symbol}\n"
                  f"Error: {e}")
        return None



# Also, update the convert_runner_to_trailing function to handle closing position when TP3 is hit:
def convert_runner_to_trailing(symbol, side, qty_runner, activation_price, ts, callback_rate_pct=1.0):
    """
    Convert remaining position to trailing stop.
    Note: TRAILING_STOP_MARKET might not be available on testnet.
    For testnet, we'll use a standard STOP_MARKET instead.
    """
    try:
        if qty_runner <= 0: 
            logging.info("No remaining position to convert to trailing stop")
            return None
        
        opposite = "SELL" if side == "BUY" else "BUY"
        
        # For testnet, use STOP_MARKET instead of TRAILING_STOP_MARKET
        logging.warning("⚠️ TRAILING_STOP_MARKET not available on testnet, using STOP_MARKET instead")
        
        # Place a regular stop at activation price
        response = client.rest_api.new_order(
            symbol=symbol,
            side=opposite,
            type="STOP_MARKET",
            quantity=round(qty_runner, 3),
            stop_price=str(round(activation_price, 1)),
            reduce_only=True,
            working_type="CONTRACT_PRICE",
            new_client_order_id=f"{symbol}_TRAIL_{ts}",
            recv_window=10000,
            time_in_force="GTC"
        )
        
        if callable(getattr(response, "data", None)):
            data = response.data()
        else:
            data = response
            
        logging.info(f"✅ Trailing stop (simulated) @ {activation_price}")
        return data
        
    except Exception as e:
        logging.error(f"convert_runner_to_trailing error: {e}")
        return None

# Now update the monitor_trade function to properly handle TP1, TP2, and TP3:
def monitor_trade(symbol, side, sl_client_id, tp1_cid, tp2_cid, tp3_cid, ts, entry_price, R, tp1_price, tp2_price, tick, price_precision):
    """
    Cleanly exits when SL or any end condition occurs.
    """
    logging.info("📡 Monitoring trade state...")
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False

    while True:
        time.sleep(3)

        # ---- STOP-LOSS STATUS (using standard order API for testnet) ----
        sl_data = get_order_status(symbol=symbol, client_order_id=sl_client_id)
        if sl_data:
            sl_status = sl_data.get("status", "")
            if sl_status in ["FILLED"]:
                logging.info("🛑 SL FILLED — exiting monitor")
                cancel_all_open_tps(symbol)
                return "SL"
            if sl_status in ["CANCELED", "REJECTED", "EXPIRED"]:
                logging.info(f"⚠️ SL terminal: {sl_status} — exiting monitor")
                cancel_all_open_tps(symbol)
                return "SL"
        else:
            logging.info("⚠️ SL order not found — exiting monitor")
            return "SL"

        # ---- POSITION CLOSED? ----
        if position_is_closed(symbol):
            logging.info("📌 Position size=0 — exiting monitor")
            cancel_all_open_tps(symbol)
            return "TP/SL"

        # ---- TP1 ----
        if not tp1_hit and check_tp_filled_with_retry(symbol, tp1_cid, start_time=ts):
            tp1_hit = True
            logging.info("🎯 TP1 hit — move SL to BE+0.2R")
            
            # Calculate new stop loss: breakeven + 0.2R
            be_plus = entry_price + (0.2 * R if side == "BUY" else -0.2 * R)
            position, pos_amt, _ = get_current_position(symbol)
            
            if position and abs(float(pos_amt)) > 0:
                # Get the current stop loss order ID
                current_sl_data = get_order_status(symbol=symbol, client_order_id=sl_client_id)
                old_order_id = current_sl_data.get("orderId") if current_sl_data else None
                
                if old_order_id:
                    # Move stop loss
                    move_stop_loss(symbol, old_order_id, side, abs(float(pos_amt)), be_plus, now_ms())
                else:
                    logging.error("❌ Could not get old SL order ID to move")
            else:
                logging.error("❌ No valid position found after TP1 hit")

        # ---- TP2 ----
        if not tp2_hit and check_tp_filled_with_retry(symbol, tp2_cid, start_time=ts):
            tp2_hit = True
            logging.info("🎯 TP2 hit — move SL to TP1 and convert runner to trailing")
            
            position, pos_amt, _ = get_current_position(symbol)
            
            if position and abs(float(pos_amt)) > 0:
                # Get the current stop loss order ID
                current_sl_data = get_order_status(symbol=symbol, client_order_id=sl_client_id)
                old_order_id = current_sl_data.get("orderId") if current_sl_data else None
                
                if old_order_id:
                    # Move stop loss to TP1 price
                    move_stop_loss(symbol, old_order_id, side, abs(float(pos_amt)), tp1_price, now_ms())
                    
                    # For testnet, skip trailing stop and just use regular stop
                    # (You can uncomment the line below if you want to use trailing stop simulation)
                    # convert_runner_to_trailing(symbol, side, abs(float(pos_amt)), tp1_price, now_ms())
                    
                    logging.info("⚠️ Skipping trailing stop on testnet, using regular stop at TP1 price")
                else:
                    logging.error("❌ Could not get old SL order ID to move")
            else:
                logging.error("❌ No valid position found after TP2 hit")

        # ---- TP3 ----
        if not tp3_hit and check_tp_filled_with_retry(symbol, tp3_cid, start_time=ts):
            tp3_hit = True
            logging.info("🏆 TP3 hit — closing all remaining position")
            
            # Cancel any open orders (including stop loss)
            cancel_all_open_tps(symbol)
            
            # Close any remaining position
            position, pos_amt, _ = get_current_position(symbol)
            if position and abs(float(pos_amt)) > 0:
                logging.info(f"Closing remaining position: {pos_amt}")
                close_open_position(symbol, side, now_ms())
            
            return "TP3"

        # ---- Optional: if all three, exit ----
        if tp1_hit and tp2_hit and tp3_hit:
            logging.info("🏁 All TPs done — exiting monitor")
            return "ALL_TP"
            
        # ---- Additional exit condition: if position is 0 and TP3 not hit yet ----
        # This handles cases where TP1 and TP2 were filled but position is already 0
        position, pos_amt, _ = get_current_position(symbol)
        if abs(float(pos_amt)) == 0 and (tp1_hit or tp2_hit):
            logging.info("📌 Position already at 0 — exiting monitor")
            cancel_all_open_tps(symbol)
            return "POSITION_CLOSED"
# ---------- EXECUTE ----------
# ---------- ENTRY ORDER VALIDATION ----------
def place_entry_order(symbol, side, qty, ts):
    """
    Place entry order with better testnet handling.
    Returns (success, filled_qty, avg_price, error_message)
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Generate UNIQUE client order ID for each attempt
            unique_ts = now_ms()
            client_order_id = f"{symbol}_ENTRY_{unique_ts}_{attempt}"
            
            logging.info(f"Attempt {attempt+1}/{max_retries}: Placing entry order: {symbol} {side} {qty} (CID: {client_order_id})")
            
            # Try MARKET order (most reliable for testnet)
            logging.info(f"Trying MARKET order for {qty} {symbol}")
            
            # First check current position before placing order - CORRECTED
            position_before, amt_before, entry_price_before = get_current_position(symbol)  # FIXED: Now 3 values
            logging.info(f"Position before order: {amt_before}")
            
            entry = client.rest_api.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=float(qty),
                new_client_order_id=client_order_id,
                recv_window=10000
            )
            
            # Get response data
            if callable(getattr(entry, "data", None)):
                entry_resp = entry.data()
            else:
                entry_resp = entry
            
            logging.info(f"Entry order response: {entry_resp}")
            
            # Extract order ID
            if hasattr(entry_resp, "orderId"):
                order_id = entry_resp.orderId
            elif hasattr(entry_resp, "order_id"):
                order_id = entry_resp.order_id
            elif isinstance(entry_resp, dict) and "orderId" in entry_resp:
                order_id = entry_resp["orderId"]
            else:
                order_id = None
            
            if order_id:
                logging.info(f"✅ Entry MARKET order placed, orderId: {order_id}")
                
                # Wait a bit for testnet to process
                time.sleep(3)
                
                # Try multiple times to check order status (testnet can be delayed)
                for status_check in range(5):
                    order_status = get_order_status(symbol, order_id=order_id)
                    if order_status:
                        status = order_status.get("status", "")
                        executed_qty = float(order_status.get("executedQty", 0) or 
                                           order_status.get("executed_qty", 0) or 
                                           order_status.get("cumQty", 0) or 0)
                        
                        logging.info(f"Order check {status_check+1}: status={status}, executed_qty={executed_qty}")
                        
                        if status == "FILLED":
                            # For testnet: check if executed_qty > 0 OR check position directly
                            if executed_qty > 0:
                                avg_price = float(order_status.get("avgPrice", 0) or 
                                                order_status.get("avg_price", 0) or 0)
                                logging.info(f"✅ Entry order filled: {executed_qty} at avg price {avg_price}")
                                return True, executed_qty, avg_price, None
                            else:
                                # Testnet anomaly: status=FILLED but executed_qty=0
                                # Check position directly
                                logging.warning("Testnet anomaly: FILLED with 0 executed_qty, checking position...")
                                position_after, amt_after, entry_price_after = get_current_position(symbol)  # FIXED: Now 3 values
                                logging.info(f"Position after order: {amt_after}")
                                
                                if abs(float(amt_after)) > abs(float(amt_before)):
                                    # Position changed - order actually filled
                                    logging.info(f"✅ Position confirmed: {amt_after} at {entry_price_after}")
                                    return True, abs(float(amt_after)), entry_price_after, None
                                elif status_check < 4:
                                    # Wait and retry
                                    time.sleep(2)
                                    continue
                                else:
                                    # After all retries, still no position
                                    return False, 0, 0, f"Order marked FILLED but no position found after {status_check+1} checks"
                    else:
                        logging.warning(f"Could not get order status, attempt {status_check+1}")
                    
                    time.sleep(2)  # Wait before next check
                
                # If we get here, order status checks failed
                # Final fallback: check position directly
                position_final, amt_final, entry_price_final = get_current_position(symbol, max_retries=3)  # FIXED: Now 3 values
                if abs(float(amt_final)) > 0.0001:
                    logging.info(f"✅ Final position check: {amt_final} at {entry_price_final}")
                    return True, abs(float(amt_final)), entry_price_final, None
                else:
                    if attempt < max_retries - 1:
                        logging.info(f"Retrying entry order... attempt {attempt+2}")
                        time.sleep(2)
                        continue
                    else:
                        return False, 0, 0, f"Order not filled after {max_retries} attempts"
            else:
                logging.warning(f"No orderId in response: {entry_resp}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        except Exception as e:
            logging.error(f"Entry order attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return False, 0, 0, str(e)
    
    return False, 0, 0, f"Failed after {max_retries} attempts"

def fill_remaining_with_market(symbol, side, remaining_qty, filled_qty, filled_avg_price, ts):
    """Fill remaining quantity with MARKET order"""
    try:
        entry = client.rest_api.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=remaining_qty,
            new_client_order_id=f"{symbol}_ENTRY_MKT_{ts}",
            recv_window=10000
        )
        
        entry_resp = entry.data() if callable(getattr(entry, "data", None)) else entry
        
        if hasattr(entry_resp, "executedQty"):
            market_filled = float(entry_resp.executedQty)
            market_avg = float(getattr(entry_resp, "avgPrice", 0))
            
            total_filled = filled_qty + market_filled
            if market_filled > 0:
                # Calculate weighted average price
                total_avg_price = ((filled_qty * filled_avg_price) + (market_filled * market_avg)) / total_filled
                logging.info(f"✅ MARKET order filled remaining: {market_filled}, total: {total_filled}, avg: {total_avg_price}")
                return True, total_filled, total_avg_price, None
                
        return False, filled_qty, filled_avg_price, "Market order for remaining failed"
    except Exception as e:
        logging.error(f"Market order for remaining failed: {e}")
        return False, filled_qty, filled_avg_price, str(e)

def wait_for_position_update(symbol, expected_qty, max_attempts=10, wait_seconds=1):
    """Wait for position to update after entry"""
    for attempt in range(max_attempts):
        try:
            position, pos_amt, entry_price = get_current_position(symbol)  # Already correct
            if position and abs(float(pos_amt)) >= expected_qty * 0.9:  # Allow 10% tolerance
                logging.info(f"✅ Position confirmed: {pos_amt} at {entry_price}")
                return True, abs(float(pos_amt)), entry_price
            else:
                logging.info(f"⏳ Waiting for position update... attempt {attempt + 1}")
                time.sleep(wait_seconds)
        except Exception as e:
            logging.error(f"Error checking position: {e}")
            time.sleep(wait_seconds)
    
    return False, 0.0, 0.0

# ---------- EXECUTE TRADE WITH BETTER ENTRY HANDLING ----------
def execute_trade(symbol="BTCUSDT", risk_usdt=5.0, leverage=20):
    ts = now_ms()
    logging.info(f"TRADE TIMESTAMP = {ts}")

    # Debug: Check balance details
    balance_info = check_balance_details()
    logging.info(f"💰 Balance Info: {balance_info}")
    send_telegram(f"💰 <b>Balance Info</b>\nWallet: {balance_info.get('wallet_balance', 0):.2f}\nAvailable: {balance_info.get('available_balance', 0):.2f}")

    # Pre-trade checks
    checks_passed, check_message = pre_trade_checks(symbol)
    if not checks_passed:
        logging.error(f"❌ Pre-trade checks failed: {check_message}")
        send_telegram(f"❌ <b>Pre-trade Checks Failed</b>\n{check_message}")
        return

    # Set leverage
    if not set_leverage(symbol, leverage):
        logging.error("Leverage setting failed, aborting trade")
        return

    # Wait for breakout
    signal, last_close, atr = get_breakout_signal(symbol)
    tries = 0
    while not signal or not last_close or not atr:
        logging.info("⏳ No breakout — waiting 5s...")
        time.sleep(5)
        signal, last_close, atr = get_breakout_signal(symbol)
        tries += 1
        if tries % 120 == 0:
            logging.info("⏳ Still waiting for breakout...")
        if tries % 30 == 0:
            checks_passed, check_message = pre_trade_checks(symbol)
            if not checks_passed:
                logging.error(f"❌ Pre-trade checks failed during wait: {check_message}")
                return

    logging.info(f"📌 Breakout Detected: {signal} @ {last_close}")
    send_telegram(f"📌 <b>Breakout</b>\n{symbol}\nSide: {signal}\nPrice: {last_close:.2f}")

    side = NewOrderSideEnum["BUY"].value if signal == "BUY" else NewOrderSideEnum["SELL"].value

    # Initial targets calculation for position sizing
    strength = 0.5
    tp1, tp2, tp3, sl, R = compute_targets_enhanced(last_close, atr, side, strength)

    # Quantity calculation with proper precision
    tick, step, price_precision, qty_precision = fetch_symbol_filters(symbol)
    qty = calculate_position_size(risk_usdt, last_close, sl)
    qty_str = round_to_step(qty, step, qty_precision)
    qty_float = float(qty_str)
    
    # Ensure minimum notional
    notional_value = qty_float * last_close
    min_notional = 100
    if notional_value < min_notional:
        required_qty = min_notional / last_close
        qty_str = round_to_step(required_qty, step, qty_precision)
        qty_float = float(qty_str)
        logging.info(f"📈 Adjusted position size to meet minimum notional: {qty_str}")
        send_telegram(f"📈 <b>Adjusted Position Size</b>\nMinimum $100 notional required\nNew size: {qty_str}")

    # Final pre-entry check
    checks_passed, check_message = pre_trade_checks(symbol)
    if not checks_passed:
        logging.error(f"❌ Final pre-trade checks failed: {check_message}")
        send_telegram(f"❌ <b>Final Check Failed</b>\n{check_message}")
        return

    # Place entry order with validation
    # Place entry order with validation
    logging.info(f"Placing entry order: {signal} qty={qty_str}")
    send_telegram(f"🚀 <b>Placing Entry Order</b>\n{symbol} {signal}\nQuantity: {qty_str}")
    
    success, filled_qty, entry_price, error_msg = place_entry_order(symbol, side, qty_str, ts)
    if not success or filled_qty <= 0:
        logging.error(f"❌ Entry order failed: {error_msg}")
        send_telegram(f"❌ <b>Entry Order Failed</b>\n{error_msg}")
        
        # Clean up any open orders
        try:
            cancel_all_open_tps(symbol)
        except:
            pass
            
        # Check if any position was accidentally opened
        position, pos_amt, entry_price = get_current_position(symbol)  # FIXED: Now 3 values
        if abs(pos_amt) > 0.0001:
            logging.warning(f"⚠️ Found unexpected position: {pos_amt}, closing it")
            close_open_position(symbol, side, ts)
            
        return
    
    # Wait for position to update in the system
    logging.info(f"✅ Entry order filled: {filled_qty} at avg price {entry_price}")
    send_telegram(f"✅ <b>Entry Order Filled</b>\n{symbol}\nSide: {signal}\nQuantity: {filled_qty}\nAvg Price: {entry_price:.2f}")
    
    # Double-check position
    pos_success, actual_qty, actual_entry_price = wait_for_position_update(symbol, filled_qty)
    if not pos_success:
        logging.error("❌ Position not found after entry order filled")
        send_telegram("❌ <b>Position Not Found</b>\nEntry order filled but position not detected")
        
        # Clean up
        cancel_all_open_tps(symbol)
        return
    
    # Use actual values from position
    # Use actual values from position (allow negative for SELL positions)
    if abs(actual_qty) > 0 and actual_entry_price > 0:
        # For SELL positions, actual_qty is negative, but we need positive for calculations
        filled_qty = abs(actual_qty)  # Use absolute value for quantity calculations
        entry_price = actual_entry_price
        logging.info(f"✅ Position verified: {filled_qty} (abs) at {entry_price}")
    else:
        logging.error(f"❌ Invalid position data: qty={actual_qty}, price={actual_entry_price}")
        send_telegram(f"❌ <b>Invalid Position Data</b>\nQty: {actual_qty}\nPrice: {actual_entry_price}")
        cancel_all_open_tps(symbol)
        return

    # Recalculate targets with actual entry price
    tp1, tp2, tp3, sl, R = compute_targets_enhanced(entry_price, atr, side, strength)
    logging.info(f"🎯 Targets recalculated:\nSL: {sl:.2f}\nTP1: {tp1:.2f}\nTP2: {tp2:.2f}\nTP3: {tp3:.2f}\nR: {R:.2f}")

    # Place SL
    sl_cid = f"{symbol}_SL_{ts}"
    sl_order = place_stop_loss(symbol, side, filled_qty, sl, ts)
    if not sl_order:
        logging.error("SL placement failed; aborting trade.")
        send_telegram("❌ SL placement failed.")
        close_open_position(symbol, side, ts)
        return

    # Place TPs
    tps = place_take_profits_enhanced(symbol, side, filled_qty, entry_price, atr, ts, strength)
    if tps:
        tp1_cid = tps[0][1] if len(tps) > 0 else None
        tp2_cid = tps[1][1] if len(tps) > 1 else None  
        tp3_cid = tps[2][1] if len(tps) > 2 else None
        logging.info(f"✅ TP orders placed: TP1={tp1_cid}, TP2={tp2_cid}, TP3={tp3_cid}")
    else:
        logging.error("No TP orders placed")
        tp1_cid, tp2_cid, tp3_cid = None, None, None
        send_telegram("❌ <b>TP Orders Failed</b>\nPlacing TP orders failed")
        close_open_position(symbol, side, ts)
        return

    # Monitor
    send_telegram(f"🔍 <b>Monitoring Trade</b>\nEntry: {entry_price:.2f}\nSL: {sl:.2f}\nTP1: {tp1:.2f}")
    logging.info("📡 Starting trade monitoring...")
    
    # Get tick size for monitor_trade
    tick, step, price_precision, qty_precision = fetch_symbol_filters(symbol)
    
    result = monitor_trade(
        symbol, side, sl_cid, tp1_cid, tp2_cid, tp3_cid, ts,
        entry_price=entry_price, R=R, tp1_price=tp1, tp2_price=tp2, 
        tick=tick, price_precision=price_precision  # Add price_precision
    )

    logging.info(f"🔚 Trade finished: {result}")
    send_telegram(f"🔚 <b>Trade Finished</b>\nResult: {result}")

    # Cleanup
    try:
        cancel_all_open_tps(symbol)
        time.sleep(1)
        close_open_position(symbol, side, now_ms())
    except Exception as e:
        logging.error(f"Cleanup error: {e}")

    logging.info("🔄 Ready for next trade.")

# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Enable continuous trade execution")
    parser.add_argument("--risk", type=float, default=5.0, help="Risk amount in USDT (default: 5.0)")
    parser.add_argument("--delay", type=int, default=10, help="Delay between trades in seconds (default: 10)")
    args = parser.parse_args()

    sync_server_time()

    if args.loop:
        while True:
            try:
                execute_trade("BTCUSDT", args.risk)
                logging.info(f"🔄 Trade cycle complete. Restarting in {args.delay} seconds...")
                send_telegram(f"🔄 <b>Trade Cycle Complete</b>\nRestarting in {args.delay} seconds...")
                time.sleep(args.delay)
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                send_telegram(f"❌ <b>Main Loop Error</b>\nError: {e}")
                time.sleep(30)  # Longer delay on error
    else:
        try:
            execute_trade("BTCUSDT", args.risk)
            logging.info("✅ Single trade execution complete.")
            send_telegram("✅ <b>Single Trade Execution Complete</b>")
        except Exception as e:
            logging.error(f"Single run error: {e}")
            send_telegram(f"❌ <b>Single Run Error</b>\nError: {e}")
