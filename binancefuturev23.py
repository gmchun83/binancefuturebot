import numpy as np

import os, time, hmac, hashlib, requests, logging
from urllib.parse import urlencode
from dotenv import load_dotenv
load_dotenv()
import requests
import argparse
# --- Telegram Alert Helper ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
    ConfigurationRestAPI,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    KlineCandlestickDataIntervalEnum,
    NewOrderSideEnum,
)

# --- Load base URL from .env ---
BASE_URL = os.getenv("BASE_PATH")
if not BASE_URL:
    raise ValueError("Missing BASE_PATH in .env file!")


# Logging configuration
logging.basicConfig(
    level=logging.INFO,  # or INFO if you prefer less detail
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/bot.log", mode='a'),
        logging.StreamHandler()
    ]
)


configuration_rest_api = ConfigurationRestAPI(
    api_key=os.getenv("API_KEY", ""),
    api_secret=os.getenv("API_SECRET", ""),
    base_path=BASE_URL,
)
client = DerivativesTradingUsdsFutures(config_rest_api=configuration_rest_api)

# NEW: Timestamp helper
def now_ms():
    return int(time.time() * 1000)

# --- Helper: Check if Binance order is filled ---
def is_order_filled(order: dict) -> bool:
    """Checks if an order is fully filled based on Binance official schema."""
    return (
        order.get("status") == "FILLED"
        and float(order.get("executedQty", 0)) >= float(order.get("origQty", 0))
    )
# ==========================
#   HELPER FUNCTIONS
# ==========================

def sync_server_time():
    try:
        base_url = BASE_URL
        response = requests.get(f"{base_url}/fapi/v1/time", timeout=5)
        response.raise_for_status()
        server_time = response.json()["serverTime"]
        local_time = int(time.time() * 1000)
        offset = server_time - local_time
        logging.info(f"Server time offset: {offset} ms")
        return offset
    except Exception as e:
        logging.error(f"Failed to get server time: {e}")
        return 0


def calculate_atr(symbol="BTCUSDT", period=14, limit=100):
    try:
        response = client.rest_api.kline_candlestick_data(
            symbol=symbol,
            interval=KlineCandlestickDataIntervalEnum["INTERVAL_1m"].value,
            limit=limit
        )
        data = response.data()
        highs = np.array([float(x[2]) for x in data])
        lows = np.array([float(x[3]) for x in data])
        closes = np.array([float(x[4]) for x in data])

        tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-period:])
        logging.info(f"Calculated ATR({period}) = {atr:.2f}")
        return atr, closes[-1]
    except Exception as e:
        logging.error(f"calculate_atr() error: {e}")
        return None, None



def send_telegram(message: str):
    """Send log or alert messages to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials not set. Skipping alert.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")

def get_market_trend(symbol="BTCUSDT", interval="1m", limit=50):
    try:
        response = client.rest_api.kline_candlestick_data(
            symbol=symbol,
            interval=KlineCandlestickDataIntervalEnum["INTERVAL_1m"].value,
            limit=limit
        )
        data = response.data()
        closes = np.array([float(c[4]) for c in data])
        sma_short = np.mean(closes[-7:])
        sma_long = np.mean(closes[-25:])
        trend = "BULLISH" if sma_short > sma_long else "BEARISH"
        logging.info(f"SMA7={sma_short:.2f}, SMA25={sma_long:.2f}, Trend={trend}")
        return trend
    except Exception as e:
        logging.error(f"get_market_trend() error: {e}")
        return None


# ==========================
#   ORDER PLACEMENT LOGIC
# ==========================

def get_order_status(symbol, order_id=None, client_order_id=None, start_time=None):
    """
    Get specific order status by filtering all_orders() - since get_order doesn't exist in USD-M Futures SDK
    """
    try:
        # Get all orders and filter - use start_time to only get recent orders
        orders = get_all_orders(symbol, start_time=start_time)
        if not orders:
            return None
        
        # Look for the specific order
        target_order = None
        for order in orders:
            # Check by order_id
            if order_id and (str(order.get('orderId')) == str(order_id) or str(order.get('order_id')) == str(order_id)):
                target_order = order
                break
            # Check by client_order_id
            if client_order_id and (order.get('clientOrderId') == client_order_id or order.get('client_order_id') == client_order_id):
                target_order = order
                break
        
        if target_order:
            status = target_order.get('status')
            client_id = target_order.get('clientOrderId') or target_order.get('client_order_id')
            logging.info(f"Order status: {status} for {client_order_id or order_id} (ClientOrderId: {client_id})")
        else:
            logging.info(f"Order not found for {client_order_id or order_id}")
            
        return target_order
        
    except Exception as e:
        logging.error(f"get_order_status() error: {e}")
        return None

def check_tp_filled(symbol, client_order_id, start_time=None):
    """
    Specifically check if a TP order is filled by its clientOrderId
    """
    order_data = get_order_status(symbol=symbol, client_order_id=client_order_id, start_time=start_time)
    if not order_data:
        return False
    
    status = order_data.get("status")
    executed_qty = float(order_data.get("executedQty", 0))
    orig_qty = float(order_data.get("origQty", 0))
    
    # Order is filled if status is FILLED and executed quantity matches original
    is_filled = (status == "FILLED" and executed_qty >= orig_qty)
    
    if is_filled:
        logging.info(f"✅ TP order filled: {client_order_id}")
    
    return is_filled

def check_tp_filled_with_retry(symbol, client_order_id, max_retries=3, start_time=None):
    """Check TP order status with retry logic"""
    for attempt in range(max_retries):
        try:
            result = check_tp_filled(symbol, client_order_id, start_time=start_time)
            return result  # Return whatever result we get, even if None
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(1)  # Brief pause between retries
    return False  # Default to not filled if all retries fail





def compute_targets(entry_price, atr, trend):
    """
    Compute Stop-Loss and Take-Profit levels based on R-multiple breakout system.
    R = |entry - stop_loss|
    TP1 = entry ± 1.5R
    TP2 = entry ± 2R
    TP3 = entry ± 3R
    """
    # Define Stop Loss at ±3×ATR for safety
    # --- Dynamic SL multiplier based on current volatility ---
    multiplier = 4.0 if atr > 100 else 2.5  # Increase buffer for high volatility

    # Define Stop Loss using swing + buffer or entry ± ATR×multiplier
    sl = entry_price - multiplier * atr if trend == "BULLISH" else entry_price + multiplier * atr

    # R = distance between entry and SL
    R = abs(entry_price - sl)

    if trend == "BULLISH":
        tp1 = entry_price + 1.5 * R
        tp2 = entry_price + 2 * R
        tp3 = entry_price + 3 * R
    else:
        tp1 = entry_price - 1.5 * R
        tp2 = entry_price - 2 * R
        tp3 = entry_price - 3 * R

    logging.info(f"TP1={tp1:.2f}, TP2={tp2:.2f}, TP3={tp3:.2f}, SL={sl:.2f}")
    return tp1, tp2, tp3, sl



def place_stop_loss(symbol, side, qty, stop_price, ts):
    """
    Place a STOP_MARKET order for USDⓈ-M Futures (with safety check to avoid immediate trigger).
    """
    try:
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


        if side == "BUY" and stop_price >= current_price:
            logging.warning(f"⛔ Skipping SL placement: stop_price {stop_price} ≥ current_price {current_price}")
            return None
        if side == "SELL" and stop_price <= current_price:
            logging.warning(f"⛔ Skipping SL placement: stop_price {stop_price} ≤ current_price {current_price}")
            return None

        # ✅ --- Safe to place stop order ---
        response = client.rest_api.new_order(
            symbol=symbol,
            side=sl_side,
            type="STOP_MARKET",
            stop_price=str(round(stop_price, 1)),
            quantity=qty,
            close_position=False,
            reduce_only=False,
            working_type="CONTRACT_PRICE",
            new_client_order_id=f"{symbol}_SL_{ts}",
            recv_window=60000
        )

        data_method = getattr(response, "data", None)
        data = data_method() if callable(data_method) else response
        if not isinstance(data, dict):
            data = dict(data) if data else {}

        logging.info(f"Raw stop-loss response: {data}")
        logging.info(f"Stop loss placed @ {stop_price}")
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
        return None



def move_stop_loss(symbol, old_order_id, side, qty, new_stop_price, ts):
    """
    Move stop-loss: place new SL first, then cancel old SL for safety.
    """
    try:
        # Fetch current mark price
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

    # Validation: avoid immediate stop-loss trigger
    if side == "BUY" and new_stop_price >= current_price:
        logging.warning(f"⛔ Invalid SL (BUY): {new_stop_price} ≥ mark {current_price}")
        return None
    if side == "SELL" and new_stop_price <= current_price:
        logging.warning(f"⛔ Invalid SL (SELL): {new_stop_price} ≤ mark {current_price}")
        return None

    # Place new stop-loss first
    # Fetch current open position
    response = client.rest_api.position_information_v2()
    positions = response.data()
    position = next((p for p in positions if getattr(p, "symbol", None) == symbol), None)
    pos_amt = abs(float(getattr(position, "positionAmt", 0))) if position else qty

    new_sl = place_stop_loss(symbol, side, pos_amt, new_stop_price)

    if new_sl:
        logging.info("✅ New SL placed successfully. Now canceling old SL.")
        try:
            client.rest_api.cancel_order(symbol=symbol, order_id=old_order_id)
        except Exception as ce:
            logging.error(f"⚠️ Error canceling old SL: {ce}")
        return new_sl
    else:
        logging.error("❌ Failed to place new SL, old SL not canceled.")
        return None


def cancel_all_open_tps(symbol="BTCUSDT"):
    """
    Cancel all open take-profit orders (LIMIT or TRAILING_STOP_MARKET)
    when Stop Loss is triggered, using Binance REST API `cancel_all_open_orders`.
    """
    try:
        # Directly call Binance Futures API to cancel all open orders for this symbol
        response = client.rest_api.cancel_all_open_orders(symbol=symbol)

        # Log rate limits and response
        rate_limits = getattr(response, "rate_limits", [])
        logging.info(f"cancel_all_open_orders() rate limits: {rate_limits}")

        # Extract response data
        data = response.data() if callable(response.data) else response
        logging.info(f"🛑 cancel_all_open_orders() response: {data}")

        logging.info(f"✅ All open TP orders cancelled for {symbol} after SL triggered.")

    except Exception as e:
        logging.error(f"cancel_all_open_tps() error: {e}")


def close_open_position(symbol="BTCUSDT", side=None, ts=None):
    """
    Force-close any remaining open position after SL or TP triggers.
    Uses MARKET order in the opposite direction of the open side.
    Fully compatible with Binance official connector (PositionInformationV2Response).
    """
    try:
        # ✅ Retrieve current positions from Binance Futures REST API
        response = client.rest_api.position_information_v2()
        rate_limits = getattr(response, "rate_limits", [])
        logging.info(f"position_information_v2() rate limits: {rate_limits}")

        # Extract actual data list
        positions = response.data()
        if not positions or not isinstance(positions, list):
            logging.warning("No valid position data returned.")
            return

        # ✅ Find the symbol’s position entry
        position = next((p for p in positions if getattr(p, "symbol", None) == symbol), None)
        if not position:
            logging.info(f"No open position found for {symbol}.")
            return

        pos_amt = float(getattr(position, "positionAmt", 0))
        if pos_amt == 0:
            logging.info(f"No open position to close for {symbol}. PositionAmt={pos_amt}")
            return

        # Determine which side to close
        entry_side = "BUY" if pos_amt > 0 else "SELL"
        close_side = "SELL" if entry_side == "BUY" else "BUY"
        qty_to_close = abs(pos_amt)

        logging.warning(f"⚠️ Forcing position close: {symbol} {qty_to_close} {close_side}")

        # ✅ Execute a market close order
        close_response = client.rest_api.new_order(
            symbol=symbol,
            side=close_side,
            type="MARKET",
            quantity=qty_to_close,
            new_client_order_id=f"{symbol}_CLOSE_{ts}",
            reduce_only=True,
            recv_window=60000
        )

        order_data = close_response.data() if callable(close_response.data) else close_response
        order_id = getattr(order_data, "order_id", None) or getattr(order_data, "orderId", "N/A")

        logging.info(f"✅ Position closed successfully for {symbol} (Order ID: {order_id})")

        # ✅ Send Telegram summary
        send_telegram(
            f"⚡ <b>Position Closed</b>\n"
            f"Symbol: {symbol}\n"
             f"Client Order ID: {symbol}_CLOSE_{ts}\n"
            f"Side: {close_side}\n"
            f"Qty: {qty_to_close}\n"
            f"Order ID: {order_id}"
        )

    except Exception as e:
        logging.error(f"close_open_position() error: {e}")



def place_take_profits(symbol, side, qty, entry_price, atr, ts, leverage=20):
    """
    Places 3 take-profit orders based on R-multiple system.
    TP1 = 1.5R, TP2 = 2R, TP3 = 3R
    """
    try:
        tp_side = "SELL" if side == "BUY" else "BUY"
        trend = "BULLISH" if side == "BUY" else "BEARISH"

        # Compute R-multiple targets
        tp1, tp2, tp3, _ = compute_targets(entry_price, atr, trend)

        tps = [
            (tp1, qty * 0.5, "TP1"),
            (tp2, qty * 0.3, "TP2"),
            (tp3, qty * 0.2, "TP3"),
        ]

        for price, amount, label in tps:
            response = client.rest_api.new_order(
                symbol=symbol,
                side=tp_side,
                type="LIMIT",
                time_in_force="GTC",
                quantity=amount,
                price=str(round(price, 1)),
                reduce_only=True,
                new_client_order_id=f"{symbol}_{label}_{ts}",
                recv_window=60000
            )

            # ✅ Official response is a typed model (not dict)
            if hasattr(response, "data") and callable(response.data):
                order_obj = response.data()
            else:
                order_obj = response

            # Safely extract order_id and other fields
            order_id = getattr(order_obj, "orderId", None) or getattr(order_obj, "order_id", "N/A")
            avg_price = getattr(order_obj, "avgPrice", None) or getattr(order_obj, "price", None)

            logging.info(f"{label} placed @ {price:.2f} (Order ID: {order_id})")

            send_telegram(
                f"🎯 <b>{label}</b> placed\n"
                f"Symbol: {symbol}\n"
                f"Client Order ID: {symbol}_{label}_{ts}\n"
                f"Price: {price:.2f}\n"
                f"Qty: {amount}\n"
                f"Order ID: {order_id}\n"
                f"Avg Price: {avg_price}"
            )


    except Exception as e:
        logging.error(f"place_take_profits() error: {e}")


# ==========================
#   MAIN BOT EXECUTION
# ==========================
def get_all_orders(symbol="BTCUSDT", start_time=None, limit=100):
    """
    Get all orders using official Binance SDK.
    Enhanced to handle different response formats.
    """
    try:
        logging.info(f"📤 Sending request to get all orders for: {symbol}")
        
        # Call the API with parameters to get recent orders
        response = client.rest_api.all_orders(
            symbol=symbol,
            limit=limit,
            start_time=start_time
        )

        rate_limits = getattr(response, "rate_limits", [])
        logging.info(f"📥 API Rate Limits: {rate_limits}")

        # Extract data using the data() method
        if hasattr(response, "data") and callable(response.data):
            raw_data = response.data()
        else:
            raw_data = response

        # Debug log
        logging.debug(f"🔍 Raw Response Type: {type(raw_data)}")
        
        # Handle different response formats - this is the key fix
        orders = []
        
        if isinstance(raw_data, list):
            # Direct list of orders (most common case)
            orders = raw_data
            logging.debug(f"Got list of {len(orders)} orders")
        else:
            # Try to extract data from response object
            logging.debug(f"Raw data structure: {dir(raw_data)}")
            
            # If it has a data attribute, use that
            if hasattr(raw_data, 'data'):
                data_attr = getattr(raw_data, 'data')
                if callable(data_attr):
                    orders = data_attr()
                else:
                    orders = data_attr
                logging.debug(f"Extracted {len(orders)} orders from data attribute")
            
            # If it's a single order object, wrap in list
            elif hasattr(raw_data, 'orderId') or hasattr(raw_data, 'order_id'):
                orders = [raw_data]
                logging.debug("Wrapped single order in list")
            
            # Last resort: try to convert to list
            else:
                try:
                    orders = list(raw_data)
                    logging.debug(f"Converted to list: {len(orders)} orders")
                except:
                    orders = []
                    logging.warning("Could not convert response to orders list")

        # Convert all orders to dictionaries for consistent handling
        formatted_orders = []
        for order in orders:
            if isinstance(order, dict):
                formatted_orders.append(order)
            elif hasattr(order, 'model_dump'):
                # Pydantic model
                formatted_orders.append(order.model_dump())
            elif hasattr(order, '__dict__'):
                # Regular object
                formatted_orders.append(vars(order))
            else:
                # Try to convert to dict
                try:
                    formatted_orders.append(dict(order))
                except:
                    logging.warning(f"Could not convert order to dict: {order}")
                    continue

        # Log some order details for debugging
        logging.info(f"✅ Retrieved {len(formatted_orders)} orders from Binance.")
        
        # Log all clientOrderIds to see what we actually got
        client_order_ids = []
        for order in formatted_orders[:10]:  # Log first 10 to avoid spam
            client_order_id = order.get('clientOrderId') or order.get('client_order_id')
            status = order.get('status', 'N/A')
            order_id = order.get('orderId') or order.get('order_id')
            if client_order_id:
                client_order_ids.append(client_order_id)
                logging.debug(f"🧾 Order: {order_id} - ClientOrderId: {client_order_id} - Status: {status}")
            else:
                logging.debug(f"🧾 Order: {order_id} - No ClientOrderId - Status: {status}")
        
        if client_order_ids:
            logging.debug(f"Found ClientOrderIds: {client_order_ids}")

        return formatted_orders

    except Exception as e:
        logging.error(f"❌ get_all_orders() error: {e}")
        return []

def summarize_pnl(entry_price, exit_price, qty, side):
    """Compute and log the profit or loss in USDT and percentage."""
    if side.upper() == "BUY":
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty

    pct = ((exit_price - entry_price) / entry_price * 100
           if side.upper() == "BUY"
           else (entry_price - exit_price) / entry_price * 100)

    logging.info(f"📊 Trade Summary:")
    logging.info(f"    Entry Price: {entry_price:.2f}")
    logging.info(f"    Exit Price : {exit_price:.2f}")
    logging.info(f"    Quantity   : {qty}")
    logging.info(f"    Side       : {side}")
    logging.info(f"    P&L (USDT) : {pnl:.4f}")
    logging.info(f"    P&L (%)    : {pct:.2f}%")
    send_telegram(f"📊 <b>Trade Summary</b>\n"
              f"Entry: {entry_price:.2f}\n"
              f"Exit: {exit_price:.2f}\n"
              f"Qty: {qty}\n"
              f"Side: {side}\n"
              f"P&L: {pnl:.4f} USDT ({pct:.2f}%)")


    return pnl, pct


def execute_trade(symbol="BTCUSDT", qty=0.01):
    TRADE_TS = now_ms()
    logging.info(f"TRADE TIMESTAMP = {TRADE_TS}")
    trend = get_market_trend(symbol)
    if not trend:
        return

    atr, entry_price = calculate_atr(symbol)
    if not atr or not entry_price:
        return

    # Widen the stop loss to 3×ATR
    tp1, tp2, tp3, sl = compute_targets(entry_price, atr, trend)
    logging.info(f"Adjusted SL = {sl:.2f}")

    side = NewOrderSideEnum["BUY"].value if trend == "BULLISH" else NewOrderSideEnum["SELL"].value

    try:
        # --- Market Entry ---
        response = client.rest_api.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            new_client_order_id=f"{symbol}_ENTRY_{TRADE_TS}",
            quantity=qty,
            recv_window=60000
        )
        logging.info(f"Main order executed: {trend} @ ~{entry_price}")

        send_telegram(f"🚀 <b>New {trend} Trade</b>\n"
              f"Symbol: {symbol}\n"
              f"Client Order ID: {symbol}_ENTRY_{TRADE_TS}\n"
              f"Entry: {entry_price:.2f}\n"
              f"Qty: {qty}\n"
              f"Side: {side}")

        # --- Initial SL ---
        sl_order = place_stop_loss(symbol, side, qty, sl, TRADE_TS)
        if not sl_order or ("orderId" not in sl_order and "order_id" not in sl_order):
            logging.error(f"Stop loss placement failed or invalid response: {sl_order}")
            return

        # --- Place Take Profits with unique IDs ---
        tp1_client_id = f"{symbol}_TP1_{TRADE_TS}"
        tp2_client_id = f"{symbol}_TP2_{TRADE_TS}"
        tp3_client_id = f"{symbol}_TP3_{TRADE_TS}"
        place_take_profits(symbol, side, qty, entry_price, atr, TRADE_TS, leverage=20)

        logging.info("Monitoring active orders for SL updates...")

        tp1_hit = False
        tp2_hit = False
        sl_order_id = sl_order.get("orderId") or sl_order.get("order_id")

        while True:
            time.sleep(5)

            # --- Enhanced TP and SL detection using direct order status checks ---
            # Use TRADE_TS as start_time to only get orders from this trade session
            tp1_filled = check_tp_filled_with_retry(symbol, tp1_client_id, start_time=TRADE_TS)
            tp2_filled = check_tp_filled_with_retry(symbol, tp2_client_id, start_time=TRADE_TS)
            tp3_filled = check_tp_filled_with_retry(symbol, tp3_client_id, start_time=TRADE_TS)

            # Check SL order
            sl_filled = False
            sl_data = None
            if sl_order_id:
                sl_data = get_order_status(symbol=symbol, order_id=sl_order_id, start_time=TRADE_TS)
                if sl_data and sl_data.get("status") == "FILLED":
                    sl_filled = True

            # --- Stop Loss Triggered ---
            if sl_filled:
                logging.warning("🚨 Stop Loss triggered! Cancelling all take-profit orders...")
                cancel_all_open_tps(symbol)
                close_open_position(symbol, side, TRADE_TS)

                # Get exit price from SL order data
                exit_price = sl
                if sl_data:
                    # Try different possible price fields
                    exit_price = (
                        sl_data.get("avgPrice") or 
                        sl_data.get("stopPrice") or 
                        sl_data.get("price") or 
                        sl
                    )
                    if exit_price == 0 or exit_price is None:  # Fallback if price not available
                        exit_price = sl
                        
                pnl, pct = summarize_pnl(entry_price, exit_price, qty, side)
                send_telegram(f"🚨 <b>STOP LOSS Triggered</b>\n"
                            f"Symbol: {symbol}\n"
                            f"Order ID: {sl_order_id}\n"
                            f"Exit: {exit_price:.2f}\n"
                            f"Qty: {qty}\n"
                            f"P&L: {pnl:.4f} USDT ({pct:.2f}%)")
                break

            # --- TP1 reached → Move SL to Break-Even ---
            if tp1_filled and not tp1_hit:
                logging.info("✅ TP1 hit → Move SL to break-even")
                send_telegram(f"✅ <b>TP1 Hit</b>\nSymbol: {symbol}\nOrder ID: {sl_order_id}\nQty: {qty}\nSL moved to break-even.")
                if sl_order_id:
                    new_sl = move_stop_loss(symbol, sl_order_id, side, qty, entry_price, TRADE_TS)
                    if new_sl:
                        sl_order = new_sl
                        sl_order_id = new_sl.get("order_id") or new_sl.get("orderId")
                        logging.info(f"✅ SL moved to break-even. New SL order ID: {sl_order_id}")
                        send_telegram(f"🛡️ <b>SL Moved to Break-Even</b>\nSymbol: {symbol}\nNew SL Order ID: {sl_order_id}" )
                tp1_hit = True

            # --- TP2 reached → Move SL to TP1 ---
            if tp2_filled and not tp2_hit:
                logging.info("✅ TP2 hit → Move SL to TP1")
                send_telegram(f"🏆 <b>TP2 Hit</b>\nSymbol: {symbol}\nOrder ID: {sl_order_id}\nQty: {qty}\nSL moved to TP1.")
                if sl_order_id:
                    new_sl = move_stop_loss(symbol, sl_order_id, side, qty, tp1, TRADE_TS)
                    if new_sl:
                        sl_order = new_sl
                        sl_order_id = new_sl.get("order_id") or new_sl.get("orderId")
                        logging.info(f"✅ SL moved to TP1. New SL order ID: {sl_order_id}")
                        send_telegram(f"🛡️ <b>SL Moved to TP1</b>\nSymbol: {symbol}\nNew SL Order ID: {sl_order_id}" )
                tp2_hit = True

            # --- TP3 reached → Close position and cancel SL ---
            if tp3_filled:
                logging.info("🏁 TP3 hit → Close position and cancel SL.")
                send_telegram(f"🎯 <b>TP3 Hit</b>\nSymbol: {symbol}\nOrder ID: {sl_order_id}\nQty: {qty}\nTP3 hit. SL canceled and position closed.")
                
                if sl_order_id:
                    try:
                        client.rest_api.cancel_order(symbol=symbol, order_id=sl_order_id)
                        logging.info(f"✅ SL order {sl_order_id} canceled after TP3.")
                        send_telegram(f"🛑 <b>SL Canceled after TP3</b>\nSymbol: {symbol}\nSL Order ID: {sl_order_id}" )
                    except Exception as ce:
                        logging.error(f"⚠️ Error canceling SL after TP3: {ce}")
                        send_telegram(f"⚠️ <b>Error Canceling SL after TP3</b>\nSymbol: {symbol}\nSL Order ID: {sl_order_id}\nError: {ce}" )

                close_open_position(symbol, side, TRADE_TS)
                break

    except Exception as e:
        logging.error(f"execute_trade() error: {e}")
        send_telegram(f"❌ <b>Trade Execution Error</b>\nSymbol: {symbol}\nError: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Enable continuous trade execution")
    args = parser.parse_args()

    sync_server_time()

    if args.loop:
        while True:
            try:
                execute_trade("BTCUSDT", 0.01)
                logging.info("🔄 Trade cycle complete. Restarting in 5 seconds...")
                send_telegram("🔄 <b>Trade Cycle Complete</b>\nRestarting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                send_telegram(f"❌ <b>Main Loop Error</b>\nError: {e}")
                time.sleep(10)
    else:
        try:
            execute_trade("BTCUSDT", 0.01)
            logging.info("✅ Single trade execution complete.")
            send_telegram("✅ <b>Single Trade Execution Complete</b>")
        except Exception as e:
            logging.error(f"Single run error: {e}")
            send_telegram(f"❌ <b>Single Run Error</b>\nError: {e}")
