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
time_offset = 0
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
    """Sync with Binance server time and apply offset"""
    try:
        base_url = BASE_URL
        response = requests.get(f"{base_url}/fapi/v1/time", timeout=5)
        response.raise_for_status()
        server_time = response.json()["serverTime"]
        local_time = int(time.time() * 1000)
        offset = server_time - local_time
        
        # Apply the offset to time functions for this session
        global time_offset
        time_offset = offset
        
        logging.info(f"🕒 Server time offset: {offset} ms (Server: {server_time}, Local: {local_time})")
        return offset
    except Exception as e:
        logging.error(f"Failed to get server time: {e}")
        return 0

# Update your now_ms() function to use the offset
def now_ms():
    """Get current timestamp with server offset applied"""
    global time_offset
    base_time = int(time.time() * 1000)
    return base_time + (time_offset if 'time_offset' in globals() else 0)

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
            
            # Enhanced logging for expired orders
            if status in ["EXPIRED", "CANCELED", "REJECTED"]:
                logging.warning(f"⚠️ Order {client_order_id or order_id} has terminal status: {status}")
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
        response = client.rest_api.new_order(
            symbol=symbol,
            side=sl_side,
            type="STOP_MARKET",
            stop_price=str(round(stop_price, 1)),
            quantity=round(qty, 3),
            close_position=False,
            reduce_only=True,
            working_type="CONTRACT_PRICE",
            new_client_order_id=f"{symbol}_SL_{ts}",  # This now uses the unique timestamp
            recv_window=10000,
            time_in_force="GTC"
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


def get_current_position(symbol="BTCUSDT", max_retries=5):
    """Get current position with enhanced debugging"""
    for attempt in range(max_retries):
        try:
            response = client.rest_api.position_information_v2()
            positions = response.data()
            
            # Debug: log all positions to see what's happening
            logging.info(f"🔍 Checking all positions (attempt {attempt + 1}):")
            for pos in positions:
                if hasattr(pos, 'symbol'):
                    symbol_name = getattr(pos, 'symbol', 'N/A')
                    pos_amt = float(getattr(pos, 'positionAmt', 0))
                    entry_price = float(getattr(pos, 'entryPrice', 0))
                    leverage = float(getattr(pos, 'leverage', 1))
                    if abs(pos_amt) > 0.0001:  # Only log significant positions
                        logging.info(f"   {symbol_name}: {pos_amt} @ {entry_price}, Leverage: {leverage}")
            
            # Find the symbol's position
            position = next((p for p in positions if getattr(p, "symbol", None) == symbol), None)
            
            if position:
                pos_amt = float(getattr(position, "positionAmt", 0))
                entry_price = float(getattr(position, "entryPrice", 0))
                leverage = float(getattr(position, "leverage", 1))
                
                logging.info(f"📊 {symbol} Position - Amount: {pos_amt}, Entry: {entry_price}, Leverage: {leverage}")
                
                if abs(pos_amt) > 0.0001:
                    logging.info(f"✅ Current position: {pos_amt} for {symbol}")
                    return position, pos_amt
                else:
                    logging.warning(f"⚠️ Position amount is zero or too small for {symbol} (amount: {pos_amt})")
            else:
                logging.warning(f"⚠️ No position found for {symbol} (attempt {attempt + 1}/{max_retries})")
            
            # Wait and retry if not final attempt
            if attempt < max_retries - 1:
                time.sleep(2)
                
        except Exception as e:
            logging.error(f"❌ get_current_position() attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None, 0

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
        position, pos_amt = get_current_position(symbol, max_retries=3)
        
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
                recv_window=10000
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
    Get all orders using official Binance SDK with enhanced time synchronization.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            logging.info(f"📤 Sending request to get all orders for: {symbol} (Attempt {attempt + 1}/{max_retries})")
            
            # Sync time before making the request
            sync_server_time()
            
            # Call the API with parameters to get recent orders
            response = client.rest_api.all_orders(
                symbol=symbol,
                limit=limit,
                start_time=start_time,
                recv_window=10000  # Increased recvWindow for better tolerance
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
            error_msg = str(e)
            logging.warning(f"❌ get_all_orders() attempt {attempt + 1} failed: {error_msg}")
            
            # Check if this is a timestamp error that we should retry
            timestamp_errors = [
                "Timestamp for this request is outside of the recvWindow",
                "recvWindow",
                "timestamp",
                "Server time",
                "time synchron"
            ]
            
            is_timestamp_error = any(error in error_msg for error in timestamp_errors)
            
            if is_timestamp_error and attempt < max_retries - 1:
                logging.info(f"🔄 Timestamp error detected, re-syncing time and retrying in 2 seconds...")
                sync_server_time()  # Re-sync time
                time.sleep(2)  # Wait a bit longer before retry
                continue
            else:
                # If it's not a timestamp error or we've exhausted retries, log and return empty
                logging.error(f"❌ get_all_orders() final error after {attempt + 1} attempts: {e}")
                return []
    
    # This should not be reached, but as a fallback
    return []

def summarize_pnl(entry_price, exit_price, qty, side):
    """Compute and log the profit or loss in USDT and percentage."""
    try:
        # Ensure all inputs are floats
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        qty = float(qty)
        side = str(side).upper()
        
        if side == "BUY":
            pnl = (exit_price - entry_price) * qty
            pct = ((exit_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
        else:
            pnl = (entry_price - exit_price) * qty
            pct = ((entry_price - exit_price) / entry_price * 100) if entry_price != 0 else 0

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
        
    except Exception as e:
        logging.error(f"Error in summarize_pnl: {e}")
        # Return default values to avoid breaking the main loop
        return 0, 0


def execute_trade(symbol="BTCUSDT", qty=0.01):
    TRADE_TS = now_ms()
    logging.info(f"TRADE TIMESTAMP = {TRADE_TS}")
    
    try:
        trend = get_market_trend(symbol)
        if not trend:
            send_telegram(f"❌ <b>Trade Aborted</b>\nSymbol: {symbol}\nReason: Could not determine market trend")
            return

        atr, entry_price = calculate_atr(symbol)
        if not atr or not entry_price:
            send_telegram(f"❌ <b>Trade Aborted</b>\nSymbol: {symbol}\nReason: Could not calculate ATR or entry price")
            return

        # Widen the stop loss to 3×ATR
        tp1, tp2, tp3, sl = compute_targets(entry_price, atr, trend)
        logging.info(f"Adjusted SL = {sl:.2f}")

        side = NewOrderSideEnum["BUY"].value if trend == "BULLISH" else NewOrderSideEnum["SELL"].value

        # --- Market Entry ---
        response = client.rest_api.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            new_client_order_id=f"{symbol}_ENTRY_{TRADE_TS}",
            quantity=qty,
            recv_window=10000
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
            send_telegram(f"❌ <b>Trade Error</b>\nSymbol: {symbol}\nReason: Stop loss placement failed")
            return

        # --- Place Take Profits with unique IDs ---
        tp1_client_id = f"{symbol}_TP1_{TRADE_TS}"
        tp2_client_id = f"{symbol}_TP2_{TRADE_TS}"
        tp3_client_id = f"{symbol}_TP3_{TRADE_TS}"
        place_take_profits(symbol, side, qty, entry_price, atr, TRADE_TS, leverage=20)

        logging.info("Monitoring active orders for SL updates...")
        send_telegram(f"🔍 <b>Monitoring Started</b>\nSymbol: {symbol}\nMonitoring TP and SL orders...")

        tp1_hit = False
        tp2_hit = False
        sl_order_id = sl_order.get("orderId") or sl_order.get("order_id")
        remaining_qty = qty
        
        # Track if we need to replace SL
        sl_needs_replacement = False
        
        while True:
            time.sleep(5)

            # --- Enhanced TP and SL detection using direct order status checks ---
            # Use TRADE_TS as start_time to only get orders from this trade session
            tp1_filled = check_tp_filled_with_retry(symbol, tp1_client_id, start_time=TRADE_TS)
            tp2_filled = check_tp_filled_with_retry(symbol, tp2_client_id, start_time=TRADE_TS)
            tp3_filled = check_tp_filled_with_retry(symbol, tp3_client_id, start_time=TRADE_TS)

            # Update remaining quantity based on filled TPs
            if tp1_filled and not tp1_hit:
                remaining_qty -= qty * 0.5
            if tp2_filled and not tp2_hit:
                remaining_qty -= qty * 0.3
            if tp3_filled:
                remaining_qty = 0

            # Check SL order with enhanced status checking
            sl_filled = False
            sl_expired = False
            sl_data = None
            if sl_order_id:
                sl_data = get_order_status(symbol=symbol, order_id=sl_order_id, start_time=TRADE_TS)
                if sl_data:
                    status = sl_data.get("status")
                    if status == "FILLED":
                        sl_filled = True
                    elif status in ["EXPIRED", "CANCELED", "REJECTED"]:
                        sl_expired = True
                        logging.warning(f"🔄 SL order {sl_order_id} has status: {status}. Needs replacement.")
                        sl_needs_replacement = True

            # --- Handle SL Replacement if Expired/Canceled ---
            if sl_needs_replacement and remaining_qty > 0:
                logging.warning("🔄 Stop Loss expired/canceled! Creating replacement SL...")
                send_telegram(f"🔄 <b>Stop Loss Replacement</b>\nSymbol: {symbol}\nOld Order ID: {sl_order_id}\nReason: Order expired/canceled")
                
                # Determine current stop price - use original SL or recalculate based on current market
                current_sl_price = sl  # Start with original SL
                
                # If TPs have been hit, adjust SL accordingly
                if tp1_hit:
                    current_sl_price = entry_price  # Break-even
                    logging.info("🔄 Replacement SL set to break-even (TP1 hit)")
                elif tp2_hit:
                    current_sl_price = tp1  # TP1 level
                    logging.info("🔄 Replacement SL set to TP1 level (TP2 hit)")
                
                # Place new SL order
                new_sl_order = place_stop_loss(symbol, side, remaining_qty, current_sl_price, f"{TRADE_TS}_replacement")
                if new_sl_order and ("orderId" in new_sl_order or "order_id" in new_sl_order):
                    sl_order = new_sl_order
                    sl_order_id = new_sl_order.get("orderId") or new_sl_order.get("order_id")
                    logging.info(f"✅ Replacement SL placed successfully. New SL order ID: {sl_order_id}")
                    send_telegram(f"✅ <b>Replacement SL Placed</b>\nSymbol: {symbol}\nNew Order ID: {sl_order_id}\nPrice: {current_sl_price:.2f}")
                    sl_needs_replacement = False
                else:
                    logging.error("❌ Failed to place replacement SL! Retrying in next cycle...")
                    # Don't reset sl_needs_replacement so we keep retrying

            # --- Stop Loss Triggered ---
            if sl_filled:
                logging.warning("🚨 Stop Loss triggered! Cancelling all take-profit orders...")
                send_telegram(f"🛑 <b>Stop Loss Triggered - Processing</b>\nSymbol: {symbol}\nOrder ID: {sl_order_id}")
                
                cancel_all_open_tps(symbol)
                close_open_position(symbol, side, TRADE_TS)

                # Get exit price from SL order data
                exit_price = sl
                if sl_data:
                    # Try different possible price fields and ensure they are converted to float
                    try:
                        exit_price = float(sl_data.get("avgPrice", 0)) or float(sl_data.get("stopPrice", 0)) or float(sl_data.get("price", 0)) or sl
                    except (ValueError, TypeError):
                        exit_price = sl
                    if exit_price == 0:  # Fallback if price not available
                        exit_price = sl
                
                # Ensure exit_price is float
                try:
                    exit_price = float(exit_price)
                except (ValueError, TypeError):
                    logging.warning(f"Could not convert exit_price to float, using SL price: {sl}")
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
            # --- TP1 reached → Move SL to Break-Even ---
            if tp1_filled and not tp1_hit:
                logging.info("✅ TP1 hit → Move SL to break-even")
                send_telegram(f"✅ <b>TP1 Hit</b>\nSymbol: {symbol}\nMoving SL to break-even.")
                
                if sl_order_id:
                    # Use local tracking instead of API for remaining quantity
                    local_remaining_qty = qty * 0.5  # After TP1, 50% remains (TP2 30% + TP3 20%)
                    
                    # Still check API to ensure position exists
                    position, api_pos_amt = get_current_position(symbol)
                    
                    if position and abs(api_pos_amt) > 0.0001:
                        # Use the actual API position amount for safety
                        actual_qty = abs(api_pos_amt)
                        logging.info(f"🔄 Moving SL using API position: {actual_qty}")
                        
                        new_sl = move_stop_loss(symbol, sl_order_id, side, actual_qty, entry_price, TRADE_TS)
                        if new_sl:
                            sl_order = new_sl
                            sl_order_id = new_sl.get("order_id") or new_sl.get("orderId")
                            logging.info(f"✅ SL moved to break-even. New SL order ID: {sl_order_id}")
                            send_telegram(f"🔄 <b>SL Updated in System</b>\nNew SL Order ID: {sl_order_id}")
                        else:
                            logging.error("❌ Failed to move SL to break-even")
                    else:
                        # If API shows no position, try with local tracking
                        logging.warning(f"⚠️ API shows no position, using local tracking: {local_remaining_qty}")
                        if local_remaining_qty > 0:
                            new_sl = move_stop_loss(symbol, sl_order_id, side, local_remaining_qty, entry_price, TRADE_TS)
                            if new_sl:
                                sl_order = new_sl
                                sl_order_id = new_sl.get("order_id") or new_sl.get("orderId")
                                logging.info(f"✅ SL moved to break-even (local tracking). New SL order ID: {sl_order_id}")
                                send_telegram(f"🔄 <b>SL Updated in System</b>\nNew SL Order ID: {sl_order_id}")
                            else:
                                logging.error("❌ Failed to move SL to break-even with local tracking")
                        else:
                            logging.error(f"❌ Cannot move SL: Both API and local tracking show no position")
                
                tp1_hit = True

            # --- TP2 reached → Move SL to TP1 ---
            if tp2_filled and not tp2_hit:
                logging.info("✅ TP2 hit → Move SL to TP1")
                send_telegram(f"🏆 <b>TP2 Hit</b>\nSymbol: {symbol}\nMoving SL to TP1.")
                
                if sl_order_id:
                    # Use local tracking instead of API for remaining quantity
                    local_remaining_qty = qty * 0.2  # After TP2, only TP3 remains (20%)
                    
                    # Still check API to ensure position exists
                    position, api_pos_amt = get_current_position(symbol)
                    
                    if position and abs(api_pos_amt) > 0.0001:
                        # Use the actual API position amount for safety
                        actual_qty = abs(api_pos_amt)
                        logging.info(f"🔄 Moving SL using API position: {actual_qty}")
                        
                        new_sl = move_stop_loss(symbol, sl_order_id, side, actual_qty, tp1, TRADE_TS)
                        if new_sl:
                            sl_order = new_sl
                            sl_order_id = new_sl.get("order_id") or new_sl.get("orderId")
                            logging.info(f"✅ SL moved to TP1. New SL order ID: {sl_order_id}")
                            send_telegram(f"🔄 <b>SL Updated in System</b>\nNew SL Order ID: {sl_order_id}")
                        else:
                            logging.error("❌ Failed to move SL to TP1")
                    else:
                        # If API shows no position, try with local tracking
                        logging.warning(f"⚠️ API shows no position, using local tracking: {local_remaining_qty}")
                        if local_remaining_qty > 0:
                            new_sl = move_stop_loss(symbol, sl_order_id, side, local_remaining_qty, tp1, TRADE_TS)
                            if new_sl:
                                sl_order = new_sl
                                sl_order_id = new_sl.get("order_id") or new_sl.get("orderId")
                                logging.info(f"✅ SL moved to TP1 (local tracking). New SL order ID: {sl_order_id}")
                                send_telegram(f"🔄 <b>SL Updated in System</b>\nNew SL Order ID: {sl_order_id}")
                            else:
                                logging.error("❌ Failed to move SL to TP1 with local tracking")
                        else:
                            logging.error(f"❌ Cannot move SL: Both API and local tracking show no position")
                
                tp2_hit = True
            # --- TP3 reached → Close position and cancel SL ---
            if tp3_filled:
                logging.info("🏁 TP3 hit → Close position and cancel SL.")
                send_telegram(f"🎯 <b>TP3 Hit</b>\nSymbol: {symbol}\nOrder ID: {sl_order_id}\nQty: {qty}\nTP3 hit. SL canceled and position closed.")
                
                if sl_order_id:
                    try:
                        client.rest_api.cancel_order(symbol=symbol, order_id=sl_order_id)
                        logging.info(f"✅ SL order {sl_order_id} canceled after TP3.")
                        send_telegram(f"🛑 <b>SL Canceled after TP3</b>\nSymbol: {symbol}\nSL Order ID: {sl_order_id}")
                    except Exception as ce:
                        logging.error(f"⚠️ Error canceling SL after TP3: {ce}")
                        send_telegram(f"⚠️ <b>Error Canceling SL after TP3</b>\nSymbol: {symbol}\nSL Order ID: {sl_order_id}\nError: {ce}")

                close_open_position(symbol, side, TRADE_TS)
                break

    except Exception as e:
        logging.error(f"execute_trade() error: {e}")
        send_telegram(f"❌ <b>Trade Execution Error</b>\nSymbol: {symbol}\nError: {str(e)}")
                
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
