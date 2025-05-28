import inspect
import json
from typing import Any, Dict, List, get_type_hints
from datetime import date, timedelta

import ta  # Importing the technical analysis library
from semantic_kernel.functions import kernel_function
from models.messages_kernel import AgentType

from helpers.fmputils import *
from helpers.yfutils import *

class TechnicalAnalysisTools:

    agent_name = AgentType.TECHNICAL.value

    # Define Company Analyst tools (functions)
    @staticmethod
    @kernel_function(description="Perform multiple technical analysis strategies using the ta library. "
                "Calculates EMA crossover, RSI, MACD (with zero-line checks), Bollinger Bands, "
                "Stochastics, ATR, ADX, and detects basic candlestick patterns. "
                "Returns JSON with analysis and a naive 'overall_rating'.")
    async def run_enhanced_technical_analysis(ticker_symbol: str) -> Dict[str, Any]:
        """
        1) Download OHLC data for the given ticker (6mo daily)
        2) Compute multiple indicators using 'ta' (EMA, RSI, MACD, Bollinger, Stochastics, ATR, ADX, etc.)
        3) Use TA-Lib to detect multiple candlestick patterns (expanded set)
        4) Fetch fundamental data (naive from yfinance) and get a simple news sentiment score
        5) Aggregate everything (technical signals, candlestick patterns, fundamentals, sentiment)
        into a final rating (buy/hold/sell) + probability
        Returns a JSON-serializable dict.
        """

        # --------------------
        # 1. Fetch Price Data
        # --------------------
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = yfUtils.get_stock_data(ticker_symbol, start_date, end_date)

            # If no data, return an empty result
        if df.empty:
            return {
                "ticker_symbol": ticker_symbol,
                "error": "No data found",
                "analysis": {}
            }

        # Ensure df is in ascending date order
        df.sort_index(ascending=True, inplace=True)

        # Prepare the result structure
        analysis_result = {
            "ticker_symbol": ticker_symbol,
            "candlestick_patterns": {},
            "indicators": {},
            "fundamentals": {},
            "news_sentiment": 0.0,
            "final_decision": {
                "score": 0,
                "max_score_possible": 0,
                "probability": 0.0,
                "rating": "hold"
            }
        }

        # -------------------------------------------------------------
        # 2. Calculate Technical Indicators (ta library)
        # -------------------------------------------------------------
        # A) EMA Cross
        short_window = 12
        long_window = 26
        df["EMA_Short"] = ta.trend.EMAIndicator(close=df["Close"], window=short_window).ema_indicator()
        df["EMA_Long"] = ta.trend.EMAIndicator(close=df["Close"], window=long_window).ema_indicator()

        # B) RSI
        df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

        # C) MACD
        macd_indicator = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd_indicator.macd()
        df["MACD_Signal"] = macd_indicator.macd_signal()
        df["MACD_Hist"] = macd_indicator.macd_diff()

        # D) Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BB_High"] = bollinger.bollinger_hband()
        df["BB_Low"] = bollinger.bollinger_lband()
        df["BB_Mid"] = bollinger.bollinger_mavg()

        # E) Stochastics
        stoch = ta.momentum.StochasticOscillator(
            high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
        )
        df["Stoch_%K"] = stoch.stoch()
        df["Stoch_%D"] = stoch.stoch_signal()

        # F) ATR
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        )
        df["ATR"] = atr_indicator.average_true_range()

        # G) ADX
        adx_indicator = ta.trend.ADXIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        )
        df["ADX"] = adx_indicator.adx()
        df["+DI"] = adx_indicator.adx_pos()
        df["-DI"] = adx_indicator.adx_neg()

        # --- Derive signals from these indicators ---
        latest_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else None

        # EMA Crossover
        ema_signal = "neutral"
        if previous_data is not None:
            was_short_below = previous_data["EMA_Short"] <= previous_data["EMA_Long"]
            is_short_above = latest_data["EMA_Short"] > latest_data["EMA_Long"]
            was_short_above = previous_data["EMA_Short"] >= previous_data["EMA_Long"]
            is_short_below = latest_data["EMA_Short"] < latest_data["EMA_Long"]
            if was_short_below and is_short_above:
                ema_signal = "bullish"
            elif was_short_above and is_short_below:
                ema_signal = "bearish"

        # RSI
        rsi_value = latest_data["RSI"]
        if rsi_value >= 70:
            rsi_signal = "overbought"
        elif rsi_value <= 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"

        # MACD Trend
        macd_value = latest_data["MACD"]
        macd_signal_line = latest_data["MACD_Signal"]
        if macd_value > macd_signal_line:
            macd_trend = "bullish"
        elif macd_value < macd_signal_line:
            macd_trend = "bearish"
        else:
            macd_trend = "neutral"

        # Bollinger
        last_close = latest_data["Close"]
        if last_close > latest_data["BB_High"]:
            bb_signal = "close_above_upper_band"
        elif last_close < latest_data["BB_Low"]:
            bb_signal = "close_below_lower_band"
        else:
            bb_signal = "within_band"

        # Stochastics
        stoch_k = latest_data["Stoch_%K"]
        if stoch_k < 20:
            stoch_signal = "oversold"
        elif stoch_k > 80:
            stoch_signal = "overbought"
        else:
            stoch_signal = "neutral"

        # ADX
        adx_value = latest_data["ADX"]
        plus_di = latest_data["+DI"]
        minus_di = latest_data["-DI"]
        adx_trend_strength = "strong_trend" if adx_value > 25 else "weak_or_sideways"
        if plus_di > minus_di:
            adx_trend_direction = "bullish_trend"
        elif plus_di < minus_di:
            adx_trend_direction = "bearish_trend"
        else:
            adx_trend_direction = "neutral_trend"

        # Populate the 'indicators' field
        analysis_result["indicators"] = {
            "close_price": float(latest_data["Close"]),
            "ema": {
                "short_ema": float(latest_data["EMA_Short"]),
                "long_ema": float(latest_data["EMA_Long"]),
                "signal": ema_signal
            },
            "rsi": {
                "value": float(rsi_value),
                "signal": rsi_signal
            },
            "macd": {
                "value": float(macd_value),
                "signal_line": float(macd_signal_line),
                "hist": float(latest_data["MACD_Hist"]),
                "trend": macd_trend
            },
            "bollinger": {
                "upper": float(latest_data["BB_High"]),
                "lower": float(latest_data["BB_Low"]),
                "mid": float(latest_data["BB_Mid"]),
                "signal": bb_signal
            },
            "stochastics": {
                "%K": float(stoch_k),
                "%D": float(latest_data["Stoch_%D"]),
                "signal": stoch_signal
            },
            "atr": float(latest_data["ATR"]),
            "adx": {
                "adx_value": float(adx_value),
                "+DI": float(plus_di),
                "-DI": float(minus_di),
                "trend_strength": adx_trend_strength,
                "trend_direction": adx_trend_direction
            }
        }

        # # -------------------------------------------------------------
        # # 3. Candlestick Patterns (TA-Lib) - Extended Set
        # # -------------------------------------------------------------
        # open_prices = df["Open"].values
        # high_prices = df["High"].values
        # low_prices = df["Low"].values
        # close_prices = df["Close"].values
        # final_index = len(df) - 1

        # # Example extended pattern set:
        # pattern_funcs = {
        #     "CDLDOJI": talib.CDLDOJI,
        #     "CDLHAMMER": talib.CDLHAMMER,
        #     "CDLHANGINGMAN": talib.CDLHANGINGMAN,
        #     "CDLENGULFING": talib.CDLENGULFING,
        #     "CDLPIERCING": talib.CDLPIERCING,
        #     "CDLSHOOTINGSTAR": talib.CDLSHOOTINGSTAR,
        #     "CDLMORNINGSTAR": talib.CDLMORNINGSTAR,
        #     "CDLEVENINGSTAR": talib.CDLEVENINGSTAR,
        #     "CDL3WHITESOLDIERS": talib.CDL3WHITESOLDIERS,
        #     "CDL3BLACKCROWS": talib.CDL3BLACKCROWS,
        # }

        # # Evaluate each pattern on the last candle
        # for name, func in pattern_funcs.items():
        #     result_array = func(open_prices, high_prices, low_prices, close_prices)
        #     pattern_value = result_array[final_index]  # typically +100, 0, or -100

        #     if pattern_value > 0:
        #         interpretation = "bullish"
        #     elif pattern_value < 0:
        #         interpretation = "bearish"
        #     else:
        #         interpretation = "none"

        #     analysis_result["candlestick_patterns"][name] = {
        #         "raw": int(pattern_value),
        #         "interpretation": interpretation
        #     }


        # ---------------
        # G) Candlestick / Pattern Detections
        # ---------------
        # Example: Hammer detection (already shown).
        # Add a second pattern, e.g., a simplistic "bullish engulfing" check
        # (This is a naive approach for demonstration.)
        pattern_detected = []

        # Hammer detection
        last_open = latest_data["Open"]
        candle_body = abs(latest_data["Close"] - last_open)
        lower_wick = min(latest_data["Close"], last_open) - latest_data["Low"]
        upper_wick = latest_data["High"] - max(latest_data["Close"], last_open)

        if (lower_wick >= 2 * candle_body) and (upper_wick <= 0.5 * candle_body):
            pattern_detected.append("possible_hammer")

        # Bullish Engulfing detection (naive approach):
        #   - Previous candle is red (close < open)
        #   - Current candle is green (close > open)
        #   - Current candle's body "engulfs" previous body
        if previous_data is not None:
            prev_body = abs(previous_data["Close"] - previous_data["Open"])
            curr_body = abs(latest_data["Close"] - latest_data["Open"])
            prev_bearish = previous_data["Close"] < previous_data["Open"]
            curr_bullish = latest_data["Close"] > latest_data["Open"]

            if prev_bearish and curr_bullish and (curr_body > prev_body) and (latest_data["Close"] > previous_data["Open"]):
                pattern_detected.append("bullish_engulfing")

        # Add more advanced patterns or integrate a dedicated pattern library as needed.

        # -------------------------------------------------------------
        # 5. Aggregate into Final Rating + Probability
        # -------------------------------------------------------------
        score = 0
        max_score = 0

        # Helper method to handle adding signals
        def add_signal(signal_str: str, weight=1):
            nonlocal score, max_score
            max_score += weight
            if signal_str in ["bullish", "oversold"]:
                score += 1 * weight
            elif signal_str in ["bearish", "overbought"]:
                score -= 1 * weight
            # "neutral" => no change

        # A) Technical Indicator Signals
        add_signal(ema_signal, weight=1)
        add_signal(rsi_signal, weight=1)
        add_signal(macd_trend, weight=1)
        add_signal(stoch_signal, weight=1)
        # ADX direction
        max_score += 1
        if adx_trend_direction == "bullish_trend":
            score += 1
        elif adx_trend_direction == "bearish_trend":
            score -= 1

        # # B) Candlestick Patterns (last candle only)
        # # - We'll add 1 for bullish pattern, -1 for bearish.
        # # - Some patterns might be more significant => use bigger weights
        # pattern_weight = 0.5  # we can weight candlestick signals less
        # for _, pdata in analysis_result["candlestick_patterns"].items():
        #     raw_interpretation = pdata["interpretation"]
        #     max_score += pattern_weight
        #     if raw_interpretation == "bullish":
        #         score += pattern_weight
        #     elif raw_interpretation == "bearish":
        #         score -= pattern_weight
        #     else:
        #         pass

        # Convert final score to probability
        if max_score == 0:
            probability = 0.5
        else:
            ratio = score / max_score  # in [-1, +1]
            probability = 0.5 + 0.5 * ratio  # in [0, 1]

        if probability >= 0.66:
            final_rating = "buy"
        elif probability <= 0.33:
            final_rating = "sell"
        else:
            final_rating = "hold"

        analysis_result["final_decision"] = {
            "score": round(score, 3),
            "max_score_possible": round(max_score, 3),
            "probability": round(probability, 3),
            "rating": final_rating
        }

        return analysis_result


    @classmethod
    def generate_tools_json_doc(cls) -> str:
        """
        Generate a JSON document containing information about all methods in the class.

        Returns:
            str: JSON string containing the methods' information
        """

        tools_list = []

        # Get all methods from the class that have the kernel_function annotation
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip this method itself and any private methods
            if name.startswith("_") or name == "generate_tools_json_doc":
                continue

            # Check if the method has the kernel_function annotation
            if hasattr(method, "__kernel_function__"):
                # Get method description from docstring or kernel_function description
                description = ""
                if hasattr(method, "__doc__") and method.__doc__:
                    description = method.__doc__.strip()

                # Get kernel_function description if available
                if hasattr(method, "__kernel_function__") and getattr(
                    method.__kernel_function__, "description", None
                ):
                    description = method.__kernel_function__.description

                # Get argument information by introspection
                sig = inspect.signature(method)
                args_dict = {}

                # Get type hints if available
                type_hints = get_type_hints(method)

                # Process parameters
                for param_name, param in sig.parameters.items():
                    # Skip first parameter 'cls' for class methods (though we're using staticmethod now)
                    if param_name in ["cls", "self"]:
                        continue

                    # Get parameter type
                    param_type = "string"  # Default type
                    if param_name in type_hints:
                        type_obj = type_hints[param_name]
                        # Convert type to string representation
                        if hasattr(type_obj, "__name__"):
                            param_type = type_obj.__name__.lower()
                        else:
                            # Handle complex types like List, Dict, etc.
                            param_type = str(type_obj).lower()
                            if "int" in param_type:
                                param_type = "int"
                            elif "float" in param_type:
                                param_type = "float"
                            elif "bool" in param_type:
                                param_type = "boolean"
                            else:
                                param_type = "string"

                    # Create parameter description
                    param_desc = param_name.replace("_", " ")
                    args_dict[param_name] = {
                        "description": param_name,
                        "title": param_name.replace("_", " ").title(),
                        "type": param_type,
                    }

                # Add the tool information to the list
                tool_entry = {
                    "agent": cls.agent_name,  # Use HR agent type
                    "function": name,
                    "description": description,
                    "arguments": json.dumps(args_dict).replace('"', "'"),
                }

                tools_list.append(tool_entry)

        # Return the JSON string representation
        return json.dumps(tools_list, ensure_ascii=False, indent=2)

    # This function does NOT have the kernel_function annotation
    # because it's meant for introspection rather than being exposed as a tool
    @classmethod
    def get_all_kernel_functions(cls) -> dict[str, Callable]:
        """
        Returns a dictionary of all methods in this class that have the @kernel_function annotation.
        This function itself is not annotated with @kernel_function.

        Returns:
            Dict[str, Callable]: Dictionary with function names as keys and function objects as values
        """
        kernel_functions = {}

        # Get all class methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip this method itself and any private/special methods
            if name.startswith("_") or name == "get_all_kernel_functions":
                continue

            # Check if the method has the kernel_function annotation
            # by looking at its __annotations__ attribute
            method_attrs = getattr(method, "__annotations__", {})
            if hasattr(method, "__kernel_function__") or "kernel_function" in str(
                method_attrs
            ):
                kernel_functions[name] = method

        return kernel_functions
