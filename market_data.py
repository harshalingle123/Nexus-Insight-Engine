"""
Nexus Bank Financial Analytics Project - Clean API Service
Yahoo Finance API with Flask Web Service (API Only)
Author: Nexus Bank Analytics Team
Date: June 2025
"""

from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import traceback
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_bank_api.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

class YahooFinanceAPI:
    """Yahoo Finance API wrapper with Flask integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.YahooFinanceAPI")
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict:
        """Get stock data and return as dictionary"""
        try:
            self.logger.info(f"Fetching stock data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Convert to dictionary format
            result = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data_points": len(data),
                "date_range": {
                    "start": data.index.min().strftime('%Y-%m-%d'),
                    "end": data.index.max().strftime('%Y-%m-%d')
                },
                "latest_price": {
                    "date": data.index[-1].strftime('%Y-%m-%d'),
                    "open": float(data['Open'].iloc[-1]),
                    "high": float(data['High'].iloc[-1]),
                    "low": float(data['Low'].iloc[-1]),
                    "close": float(data['Close'].iloc[-1]),
                    "volume": int(data['Volume'].iloc[-1])
                },
                "statistics": {
                    "avg_price": float(data['Close'].mean()),
                    "max_price": float(data['Close'].max()),
                    "min_price": float(data['Close'].min()),
                    "volatility": float(data['Close'].std()),
                    "total_volume": int(data['Volume'].sum())
                },
                "historical_data": [
                    {
                        "date": idx.strftime('%Y-%m-%d'),
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": int(row['Volume'])
                    }
                    for idx, row in data.iterrows()
                ]
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            raise
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information"""
        try:
            self.logger.info(f"Fetching company info for {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            result = {
                "symbol": symbol,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "country": info.get("country", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "employee_count": info.get("fullTimeEmployees", 0),
                "website": info.get("website", "N/A"),
                "business_summary": info.get("longBusinessSummary", "N/A"),
                "financials": {
                    "revenue": info.get("totalRevenue", 0),
                    "gross_profit": info.get("grossProfits", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "forward_pe": info.get("forwardPE", 0),
                    "dividend_yield": info.get("dividendYield", 0),
                    "book_value": info.get("bookValue", 0),
                    "price_to_book": info.get("priceToBook", 0),
                    "debt_to_equity": info.get("debtToEquity", 0)
                },
                "trading_info": {
                    "current_price": info.get("currentPrice", 0),
                    "previous_close": info.get("previousClose", 0),
                    "day_high": info.get("dayHigh", 0),
                    "day_low": info.get("dayLow", 0),
                    "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                    "52_week_low": info.get("fiftyTwoWeekLow", 0),
                    "volume": info.get("volume", 0),
                    "avg_volume": info.get("averageVolume", 0)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            raise
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict:
        """Get data for multiple stocks"""
        try:
            self.logger.info(f"Fetching data for multiple stocks: {symbols}")
            results = {}
            for symbol in symbols:
                try:
                    results[symbol] = self.get_stock_data(symbol, period)
                except Exception as e:
                    results[symbol] = {"error": str(e)}
            
            return {
                "symbols": symbols,
                "period": period,
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "summary": {
                    "total_symbols": len(symbols),
                    "successful": len([r for r in results.values() if "error" not in r]),
                    "failed": len([r for r in results.values() if "error" in r])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching multiple stocks: {str(e)}")
            raise

    def get_market_summary(self) -> Dict:
        """Get major market indices summary"""
        try:
            indices = {
                "S&P 500": "^GSPC",
                "Dow Jones": "^DJI", 
                "NASDAQ": "^IXIC",
                "Russell 2000": "^RUT",
                "VIX": "^VIX"
            }
            
            results = {}
            for name, symbol in indices.items():
                try:
                    data = self.get_stock_data(symbol, period="1d")
                    results[name] = {
                        "symbol": symbol,
                        "current_price": data["latest_price"]["close"],
                        "change": data["latest_price"]["close"] - data["latest_price"]["open"],
                        "change_percent": ((data["latest_price"]["close"] - data["latest_price"]["open"]) / data["latest_price"]["open"]) * 100
                    }
                except Exception as e:
                    results[name] = {"error": str(e)}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_indices": results
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching market summary: {str(e)}")
            raise

# Initialize the Yahoo Finance API
yahoo_api = YahooFinanceAPI()

def handle_errors(f):
    """Decorator for error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    return decorated_function

# API Endpoints
@app.route('/api/stock/<symbol>')
@handle_errors
def get_stock_data(symbol):
    """Get stock data for a specific symbol
    
    Parameters:
    - symbol: Stock symbol (e.g., AAPL, GOOGL)
    - period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    - interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    """
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    data = yahoo_api.get_stock_data(symbol.upper(), period, interval)
    return jsonify(data)

@app.route('/api/forex/<pair>')
@handle_errors
def get_forex_data(pair):
    """Get forex data for a currency pair
    
    Parameters:
    - pair: Currency pair (e.g., EURUSD=X, GBPUSD=X)
    - period: Time period (default: 1y)
    """
    period = request.args.get('period', '1y')
    
    # Ensure proper forex format
    if not pair.endswith('=X'):
        pair += '=X'
    
    data = yahoo_api.get_stock_data(pair.upper(), period)
    return jsonify(data)

@app.route('/api/crypto/<symbol>')
@handle_errors
def get_crypto_data(symbol):
    """Get cryptocurrency data
    
    Parameters:
    - symbol: Crypto symbol (e.g., BTC-USD, ETH-USD)
    - period: Time period (default: 1y)
    """
    period = request.args.get('period', '1y')
    
    # Ensure proper crypto format
    if '-USD' not in symbol.upper():
        symbol += '-USD'
    
    data = yahoo_api.get_stock_data(symbol.upper(), period)
    return jsonify(data)

@app.route('/api/index/<symbol>')
@handle_errors
def get_index_data(symbol):
    """Get market index data
    
    Parameters:
    - symbol: Index symbol (e.g., ^GSPC, ^DJI, ^IXIC)
    - period: Time period (default: 1y)
    """
    period = request.args.get('period', '1y')
    
    # Ensure proper index format
    if not symbol.startswith('^'):
        symbol = '^' + symbol
    
    data = yahoo_api.get_stock_data(symbol.upper(), period)
    return jsonify(data)

@app.route('/api/company/<symbol>')
@handle_errors
def get_company_info(symbol):
    """Get detailed company information"""
    data = yahoo_api.get_company_info(symbol.upper())
    return jsonify(data)

@app.route('/api/multiple_stocks', methods=['POST'])
@handle_errors
def get_multiple_stocks():
    """Get data for multiple stocks
    
    Body: {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "period": "1y"
    }
    """
    data = request.json
    symbols = data.get('symbols', [])
    period = data.get('period', '1y')
    
    if not symbols:
        return jsonify({
            "error": "No symbols provided",
            "expected_format": {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "period": "1y"
            }
        }), 400
    
    result = yahoo_api.get_multiple_stocks(symbols, period)
    return jsonify(result)

@app.route('/api/market_summary')
@handle_errors
def get_market_summary():
    """Get summary of major market indices"""
    data = yahoo_api.get_market_summary()
    return jsonify(data)

@app.route('/api/popular_stocks')
@handle_errors
def get_popular_stocks():
    """Get data for popular stocks"""
    popular_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    period = request.args.get('period', '1mo')
    
    result = yahoo_api.get_multiple_stocks(popular_symbols, period)
    return jsonify(result)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Nexus Bank Yahoo Finance API",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    })

@app.route('/api/endpoints')
def list_endpoints():
    """List all available API endpoints"""
    return jsonify({
        "service": "Nexus Bank Yahoo Finance API",
        "version": "2.0.0",
        "endpoints": {
            "GET /api/stock/{symbol}": "Get stock data with optional period and interval parameters",
            "GET /api/forex/{pair}": "Get forex exchange rates",
            "GET /api/crypto/{symbol}": "Get cryptocurrency data", 
            "GET /api/index/{symbol}": "Get market index data",
            "GET /api/company/{symbol}": "Get detailed company information",
            "POST /api/multiple_stocks": "Get data for multiple stocks (requires JSON body)",
            "GET /api/market_summary": "Get summary of major market indices",
            "GET /api/popular_stocks": "Get data for popular stocks",
            "GET /api/health": "Health check endpoint",
            "GET /api/endpoints": "List all available endpoints"
        },
        "parameters": {
            "period": "1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max",
            "interval": "1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo"
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": "/api/endpoints",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("🏦 Starting Nexus Bank Yahoo Finance API Service...")
    print("=" * 60)
    print("🚀 API Endpoints available at: http://localhost:5000")
    print("📋 List endpoints: GET /api/endpoints")
    print("❤️  Health check: GET /api/health")
    print("📊 Stock data: GET /api/stock/{symbol}?period=1y")
    print("🏢 Company info: GET /api/company/{symbol}")
    print("💱 Forex data: GET /api/forex/{pair}")
    print("₿  Crypto data: GET /api/crypto/{symbol}")
    print("📈 Index data: GET /api/index/{symbol}")
    print("📊 Multiple stocks: POST /api/multiple_stocks")
    print("=" * 60)
    print("📝 Logs: nexus_bank_api.log")
    print("🔄 Starting Flask development server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)