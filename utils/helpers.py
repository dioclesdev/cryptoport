#!/usr/bin/env python3
"""
Helpers module for Crypto Bullrun Analyzer

This module provides utility functions and helpers for the Crypto Bullrun Analyzer
project, including data formatting, validation, and common operations.
"""

import os
import re
import json
import time
import hashlib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('helpers')

# Constants
CRYPTO_SYMBOLS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'Binance Coin',
    'SOL': 'Solana',
    'XRP': 'XRP',
    'ADA': 'Cardano',
    'AVAX': 'Avalanche',
    'DOT': 'Polkadot',
    'MATIC': 'Polygon',
    'DOGE': 'Dogecoin',
    'LTC': 'Litecoin',
    'LINK': 'Chainlink',
    'UNI': 'Uniswap',
    'ATOM': 'Cosmos',
    'XLM': 'Stellar',
    # Add more as needed
}

# Currency formatting helpers
def format_price(value: Union[float, int, str], decimals: int = None) -> str:
    """
    Format price with appropriate decimal places.
    
    Args:
        value: Price value to format
        decimals: Number of decimal places (adaptive if None)
        
    Returns:
        Formatted price string
    """
    if value is None:
        return "N/A"
        
    try:
        # Convert to float
        num_value = float(value)
        
        # Determine decimals automatically if not specified
        if decimals is None:
            if num_value < 0.001:
                decimals = 6
            elif num_value < 0.01:
                decimals = 5
            elif num_value < 0.1:
                decimals = 4
            elif num_value < 1:
                decimals = 3
            elif num_value < 1000:
                decimals = 2
            else:
                decimals = 2
                
        # Format the number
        return f"${num_value:.{decimals}f}"
        
    except (ValueError, TypeError):
        return str(value)

def format_large_number(value: Union[float, int, str], precision: int = 2) -> str:
    """
    Format large numbers with K, M, B, T suffixes.
    
    Args:
        value: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string with suffix
    """
    if value is None:
        return "N/A"
        
    try:
        num_value = float(value)
        
        # Define suffixes and thresholds
        suffixes = ['', 'K', 'M', 'B', 'T']
        suffix_idx = 0
        
        while num_value >= 1000 and suffix_idx < len(suffixes) - 1:
            num_value /= 1000
            suffix_idx += 1
            
        # Format with appropriate precision
        if num_value >= 100:
            precision = 0
        elif num_value >= 10:
            precision = 1
            
        return f"{num_value:.{precision}f}{suffixes[suffix_idx]}"
        
    except (ValueError, TypeError):
        return str(value)

def format_percent(value: Union[float, int, str], include_sign: bool = True) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage value to format
        include_sign: Whether to include + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
        
    try:
        num_value = float(value)
        
        if include_sign and num_value > 0:
            return f"+{num_value:.2f}%"
        else:
            return f"{num_value:.2f}%"
            
    except (ValueError, TypeError):
        return str(value)

# Data validation and cleaning
def clean_symbol(symbol: str) -> str:
    """
    Clean and normalize cryptocurrency symbol.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        Cleaned symbol
    """
    if not symbol:
        return ""
        
    # Remove any non-alphanumeric characters except -
    clean = re.sub(r'[^A-Za-z0-9\-]', '', str(symbol))
    
    # Convert to uppercase
    return clean.upper()

def validate_crypto_symbol(symbol: str) -> bool:
    """
    Validate if a string is a valid cryptocurrency symbol.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol:
        return False
        
    # Clean the symbol first
    clean = clean_symbol(symbol)
    
    # Basic validation rules
    if len(clean) < 2 or len(clean) > 10:
        return False
        
    # Must be alphanumeric
    if not re.match(r'^[A-Z0-9\-]+$', clean):
        return False
        
    # Known stablecoins pattern
    stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'USDD', 'GUSD', 'USDP', 'FRAX']
    if clean in stablecoins:
        return True
        
    # Known symbols
    if clean in CRYPTO_SYMBOLS:
        return True
        
    # More advanced checks could be added here
    
    return True

def get_coin_name(symbol: str) -> str:
    """
    Get full name for a cryptocurrency symbol.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        Full name if known, otherwise the symbol
    """
    clean = clean_symbol(symbol)
    return CRYPTO_SYMBOLS.get(clean, clean)

def is_bitcoin_address(address: str) -> bool:
    """
    Basic validation for Bitcoin addresses.
    
    Args:
        address: Address to validate
        
    Returns:
        True if likely a Bitcoin address
    """
    if not address:
        return False
        
    # Legacy addresses
    if re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):
        return True
        
    # SegWit addresses
    if re.match(r'^bc1[ac-hj-np-z02-9]{39,59}$', address):
        return True
        
    return False

def is_ethereum_address(address: str) -> bool:
    """
    Basic validation for Ethereum addresses.
    
    Args:
        address: Address to validate
        
    Returns:
        True if likely an Ethereum address
    """
    if not address:
        return False
        
    # Check format (0x followed by 40 hex chars)
    return bool(re.match(r'^0x[a-fA-F0-9]{40}$', address))

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted float or default
    """
    if value is None:
        return default
        
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def truncate_string(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated string
    """
    if not text:
        return ""
        
    if len(text) <= max_length:
        return text
        
    return text[:max_length - len(suffix)] + suffix

# File and path utilities
def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists.
    
    Args:
        directory: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False

def safe_filename(filename: str) -> str:
    """
    Convert a string to a safe filename.
    
    Args:
        filename: Input filename
        
    Returns:
        Safe filename
    """
    # Remove invalid characters
    safe = re.sub(r'[^\w\s\-\.]', '', filename)
    
    # Replace spaces with underscores
    safe = safe.replace(' ', '_')
    
    # Ensure it's not empty
    if not safe:
        safe = 'unnamed_file'
        
    return safe

def generate_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate a timestamp string.
    
    Args:
        format_str: Format string for strftime
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format_str)

def file_age_hours(filepath: str) -> Optional[float]:
    """
    Get age of a file in hours.
    
    Args:
        filepath: Path to file
        
    Returns:
        Age in hours or None if file doesn't exist
    """
    try:
        if not os.path.exists(filepath):
            return None
            
        mtime = os.path.getmtime(filepath)
        age_seconds = time.time() - mtime
        return age_seconds / 3600
    except Exception:
        return None

# Data processing helpers
def extract_coin_metrics(data: Dict) -> Dict:
    """
    Extract key metrics from coin data.
    
    Args:
        data: Coin data dictionary
        
    Returns:
        Dictionary with key metrics
    """
    metrics = {}
    
    # Safe extraction with defaults
    metrics['symbol'] = data.get('symbol', '').upper()
    metrics['name'] = data.get('name', '')
    
    # Price and market data
    metrics['price_usd'] = safe_float(data.get('usd_price', data.get('current_price_usd', 0)))
    metrics['market_cap'] = safe_float(data.get('usd_market_cap', data.get('market_cap_usd', 0)))
    metrics['volume_24h'] = safe_float(data.get('usd_volume_24h', data.get('volume_24h_usd', 0)))
    
    # Price changes
    metrics['price_change_24h'] = safe_float(data.get('usd_percent_change_24h', 0))
    metrics['price_change_7d'] = safe_float(data.get('usd_percent_change_7d', 0))
    metrics['price_change_30d'] = safe_float(data.get('usd_percent_change_30d', 0))
    
    # ATH data
    metrics['ath_price'] = safe_float(data.get('usd_ath', 0))
    metrics['ath_date'] = data.get('ath_date', '')
    metrics['ath_percent_down'] = safe_float(data.get('high_distance_percent', 0))
    
    # Rank and supply
    metrics['market_cap_rank'] = data.get('cmc_rank', data.get('market_cap_rank', 0))
    metrics['circulating_supply'] = safe_float(data.get('circulating_supply', 0))
    metrics['total_supply'] = safe_float(data.get('total_supply', 0))
    metrics['max_supply'] = safe_float(data.get('max_supply', 0))
    
    return metrics

def normalize_analysis_result(result: Dict) -> Dict:
    """
    Normalize analysis result format for consistent processing.
    
    Args:
        result: Analysis result dictionary
        
    Returns:
        Normalized result dictionary
    """
    normalized = {}
    
    # Basic info
    normalized['symbol'] = result.get('symbol', '').upper()
    normalized['name'] = result.get('name', '')
    
    # Handle different score formats
    if 'bullrun_score' in result:
        normalized['total_score'] = result['bullrun_score']
    elif 'total_score' in result:
        normalized['bullrun_score'] = result['total_score']
    else:
        # Default
        normalized['total_score'] = 0
        normalized['bullrun_score'] = 0
    
    # Handle potential formats
    normalized['bullrun_potential'] = result.get('bullrun_potential', 'Unknown')
    
    # Price information
    if 'current_price_usd' in result:
        normalized['current_price'] = result['current_price_usd']
    elif 'current_price' in result:
        normalized['current_price_usd'] = result['current_price']
    else:
        normalized['current_price'] = 0
        normalized['current_price_usd'] = 0
    
    # Market cap information
    if 'market_cap_usd' in result:
        normalized['market_cap'] = result['market_cap_usd']
    elif 'market_cap' in result:
        normalized['market_cap_usd'] = result['market_cap']
    else:
        normalized['market_cap'] = 0
        normalized['market_cap_usd'] = 0
    
    # Other fields
    normalized['high_distance_percent'] = result.get('high_distance_percent', 0)
    normalized['market_cap_rank'] = result.get('market_cap_rank', 0)
    normalized['volume_24h_usd'] = result.get('volume_24h_usd', result.get('volume_24h', 0))
    
    # Analysis scores
    normalized['scores'] = result.get('scores', {})
    
    # Analysis date
    normalized['analysis_date'] = result.get('analysis_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return normalized

def calculate_potential_category(score: float) -> str:
    """
    Calculate potential category based on score.
    
    Args:
        score: Bullrun score (0-1)
        
    Returns:
        Category string
    """
    if score >= 0.8:
        return "Sehr hoch"
    elif score >= 0.7:
        return "Hoch"
    elif score >= 0.6:
        return "Überdurchschnittlich"
    elif score >= 0.5:
        return "Durchschnittlich"
    elif score >= 0.4:
        return "Unterdurchschnittlich"
    else:
        return "Niedrig"

def categorize_results(results: List[Dict]) -> Dict:
    """
    Categorize analysis results by potential.
    
    Args:
        results: List of analysis results
        
    Returns:
        Dictionary with categorized results
    """
    categories = {
        "very_high": [],
        "high": [],
        "above_average": [],
        "average": [],
        "below_average": [],
        "low": []
    }
    
    for result in results:
        score = result.get('total_score', result.get('bullrun_score', 0))
        symbol = result.get('symbol', '').upper()
        
        if score >= 0.8:
            categories["very_high"].append(symbol)
        elif score >= 0.7:
            categories["high"].append(symbol)
        elif score >= 0.6:
            categories["above_average"].append(symbol)
        elif score >= 0.5:
            categories["average"].append(symbol)
        elif score >= 0.4:
            categories["below_average"].append(symbol)
        else:
            categories["low"].append(symbol)
    
    return categories

# Chart and visualization helpers
def generate_chart_colors(scores: List[float]) -> List[str]:
    """
    Generate color codes based on scores.
    
    Args:
        scores: List of score values
        
    Returns:
        List of color codes
    """
    colors = []
    
    for score in scores:
        if score >= 0.8:
            colors.append('#1a5490')  # Dark blue - very high
        elif score >= 0.7:
            colors.append('#28a745')  # Green - high
        elif score >= 0.6:
            colors.append('#fd7e14')  # Orange - above average
        elif score >= 0.5:
            colors.append('#ffc107')  # Yellow - average
        elif score >= 0.4:
            colors.append('#6f42c1')  # Purple - below average
        else:
            colors.append('#dc3545')  # Red - low
    
    return colors

def score_to_color(score: float) -> str:
    """
    Convert score to color code.
    
    Args:
        score: Bullrun score
        
    Returns:
        Hex color code
    """
    if score >= 0.8:
        return '#1a5490'  # Dark blue
    elif score >= 0.7:
        return '#28a745'  # Green
    elif score >= 0.6:
        return '#fd7e14'  # Orange
    elif score >= 0.5:
        return '#ffc107'  # Yellow
    elif score >= 0.4:
        return '#6f42c1'  # Purple
    else:
        return '#dc3545'  # Red

def generate_score_badge_html(score: float) -> str:
    """
    Generate HTML for score badge.
    
    Args:
        score: Bullrun score
        
    Returns:
        HTML string
    """
    color_class = ""
    if score >= 0.7:
        color_class = "bg-success"
    elif score >= 0.5:
        color_class = "bg-warning"
    else:
        color_class = "bg-danger"
    
    return f'<span class="badge {color_class}">{score:.3f}</span>'

def generate_potential_badge_html(potential: str) -> str:
    """
    Generate HTML for potential badge.
    
    Args:
        potential: Potential category
        
    Returns:
        HTML string
    """
    color_class = ""
    if potential in ["Sehr hoch", "Hoch"]:
        color_class = "bg-success"
    elif potential in ["Überdurchschnittlich", "Durchschnittlich"]:
        color_class = "bg-warning"
    else:
        color_class = "bg-danger"
    
    return f'<span class="badge {color_class}">{potential}</span>'

# Import/Export helpers
def read_text_file_lines(filepath: str) -> List[str]:
    """
    Read lines from a text file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        List of lines
    """
    lines = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
    
    return lines

def parse_csv_line(line: str) -> Tuple[str, float, float]:
    """
    Parse CSV line in format Symbol,Amount,Price.
    
    Args:
        line: CSV line
        
    Returns:
        Tuple of (symbol, amount, price)
    """
    parts = line.split(',')
    
    if len(parts) < 3:
        return None, 0, 0
    
    symbol = clean_symbol(parts[0])
    
    try:
        amount = float(parts[1].strip())
        price = float(parts[2].strip())
    except ValueError:
        return symbol, 0, 0
    
    return symbol, amount, price

def load_portfolio_from_csv(filepath: str) -> Dict:
    """
    Load portfolio from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Portfolio dictionary
    """
    portfolio = {"holdings": {}, "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    try:
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            if 'Symbol' in df.columns and 'Amount' in df.columns and 'Price' in df.columns:
                symbol = clean_symbol(row['Symbol'])
                amount = safe_float(row['Amount'])
                price = safe_float(row['Price'])
                
                if symbol and amount > 0:
                    portfolio['holdings'][symbol] = {
                        "amount": amount,
                        "avg_price": price
                    }
            elif len(row) >= 3:  # Assume first 3 columns are Symbol, Amount, Price
                symbol = clean_symbol(row.iloc[0])
                amount = safe_float(row.iloc[1])
                price = safe_float(row.iloc[2])
                
                if symbol and amount > 0:
                    portfolio['holdings'][symbol] = {
                        "amount": amount,
                        "avg_price": price
                    }
    except Exception as e:
        logger.error(f"Error loading portfolio from CSV {filepath}: {e}")
    
    return portfolio

# Web and API helpers
def parse_sort_parameter(sort_param: str, default: str = 'total_score', reverse: bool = True) -> Tuple[str, bool]:
    """
    Parse sort parameter for API requests.
    
    Args:
        sort_param: Sort parameter string
        default: Default sort field
        reverse: Default sort direction
        
    Returns:
        Tuple of (sort_field, reverse)
    """
    if not sort_param:
        return default, reverse
    
    # Check for direction prefix
    direction = reverse
    field = sort_param
    
    if sort_param.startswith('-'):
        direction = False
        field = sort_param[1:]
    elif sort_param.startswith('+'):
        direction = True
        field = sort_param[1:]
    
    # Map common field names
    field_mapping = {
        'score': 'total_score',
        'bullrun_score': 'total_score',
        'price': 'current_price_usd',
        'market_cap': 'market_cap_usd',
        'volume': 'volume_24h_usd',
        'change': 'price_change_24h',
        'rank': 'market_cap_rank'
    }
    
    normalized_field = field_mapping.get(field, field)
    
    # Validate field exists (optional)
    valid_fields = list(field_mapping.values()) + list(field_mapping.keys())
    if normalized_field not in valid_fields:
        return default, reverse
    
    return normalized_field, direction

def format_api_response(data: Any, success: bool = True, message: str = "") -> Dict:
    """
    Format API response in a standard structure.
    
    Args:
        data: Response data
        success: Success flag
        message: Optional message
        
    Returns:
        Formatted API response
    """
    return {
        "success": success,
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "data": data
    }

# Misc helpers
def generate_csv_export(data: List[Dict], filename: str) -> str:
    """
    Generate CSV export file.
    
    Args:
        data: List of dictionaries to export
        filename: Output filename
        
    Returns:
        Path to generated CSV file
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    except Exception as e:
        logger.error(f"Error generating CSV export: {e}")
        return ""

def calculate_portfolio_metrics(portfolio: Dict, current_prices: Dict) -> Dict:
    """
    Calculate portfolio metrics.
    
    Args:
        portfolio: Portfolio dictionary
        current_prices: Dictionary of current prices by symbol
        
    Returns:
        Dictionary with portfolio metrics
    """
    metrics = {
        "total_value": 0,
        "total_cost": 0,
        "profit_loss": 0,
        "profit_loss_percent": 0,
        "assets_count": len(portfolio.get("holdings", {})),
        "profitable_assets": 0,
        "losing_assets": 0
    }
    
    for symbol, data in portfolio.get("holdings", {}).items():
        amount = data.get("amount", 0)
        avg_price = data.get("avg_price", 0)
        current_price = current_prices.get(symbol, 0)
        
        cost_basis = amount * avg_price
        current_value = amount * current_price
        
        metrics["total_cost"] += cost_basis
        metrics["total_value"] += current_value
        
        if current_value > cost_basis:
            metrics["profitable_assets"] += 1
        elif current_value < cost_basis:
            metrics["losing_assets"] += 1
    
    # Calculate profit/loss
    metrics["profit_loss"] = metrics["total_value"] - metrics["total_cost"]
    
    if metrics["total_cost"] > 0:
        metrics["profit_loss_percent"] = (metrics["profit_loss"] / metrics["total_cost"]) * 100
    
    return metrics

def calculate_timestamp_diff(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Calculate human-readable time difference from timestamp.
    
    Args:
        timestamp_str: Timestamp string
        format_str: Format string for strptime
        
    Returns:
        Human-readable time difference
    """
    try:
        timestamp = datetime.strptime(timestamp_str, format_str)
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 30:
            months = diff.days // 30
            return f"{months} {'month' if months == 1 else 'months'} ago"
        elif diff.days > 0:
            return f"{diff.days} {'day' if diff.days == 1 else 'days'} ago"
        elif diff.seconds // 3600 > 0:
            hours = diff.seconds // 3600
            return f"{hours} {'hour' if hours == 1 else 'hours'} ago"
        elif diff.seconds // 60 > 0:
            minutes = diff.seconds // 60
            return f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago"
        else:
            return "just now"
    except Exception:
        return "unknown"

# Testing function
if __name__ == "__main__":
    """Run basic tests on helper functions"""
    
    # Test price formatting
    print("Price formatting tests:")
    print(f"0.00012345 -> {format_price(0.00012345)}")
    print(f"0.0012345 -> {format_price(0.0012345)}")
    print(f"0.012345 -> {format_price(0.012345)}")
    print(f"0.12345 -> {format_price(0.12345)}")
    print(f"1.2345 -> {format_price(1.2345)}")
    print(f"12.345 -> {format_price(12.345)}")
    print(f"123.45 -> {format_price(123.45)}")
    print(f"1234.5 -> {format_price(1234.5)}")
    print(f"12345 -> {format_price(12345)}")
    
    print("\nLarge number formatting tests:")
    print(f"123 -> {format_large_number(123)}")
    print(f"1234 -> {format_large_number(1234)}")
    print(f"12345 -> {format_large_number(12345)}")
    print(f"123456 -> {format_large_number(123456)}")
    print(f"1234567 -> {format_large_number(1234567)}")
    print(f"12345678 -> {format_large_number(12345678)}")
    print(f"123456789 -> {format_large_number(123456789)}")
    print(f"1234567890 -> {format_large_number(1234567890)}")
    
    print("\nPercent formatting tests:")
    print(f"10.5 -> {format_percent(10.5)}")
    print(f"-5.25 -> {format_percent(-5.25)}")
    print(f"0 -> {format_percent(0)}")
    
    print("\nSymbol cleaning tests:")
    print(f"btc -> {clean_symbol('btc')}")
    print(f"ETH-USD -> {clean_symbol('ETH-USD')}")
    print(f"sol/usd -> {clean_symbol('sol/usd')}")
    print(f"$BNB -> {clean_symbol('$BNB')}")
    
    print("\nAddress validation tests:")
    btc_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    eth_address = "0x742F96B7e9a20A9E4A34F1F3d1CF4d7e7C5dA845"
    print(f"{btc_address} is BTC: {is_bitcoin_address(btc_address)}")
    print(f"{eth_address} is ETH: {is_ethereum_address(eth_address)}")
    
    print("\nTimestamp diff tests:")
    now = datetime.now()
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    month_ago = (now - timedelta(days=32)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{yesterday} -> {calculate_timestamp_diff(yesterday)}")
    print(f"{week_ago} -> {calculate_timestamp_diff(week_ago)}")
    print(f"{month_ago} -> {calculate_timestamp_diff(month_ago)}")
    
    print("\nBadge HTML tests:")
    print(f"Score 0.85 -> {generate_score_badge_html(0.85)}")
    print(f"Score 0.65 -> {generate_score_badge_html(0.65)}")
    print(f"Score 0.35 -> {generate_score_badge_html(0.35)}")
    
    print("\nColor generation tests:")
    scores = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
    print(f"Colors for {scores}: {generate_chart_colors(scores)}")
