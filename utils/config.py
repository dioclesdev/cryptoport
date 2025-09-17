# utils/config.py
"""
Configuration Management for CryptoPort
"""

import os
from pathlib import Path

class Config:
    """Base Configuration Class"""
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        
    def get_config(self, env='development'):
        """Get configuration based on environment"""
        configs = {
            'development': DevelopmentConfig(),
            'production': ProductionConfig(), 
            'testing': TestingConfig(),
            'default': DevelopmentConfig()
        }
        return configs.get(env, DevelopmentConfig()).get_dict()

class BaseConfig:
    """Base Configuration"""
    
    SECRET_KEY = os.environ.get('SECRET_KEY', '')
    
    # Cache Configuration
    CACHE_DIRS = [
        os.path.join(os.path.expanduser('~'), '.cryptoport', 'cache'),
        os.path.join(os.path.dirname(__file__), '..', 'cache'),
        'cache'
    ]
    CACHE_MAX_AGE_HOURS = 24
    NO_RETRY_API_CALLS = True
    
    # API Configuration
    COINMARKETCAP_API_KEY = os.environ.get('COINMARKETCAP_API_KEY', '')
    MORALIS_API_KEY = os.environ.get('MORALIS_API_KEY', '')
    
    # Email Configuration
    SMTP_SERVER = os.environ.get('SMTP_SERVER', '')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
    
    # Dashboard Configuration
    AUTO_REFRESH_MINUTES = 30
    MAX_COINS_DISPLAY = 50
    CHART_STYLE = 'seaborn-v0_8'
    
    def get_dict(self):
        """Return configuration as dictionary"""
        return {key: value for key, value in self.__class__.__dict__.items() 
                if not key.startswith('_') and not callable(value)}

class DevelopmentConfig(BaseConfig):
    """Development Configuration"""
    DEBUG = True
    TESTING = False
    
class ProductionConfig(BaseConfig):
    """Production Configuration"""
    DEBUG = False
    TESTING = False
    
class TestingConfig(BaseConfig):
    """Testing Configuration"""
    DEBUG = True
    TESTING = True


# utils/helpers.py
"""
Helper Functions for CryptoPort
"""

import locale
from typing import Any, Optional

def format_currency(value: Optional[float], currency: str = 'USD') -> str:
    """Format currency with proper symbols and thousands separators"""
    if value is None or value == 0:
        return f"$0.00"
    
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.2f}K"
    elif abs(value) >= 1:
        return f"${value:.2f}"
    else:
        return f"${value:.6f}"

def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    """Format percentage with proper sign and decimals"""
    if value is None:
        return "N/A"
    
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"

def format_score(value: Optional[float]) -> str:
    """Format bullrun score"""
    if value is None:
        return "N/A"
    return f"{value:.3f}"

def get_score_badge_class(score: Optional[float]) -> str:
    """Get Bootstrap badge class for score"""
    if score is None:
        return "badge-secondary"
    
    if score >= 0.8:
        return "badge-success"
    elif score >= 0.7:
        return "badge-info" 
    elif score >= 0.6:
        return "badge-warning"
    elif score >= 0.5:
        return "badge-light"
    else:
        return "badge-danger"

def get_potential_class(potential: str) -> str:
    """Get CSS class for potential category"""
    potential_classes = {
        'Sehr hoch': 'text-success',
        'Hoch': 'text-info',
        'Überdurchschnittlich': 'text-warning', 
        'Durchschnittlich': 'text-secondary',
        'Unterdurchschnittlich': 'text-muted',
        'Niedrig': 'text-danger'
    }
    return potential_classes.get(potential, 'text-secondary')

def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safe division with fallback"""
    try:
        return numerator / denominator if denominator != 0 else fallback
    except (TypeError, ZeroDivisionError):
        return fallback

def truncate_address(address: str, start_chars: int = 6, end_chars: int = 4) -> str:
    """Truncate crypto address for display"""
    if not address or len(address) <= start_chars + end_chars:
        return address
    return f"{address[:start_chars]}...{address[-end_chars:]}"

def calculate_portfolio_metrics(holdings: dict) -> dict:
    """Calculate portfolio summary metrics"""
    if not holdings:
        return {
            'total_value': 0,
            'total_coins': 0,
            'total_profit_loss': 0,
            'best_performer': None,
            'worst_performer': None
        }
    
    total_value = sum(holding.get('current_value', 0) for holding in holdings.values())
    total_coins = len(holdings)
    total_profit_loss = sum(holding.get('profit_loss', 0) for holding in holdings.values())
    
    # Best/worst performers
    performers = [(symbol, data.get('profit_loss_percent', 0)) 
                 for symbol, data in holdings.items() 
                 if data.get('profit_loss_percent') is not None]
    
    best_performer = max(performers, key=lambda x: x[1])[0] if performers else None
    worst_performer = min(performers, key=lambda x: x[1])[0] if performers else None
    
    return {
        'total_value': total_value,
        'total_coins': total_coins, 
        'total_profit_loss': total_profit_loss,
        'best_performer': best_performer,
        'worst_performer': worst_performer
    }
