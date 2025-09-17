#!/usr/bin/env python3
"""
Crypto Bullrun Analyzer - Flask Web Application

This module provides the web interface for the Crypto Bullrun Analyzer,
including dashboard, portfolio management, analysis tools, and more.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import csv
from flask import Response

def set_working_directory():
    """Set working directory to the directory of app.py file."""
    try:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        
        if os.getcwd() != current_dir:
            os.chdir(current_dir)
            print(f"Working directory changed to: {current_dir}")
        else:
            print(f"Already in correct directory: {current_dir}")
            
        return current_dir
    except Exception as e:
        print(f"Warning: Could not set working directory: {e}")
        return os.getcwd()

# Set working directory first
PROJECT_DIR = set_working_directory()

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Import project modules
try:
    from cache_service import get_cache_service, cache_api_response
except ImportError:
    # Fallback cache implementation
    class FallbackCacheService:
        def __init__(self, config=None):
            self.cache_dirs = ['crypto_cache']
            os.makedirs('crypto_cache', exist_ok=True)
        
        def load_from_cache(self, key, extensions=None, max_age_hours=24):
            return None, False
        
        def save_to_cache(self, key, data, extension='json'):
            return ""
        
        def invalidate_cache(self, key):
            pass
        
        def get_cache_info(self):
            return {
                'directories': [{'path': d, 'exists': True, 'files': 0, 'size_mb': 0.0} for d in self.cache_dirs], 
                'total_files': 0,
                'file_count': {'total': 0}
            }
    
    def get_cache_service(config=None):
        return FallbackCacheService(config)
    
    def cache_api_response(prefix, identifier, callback, ttl_hours=24, force_refresh=False):
        return callback()

try:
    import helpers
except ImportError:
    # Minimal helpers fallback
    class helpers:
        @staticmethod
        def generate_chart_colors(scores):
            return ['#28a745' if score >= 0.7 else '#ffc107' if score >= 0.5 else '#dc3545' for score in scores]

# Try to import the analyzer
try:
    from crypto_analyzer import CryptoBullrunAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("Warning: crypto_analyzer.py not found. Running in demo mode.")

# Try to import Top200 analyzer
try:
    from top200_analyzer import AutonomousTop200Analyzer
    TOP200_AVAILABLE = True
except ImportError:
    TOP200_AVAILABLE = False
    print("Warning: top200_analyzer.py not found. Top200 analysis unavailable.")

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'crypto_analyzer_dashboard_2025')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

# Global settings and state
CONFIG = {
    'auto_refresh_minutes': 30,
    'cache_max_age_hours': 24,
    'chart_style': 'default',
    'max_coins_display': 50,
    'use_cache_fallback': True,
    'demo_mode': not ANALYZER_AVAILABLE
}

DASHBOARD_STATE = {
    'last_update': None,
    'watchlist_analysis': None,
    'portfolio_analysis': None,
    'top200_analysis': None,
    'cache_status': 'Loading...'
}

# Initialize cache service
cache_service = get_cache_service({
    'cache_directories': [
        'crypto_cache',
        os.path.join(os.path.expanduser('~'), '.cache', 'crypto_analyzer')
    ],
    'default_ttl_hours': CONFIG['cache_max_age_hours'],
    'auto_cleanup_on_start': True
})

# Initialize analyzers
analyzer = None
top200_analyzer = None

#
# Template Filters
#

@app.template_filter('score_class')
def score_class_filter(score):
    """Get CSS class for bullrun score"""
    if score is None:
        return 'bg-secondary'
    
    try:
        score = float(score)
        if score >= 0.8:
            return 'bg-success'
        elif score >= 0.7:
            return 'bg-info'
        elif score >= 0.6:
            return 'bg-warning'
        elif score >= 0.5:
            return 'bg-secondary'
        elif score >= 0.4:
            return 'bg-dark'
        else:
            return 'bg-danger'
    except (ValueError, TypeError):
        return 'bg-secondary'

@app.template_filter('potential_class')
def potential_class_filter(potential):
    """Get badge class for potential category"""
    if not potential:
        return 'bg-secondary'
    
    potential_map = {
        'Sehr hoch': 'bg-success',
        'Hoch': 'bg-info',
        'Überdurchschnittlich': 'bg-warning text-dark',
        'Durchschnittlich': 'bg-secondary',
        'Unterdurchschnittlich': 'bg-dark',
        'Niedrig': 'bg-danger'
    }
    return potential_map.get(potential, 'bg-secondary')

@app.template_filter('potential_badge_class')
def potential_badge_class_filter(potential):
    """Get badge class for potential category - alternative name"""
    return potential_class_filter(potential)  # Use existing filter

@app.template_filter('currency')
def currency_filter(value):
    """Format currency values"""
    if value is None or value == 0:
        return "$0.00"
    
    try:
        value = float(value)
        abs_value = abs(value)
        
        if abs_value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif abs_value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs_value >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs_value >= 1e3:
            return f"${value/1e3:.2f}K"
        elif abs_value >= 1:
            return f"${value:.2f}"
        else:
            return f"${value:.6f}"
    except (ValueError, TypeError):
        return "$0.00"

@app.template_filter('percentage')
def percentage_filter(value):
    """Format percentage values"""
    if value is None:
        return "N/A"
    
    try:
        value = float(value)
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.2f}%"
    except (ValueError, TypeError):
        return "N/A"

@app.template_filter('format_score')
def format_score_filter(score):
    """Format bullrun score for display"""
    if score is None:
        return "N/A"
    
    try:
        score = float(score)
        return f"{score:.3f}"
    except (ValueError, TypeError):
        return "N/A"

@app.template_filter('now')
def format_now(format_str):
    """Template filter to format current date"""
    return datetime.now().strftime(format_str)

#
# Helper Functions
#

def initialize_analyzers():
    """Initialize the crypto analyzers"""
    global analyzer, top200_analyzer, DASHBOARD_STATE
    
    if ANALYZER_AVAILABLE:
        try:
            analyzer = CryptoBullrunAnalyzer()
            DASHBOARD_STATE['cache_status'] = 'Live + Cache'
            logger.info("CryptoBullrunAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CryptoBullrunAnalyzer: {e}")
            DASHBOARD_STATE['cache_status'] = 'Error'
    else:
        DASHBOARD_STATE['cache_status'] = 'Demo Mode'
        
    if TOP200_AVAILABLE:
        try:
            top200_analyzer = AutonomousTop200Analyzer()
            logger.info("Top200 Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Top200 Analyzer: {e}")

def create_demo_data():
    """Create demo data when no analyzer is available"""
    return {
        'coins': [
            {
                'symbol': 'BTC', 
                'name': 'Bitcoin', 
                'bullrun_score': 0.85, 
                'total_score': 0.85, 
                'bullrun_potential': 'Sehr hoch', 
                'current_price': 45000.0, 
                'current_price_usd': 45000.0,
                'market_cap_usd': 800000000000, 
                'volume_24h_usd': 28000000000
            },
            {
                'symbol': 'ETH', 
                'name': 'Ethereum', 
                'bullrun_score': 0.78, 
                'total_score': 0.78, 
                'bullrun_potential': 'Hoch', 
                'current_price': 3200.0, 
                'current_price_usd': 3200.0,
                'market_cap_usd': 380000000000, 
                'volume_24h_usd': 15000000000
            },
            {
                'symbol': 'SOL', 
                'name': 'Solana', 
                'bullrun_score': 0.72, 
                'total_score': 0.72, 
                'bullrun_potential': 'Hoch', 
                'current_price': 150.0, 
                'current_price_usd': 150.0,
                'market_cap_usd': 55000000000, 
                'volume_24h_usd': 3500000000
            },
            {
                'symbol': 'ADA', 
                'name': 'Cardano', 
                'bullrun_score': 0.65, 
                'total_score': 0.65, 
                'bullrun_potential': 'Überdurchschnittlich', 
                'current_price': 0.45, 
                'current_price_usd': 0.45,
                'market_cap_usd': 15000000000, 
                'volume_24h_usd': 950000000
            },
            {
                'symbol': 'DOT', 
                'name': 'Polkadot', 
                'bullrun_score': 0.58, 
                'total_score': 0.58, 
                'bullrun_potential': 'Durchschnittlich', 
                'current_price': 7.2, 
                'current_price_usd': 7.2,
                'market_cap_usd': 8500000000, 
                'volume_24h_usd': 350000000
            },
        ],
        'potential_categories': {
            'very_high': ['BTC'],
            'high': ['ETH', 'SOL'],
            'above_average': ['ADA'],
            'average': ['DOT'],
            'below_average': [],
            'low': []
        }
    }

def normalize_coin_data(coin_data):
    """Normalize coin data for consistent template usage"""
    if not coin_data:
        return coin_data
    
    normalized = {}
    
    # Basic fields
    normalized['symbol'] = coin_data.get('symbol', '')
    normalized['name'] = coin_data.get('name', '')
    
    # Score fields (with fallback)
    score = coin_data.get('bullrun_score', coin_data.get('total_score', 0))
    normalized['bullrun_score'] = float(score) if score is not None else 0.0
    normalized['total_score'] = normalized['bullrun_score']
    
    # Potential
    normalized['bullrun_potential'] = coin_data.get('bullrun_potential', 'Unbekannt')
    
    # Price fields (with fallback)
    price = coin_data.get('current_price_usd', coin_data.get('current_price', 0))
    normalized['current_price'] = float(price) if price is not None else 0.0
    normalized['current_price_usd'] = normalized['current_price']
    
    # Market Cap (with fallback)
    market_cap = coin_data.get('market_cap_usd', coin_data.get('market_cap', 0))
    normalized['market_cap'] = float(market_cap) if market_cap is not None else 0.0
    normalized['market_cap_usd'] = normalized['market_cap']
    
    # Additional fields
    volume = coin_data.get('volume_24h_usd', coin_data.get('volume_24h', 0))
    normalized['volume_24h_usd'] = float(volume) if volume is not None else 0.0
    
    normalized['high_distance_percent'] = float(coin_data.get('high_distance_percent', 0)) if coin_data.get('high_distance_percent') is not None else 0.0
    normalized['market_cap_rank'] = coin_data.get('market_cap_rank', coin_data.get('cmc_rank', 0))
    
    # Copy other fields
    for key, value in coin_data.items():
        if key not in normalized:
            normalized[key] = value
    
    return normalized

def analyze_watchlist_safe(force_refresh=False):
    """Analyze watchlist using cache when possible"""
    # Try to load from cache first
    if not force_refresh:
        try:
            cached_analysis, success = cache_service.load_from_cache('watchlist_analysis', ['json'], CONFIG['cache_max_age_hours'])
            if success and cached_analysis:
                logger.info("Using cached watchlist analysis")
                return cached_analysis
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
    
    # Try real analyzer if available
    if analyzer:
        try:
            analysis = analyzer.analyze_watchlist()
            if "error" not in analysis:
                # Save to cache
                cache_service.save_to_cache('watchlist_analysis', analysis, 'json')
                return analysis
        except Exception as e:
            logger.error(f"Real analyzer failed, using fallback: {e}")
    
    # Create demo data if nothing available
    return create_demo_data()

def load_top200_analysis():
    """Load the most recent Top200 analysis from cache"""
    # First try direct cache
    try:
        cached_analysis, success = cache_service.load_from_cache('top200_analysis', ['json'], CONFIG['cache_max_age_hours'] * 2)
        if success and cached_analysis:
            return cached_analysis
    except Exception as e:
        logger.warning(f"Top200 cache load failed: {e}")
    
    # Then try to find analysis files
    analysis_files = get_analysis_files()
    
    # Look specifically for Top200 files first
    for file_info in analysis_files:
        if 'top200' in file_info['name'].lower():
            try:
                file_path = file_info['path']
                
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        analysis_data = csv_to_analysis_format(df)
                        if analysis_data:
                            logger.info(f"Loaded Top200 analysis from {file_info['name']}")
                            return analysis_data
                                
            except Exception as e:
                logger.error(f"Error loading Top200 analysis from {file_info['name']}: {e}")
                continue
    
    return None

def get_analysis_files():
    """Get available analysis files"""
    analysis_files = []
    
    # Define possible cache dirs - safer approach
    try:
        if hasattr(cache_service, 'cache_dirs'):
            cache_dirs = cache_service.cache_dirs
        else:
            cache_dirs = ['crypto_cache']
    except Exception:
        cache_dirs = ['crypto_cache']
    
    # Add additional directories
    cache_dirs = cache_dirs + [
        os.path.join('novumLabs', '13_Crypto Analyzer', 'analysis'),
        os.path.join('analysis_results'),
        'analysis_results'
    ]
    
    for cache_dir in cache_dirs:
        try:
            if not os.path.exists(cache_dir):
                continue
                
            # Look for various analysis file patterns
            patterns = [
                "top_bullrun_analysis_*.csv",
                "top200_analysis_*.csv",
                "top200_analysis_*.html",
                "crypto_report_*.csv", 
                "*analysis*.csv"
            ]
            
            for pattern in patterns:
                import glob
                files = glob.glob(os.path.join(cache_dir, pattern))
                for file_path in files:
                    try:
                        file_time = os.path.getmtime(file_path)
                        analysis_files.append({
                            'path': file_path,
                            'name': os.path.basename(file_path),
                            'modified': datetime.fromtimestamp(file_time),
                            'size': os.path.getsize(file_path),
                            'type': 'Top200' if 'top200' in file_path.lower() else 'Standard'
                        })
                    except Exception:
                        continue
        except Exception:
            continue
    
    # Sort by modification time, newest first
    analysis_files.sort(key=lambda x: x['modified'], reverse=True)
    return analysis_files

def csv_to_analysis_format(df):
    """Convert CSV analysis to expected format"""
    try:
        coins = []
        potential_categories = {
            "very_high": [],
            "high": [],
            "above_average": [],
            "average": [],
            "below_average": [],
            "low": []
        }
        
        for _, row in df.iterrows():
            symbol = str(row.get('Symbol', row.get('symbol', '')))
            name = str(row.get('Name', row.get('name', symbol)))
            score = float(row.get('Bullrun_Score', row.get('bullrun_score', row.get('Score', 0))))
            potential = str(row.get('Bullrun_Potential', row.get('bullrun_potential', row.get('Potential', 'Unknown'))))
            price = float(row.get('Price_USD', row.get('current_price', row.get('Preis_USD', 0))))
            market_cap = float(row.get('Market_Cap_USD', row.get('market_cap_usd', 0)))
            
            coin_data = {
                'symbol': symbol,
                'name': name,
                'bullrun_score': score,
                'total_score': score,
                'bullrun_potential': potential,
                'current_price': price,
                'current_price_usd': price,
                'market_cap_usd': market_cap
            }
            coins.append(coin_data)
            
            # Categorize
            if score >= 0.8:
                potential_categories["very_high"].append(symbol)
            elif score >= 0.7:
                potential_categories["high"].append(symbol)
            elif score >= 0.6:
                potential_categories["above_average"].append(symbol)
            elif score >= 0.5:
                potential_categories["average"].append(symbol)
            elif score >= 0.4:
                potential_categories["below_average"].append(symbol)
            else:
                potential_categories["low"].append(symbol)
        
        # Sort by score
        coins.sort(key=lambda x: x['bullrun_score'], reverse=True)
        
        return {
            'coins': coins,
            'potential_categories': potential_categories
        }
        
    except Exception as e:
        logger.error(f"Error converting CSV to analysis format: {e}")
        return None

def update_dashboard_data(force_refresh=False):
    """Update all dashboard data using cache-first approach"""
    global DASHBOARD_STATE
    
    try:
        logger.info("Updating dashboard data (cache-first)...")
        
        if analyzer:
            # Get watchlist analysis (cache-first)
            watchlist_analysis = analyze_watchlist_safe(force_refresh)
            
            # Normalize all coin data
            if watchlist_analysis and watchlist_analysis.get('coins'):
                watchlist_analysis['coins'] = [normalize_coin_data(coin) for coin in watchlist_analysis['coins']]
            
            DASHBOARD_STATE['watchlist_analysis'] = watchlist_analysis
            
            # Try portfolio analysis if real analyzer available
            try:
                portfolio_analysis = analyzer.analyze_portfolio()
                if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                    portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                    DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
            except Exception as e:
                logger.error(f"Error analyzing portfolio: {e}")
                DASHBOARD_STATE['portfolio_analysis'] = None
            
            # Load Top200 analysis from cache
            top200_analysis = load_top200_analysis()
            if top200_analysis and top200_analysis.get('coins'):
                top200_analysis['coins'] = [normalize_coin_data(coin) for coin in top200_analysis['coins']]
            DASHBOARD_STATE['top200_analysis'] = top200_analysis
            
        else:
            # Demo mode - data is already normalized
            DASHBOARD_STATE['watchlist_analysis'] = create_demo_data()
            DASHBOARD_STATE['portfolio_analysis'] = None
            DASHBOARD_STATE['top200_analysis'] = None
        
        DASHBOARD_STATE['last_update'] = datetime.now().isoformat()
        logger.info("Dashboard data updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating dashboard data: {e}")
        return False

def generate_chart_colors(scores):
    """Generate color codes based on scores"""
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

def generate_charts(analysis_result):
    """Generate base64 encoded charts for the analysis result"""
    charts = {}
    
    if not analysis_result:
        return charts
    
    try:
        valid_coins = [coin for coin in analysis_result.get("coins", []) 
                      if coin.get("bullrun_score") is not None or coin.get("total_score") is not None]
        
        if not valid_coins:
            return charts
        
        # 1. Top 10 Coins Bar Chart
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_coins = sorted(valid_coins, key=lambda x: x.get("bullrun_score", x.get("total_score", 0)), reverse=True)[:10]
        symbols = [coin["symbol"] for coin in top_coins]
        scores = [coin.get("bullrun_score", coin.get("total_score", 0)) for coin in top_coins]
        
        # Color coding
        colors = generate_chart_colors(scores)
        
        bars = ax.bar(symbols, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add score values on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Top 10 Coins - Bullrun Scores', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Bullrun Score', fontsize=12)
        ax.set_xlabel('Cryptocurrency', fontsize=12)
        ax.set_ylim(0, max(scores) * 1.1 if scores else 1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        charts['top_coins'] = create_chart_base64(fig)
        
        # 2. Potential Distribution Pie Chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        categories = ['Sehr hoch', 'Hoch', 'Überdurchschnittlich', 
                     'Durchschnittlich', 'Unterdurchschnittlich', 'Niedrig']
        values = [
            len(analysis_result["potential_categories"]["very_high"]),
            len(analysis_result["potential_categories"]["high"]),
            len(analysis_result["potential_categories"]["above_average"]),
            len(analysis_result["potential_categories"]["average"]),
            len(analysis_result["potential_categories"]["below_average"]),
            len(analysis_result["potential_categories"]["low"])
        ]
        colors_pie = ['#2E8B57', '#32CD32', '#FFD700', '#FFA500', '#FF8C00', '#FF6B6B']
        
        # Only categories with values > 0
        non_zero_data = [(cat, val, col) for cat, val, col in zip(categories, values, colors_pie) if val > 0]
        
        if non_zero_data:
            categories, values, colors_pie = zip(*non_zero_data)
            
            wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors_pie,
                                             autopct='%1.1f%%', startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Potential Distribution', fontsize=16, fontweight='bold')
            
        charts['distribution'] = create_chart_base64(fig)
        
        logger.info(f"Generated {len(charts)} charts successfully")
        
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
    
    return charts

def create_chart_base64(fig):
    """Convert matplotlib figure to base64 encoded string"""
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return f"data:image/png;base64,{chart_base64}"
    except Exception as e:
        plt.close(fig)
        logger.error(f"Error creating chart: {e}")
        return None

#
# Template Helper Functions
#

def get_score_class_helper(score):
    """Helper function for template score classes"""
    if score is None:
        return 'score-avg'
    
    try:
        score = float(score)
        if score >= 0.8:
            return 'score-very-high'
        elif score >= 0.7:
            return 'score-high'
        elif score >= 0.6:
            return 'score-above-avg'
        elif score >= 0.5:
            return 'score-avg'
        elif score >= 0.4:
            return 'score-below-avg'
        else:
            return 'score-low'
    except (ValueError, TypeError):
        return 'score-avg'

def get_potential_class_helper(potential):
    """Helper function for template potential classes"""
    if not potential:
        return 'badge-durchschnittlich'
    
    potential_map = {
        'Sehr hoch': 'badge-sehr-hoch',
        'Hoch': 'badge-hoch',
        'Überdurchschnittlich': 'badge-ueberdurchschnittlich',
        'Durchschnittlich': 'badge-durchschnittlich',
        'Unterdurchschnittlich': 'badge-unterdurchschnittlich',
        'Niedrig': 'badge-niedrig'
    }
    return potential_map.get(potential, 'badge-durchschnittlich')

#
# Context Processors and Template Globals
#

@app.context_processor
def inject_global_variables():
    """Inject global variables into templates"""
    return {
        'cache_status': DASHBOARD_STATE['cache_status'],
        'auto_refresh_enabled': True,
        'auto_refresh_minutes': CONFIG['auto_refresh_minutes'],
        'last_update': DASHBOARD_STATE['last_update'],
        'app_name': 'Crypto Bullrun Analyzer',
        'app_version': '2.0.0',
        'current_time': datetime.now(),
        'analyzer_available': ANALYZER_AVAILABLE,
        'top200_available': TOP200_AVAILABLE,
        'getScoreClass': get_score_class_helper,
        'getPotentialClass': get_potential_class_helper
    }

#
# Flask Routes
#

@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        # Update data if needed
        if (not DASHBOARD_STATE['last_update'] or 
            datetime.fromisoformat(DASHBOARD_STATE['last_update']) < 
            datetime.now() - timedelta(minutes=CONFIG['auto_refresh_minutes'])):
            update_dashboard_data()
        
        # Generate charts
        charts = generate_charts(DASHBOARD_STATE['watchlist_analysis'])
        
        # Prepare statistics
        stats = {}
        
        if DASHBOARD_STATE['watchlist_analysis']:
            analysis = DASHBOARD_STATE['watchlist_analysis']
            valid_coins = [coin for coin in analysis.get("coins", []) 
                          if coin.get("bullrun_score") is not None or coin.get("total_score") is not None]
            
            stats = {
                'total_coins': len(valid_coins),
                'avg_score': sum(coin.get("bullrun_score", coin.get("total_score", 0)) for coin in valid_coins) / len(valid_coins) if valid_coins else 0,
                'top_performers': len([coin for coin in valid_coins if coin.get("bullrun_score", coin.get("total_score", 0)) >= 0.7]),
                'very_high_potential': len(analysis["potential_categories"]["very_high"])
            }
        
        return render_template(
            'dashboard.html',
            active_page='dashboard',
            charts=charts,
            stats=stats,
            watchlist_analysis=DASHBOARD_STATE['watchlist_analysis'],
            last_update=DASHBOARD_STATE['last_update']
        )
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        return render_template(
            'dashboard.html',
            active_page='dashboard',
            error_message=f"An error occurred: {str(e)}",
            last_update=datetime.now().isoformat()
        )

@app.route('/top200')
def top200_page():
    """Top 200 Analysis Page"""
    # Get latest Top200 analysis
    top200_data = DASHBOARD_STATE.get('top200_analysis')
    analysis_files = get_analysis_files()
    
    # Filter for Top200 files only
    top200_files = [f for f in analysis_files if 'top200' in f['name'].lower()]
    
    return render_template(
        'top200.html',
        active_page='top200',
        top200_data=top200_data,
        top200_files=top200_files[:10],
        analysis_files=analysis_files[:20],
        top200_available=TOP200_AVAILABLE
    )

@app.route('/watchlist')
def watchlist_page():
    """Watchlist management page"""
    watchlist_coins = []
    coin_data = {}
    
    try:
        if analyzer:
            try:
                watchlist = analyzer.load_watchlist()
                watchlist_coins = watchlist.get("coins", [])
                
                # Load additional coin data (names, etc.)
                for symbol in watchlist_coins[:10]:  # Only first 10 for performance
                    try:
                        coin_data[symbol] = analyzer.fetch_coin_data(symbol)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error loading watchlist: {e}")
    except Exception as e:
        logger.error(f"Critical error in watchlist_page: {e}")
    
    return render_template(
        'watchlist.html',
        active_page='watchlist',
        watchlist_coins=watchlist_coins,
        coin_data=coin_data,
        analysis=DASHBOARD_STATE.get('watchlist_analysis')
    )

@app.route('/portfolio')
def portfolio_page():
    """Portfolio management and analysis page"""
    portfolio = {}
    
    if analyzer:
        try:
            portfolio = analyzer.load_portfolio()
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
    
    return render_template(
        'portfolio.html',
        active_page='portfolio',
        portfolio=portfolio,
        portfolio_analysis=DASHBOARD_STATE.get('portfolio_analysis')
    )

@app.route('/hardware')
def hardware_page():
    """Hardware wallet integration page"""
    return render_template(
        'hardware.html',
        active_page='hardware'
    )

@app.route('/cache-status')
def cache_status_page():
    """Display cache status and available data"""
    try:
        # Sichere Cache-Info Abfrage
        cache_info = {}
        if hasattr(cache_service, 'get_cache_info') and callable(getattr(cache_service, 'get_cache_info')):
            try:
                cache_info = cache_service.get_cache_info()
            except Exception as e:
                logger.error(f"Error getting cache info: {e}")
                cache_info = {
                    'directories': [{'path': d, 'exists': True, 'files': 0, 'size_mb': 0.0} for d in getattr(cache_service, 'cache_dirs', ['crypto_cache'])],
                    'total_files': 0,
                    'file_count': {'total': 0}
                }
        else:
            # Fallback für einfache Cache-Implementierung
            cache_dirs = getattr(cache_service, 'cache_dirs', ['crypto_cache'])
            cache_info = {
                'directories': [{'path': d, 'exists': os.path.exists(d), 'files': 0, 'size_mb': 0.0} for d in cache_dirs],
                'total_files': 0,
                'file_count': {'total': 0}
            }
        
        # Analysis files sicher abrufen
        analysis_files = []
        try:
            analysis_files = get_analysis_files()
        except Exception as e:
            logger.error(f"Error getting analysis files: {e}")
            analysis_files = []
        
        # Files kategorisieren
        top200_files = [f for f in analysis_files if 'top200' in f.get('name', '').lower()]
        standard_files = [f for f in analysis_files if 'top200' not in f.get('name', '').lower()]
        
        return render_template(
            'cache_status.html',
            active_page='cache',
            cache_info=cache_info,
            top200_files=top200_files[:20],
            standard_files=standard_files[:20], 
            total_files=len(analysis_files),
            analyzer_available=ANALYZER_AVAILABLE,
            top200_available=TOP200_AVAILABLE
        )
        
    except Exception as e:
        logger.error(f"Critical error in cache_status_page: {e}")
        # Minimal fallback
        return render_template(
            'cache_status.html',
            active_page='cache',
            cache_info={'directories': [], 'total_files': 0},
            top200_files=[],
            standard_files=[],
            total_files=0,
            analyzer_available=False,
            top200_available=False,
            error_message=f"Cache status error: {str(e)}"
        )

@app.route('/load-analysis')
def load_analysis():
    """Load specific analysis file"""
    file_path = request.args.get('file')
    if not file_path or not os.path.exists(file_path):
        flash('File not found', 'error')
        return redirect(url_for('cache_status_page'))
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            analysis_data = csv_to_analysis_format(df)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                analysis_data = json.load(f)
        else:
            flash('Unsupported file format', 'error')
            return redirect(url_for('cache_status_page'))
        
        if analysis_data:
            # Normalize data
            if analysis_data.get('coins'):
                analysis_data['coins'] = [normalize_coin_data(coin) for coin in analysis_data['coins']]
            
            # Determine if this is Top200 data
            if 'top200' in file_path.lower():
                DASHBOARD_STATE['top200_analysis'] = analysis_data
                flash(f'Top200 analysis loaded from {os.path.basename(file_path)}', 'success')
                redirect_target = 'top200_page'
            else:
                DASHBOARD_STATE['watchlist_analysis'] = analysis_data
                flash(f'Analysis data loaded from {os.path.basename(file_path)}', 'success')
                redirect_target = 'dashboard'
                
            DASHBOARD_STATE['last_update'] = datetime.now().isoformat()
            return redirect(url_for(redirect_target))
        else:
            flash('Invalid analysis data format', 'error')
            
    except Exception as e:
        flash(f'Error loading analysis: {str(e)}', 'error')
    
    return redirect(url_for('cache_status_page'))

@app.route('/docs')
def docs_page():
    """Documentation page"""
    try:
        # Get system status for display
        system_info = {
            'analyzer_available': ANALYZER_AVAILABLE,
            'top200_available': TOP200_AVAILABLE,
            'cache_status': DASHBOARD_STATE['cache_status'],
            'total_files': len(get_analysis_files()),
            'version': '2.0.0'
        }
        
        return render_template(
            'docs.html',
            active_page='docs',
            system_info=system_info,
            cache_status=DASHBOARD_STATE['cache_status']
        )
    except Exception as e:
        logger.error(f"Error loading documentation page: {e}")
        return render_template(
            'docs.html',
            active_page='docs',
            system_info={},
            cache_status='Error'
        )

#
# API Routes
#

@app.route('/api/refresh')
def api_refresh():
    """API endpoint to refresh data"""
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    success = update_dashboard_data(force_refresh)
    message = 'Data refreshed successfully' if success else 'Error refreshing data'
    
    return jsonify({
        'success': success,
        'last_update': DASHBOARD_STATE['last_update'],
        'message': message
    })

@app.route('/api/analyze-watchlist', methods=['POST'])
def api_analyze_watchlist():
    """Analyze watchlist"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        force_refresh = request.json.get('force_refresh', False) if request.is_json else False
        analysis = analyze_watchlist_safe(force_refresh)
        
        # Normalize data
        if analysis and analysis.get('coins'):
            analysis['coins'] = [normalize_coin_data(coin) for coin in analysis['coins']]
        
        DASHBOARD_STATE['watchlist_analysis'] = analysis
        DASHBOARD_STATE['last_update'] = datetime.now().isoformat()
        return jsonify({'success': True, 'message': 'Analysis completed'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Analysis error: {str(e)}'})

@app.route('/api/run-top200', methods=['POST'])
def api_run_top200():
    """API endpoint to run Top200 analysis"""
    if not TOP200_AVAILABLE or not top200_analyzer:
        return jsonify({'success': False, 'message': 'Top200 analyzer not available'})
    
    try:
        # Run the analysis
        result = top200_analyzer.run_analysis()
        
        if result and result.get('successful_analyses', 0) > 0:
            # Update dashboard data with new analysis
            new_analysis = load_top200_analysis()
            if new_analysis and new_analysis.get('coins'):
                new_analysis['coins'] = [normalize_coin_data(coin) for coin in new_analysis['coins']]
                DASHBOARD_STATE['top200_analysis'] = new_analysis
                DASHBOARD_STATE['last_update'] = datetime.now().isoformat()
            
            return jsonify({
                'success': True,
                'successful_analyses': result['successful_analyses'],
                'failed_analyses': result['failed_analyses'],
                'success_rate': result['success_rate'],
                'duration_seconds': result['duration_seconds'],
                'message': f'Top200 analysis completed successfully'
            })
        else:
            return jsonify({'success': False, 'message': 'Analysis failed to complete or produced no results'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Analysis error: {str(e)}'})

@app.route('/watchlist/add', methods=['POST'])
def watchlist_add():
    """Add coin to watchlist"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbol = request.json.get('symbol', '').upper().strip()
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        success = analyzer.add_to_watchlist(symbol)
        
        if success:
            # Invalidate cache
            cache_service.invalidate_cache('watchlist_analysis')
            return jsonify({'success': True, 'message': f'{symbol} added to watchlist'})
        else:
            return jsonify({'success': False, 'message': f'Could not add {symbol}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/watchlist/remove', methods=['POST'])
def watchlist_remove():
    """Remove coin from watchlist"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbol = request.json.get('symbol', '').upper().strip()
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        success = analyzer.remove_from_watchlist(symbol)
        
        if success:
            # Invalidate cache
            cache_service.invalidate_cache('watchlist_analysis')
            return jsonify({'success': True, 'message': f'{symbol} removed from watchlist'})
        else:
            return jsonify({'success': False, 'message': f'{symbol} not found in watchlist'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/watchlist/bulk-add', methods=['POST'])
def watchlist_bulk_add():
    """Add multiple coins to watchlist"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbols = request.json.get('symbols', [])
        if not symbols:
            return jsonify({'success': False, 'message': 'No symbols provided'})
        
        added_count = 0
        skipped_count = 0
        
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if symbol:
                try:
                    if analyzer.add_to_watchlist(symbol):
                        added_count += 1
                    else:
                        skipped_count += 1
                except Exception:
                    skipped_count += 1
        
        # Invalidate cache if any coins were added
        if added_count > 0:
            cache_service.invalidate_cache('watchlist_analysis')
            
        return jsonify({
            'success': True, 
            'added_count': added_count,
            'skipped_count': skipped_count,
            'message': f'{added_count} coins added, {skipped_count} skipped'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/portfolio/add', methods=['POST'])
def portfolio_add():
    """Add coin to portfolio"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbol = request.json.get('symbol', '').upper().strip()
        amount = request.json.get('amount', 0)
        avg_price = request.json.get('avg_price', 0)
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        try:
            amount = float(amount)
            avg_price = float(avg_price)
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid amount or price'})
        
        success = analyzer.add_to_portfolio(symbol, amount, avg_price)
        
        if success:
            # Update portfolio analysis
            try:
                portfolio_analysis = analyzer.analyze_portfolio()
                if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                    portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                    DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
            except Exception:
                pass
                
            return jsonify({'success': True, 'message': f'{symbol} added to portfolio'})
        else:
            return jsonify({'success': False, 'message': f'Could not add {symbol}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/portfolio/remove', methods=['POST'])
def portfolio_remove():
    """Remove coin from portfolio"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbol = request.json.get('symbol', '').upper().strip()
        amount = request.json.get('amount', None)
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        if amount is not None:
            try:
                amount = float(amount)
            except ValueError:
                return jsonify({'success': False, 'message': 'Invalid amount'})
        
        success = analyzer.remove_from_portfolio(symbol, amount)
        
        if success:
            # Update portfolio analysis
            try:
                portfolio_analysis = analyzer.analyze_portfolio()
                if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                    portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                    DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
            except Exception:
                pass
                
            return jsonify({'success': True, 'message': f'{symbol} removed from portfolio'})
        else:
            return jsonify({'success': False, 'message': f'{symbol} not found in portfolio'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# Add new portfolio routes here
@app.route('/portfolio/update', methods=['POST'])
def portfolio_update():
    """Update existing portfolio position"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbol = request.json.get('symbol', '').upper().strip()
        amount = request.json.get('amount', 0)
        avg_price = request.json.get('avg_price', 0)
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        try:
            amount = float(amount)
            avg_price = float(avg_price)
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid amount or price'})
        
        if amount < 0 or avg_price < 0:
            return jsonify({'success': False, 'message': 'Amount and price must be positive'})
        
        # Check if position exists
        current_position = analyzer.get_portfolio_position(symbol)
        if not current_position:
            return jsonify({'success': False, 'message': f'{symbol} not found in portfolio'})
        
        # Update position (set mode - overwrites existing)
        success = analyzer.add_to_portfolio(symbol, amount, avg_price, mode="set")
        
        if success:
            # Update portfolio analysis
            try:
                portfolio_analysis = analyzer.analyze_portfolio()
                if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                    portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                    DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
            except Exception:
                pass
                
            return jsonify({'success': True, 'message': f'{symbol} position updated successfully'})
        else:
            return jsonify({'success': False, 'message': f'Could not update {symbol}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/portfolio/modify-amount', methods=['POST'])
def portfolio_modify_amount():
    """Modify amount in portfolio position"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbol = request.json.get('symbol', '').upper().strip()
        new_amount = request.json.get('new_amount', 0)
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        try:
            new_amount = float(new_amount)
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid amount'})
        
        if new_amount < 0:
            return jsonify({'success': False, 'message': 'Amount must be positive'})
        
        # Check if position exists
        current_position = analyzer.get_portfolio_position(symbol)
        if not current_position:
            return jsonify({'success': False, 'message': f'{symbol} not found in portfolio'})
        
        # Modify amount only (keep same avg price)
        success = analyzer.modify_portfolio_amount(symbol, new_amount, "set")
        
        if success:
            # Update portfolio analysis
            try:
                portfolio_analysis = analyzer.analyze_portfolio()
                if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                    portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                    DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
            except Exception:
                pass
                
            if new_amount == 0:
                return jsonify({'success': True, 'message': f'{symbol} removed from portfolio'})
            else:
                return jsonify({'success': True, 'message': f'{symbol} amount updated to {new_amount}'})
        else:
            return jsonify({'success': False, 'message': f'Could not modify {symbol}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/portfolio/get-position/<symbol>')
def portfolio_get_position(symbol):
    """Get portfolio position details"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        symbol = symbol.upper().strip()
        position = analyzer.get_portfolio_position(symbol)
        
        if position:
            return jsonify({
                'success': True,
                'position': position
            })
        else:
            return jsonify({
                'success': False, 
                'message': f'{symbol} not found in portfolio'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/portfolio/add-mode', methods=['POST'])
def portfolio_add_mode():
    """Add coin to portfolio with specified mode (add/set)"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})
            
        symbol = request.json.get('symbol', '').upper().strip()
        amount = request.json.get('amount', 0)
        avg_price = request.json.get('avg_price', 0)
        mode = request.json.get('mode', 'add')  # 'add' or 'set'
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        if mode not in ['add', 'set']:
            return jsonify({'success': False, 'message': 'Mode must be "add" or "set"'})
        
        try:
            amount = float(amount)
            avg_price = float(avg_price)
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid amount or price'})
        
        if amount <= 0 or avg_price <= 0:
            return jsonify({'success': False, 'message': 'Amount and price must be positive'})
        
        # Use the new add_to_portfolio with mode parameter
        success = analyzer.add_to_portfolio(symbol, amount, avg_price, mode=mode)
        
        if success:
            # Update portfolio analysis
            try:
                portfolio_analysis = analyzer.analyze_portfolio()
                if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                    portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                    DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
            except Exception:
                pass
                
            action = "added to" if mode == "add" else "set in"
            return jsonify({'success': True, 'message': f'{symbol} {action} portfolio'})
        else:
            return jsonify({'success': False, 'message': f'Could not process {symbol}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/export-portfolio')
def api_export_portfolio():
    """Export portfolio to CSV"""
    if not analyzer:
        flash('Analyzer not available', 'error')
        return redirect(url_for('portfolio_page'))
    
    try:
        portfolio = analyzer.load_portfolio()
        portfolio_analysis = analyzer.analyze_portfolio()
        
        if "error" in portfolio_analysis or not portfolio_analysis.get('coins'):
            flash('No portfolio data to export', 'warning')
            return redirect(url_for('portfolio_page'))
        
        # Create CSV data
        csv_data = "Symbol,Name,Amount,Avg Price,Current Price,Value,P/L,P/L %,Bullrun Score\n"
        
        for coin in portfolio_analysis['coins']:
            csv_data += f"{coin['symbol']},{coin['name']},{coin['amount']},{coin['avg_price']},"
            csv_data += f"{coin['current_price']},{coin['value']},{coin['profit_loss']},"
            csv_data += f"{coin['profit_loss_percent']},{coin['bullrun_score']}\n"
        
        # Create response
        output = io.StringIO()
        output.write(csv_data)
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=portfolio_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
    except Exception as e:
        flash(f'Error exporting portfolio: {str(e)}', 'error')
        return redirect(url_for('portfolio_page'))

@app.route('/portfolio/import', methods=['POST'])
def portfolio_import():
    """Import portfolio from CSV file"""
    if not analyzer:
        return jsonify({'success': False, 'message': 'Analyzer not available'})
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'message': 'File must be CSV format'})
        
        # Read CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.DictReader(stream)
        
        imported = 0
        errors = []
        
        for row in csv_reader:
            try:
                symbol = row.get('Symbol', '').strip().upper()
                amount = float(row.get('Amount', 0))
                price = float(row.get('Price', row.get('Avg Price', 0)))
                
                if symbol and amount > 0 and price > 0:
                    if analyzer.add_to_portfolio(symbol, amount, price, mode="set"):
                        imported += 1
                    else:
                        errors.append(f"{symbol}: Failed to add")
                else:
                    errors.append(f"Row {csv_reader.line_num}: Invalid data")
            except Exception as e:
                errors.append(f"Row {csv_reader.line_num}: {str(e)}")
        
        # Update portfolio analysis
        try:
            portfolio_analysis = analyzer.analyze_portfolio()
            if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
        except Exception:
            pass
        
        if imported > 0:
            message = f'Successfully imported {imported} positions'
            if errors:
                message += f' ({len(errors)} errors)'
            return jsonify({'success': True, 'message': message, 'errors': errors[:5]})
        else:
            return jsonify({'success': False, 'message': 'No positions imported', 'errors': errors[:5]})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Import error: {str(e)}'})

@app.route('/api/hardware-scan', methods=['POST'])
def api_hardware_scan():
    """Scan hardware wallets"""
    try:
        wallet_type = request.json.get('wallet_type', 'manual')
        addresses = request.json.get('addresses', [])
        
        # Import hardware wallet module dynamically
        try:
            from hardware_wallet_integration import HardwareWalletIntegration
            wallet_scanner = HardwareWalletIntegration()
            
            results = []
            
            if wallet_type == 'ledger':
                results = wallet_scanner.read_ledger_wallet()
            elif wallet_type == 'trezor':
                results = wallet_scanner.read_trezor_wallet()
            elif wallet_type == 'manual':
                results = wallet_scanner.read_manual_addresses(addresses)
            
            # Add USD values
            results_with_value = wallet_scanner.add_usd_values(results)
            
            return jsonify({
                'success': True,
                'results': [r.__dict__ for r in results_with_value],
                'total_addresses': len(results),
                'total_value': sum(r.usd_value or 0 for r in results_with_value)
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'message': 'Hardware wallet integration not available',
                'error': 'module_not_found'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error scanning wallets: {str(e)}',
            'error': 'scan_failed'
        })

@app.route('/api/hardware-import', methods=['POST'])
def api_hardware_import():
    """Import hardware wallet balances to portfolio"""
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'message': 'Analyzer not available',
                'error': 'analyzer_not_found'
            })
            
        balances = request.json.get('balances', [])
        
        if not balances:
            return jsonify({
                'success': False,
                'message': 'No balances provided',
                'error': 'no_data'
            })
            
        # Import hardware wallet module dynamically
        try:
            from hardware_wallet_integration import HardwareWalletIntegration
            wallet_importer = HardwareWalletIntegration()
            
            # Convert balances from JSON to objects
            from dataclasses import dataclass
            
            @dataclass
            class WalletBalance:
                symbol: str
                balance: float
                address: str
                network: str
                usd_value: Optional[float] = None
            
            balance_objects = []
            for b in balances:
                balance_objects.append(WalletBalance(
                    symbol=b.get('symbol', ''),
                    balance=float(b.get('balance', 0)),
                    address=b.get('address', ''),
                    network=b.get('network', ''),
                    usd_value=float(b.get('usd_value', 0))
                ))
            
            success = wallet_importer.import_to_analyzer(balance_objects)
            
            if success:
                # Update portfolio analysis
                try:
                    portfolio_analysis = analyzer.analyze_portfolio()
                    if "error" not in portfolio_analysis and portfolio_analysis.get('coins'):
                        portfolio_analysis['coins'] = [normalize_coin_data(coin) for coin in portfolio_analysis['coins']]
                        DASHBOARD_STATE['portfolio_analysis'] = portfolio_analysis
                except Exception:
                    pass
                    
                return jsonify({
                    'success': True,
                    'message': f'Successfully imported {len(balances)} balances to portfolio'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Import failed',
                    'error': 'import_failed'
                })
                
        except ImportError:
            return jsonify({
                'success': False,
                'message': 'Hardware wallet integration not available',
                'error': 'module_not_found'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error importing balances: {str(e)}',
            'error': 'import_failed'
        })

#
# Error Handlers
#

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page Not Found",
                         error_description="The page you're looking for doesn't exist."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal Server Error",
                         error_description="Something went wrong on our end."), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle general exceptions with improved debugging"""
    import traceback
    error_details = traceback.format_exc()
    
    # Log specific error types
    error_str = str(e)
    if "potential_badge_class" in error_str:
        logger.error("Template filter error - potential_badge_class filter should now be available")
    elif "file_count" in error_str:
        logger.error("Cache service error - check cache_service implementation")
    elif "socket.io" in error_str:
        logger.error("Socket.IO error - remove Socket.IO references from templates")
    else:
        logger.error(f"Unhandled exception: {e}\n{error_details}")
    
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Application Error",
                         error_description="An unexpected error occurred.",
                         error_details=error_str if app.debug else None), 500

#
# Application Initialization
#

# Initialize application components
initialize_analyzers()
update_dashboard_data()

if __name__ == '__main__':
    """Start the web application"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Bullrun Analyzer Web Interface')
    parser.add_argument('--host', default='127.0.0.1', help='Host address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    
    args = parser.parse_args()
    
    if args.no_cache:
        CONFIG['use_cache_fallback'] = False
    
    print("=" * 70)
    print("CRYPTO BULLRUN ANALYZER - WEB INTERFACE")
    print("=" * 70)
    
    print(f"Status:")
    print(f"  Analyzer available: {'✓' if ANALYZER_AVAILABLE else '✗'}")
    print(f"  Top200 analyzer available: {'✓' if TOP200_AVAILABLE else '✗'}")
    print(f"  Cache enabled: {'✓' if CONFIG['use_cache_fallback'] else '✗'}")
    print(f"  Running in: {DASHBOARD_STATE['cache_status']}")
    
    print(f"\nStarting web server at: http://{args.host}:{args.port}")
    print(f"Press Ctrl+C to exit")
    
    app.run(host=args.host, port=args.port, debug=args.debug)