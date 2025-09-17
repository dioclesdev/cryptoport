# blueprints/dashboard.py
"""
Dashboard Blueprint - Main Overview
"""

from flask import Blueprint, render_template, request, jsonify, g
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
def index():
    """Main Dashboard Page"""
    try:
        # Get watchlist analysis
        watchlist_analysis = g.analyzer.analyze_watchlist_safe()
        
        # Get portfolio analysis (optional)
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        # Generate charts
        charts = generate_dashboard_charts(watchlist_analysis)
        
        # Calculate statistics
        stats = calculate_dashboard_stats(watchlist_analysis, portfolio_analysis)
        
        return render_template('dashboard/index.html',
                             watchlist_analysis=watchlist_analysis,
                             portfolio_analysis=portfolio_analysis,
                             charts=charts,
                             stats=stats)
                             
    except Exception as e:
        return render_template('dashboard/error.html', error=str(e))

@dashboard_bp.route('/overview')
def overview():
    """Dashboard Overview Page"""
    try:
        # Get cache status
        cache_status = g.cache.get_status()
        
        # Get recent analysis files
        analysis_files = g.cache.get_analysis_files()[:5]
        
        # System status
        system_status = {
            'analyzer_available': g.analyzer.is_available(),
            'top200_available': hasattr(g.analyzer, 'top200_analyzer') and g.analyzer.top200_analyzer,
            'cache_directories': len(cache_status['directories']),
            'total_cache_files': cache_status['total_files'],
            'last_update': datetime.now()
        }
        
        return render_template('dashboard/overview.html',
                             cache_status=cache_status,
                             analysis_files=analysis_files,
                             system_status=system_status)
                             
    except Exception as e:
        return render_template('dashboard/error.html', error=str(e))

def generate_dashboard_charts(watchlist_analysis):
    """Generate charts for dashboard"""
    if not watchlist_analysis or not watchlist_analysis.get('coins'):
        return {}
    
    charts = {}
    
    try:
        valid_coins = [coin for coin in watchlist_analysis['coins'] 
                      if coin.get('bullrun_score') is not None]
        
        if not valid_coins:
            return charts
        
        # 1. Top 10 Coins Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_coins = sorted(valid_coins, key=lambda x: x['bullrun_score'], reverse=True)[:10]
        symbols = [coin['symbol'] for coin in top_coins]
        scores = [coin['bullrun_score'] for coin in top_coins]
        
        # Color coding
        colors = ['#28a745' if score >= 0.7 else '#ffc107' if score >= 0.5 else '#dc3545' 
                 for score in scores]
        
        bars = ax.bar(symbols, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add score values on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Top 10 Coins - Bullrun Scores', fontsize=14, fontweight='bold')
        ax.set_ylabel('Bullrun Score')
        ax.set_ylim(0, max(scores) * 1.1 if scores else 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        charts['top_coins'] = create_chart_base64(fig)
        
        # 2. Score Distribution Pie Chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        categories = ['Very High (≥0.8)', 'High (≥0.7)', 'Above Avg (≥0.6)', 
                     'Average (≥0.5)', 'Below Avg (≥0.4)', 'Low (<0.4)']
        
        values = [
            len([c for c in valid_coins if c['bullrun_score'] >= 0.8]),
            len([c for c in valid_coins if 0.7 <= c['bullrun_score'] < 0.8]),
            len([c for c in valid_coins if 0.6 <= c['bullrun_score'] < 0.7]),
            len([c for c in valid_coins if 0.5 <= c['bullrun_score'] < 0.6]),
            len([c for c in valid_coins if 0.4 <= c['bullrun_score'] < 0.5]),
            len([c for c in valid_coins if c['bullrun_score'] < 0.4])
        ]
        
        colors = ['#28a745', '#20c997', '#ffc107', '#fd7e14', '#6f42c1', '#dc3545']
        
        # Only show categories with values > 0
        non_zero_data = [(cat, val, col) for cat, val, col in zip(categories, values, colors) if val > 0]
        
        if non_zero_data:
            categories, values, colors = zip(*non_zero_data)
            
            wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors,
                                             autopct='%1.1f%%', startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
            
            charts['distribution'] = create_chart_base64(fig)
        
    except Exception as e:
        print(f"❌ Error generating dashboard charts: {e}")
    
    return charts

def calculate_dashboard_stats(watchlist_analysis, portfolio_analysis):
    """Calculate dashboard statistics"""
    stats = {
        'watchlist': {
            'total_coins': 0,
            'average_score': 0.0,
            'top_performers': 0,
            'very_high_potential': 0
        },
        'portfolio': {
            'total_value': 0.0,
            'total_profit_loss': 0.0,
            'profit_loss_percent': 0.0,
            'total_holdings': 0
        }
    }
    
    # Watchlist stats
    if watchlist_analysis and watchlist_analysis.get('coins'):
        valid_coins = [coin for coin in watchlist_analysis['coins'] 
                      if coin.get('bullrun_score') is not None]
        
        if valid_coins:
            stats['watchlist']['total_coins'] = len(valid_coins)
            stats['watchlist']['average_score'] = sum(coin['bullrun_score'] for coin in valid_coins) / len(valid_coins)
            stats['watchlist']['top_performers'] = len([coin for coin in valid_coins if coin['bullrun_score'] >= 0.7])
            stats['watchlist']['very_high_potential'] = len([coin for coin in valid_coins if coin['bullrun_score'] >= 0.8])
    
    # Portfolio stats
    if portfolio_analysis and 'error' not in portfolio_analysis:
        stats['portfolio']['total_value'] = portfolio_analysis.get('total_value', 0)
        stats['portfolio']['total_profit_loss'] = portfolio_analysis.get('profit_loss', 0)
        stats['portfolio']['profit_loss_percent'] = portfolio_analysis.get('profit_loss_percent', 0)
        stats['portfolio']['total_holdings'] = len(portfolio_analysis.get('coins', []))
    
    return stats

def create_chart_base64(fig):
    """Convert matplotlib figure to base64 for web display"""
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
        print(f"❌ Error creating chart: {e}")
        return None


# blueprints/watchlist.py
"""
Watchlist Blueprint - Watchlist Management
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, g

watchlist_bp = Blueprint('watchlist', __name__)

@watchlist_bp.route('/')
def index():
    """Watchlist Management Page"""
    try:
        # Get current watchlist
        watchlist_coins = g.analyzer.get_watchlist_coins()
        
        # Get cached data for display
        coin_details = {}
        for symbol in watchlist_coins[:20]:  # Limit for performance
            cached_data = g.cache.get_cached_coin_data(symbol)
            if cached_data:
                coin_details[symbol] = {
                    'name': cached_data.get('name', symbol),
                    'price': cached_data.get('usd_price', 0)
                }
        
        return render_template('watchlist/index.html',
                             watchlist_coins=watchlist_coins,
                             coin_details=coin_details)
                             
    except Exception as e:
        flash(f'Error loading watchlist: {str(e)}', 'error')
        return render_template('watchlist/index.html', 
                             watchlist_coins=[], coin_details={})

@watchlist_bp.route('/add', methods=['POST'])
def add_coin():
    """Add coin to watchlist"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        success = g.analyzer.add_to_watchlist(symbol)
        
        if success:
            return jsonify({'success': True, 'message': f'{symbol} added to watchlist'})
        else:
            return jsonify({'success': False, 'message': f'Failed to add {symbol}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@watchlist_bp.route('/remove', methods=['POST'])
def remove_coin():
    """Remove coin from watchlist"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        success = g.analyzer.remove_from_watchlist(symbol)
        
        if success:
            return jsonify({'success': True, 'message': f'{symbol} removed from watchlist'})
        else:
            return jsonify({'success': False, 'message': f'{symbol} not found in watchlist'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@watchlist_bp.route('/bulk-add', methods=['POST'])
def bulk_add():
    """Add multiple coins to watchlist"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'success': False, 'message': 'No symbols provided'})
        
        added_count = 0
        skipped_count = 0
        
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if symbol:
                try:
                    if g.analyzer.add_to_watchlist(symbol):
                        added_count += 1
                    else:
                        skipped_count += 1
                except:
                    skipped_count += 1
        
        return jsonify({
            'success': True,
            'added_count': added_count,
            'skipped_count': skipped_count,
            'message': f'{added_count} coins added, {skipped_count} skipped'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@watchlist_bp.route('/analyze', methods=['POST'])
def analyze():
    """Analyze current watchlist"""
    try:
        analysis = g.analyzer.analyze_watchlist_safe()
        
        if "error" in analysis:
            return jsonify({'success': False, 'message': analysis['error']})
        
        # Cache the analysis result
        g.cache.save_analysis(analysis, 'watchlist_analysis.json')
        
        return jsonify({
            'success': True, 
            'message': 'Watchlist analysis completed',
            'total_coins': len(analysis.get('coins', [])),
            'average_score': sum(coin.get('bullrun_score', 0) 
                               for coin in analysis.get('coins', [])) / max(len(analysis.get('coins', [])), 1)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Analysis error: {str(e)}'})


# blueprints/portfolio.py
"""
Portfolio Blueprint - Portfolio Management
"""

from flask import Blueprint, render_template, request, jsonify, flash, g

portfolio_bp = Blueprint('portfolio', __name__)

@portfolio_bp.route('/')
def index():
    """Portfolio Overview Page"""
    try:
        # Get portfolio analysis
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        # Generate portfolio charts if data available
        charts = {}
        if portfolio_analysis and 'error' not in portfolio_analysis:
            charts = generate_portfolio_charts(portfolio_analysis)
        
        return render_template('portfolio/index.html',
                             portfolio_analysis=portfolio_analysis,
                             charts=charts)
                             
    except Exception as e:
        flash(f'Error loading portfolio: {str(e)}', 'error')
        return render_template('portfolio/index.html', 
                             portfolio_analysis=None, charts={})

@portfolio_bp.route('/add', methods=['GET', 'POST'])
def add_holding():
    """Add holding to portfolio"""
    if request.method == 'GET':
        return render_template('portfolio/add.html')
    
    try:
        data = request.get_json() if request.is_json else request.form
        
        symbol = data.get('symbol', '').upper().strip()
        amount = float(data.get('amount', 0))
        avg_price = float(data.get('avg_price', 0))
        
        if not symbol or amount <= 0 or avg_price <= 0:
            return jsonify({'success': False, 'message': 'Invalid input data'})
        
        success = g.analyzer.add_to_portfolio(symbol, amount, avg_price)
        
        if success:
            return jsonify({'success': True, 'message': f'{symbol} added to portfolio'})
        else:
            return jsonify({'success': False, 'message': f'Failed to add {symbol}'})
            
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid number format'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@portfolio_bp.route('/performance')
def performance():
    """Portfolio Performance Analysis"""
    try:
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        if not portfolio_analysis or 'error' in portfolio_analysis:
            flash('No portfolio data available', 'info')
            return render_template('portfolio/performance.html', 
                                 portfolio_analysis=None)
        
        # Generate performance metrics
        performance_data = calculate_performance_metrics(portfolio_analysis)
        
        return render_template('portfolio/performance.html',
                             portfolio_analysis=portfolio_analysis,
                             performance_data=performance_data)
                             
    except Exception as e:
        flash(f'Error loading performance data: {str(e)}', 'error')
        return render_template('portfolio/performance.html', 
                             portfolio_analysis=None)

def generate_portfolio_charts(portfolio_analysis):
    """Generate portfolio visualization charts"""
    charts = {}
    
    try:
        coins = portfolio_analysis.get('coins', [])
        if not coins:
            return charts
        
        # Portfolio Allocation Pie Chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = [f"{coin['symbol']} (${coin['value']:,.0f})" for coin in coins]
        values = [coin['value'] for coin in coins]
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                         startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Portfolio Allocation by Value', fontsize=14, fontweight='bold')
        charts['allocation'] = create_chart_base64(fig)
        
        # Profit/Loss Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        symbols = [coin['symbol'] for coin in coins]
        profit_loss = [coin.get('profit_loss_percent', 0) for coin in coins]
        
        colors = ['#28a745' if pl >= 0 else '#dc3545' for pl in profit_loss]
        
        bars = ax.bar(symbols, profit_loss, color=colors, alpha=0.8)
        
        # Add percentage labels
        for bar, pl in zip(bars, profit_loss):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (1 if height >= 0 else -3),
                   f'{pl:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_title('Profit/Loss by Coin (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Profit/Loss (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        charts['profit_loss'] = create_chart_base64(fig)
        
    except Exception as e:
        print(f"❌ Error generating portfolio charts: {e}")
    
    return charts

def calculate_performance_metrics(portfolio_analysis):
    """Calculate additional performance metrics"""
    coins = portfolio_analysis.get('coins', [])
    if not coins:
        return {}
    
    # Best and worst performers
    performers = [(coin['symbol'], coin.get('profit_loss_percent', 0)) 
                 for coin in coins if coin.get('profit_loss_percent') is not None]
    
    best_performer = max(performers, key=lambda x: x[1]) if performers else None
    worst_performer = min(performers, key=lambda x: x[1]) if performers else None
    
    # Portfolio concentration (top 3 holdings percentage)
    total_value = portfolio_analysis.get('total_value', 1)
    top_3_value = sum(sorted([coin['value'] for coin in coins], reverse=True)[:3])
    concentration = (top_3_value / total_value * 100) if total_value > 0 else 0
    
    return {
        'best_performer': best_performer,
        'worst_performer': worst_performer,
        'concentration': concentration,
        'total_holdings': len(coins),
        'profitable_holdings': len([c for c in coins if c.get('profit_loss', 0) > 0])
    }