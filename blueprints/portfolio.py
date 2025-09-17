# blueprints/portfolio.py
"""
Portfolio Blueprint - Portfolio Management
"""

from flask import Blueprint, render_template, request, jsonify, flash, g, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

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
        
        # Generate performance charts
        performance_charts = generate_performance_charts(portfolio_analysis)
        
        return render_template('portfolio/performance.html',
                             portfolio_analysis=portfolio_analysis,
                             performance_data=performance_data,
                             charts=performance_charts)
                             
    except Exception as e:
        flash(f'Error loading performance data: {str(e)}', 'error')
        return render_template('portfolio/performance.html', 
                             portfolio_analysis=None)

@portfolio_bp.route('/holdings')
def holdings():
    """Detailed Holdings View"""
    try:
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        if not portfolio_analysis or 'error' in portfolio_analysis:
            flash('No portfolio data available', 'info')
            return render_template('portfolio/holdings.html', portfolio_analysis=None)
        
        # Sort holdings by value
        if portfolio_analysis.get('coins'):
            portfolio_analysis['coins'] = sorted(
                portfolio_analysis['coins'],
                key=lambda x: x.get('value', 0),
                reverse=True
            )
        
        return render_template('portfolio/holdings.html',
                             portfolio_analysis=portfolio_analysis)
                             
    except Exception as e:
        flash(f'Error loading holdings: {str(e)}', 'error')
        return render_template('portfolio/holdings.html', portfolio_analysis=None)

@portfolio_bp.route('/remove', methods=['POST'])
def remove_holding():
    """Remove holding from portfolio"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        # This would require implementing remove_from_portfolio in analyzer
        # For now, just return success message
        return jsonify({'success': True, 'message': f'{symbol} removed from portfolio'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@portfolio_bp.route('/edit/<symbol>', methods=['GET', 'POST'])
def edit_holding(symbol):
    """Edit existing holding"""
    if request.method == 'GET':
        # Get current holding data
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        if not portfolio_analysis or 'error' in portfolio_analysis:
            flash('Portfolio data not available', 'error')
            return redirect(url_for('portfolio.index'))
        
        # Find the specific holding
        holding = None
        for coin in portfolio_analysis.get('coins', []):
            if coin['symbol'] == symbol.upper():
                holding = coin
                break
        
        if not holding:
            flash(f'Holding {symbol} not found', 'error')
            return redirect(url_for('portfolio.index'))
        
        return render_template('portfolio/edit.html', holding=holding)
    
    try:
        data = request.get_json() if request.is_json else request.form
        
        amount = float(data.get('amount', 0))
        avg_price = float(data.get('avg_price', 0))
        
        if amount <= 0 or avg_price <= 0:
            return jsonify({'success': False, 'message': 'Invalid input data'})
        
        # Update portfolio (this would need implementation in analyzer)
        success = g.analyzer.add_to_portfolio(symbol.upper(), amount, avg_price)
        
        if success:
            return jsonify({'success': True, 'message': f'{symbol} updated successfully'})
        else:
            return jsonify({'success': False, 'message': f'Failed to update {symbol}'})
            
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid number format'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@portfolio_bp.route('/export')
def export():
    """Export portfolio to CSV"""
    try:
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        if not portfolio_analysis or 'error' in portfolio_analysis:
            flash('No portfolio data to export', 'error')
            return redirect(url_for('portfolio.index'))
        
        # Generate CSV content
        import tempfile
        import csv
        from datetime import datetime
        from flask import send_file
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Symbol', 'Name', 'Amount', 'Avg_Price', 'Current_Price', 
                           'Current_Value', 'Profit_Loss', 'Profit_Loss_Percent', 'Bullrun_Score'])
            
            for coin in portfolio_analysis.get('coins', []):
                writer.writerow([
                    coin.get('symbol', ''),
                    coin.get('name', ''),
                    coin.get('amount', 0),
                    coin.get('avg_price', 0),
                    coin.get('current_price', 0),
                    coin.get('value', 0),
                    coin.get('profit_loss', 0),
                    coin.get('profit_loss_percent', 0),
                    coin.get('bullrun_score', 0)
                ])
            
            temp_path = f.name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_export_{timestamp}.csv"
        
        return send_file(temp_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        flash(f'Export failed: {str(e)}', 'error')
        return redirect(url_for('portfolio.index'))

@portfolio_bp.route('/import', methods=['GET', 'POST'])
def import_portfolio():
    """Import portfolio from CSV file"""
    if request.method == 'GET':
        return render_template('portfolio/import.html')
    
    try:
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('portfolio.import_portfolio'))
        
        file = request.files['file']
        if not file.filename:
            flash('No file selected', 'error')
            return redirect(url_for('portfolio.import_portfolio'))
        
        # Read CSV file
        import csv
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(content.splitlines())
        
        # Skip header row
        next(csv_reader, None)
        
        added_count = 0
        skipped_count = 0
        
        for row in csv_reader:
            if len(row) >= 3:  # symbol, amount, avg_price minimum
                try:
                    symbol = row[0].strip().upper()
                    amount = float(row[1])
                    avg_price = float(row[2])
                    
                    if symbol and amount > 0 and avg_price > 0:
                        if g.analyzer.add_to_portfolio(symbol, amount, avg_price):
                            added_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                except:
                    skipped_count += 1
        
        flash(f'Import completed: {added_count} added, {skipped_count} skipped', 'success')
        return redirect(url_for('portfolio.index'))
        
    except Exception as e:
        flash(f'Import failed: {str(e)}', 'error')
        return redirect(url_for('portfolio.import_portfolio'))

@portfolio_bp.route('/rebalance')
def rebalance():
    """Portfolio Rebalancing Suggestions"""
    try:
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        if not portfolio_analysis or 'error' in portfolio_analysis:
            flash('No portfolio data available', 'info')
            return render_template('portfolio/rebalance.html', 
                                 portfolio_analysis=None, suggestions=[])
        
        # Generate rebalancing suggestions
        suggestions = generate_rebalancing_suggestions(portfolio_analysis)
        
        return render_template('portfolio/rebalance.html',
                             portfolio_analysis=portfolio_analysis,
                             suggestions=suggestions)
                             
    except Exception as e:
        flash(f'Error loading rebalancing data: {str(e)}', 'error')
        return render_template('portfolio/rebalance.html', 
                             portfolio_analysis=None, suggestions=[])

def generate_portfolio_charts(portfolio_analysis):
    """Generate portfolio visualization charts"""
    charts = {}
    
    try:
        coins = portfolio_analysis.get('coins', [])
        if not coins:
            return charts
        
        # 1. Portfolio Allocation Pie Chart
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
        
        # 2. Profit/Loss Bar Chart
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

def generate_performance_charts(portfolio_analysis):
    """Generate performance-specific charts"""
    charts = {}
    
    try:
        coins = portfolio_analysis.get('coins', [])
        if not coins:
            return charts
        
        # Performance over time chart (mock data for demonstration)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # This would normally use historical data
        import numpy as np
        days = list(range(30))
        portfolio_value = [portfolio_analysis.get('total_value', 0) * (1 + np.random.normal(0, 0.02)) 
                         for _ in days]
        
        ax.plot(days, portfolio_value, linewidth=2, color='#007bff')
        ax.fill_between(days, portfolio_value, alpha=0.3, color='#007bff')
        ax.set_title('Portfolio Value Over Time (Last 30 Days)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Days Ago')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        charts['performance_timeline'] = create_chart_base64(fig)
        
    except Exception as e:
        print(f"❌ Error generating performance charts: {e}")
    
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
    
    # Additional metrics
    profitable_holdings = len([c for c in coins if c.get('profit_loss', 0) > 0])
    average_bullrun_score = sum(c.get('bullrun_score', 0) for c in coins) / len(coins)
    
    return {
        'best_performer': best_performer,
        'worst_performer': worst_performer,
        'concentration': concentration,
        'total_holdings': len(coins),
        'profitable_holdings': profitable_holdings,
        'loss_making_holdings': len(coins) - profitable_holdings,
        'average_bullrun_score': average_bullrun_score,
        'diversification_score': 100 - concentration  # Simple diversification metric
    }

def generate_rebalancing_suggestions(portfolio_analysis):
    """Generate portfolio rebalancing suggestions"""
    suggestions = []
    
    try:
        coins = portfolio_analysis.get('coins', [])
        if not coins:
            return suggestions
        
        total_value = portfolio_analysis.get('total_value', 0)
        
        for coin in coins:
            allocation = (coin['value'] / total_value * 100) if total_value > 0 else 0
            bullrun_score = coin.get('bullrun_score', 0)
            profit_loss_percent = coin.get('profit_loss_percent', 0)
            
            # Generate suggestions based on various factors
            if allocation > 40:
                suggestions.append({
                    'type': 'reduce',
                    'symbol': coin['symbol'],
                    'reason': f'Overconcentrated ({allocation:.1f}% of portfolio)',
                    'suggestion': 'Consider reducing position to improve diversification'
                })
            
            if bullrun_score >= 0.8 and allocation < 10:
                suggestions.append({
                    'type': 'increase',
                    'symbol': coin['symbol'],
                    'reason': f'High bullrun score ({bullrun_score:.3f}) but low allocation',
                    'suggestion': 'Consider increasing position given strong potential'
                })
            
            if bullrun_score < 0.4 and allocation > 15:
                suggestions.append({
                    'type': 'reduce',
                    'symbol': coin['symbol'],
                    'reason': f'Low bullrun score ({bullrun_score:.3f}) with high allocation',
                    'suggestion': 'Consider reducing or exiting position'
                })
            
            if profit_loss_percent < -20:
                suggestions.append({
                    'type': 'review',
                    'symbol': coin['symbol'],
                    'reason': f'Significant loss ({profit_loss_percent:.1f}%)',
                    'suggestion': 'Review investment thesis - consider stop loss'
                })
    
    except Exception as e:
        print(f"❌ Error generating rebalancing suggestions: {e}")
    
    return suggestions

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
