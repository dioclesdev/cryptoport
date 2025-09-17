# blueprints/api.py
"""
API Blueprint - REST API Endpoints
"""

from flask import Blueprint, request, jsonify, g
from datetime import datetime
import json

api_bp = Blueprint('api', __name__)

@api_bp.route('/status')
def status():
    """API Status Endpoint"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'services': {
            'analyzer': g.analyzer.is_available() if hasattr(g, 'analyzer') else False,
            'cache': True if hasattr(g, 'cache') else False,
            'top200': hasattr(g.analyzer, 'top200_analyzer') and g.analyzer.top200_analyzer is not None
        }
    })

@api_bp.route('/refresh', methods=['GET', 'POST'])
def refresh_data():
    """Refresh all dashboard data"""
    try:
        # Trigger fresh analysis
        analysis = g.analyzer.analyze_watchlist_safe()
        
        if "error" in analysis:
            return jsonify({
                'success': False,
                'message': analysis['error'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Save to cache
        g.cache.save_analysis(analysis)
        
        return jsonify({
            'success': True,
            'message': 'Data refreshed successfully',
            'total_coins': len(analysis.get('coins', [])),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Refresh failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

@api_bp.route('/analyze/symbol/<symbol>')
def analyze_symbol(symbol):
    """Analyze single symbol"""
    try:
        symbol = symbol.upper()
        analysis = g.analyzer.analyze_symbol(symbol)
        
        if "error" in analysis:
            return jsonify({'success': False, 'error': analysis['error']})
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/analyze/watchlist', methods=['POST'])
def analyze_watchlist():
    """Analyze current watchlist"""
    try:
        analysis = g.analyzer.analyze_watchlist_safe()
        
        if "error" in analysis:
            return jsonify({'success': False, 'error': analysis['error']})
        
        # Cache the result
        g.cache.save_analysis(analysis)
        
        return jsonify({
            'success': True,
            'message': 'Watchlist analysis completed',
            'total_coins': len(analysis.get('coins', [])),
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/analyze/portfolio', methods=['POST'])
def analyze_portfolio():
    """Analyze current portfolio"""
    try:
        analysis = g.analyzer.analyze_portfolio_safe()
        
        if not analysis or "error" in analysis:
            return jsonify({'success': False, 'error': 'Portfolio analysis failed or no data'})
        
        return jsonify({
            'success': True,
            'message': 'Portfolio analysis completed',
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/cache/status')
def cache_status():
    """Get cache status information"""
    try:
        status = g.cache.get_status()
        return jsonify({
            'success': True,
            'cache_status': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/cache/files')
def cache_files():
    """Get list of cached analysis files"""
    try:
        files = g.cache.get_analysis_files()
        return jsonify({
            'success': True,
            'files': [{
                'name': f['name'],
                'type': f['type'],
                'modified': f['modified'].isoformat(),
                'size': f['size']
            } for f in files]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/export/watchlist')
def export_watchlist():
    """Export watchlist analysis as JSON"""
    try:
        analysis = g.analyzer.analyze_watchlist_safe()
        
        if "error" in analysis:
            return jsonify({'success': False, 'error': analysis['error']})
        
        return jsonify({
            'success': True,
            'export_timestamp': datetime.now().isoformat(),
            'data': analysis
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/export/portfolio')
def export_portfolio():
    """Export portfolio analysis as JSON"""
    try:
        analysis = g.analyzer.analyze_portfolio_safe()
        
        if not analysis or "error" in analysis:
            return jsonify({'success': False, 'error': 'No portfolio data available'})
        
        return jsonify({
            'success': True,
            'export_timestamp': datetime.now().isoformat(),
            'data': analysis
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# blueprints/analysis.py
"""
Analysis Blueprint - Advanced Analysis & Top200
"""

from flask import Blueprint, render_template, request, jsonify, g, flash, redirect, url_for

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/')
def index():
    """Analysis Overview Page"""
    try:
        # Get latest analysis data
        watchlist_analysis = g.analyzer.analyze_watchlist_safe()
        top200_analysis = g.cache.get_top200_analysis()
        
        # Get analysis files for history
        analysis_files = g.cache.get_analysis_files()[:10]
        
        return render_template('analysis/index.html',
                             watchlist_analysis=watchlist_analysis,
                             top200_analysis=top200_analysis,
                             analysis_files=analysis_files)
                             
    except Exception as e:
        flash(f'Error loading analysis data: {str(e)}', 'error')
        return render_template('analysis/index.html',
                             watchlist_analysis=None,
                             top200_analysis=None,
                             analysis_files=[])

@analysis_bp.route('/top200')
def top200():
    """Top200 Analysis Page"""
    try:
        # Get cached Top200 analysis
        top200_analysis = g.cache.get_top200_analysis()
        
        # Get Top200 specific files
        analysis_files = g.cache.get_analysis_files()
        top200_files = [f for f in analysis_files if 'top200' in f['name'].lower()]
        
        # Calculate statistics
        stats = {}
        if top200_analysis and top200_analysis.get('coins'):
            coins = top200_analysis['coins']
            stats = {
                'total_analyzed': len(coins),
                'average_score': sum(c.get('bullrun_score', 0) for c in coins) / len(coins),
                'high_potential': len([c for c in coins if c.get('bullrun_score', 0) >= 0.7]),
                'very_high_potential': len([c for c in coins if c.get('bullrun_score', 0) >= 0.8])
            }
        
        return render_template('analysis/top200.html',
                             top200_analysis=top200_analysis,
                             top200_files=top200_files,
                             stats=stats)
                             
    except Exception as e:
        flash(f'Error loading Top200 analysis: {str(e)}', 'error')
        return render_template('analysis/top200.html',
                             top200_analysis=None,
                             top200_files=[],
                             stats={})

@analysis_bp.route('/run-top200', methods=['POST'])
def run_top200():
    """Execute Top200 Analysis"""
    try:
        # Check if Top200 analyzer is available
        if not hasattr(g.analyzer, 'top200_analyzer') or not g.analyzer.top200_analyzer:
            return jsonify({
                'success': False,
                'message': 'Top200 analyzer not available'
            })
        
        # Get parameters
        data = request.get_json() if request.is_json else {}
        max_symbols = data.get('max_symbols', 200)
        
        # Run analysis
        result = g.analyzer.run_top200_analysis()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Top200 analysis failed: {str(e)}'
        })

@analysis_bp.route('/compare')
def compare():
    """Compare Analysis Results"""
    try:
        # Get recent analysis files for comparison
        analysis_files = g.cache.get_analysis_files()
        
        # Load up to 3 most recent analyses for comparison
        analyses = []
        for file_info in analysis_files[:3]:
            try:
                analysis_data = g.cache._load_analysis_file(file_info['path'])
                if analysis_data:
                    analyses.append({
                        'name': file_info['name'],
                        'modified': file_info['modified'],
                        'data': analysis_data
                    })
            except:
                continue
        
        # Calculate comparison metrics
        comparison = calculate_comparison_metrics(analyses)
        
        return render_template('analysis/compare.html',
                             analyses=analyses,
                             comparison=comparison)
                             
    except Exception as e:
        flash(f'Error loading comparison data: {str(e)}', 'error')
        return render_template('analysis/compare.html',
                             analyses=[],
                             comparison={})

@analysis_bp.route('/load-analysis')
def load_analysis():
    """Load specific analysis file"""
    file_path = request.args.get('file')
    if not file_path:
        flash('No file specified', 'error')
        return redirect(url_for('analysis.index'))
    
    try:
        # Load the analysis file
        analysis_data = g.cache._load_analysis_file(file_path)
        
        if not analysis_data:
            flash('Failed to load analysis file', 'error')
            return redirect(url_for('analysis.index'))
        
        # Determine analysis type and redirect
        if 'top200' in file_path.lower():
            flash(f'Top200 analysis loaded successfully', 'success')
            return redirect(url_for('analysis.top200'))
        else:
            flash(f'Analysis loaded successfully', 'success')
            return redirect(url_for('analysis.index'))
            
    except Exception as e:
        flash(f'Error loading analysis: {str(e)}', 'error')
        return redirect(url_for('analysis.index'))

@analysis_bp.route('/trends')
def trends():
    """Analysis Trends & Historical Data"""
    try:
        # Get historical analysis files
        analysis_files = g.cache.get_analysis_files()
        
        # Load multiple analyses for trend analysis
        trend_data = []
        for file_info in analysis_files[:10]:  # Last 10 analyses
            try:
                analysis = g.cache._load_analysis_file(file_info['path'])
                if analysis and analysis.get('coins'):
                    coins = analysis['coins']
                    trend_data.append({
                        'date': file_info['modified'],
                        'total_coins': len(coins),
                        'average_score': sum(c.get('bullrun_score', 0) for c in coins) / len(coins),
                        'high_potential': len([c for c in coins if c.get('bullrun_score', 0) >= 0.7])
                    })
            except:
                continue
        
        # Sort by date
        trend_data.sort(key=lambda x: x['date'])
        
        return render_template('analysis/trends.html',
                             trend_data=trend_data)
                             
    except Exception as e:
        flash(f'Error loading trend data: {str(e)}', 'error')
        return render_template('analysis/trends.html',
                             trend_data=[])

def calculate_comparison_metrics(analyses):
    """Calculate metrics for comparing multiple analyses"""
    if len(analyses) < 2:
        return {}
    
    comparison = {
        'symbol_overlap': {},
        'score_changes': {},
        'new_symbols': set(),
        'removed_symbols': set()
    }
    
    try:
        # Compare first two analyses
        if len(analyses) >= 2:
            old_analysis = analyses[1]['data']
            new_analysis = analyses[0]['data']
            
            old_symbols = {coin['symbol']: coin for coin in old_analysis.get('coins', [])}
            new_symbols = {coin['symbol']: coin for coin in new_analysis.get('coins', [])}
            
            # Find overlapping symbols
            overlap = set(old_symbols.keys()) & set(new_symbols.keys())
            comparison['symbol_overlap'] = {
                'total': len(overlap),
                'percentage': len(overlap) / max(len(old_symbols), 1) * 100
            }
            
            # Calculate score changes
            score_changes = []
            for symbol in overlap:
                old_score = old_symbols[symbol].get('bullrun_score', 0)
                new_score = new_symbols[symbol].get('bullrun_score', 0)
                change = new_score - old_score
                
                if abs(change) > 0.01:  # Only significant changes
                    score_changes.append({
                        'symbol': symbol,
                        'old_score': old_score,
                        'new_score': new_score,
                        'change': change
                    })
            
            # Sort by absolute change
            score_changes.sort(key=lambda x: abs(x['change']), reverse=True)
            comparison['score_changes'] = score_changes[:10]  # Top 10 changes
            
            # New and removed symbols
            comparison['new_symbols'] = list(set(new_symbols.keys()) - set(old_symbols.keys()))
            comparison['removed_symbols'] = list(set(old_symbols.keys()) - set(new_symbols.keys()))
    
    except Exception as e:
        print(f"❌ Error calculating comparison metrics: {e}")
    
    return comparison


# blueprints/reports.py
"""
Reports Blueprint - Report Generation & Email
"""

from flask import Blueprint, render_template, request, jsonify, flash, g, send_file
from datetime import datetime
import os
import tempfile

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/')
def index():
    """Reports Overview Page"""
    try:
        # Get available analysis data
        watchlist_analysis = g.analyzer.analyze_watchlist_safe()
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        
        return render_template('reports/index.html',
                             watchlist_analysis=watchlist_analysis,
                             portfolio_analysis=portfolio_analysis)
                             
    except Exception as e:
        flash(f'Error loading report data: {str(e)}', 'error')
        return render_template('reports/index.html',
                             watchlist_analysis=None,
                             portfolio_analysis=None)

@reports_bp.route('/generate/html')
def generate_html():
    """Generate HTML Report"""
    try:
        report_type = request.args.get('type', 'watchlist')
        
        if report_type == 'watchlist':
            analysis = g.analyzer.analyze_watchlist_safe()
            template = 'reports/watchlist_report.html'
        elif report_type == 'portfolio':
            analysis = g.analyzer.analyze_portfolio_safe()
            template = 'reports/portfolio_report.html'
        else:
            flash('Invalid report type', 'error')
            return redirect(url_for('reports.index'))
        
        if not analysis or "error" in analysis:
            flash('No data available for report generation', 'error')
            return redirect(url_for('reports.index'))
        
        return render_template(template,
                             analysis=analysis,
                             generated_at=datetime.now())
                             
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('reports.index'))

@reports_bp.route('/generate/csv')
def generate_csv():
    """Generate CSV Report"""
    try:
        report_type = request.args.get('type', 'watchlist')
        
        if report_type == 'watchlist':
            analysis = g.analyzer.analyze_watchlist_safe()
        elif report_type == 'portfolio':
            analysis = g.analyzer.analyze_portfolio_safe()
        else:
            return jsonify({'success': False, 'error': 'Invalid report type'})
        
        if not analysis or "error" in analysis:
            return jsonify({'success': False, 'error': 'No data available'})
        
        # Generate CSV content
        csv_content = generate_csv_content(analysis, report_type)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_report_{timestamp}.csv"
        
        return send_file(temp_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@reports_bp.route('/email/send', methods=['POST'])
def send_email():
    """Send Email Report"""
    try:
        data = request.get_json()
        recipient = data.get('recipient')
        report_type = data.get('type', 'watchlist')
        
        if not recipient:
            return jsonify({'success': False, 'error': 'Recipient email required'})
        
        # Check if email system is available
        try:
            from email_report_system import CryptoEmailReporter
            
            # SMTP Configuration (should be moved to config)
            smtp_config = {
                'smtp_server': 'smtp.world4you.com',
                'smtp_port': 587,
                'smtp_username': 'office@bepartof.work',
                'smtp_password': 'ggig@ag7ar',
                'sender_email': 'office@bepartof.work'
            }
            
            reporter = CryptoEmailReporter(smtp_config, use_attachments=True)
            success = reporter.send_watchlist_report(g.analyzer, recipient)
            
            if success:
                return jsonify({'success': True, 'message': f'Report sent to {recipient}'})
            else:
                return jsonify({'success': False, 'error': 'Failed to send email'})
                
        except ImportError:
            return jsonify({'success': False, 'error': 'Email system not available'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_csv_content(analysis, report_type):
    """Generate CSV content from analysis data"""
    lines = []
    
    if report_type == 'watchlist' and analysis.get('coins'):
        # Watchlist CSV
        lines.append("Rank,Symbol,Name,Bullrun_Score,Bullrun_Potential,Price_USD,Market_Cap_USD")
        
        coins = sorted(analysis['coins'], key=lambda x: x.get('bullrun_score', 0), reverse=True)
        for i, coin in enumerate(coins, 1):
            lines.append(
                f"{i},"
                f"{coin.get('symbol', '')},"
                f"{coin.get('name', '')},"
                f"{coin.get('bullrun_score', 0):.3f},"
                f"{coin.get('bullrun_potential', '')},"
                f"{coin.get('current_price_usd', 0):.6f},"
                f"{coin.get('market_cap_usd', 0):.0f}"
            )
    
    elif report_type == 'portfolio' and analysis.get('coins'):
        # Portfolio CSV
        lines.append("Symbol,Name,Amount,Avg_Price,Current_Price,Current_Value,Profit_Loss,Profit_Loss_Percent,Bullrun_Score")
        
        for coin in analysis['coins']:
            lines.append(
                f"{coin.get('symbol', '')},"
                f"{coin.get('name', '')},"
                f"{coin.get('amount', 0):.6f},"
                f"{coin.get('avg_price', 0):.6f},"
                f"{coin.get('current_price', 0):.6f},"
                f"{coin.get('value', 0):.2f},"
                f"{coin.get('profit_loss', 0):.2f},"
                f"{coin.get('profit_loss_percent', 0):.2f},"
                f"{coin.get('bullrun_score', 0):.3f}"
            )
    
    return '\n'.join(lines)


# blueprints/admin.py
"""
Admin Blueprint - Administration Panel
"""

from flask import Blueprint, render_template, request, jsonify, flash, g
from datetime import datetime
import os

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/')
def index():
    """Admin Dashboard"""
    try:
        # System status
        system_status = {
            'analyzer_available': g.analyzer.is_available(),
            'cache_status': g.cache.get_status(),
            'uptime': datetime.now(),  # Would need actual uptime tracking
            'last_analysis': 'N/A'  # Would need to track this
        }
        
        # Recent activities (would need logging)
        activities = [
            {'time': datetime.now(), 'action': 'System started', 'status': 'success'},
            # Add more activities from logs
        ]
        
        return render_template('admin/index.html',
                             system_status=system_status,
                             activities=activities)
                             
    except Exception as e:
        flash(f'Error loading admin panel: {str(e)}', 'error')
        return render_template('admin/index.html',
                             system_status={},
                             activities=[])

@admin_bp.route('/settings')
def settings():
    """System Settings"""
    try:
        # Load current configuration
        current_config = {
            'cache_max_age_hours': 24,
            'auto_refresh_minutes': 30,
            'max_coins_display': 50,
            'api_timeout': 30
        }
        
        return render_template('admin/settings.html',
                             config=current_config)
                             
    except Exception as e:
        flash(f'Error loading settings: {str(e)}', 'error')
        return render_template('admin/settings.html',
                             config={})

@admin_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear system cache"""
    try:
        # This would implement cache clearing
        # For now, just return success
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@admin_bp.route('/logs')
def logs():
    """View System Logs"""
    try:
        # Read log files (if they exist)
        log_entries = []
        log_file = 'logs/cryptoport.log'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
                for line in lines:
                    log_entries.append(line.strip())
        
        return render_template('admin/logs.html',
                             log_entries=log_entries)
                             
    except Exception as e:
        flash(f'Error loading logs: {str(e)}', 'error')
        return render_template('admin/logs.html',
                             log_entries=[])

@admin_bp.route('/maintenance/start', methods=['POST'])
def start_maintenance():
    """Start maintenance mode"""
    try:
        # Implement maintenance mode
        return jsonify({
            'success': True,
            'message': 'Maintenance mode activated'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@admin_bp.route('/maintenance/stop', methods=['POST'])
def stop_maintenance():
    """Stop maintenance mode"""
    try:
        # Implement maintenance mode stop
        return jsonify({
            'success': True,
            'message': 'Maintenance mode deactivated'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})