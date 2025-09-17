# blueprints/reports.py
"""
Reports Blueprint - Report Generation & Email
"""

from flask import Blueprint, render_template, request, jsonify, flash, g, send_file, redirect, url_for
from datetime import datetime
import os
import tempfile
import csv

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/')
def index():
    """Reports Overview Page"""
    try:
        # Get available analysis data
        watchlist_analysis = g.analyzer.analyze_watchlist_safe()
        portfolio_analysis = g.analyzer.analyze_portfolio_safe()
        top200_analysis = g.cache.get_top200_analysis()
        
        # Get recent reports/exports
        analysis_files = g.cache.get_analysis_files()[:5]
        
        # Calculate report statistics
        report_stats = {
            'watchlist_available': watchlist_analysis and 'error' not in watchlist_analysis,
            'portfolio_available': portfolio_analysis and 'error' not in portfolio_analysis,
            'top200_available': top200_analysis is not None,
            'total_coins_watchlist': len(watchlist_analysis.get('coins', [])) if watchlist_analysis else 0,
            'total_holdings': len(portfolio_analysis.get('coins', [])) if portfolio_analysis else 0,
            'total_top200': len(top200_analysis.get('coins', [])) if top200_analysis else 0
        }
        
        return render_template('reports/index.html',
                             watchlist_analysis=watchlist_analysis,
                             portfolio_analysis=portfolio_analysis,
                             top200_analysis=top200_analysis,
                             recent_files=analysis_files,
                             report_stats=report_stats)
                             
    except Exception as e:
        flash(f'Error loading report data: {str(e)}', 'error')
        return render_template('reports/index.html',
                             watchlist_analysis=None,
                             portfolio_analysis=None,
                             top200_analysis=None,
                             recent_files=[],
                             report_stats={})

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
        elif report_type == 'top200':
            analysis = g.cache.get_top200_analysis()
            template = 'reports/top200_report.html'
        else:
            flash('Invalid report type', 'error')
            return redirect(url_for('reports.index'))
        
        if not analysis or "error" in analysis:
            flash('No data available for report generation', 'error')
            return redirect(url_for('reports.index'))
        
        # Generate report statistics
        report_metadata = generate_report_metadata(analysis, report_type)
        
        return render_template(template,
                             analysis=analysis,
                             generated_at=datetime.now(),
                             metadata=report_metadata)
                             
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
        elif report_type == 'top200':
            analysis = g.cache.get_top200_analysis()
        else:
            return jsonify({'success': False, 'error': 'Invalid report type'})
        
        if not analysis or "error" in analysis:
            return jsonify({'success': False, 'error': 'No data available'})
        
        # Generate CSV content
        csv_content = generate_csv_content(analysis, report_type)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write(csv_content)
            temp_path = f.name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_report_{timestamp}.csv"
        
        return send_file(temp_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@reports_bp.route('/generate/excel')
def generate_excel():
    """Generate Excel Report with multiple sheets"""
    try:
        import pandas as pd
        from io import BytesIO
        
        report_type = request.args.get('type', 'comprehensive')
        
        # Create Excel writer
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Watchlist sheet
            watchlist_analysis = g.analyzer.analyze_watchlist_safe()
            if watchlist_analysis and 'error' not in watchlist_analysis:
                df_watchlist = create_dataframe_from_analysis(watchlist_analysis, 'watchlist')
                df_watchlist.to_excel(writer, sheet_name='Watchlist', index=False)
            
            # Portfolio sheet
            portfolio_analysis = g.analyzer.analyze_portfolio_safe()
            if portfolio_analysis and 'error' not in portfolio_analysis:
                df_portfolio = create_dataframe_from_analysis(portfolio_analysis, 'portfolio')
                df_portfolio.to_excel(writer, sheet_name='Portfolio', index=False)
            
            # Top200 sheet
            top200_analysis = g.cache.get_top200_analysis()
            if top200_analysis:
                df_top200 = create_dataframe_from_analysis(top200_analysis, 'top200')
                df_top200.to_excel(writer, sheet_name='Top200', index=False)
            
            # Summary sheet
            summary_data = create_summary_data(watchlist_analysis, portfolio_analysis, top200_analysis)
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crypto_report_{timestamp}.xlsx"
        
        return send_file(BytesIO(output.read()), as_attachment=True, 
                        download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    except ImportError:
        return jsonify({'success': False, 'error': 'Excel export not available - install openpyxl'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@reports_bp.route('/email/send', methods=['POST'])
def send_email():
    """Send Email Report"""
    try:
        data = request.get_json()
        recipient = data.get('recipient')
        report_type = data.get('type', 'watchlist')
        include_charts = data.get('include_charts', True)
        
        if not recipient:
            return jsonify({'success': False, 'error': 'Recipient email required'})
        
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, recipient):
            return jsonify({'success': False, 'error': 'Invalid email format'})
        
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
            
            reporter = CryptoEmailReporter(smtp_config, use_attachments=include_charts)
            
            if report_type == 'watchlist':
                success = reporter.send_watchlist_report(g.analyzer, recipient)
            elif report_type == 'portfolio':
                # Would need to implement portfolio email report
                success = False
                return jsonify({'success': False, 'error': 'Portfolio email reports not yet implemented'})
            else:
                return jsonify({'success': False, 'error': 'Invalid report type'})
            
            if success:
                return jsonify({'success': True, 'message': f'Report sent to {recipient}'})
            else:
                return jsonify({'success': False, 'error': 'Failed to send email'})
                
        except ImportError:
            return jsonify({'success': False, 'error': 'Email system not available'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@reports_bp.route('/schedule')
def schedule():
    """Report Scheduling Page"""
    try:
        # Get current scheduled reports (mock data for now)
        scheduled_reports = [
            {
                'id': 1,
                'type': 'watchlist',
                'recipient': 'user@example.com',
                'frequency': 'daily',
                'time': '08:00',
                'active': True,
                'last_sent': datetime.now(),
                'next_run': datetime.now()
            }
        ]
        
        return render_template('reports/schedule.html', 
                             scheduled_reports=scheduled_reports)
    except Exception as e:
        flash(f'Error loading schedule page: {str(e)}', 'error')
        return redirect(url_for('reports.index'))

@reports_bp.route('/schedule/add', methods=['POST'])
def add_schedule():
    """Add scheduled report"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['type', 'recipient', 'frequency']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Missing {field}'})
        
        # Create scheduled report entry (this would save to database)
        schedule_entry = {
            'type': data['type'],
            'recipient': data['recipient'],
            'frequency': data['frequency'],
            'time': data.get('time', '08:00'),
            'active': True,
            'created_at': datetime.now()
        }
        
        # In a real application, this would be saved to database
        return jsonify({
            'success': True,
            'message': 'Scheduled report added successfully',
            'schedule_id': 'mock_id_123'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@reports_bp.route('/templates')
def templates():
    """Report Templates Management"""
    try:
        # Get available templates
        templates_info = [
            {
                'id': 'standard_watchlist',
                'name': 'Standard Watchlist Report',
                'description': 'Basic watchlist analysis with top performers',
                'type': 'watchlist',
                'sections': ['overview', 'top_coins', 'score_distribution']
            },
            {
                'id': 'detailed_portfolio',
                'name': 'Detailed Portfolio Report',
                'description': 'Comprehensive portfolio performance analysis',
                'type': 'portfolio',
                'sections': ['summary', 'holdings', 'performance', 'recommendations']
            },
            {
                'id': 'market_overview',
                'name': 'Market Overview Report',
                'description': 'Top 200 market analysis with trends',
                'type': 'top200',
                'sections': ['market_stats', 'top_performers', 'sector_analysis']
            },
            {
                'id': 'executive_summary',
                'name': 'Executive Summary',
                'description': 'High-level overview for decision makers',
                'type': 'comprehensive',
                'sections': ['key_metrics', 'recommendations', 'risk_analysis']
            }
        ]
        
        return render_template('reports/templates.html', templates=templates_info)
    except Exception as e:
        flash(f'Error loading templates: {str(e)}', 'error')
        return redirect(url_for('reports.index'))

@reports_bp.route('/custom')
def custom():
    """Custom Report Builder"""
    try:
        # Available report sections
        available_sections = {
            'watchlist': [
                {'id': 'overview', 'name': 'Overview Statistics', 'description': 'Basic stats and metrics'},
                {'id': 'top_performers', 'name': 'Top Performers', 'description': 'Highest scoring coins'},
                {'id': 'score_distribution', 'name': 'Score Distribution', 'description': 'Score histogram'},
                {'id': 'potential_breakdown', 'name': 'Potential Categories', 'description': 'Coins by potential level'},
                {'id': 'detailed_analysis', 'name': 'Detailed Analysis', 'description': 'Complete coin breakdown'}
            ],
            'portfolio': [
                {'id': 'summary', 'name': 'Portfolio Summary', 'description': 'Total value and P&L'},
                {'id': 'allocation', 'name': 'Asset Allocation', 'description': 'Portfolio distribution'},
                {'id': 'performance', 'name': 'Performance Analysis', 'description': 'Returns and metrics'},
                {'id': 'risk_analysis', 'name': 'Risk Analysis', 'description': 'Risk metrics and volatility'},
                {'id': 'recommendations', 'name': 'Recommendations', 'description': 'Rebalancing suggestions'}
            ]
        }
        
        return render_template('reports/custom.html', sections=available_sections)
    except Exception as e:
        flash(f'Error loading custom builder: {str(e)}', 'error')
        return redirect(url_for('reports.index'))

@reports_bp.route('/generate/custom', methods=['POST'])
def generate_custom():
    """Generate custom report"""
    try:
        data = request.get_json()
        report_config = data.get('config', {})
        output_format = data.get('format', 'html')
        
        # Validate configuration
        if not report_config.get('sections'):
            return jsonify({'success': False, 'error': 'No sections selected'})
        
        # Build custom report based on configuration
        report_data = build_custom_report(report_config)
        
        if output_format == 'html':
            html_content = render_custom_html_report(report_data, report_config)
            return jsonify({
                'success': True,
                'html_content': html_content,
                'report_title': report_config.get('title', 'Custom Report')
            })
        elif output_format == 'csv':
            csv_content = render_custom_csv_report(report_data, report_config)
            return jsonify({
                'success': True,
                'csv_content': csv_content,
                'filename': f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            })
        else:
            return jsonify({'success': False, 'error': 'Unsupported output format'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@reports_bp.route('/history')
def history():
    """Report History and Archives"""
    try:
        # Get analysis files as report history
        analysis_files = g.cache.get_analysis_files()
        
        # Categorize by type
        history_by_type = {
            'watchlist': [],
            'top200': [],
            'portfolio': [],
            'other': []
        }
        
        for file_info in analysis_files:
            if 'watchlist' in file_info['name'].lower():
                history_by_type['watchlist'].append(file_info)
            elif 'top200' in file_info['name'].lower():
                history_by_type['top200'].append(file_info)
            elif 'portfolio' in file_info['name'].lower():
                history_by_type['portfolio'].append(file_info)
            else:
                history_by_type['other'].append(file_info)
        
        return render_template('reports/history.html', history=history_by_type)
    except Exception as e:
        flash(f'Error loading report history: {str(e)}', 'error')
        return redirect(url_for('reports.index'))

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
                f'"{coin.get('name', '')}",'
                f"{coin.get('bullrun_score', 0):.3f},"
                f'"{coin.get('bullrun_potential', '')}",'
                f"{coin.get('current_price_usd', 0):.6f},"
                f"{coin.get('market_cap_usd', 0):.0f}"
            )
    
    elif report_type == 'portfolio' and analysis.get('coins'):
        # Portfolio CSV
        lines.append("Symbol,Name,Amount,Avg_Price,Current_Price,Current_Value,Profit_Loss,Profit_Loss_Percent,Bullrun_Score")
        
        for coin in analysis['coins']:
            lines.append(
                f"{coin.get('symbol', '')},"
                f'"{coin.get('name', '')}",'
                f"{coin.get('amount', 0):.6f},"
                f"{coin.get('avg_price', 0):.6f},"
                f"{coin.get('current_price', 0):.6f},"
                f"{coin.get('value', 0):.2f},"
                f"{coin.get('profit_loss', 0):.2f},"
                f"{coin.get('profit_loss_percent', 0):.2f},"
                f"{coin.get('bullrun_score', 0):.3f}"
            )
    
    elif report_type == 'top200' and analysis.get('coins'):
        # Top200 CSV
        lines.append("Rank,Symbol,Name,Bullrun_Score,Bullrun_Potential,Price_USD,Market_Cap_USD")
        
        coins = sorted(analysis['coins'], key=lambda x: x.get('bullrun_score', 0), reverse=True)
        for i, coin in enumerate(coins, 1):
            lines.append(
                f"{i},"
                f"{coin.get('symbol', '')},"
                f'"{coin.get('name', '')}",'
                f"{coin.get('bullrun_score', 0):.3f},"
                f'"{coin.get('bullrun_potential', '')}",'
                f"{coin.get('current_price_usd', coin.get('current_price', 0)):.6f},"
                f"{coin.get('market_cap_usd', 0):.0f}"
            )
    
    return '\n'.join(lines)

def generate_report_metadata(analysis, report_type):
    """Generate metadata for reports"""
    metadata = {
        'generated_at': datetime.now(),
        'report_type': report_type,
        'total_coins': 0,
        'average_score': 0,
        'top_performer': None,
        'data_source': analysis.get('data_source', 'live')
    }
    
    try:
        if analysis.get('coins'):
            coins = analysis['coins']
            metadata['total_coins'] = len(coins)
            
            scores = [c.get('bullrun_score', 0) for c in coins]
            if scores:
                metadata['average_score'] = sum(scores) / len(scores)
                top_coin = max(coins, key=lambda x: x.get('bullrun_score', 0))
                metadata['top_performer'] = {
                    'symbol': top_coin.get('symbol', ''),
                    'name': top_coin.get('name', ''),
                    'score': top_coin.get('bullrun_score', 0)
                }
    except:
        pass
    
    return metadata

def create_dataframe_from_analysis(analysis, analysis_type):
    """Create pandas DataFrame from analysis data"""
    import pandas as pd
    
    if not analysis.get('coins'):
        return pd.DataFrame()
    
    data = []
    for coin in analysis['coins']:
        row = {
            'Symbol': coin.get('symbol', ''),
            'Name': coin.get('name', ''),
            'Bullrun_Score': coin.get('bullrun_score', 0),
            'Bullrun_Potential': coin.get('bullrun_potential', ''),
            'Price_USD': coin.get('current_price_usd', coin.get('current_price', 0)),
            'Market_Cap_USD': coin.get('market_cap_usd', 0)
        }
        
        # Add analysis-type specific columns
        if analysis_type == 'portfolio':
            row.update({
                'Amount': coin.get('amount', 0),
                'Avg_Price': coin.get('avg_price', 0),
                'Current_Value': coin.get('value', 0),
                'Profit_Loss': coin.get('profit_loss', 0),
                'Profit_Loss_Percent': coin.get('profit_loss_percent', 0)
            })
        
        data.append(row)
    
    return pd.DataFrame(data)

def create_summary_data(watchlist_analysis, portfolio_analysis, top200_analysis):
    """Create summary data for comprehensive reports"""
    summary = []
    
    # Watchlist summary
    if watchlist_analysis and 'error' not in watchlist_analysis:
        coins = watchlist_analysis.get('coins', [])
        summary.append({
            'Category': 'Watchlist',
            'Total_Coins': len(coins),
            'Average_Score': sum(c.get('bullrun_score', 0) for c in coins) / len(coins) if coins else 0,
            'High_Potential': len([c for c in coins if c.get('bullrun_score', 0) >= 0.7]),
            'Very_High_Potential': len([c for c in coins if c.get('bullrun_score', 0) >= 0.8])
        })
    
    # Portfolio summary
    if portfolio_analysis and 'error' not in portfolio_analysis:
        summary.append({
            'Category': 'Portfolio',
            'Total_Value': portfolio_analysis.get('total_value', 0),
            'Total_Profit_Loss': portfolio_analysis.get('profit_loss', 0),
            'Profit_Loss_Percent': portfolio_analysis.get('profit_loss_percent', 0),
            'Holdings_Count': len(portfolio_analysis.get('coins', []))
        })
    
    # Top200 summary
    if top200_analysis:
        coins = top200_analysis.get('coins', [])
        summary.append({
            'Category': 'Top200',
            'Total_Analyzed': len(coins),
            'Average_Score': sum(c.get('bullrun_score', 0) for c in coins) / len(coins) if coins else 0,
            'Market_Leaders': len([c for c in coins if c.get('bullrun_score', 0) >= 0.8])
        })
    
    return summary

def build_custom_report(config):
    """Build custom report data based on configuration"""
    report_data = {}
    
    try:
        report_type = config.get('type', 'watchlist')
        sections = config.get('sections', [])
        
        # Get base data
        if report_type == 'watchlist':
            base_analysis = g.analyzer.analyze_watchlist_safe()
        elif report_type == 'portfolio':
            base_analysis = g.analyzer.analyze_portfolio_safe()
        else:
            base_analysis = None
        
        # Build sections
        for section in sections:
            if section == 'overview' and base_analysis:
                report_data['overview'] = generate_overview_section(base_analysis)
            elif section == 'top_performers' and base_analysis:
                report_data['top_performers'] = generate_top_performers_section(base_analysis)
            # Add more sections as needed
    
    except Exception as e:
        print(f"Error building custom report: {e}")
    
    return report_data

def generate_overview_section(analysis):
    """Generate overview section for custom reports"""
    if not analysis.get('coins'):
        return {}
    
    coins = analysis['coins']
    return {
        'total_coins': len(coins),
        'average_score': sum(c.get('bullrun_score', 0) for c in coins) / len(coins),
        'high_potential_count': len([c for c in coins if c.get('bullrun_score', 0) >= 0.7]),
        'score_range': {
            'min': min(c.get('bullrun_score', 0) for c in coins),
            'max': max(c.get('bullrun_score', 0) for c in coins)
        }
    }

def generate_top_performers_section(analysis):
    """Generate top performers section for custom reports"""
    if not analysis.get('coins'):
        return []
    
    coins = sorted(analysis['coins'], key=lambda x: x.get('bullrun_score', 0), reverse=True)
    return coins[:10]  # Top 10 performers

def render_custom_html_report(report_data, config):
    """Render custom report as HTML"""
    html_parts = ['<html><head><title>Custom Report</title></head><body>']
    html_parts.append(f'<h1>{config.get("title", "Custom Report")}</h1>')
    
    for section_name, section_data in report_data.items():
        html_parts.append(f'<h2>{section_name.replace("_", " ").title()}</h2>')
        
        if section_name == 'overview' and isinstance(section_data, dict):
            html_parts.append('<ul>')
            for key, value in section_data.items():
                html_parts.append(f'<li>{key.replace("_", " ").title()}: {value}</li>')
            html_parts.append('</ul>')
        elif section_name == 'top_performers' and isinstance(section_data, list):
            html_parts.append('<table border="1"><tr><th>Symbol</th><th>Name</th><th>Score</th></tr>')
            for coin in section_data:
                html_parts.append(f'<tr><td>{coin.get("symbol", "")}</td><td>{coin.get("name", "")}</td><td>{coin.get("bullrun_score", 0):.3f}</td></tr>')
            html_parts.append('</table>')
    
    html_parts.append('</body></html>')
    return ''.join(html_parts)

def render_custom_csv_report(report_data, config):
    """Render custom report as CSV"""
    lines = [f"# {config.get('title', 'Custom Report')}"]
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for section_name, section_data in report_data.items():
        lines.append(f"# {section_name.replace('_', ' ').title()}")
        
        if section_name == 'top_performers' and isinstance(section_data, list):
            lines.append("Symbol,Name,Bullrun_Score,Bullrun_Potential")
            for coin in section_data:
                lines.append(f"{coin.get('symbol', '')},{coin.get('name', '')},{coin.get('bullrun_score', 0):.3f},{coin.get('bullrun_potential', '')}")
        
        lines.append("")
    
    return '\n'.join(lines)
