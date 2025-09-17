# blueprints/admin.py
"""
Admin Blueprint - Administration Panel
"""

from flask import Blueprint, render_template, request, jsonify, flash, g
from datetime import datetime, timedelta
import os
import sys

admin_bp = Blueprint('admin', __name__)

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System monitoring will be limited.")

@admin_bp.route('/')
def index():
    """Admin Dashboard"""
    try:
        # System status
        system_status = {
            'analyzer_available': g.analyzer.is_available(),
            'cache_status': g.cache.get_status(),
            'uptime': get_system_uptime(),
            'memory_usage': get_memory_usage(),
            'disk_usage': get_disk_usage(),
            'cpu_usage': get_cpu_usage(),
            'last_analysis': get_last_analysis_time(),
            'python_version': sys.version.split()[0],
            'platform': sys.platform
        }
        
        # Service status
        service_status = {
            'analyzer_service': hasattr(g, 'analyzer') and g.analyzer is not None,
            'cache_service': hasattr(g, 'cache') and g.cache is not None,
            'top200_available': hasattr(g.analyzer, 'top200_analyzer') and g.analyzer.top200_analyzer is not None if hasattr(g, 'analyzer') else False,
            'email_system': check_email_system_availability(),
            'api_endpoints': True  # Always available if admin panel loads
        }
        
        # Recent activities (simplified - would need proper logging)
        activities = generate_recent_activities()