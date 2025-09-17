# blueprints/analysis.py
"""
Analysis Blueprint - Advanced Analysis & Top200
"""

from flask import Blueprint, render_template, request, jsonify, g, flash, redirect, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/')
def index():
    """Analysis Overview Page"""
    try:
        # Get latest analysis data
        watchlist_analysis = g.analyzer.analyze_watchlist_safe()
        top200_analysis = g.cache.get_top200_analysis()
        
        # Get analysis files for history
        analysis_