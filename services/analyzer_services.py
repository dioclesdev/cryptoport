# services/analyzer_service.py
"""
Analyzer Service - Business Logic Layer
"""

import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from crypto_analyzer import CryptoBullrunAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

try:
    from top200_analyzer import AutonomousTop200Analyzer  
    TOP200_AVAILABLE = True
except ImportError:
    TOP200_AVAILABLE = False

class AnalyzerService:
    """Service for Crypto Analysis Operations"""
    
    def __init__(self, cache_service):
        self.cache_service = cache_service
        self.analyzer = None
        self.top200_analyzer = None
        self._last_update = {}
        
        # Initialize analyzers
        self._init_analyzers()
        
    def _init_analyzers(self):
        """Initialize analyzer instances"""
        if ANALYZER_AVAILABLE:
            try:
                self.analyzer = CryptoBullrunAnalyzer()
                print("✅ CryptoBullrunAnalyzer initialized")
            except Exception as e:
                print(f"❌ Failed to initialize analyzer: {e}")
                
        if TOP200_AVAILABLE:
            try:
                self.top200_analyzer = AutonomousTop200Analyzer()
                print("✅ Top200Analyzer initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Top200 analyzer: {e}")
    
    def is_available(self) -> bool:
        """Check if analyzer is available"""
        return self.analyzer is not None
    
    def analyze_watchlist_safe(self) -> Dict:
        """Analyze watchlist with cache fallback"""
        try:
            # Try cache first
            cached_data = self.cache_service.get_latest_analysis()
            if cached_data:
                print("📦 Using cached watchlist analysis")
                return cached_data
            
            # Try real analyzer if available
            if self.analyzer:
                try:
                    analysis = self.analyzer.analyze_watchlist()
                    if "error" not in analysis:
                        # Cache the result
                        self.cache_service.save_analysis(analysis)
                        return analysis
                except Exception as e:
                    print(f"⚠️ Real analyzer failed: {e}")
            
            # Fallback to demo data
            return self._create_demo_watchlist_data()
            
        except Exception as e:
            print(f"❌ Watchlist analysis failed: {e}")
            return {"error": "Analysis failed", "coins": []}
    
    def analyze_portfolio_safe(self) -> Optional[Dict]:
        """Analyze portfolio with cache fallback"""
        if not self.analyzer:
            return None
            
        try:
            portfolio_data = self.analyzer.analyze_portfolio()
            if "error" not in portfolio_data:
                return portfolio_data
        except Exception as e:
            print(f"⚠️ Portfolio analysis failed: {e}")
            
        return None
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze single symbol with cache fallback"""
        # Check cache first
        cached_coin_data = self.cache_service.get_cached_coin_data(symbol)
        if cached_coin_data:
            # Use cached data to create analysis
            return self._create_analysis_from_cache(cached_coin_data)
        
        # Try real analyzer
        if self.analyzer:
            try:
                return self.analyzer.analyze_symbol(symbol)
            except Exception as e:
                print(f"⚠️ Symbol analysis failed for {symbol}: {e}")
        
        return {"error": f"Analysis failed for {symbol}"}
    
    def run_top200_analysis(self) -> Dict:
        """Run Top200 analysis"""
        if not self.top200_analyzer:
            return {"success": False, "message": "Top200 analyzer not available"}
        
        try:
            print("🚀 Starting Top200 analysis...")
            result = self.top200_analyzer.run_analysis()
            
            if result and result.get('successful_analyses', 0) > 0:
                # Reload cached data
                new_analysis = self.cache_service.get_top200_analysis()
                return {
                    "success": True,
                    "successful_analyses": result['successful_analyses'],
                    "failed_analyses": result['failed_analyses'],
                    "success_rate": result['success_rate'],
                    "duration_seconds": result['duration_seconds'],
                    "analysis_data": new_analysis
                }
            else:
                return {"success": False, "message": "Analysis completed but no results"}
                
        except Exception as e:
            return {"success": False, "message": f"Analysis error: {str(e)}"}
    
    def get_watchlist_coins(self) -> List[str]:
        """Get current watchlist coins"""
        if self.analyzer:
            try:
                watchlist = self.analyzer.load_watchlist()
                return watchlist.get("coins", [])
            except:
                pass
        
        # Fallback default coins
        return ["BTC", "ETH", "SOL", "ADA", "DOT", "MATIC"]
    
    def add_to_watchlist(self, symbol: str) -> bool:
        """Add coin to watchlist"""
        if not self.analyzer:
            return False
        
        try:
            return self.analyzer.add_to_watchlist(symbol.upper())
        except Exception as e:
            print(f"❌ Failed to add {symbol} to watchlist: {e}")
            return False
    
    def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove coin from watchlist"""
        if not self.analyzer:
            return False
        
        try:
            return self.analyzer.remove_from_watchlist(symbol.upper())
        except Exception as e:
            print(f"❌ Failed to remove {symbol} from watchlist: {e}")
            return False
    
    def add_to_portfolio(self, symbol: str, amount: float, avg_price: float) -> bool:
        """Add coin to portfolio"""
        if not self.analyzer:
            return False
        
        try:
            return self.analyzer.add_to_portfolio(symbol.upper(), amount, avg_price)
        except Exception as e:
            print(f"❌ Failed to add {symbol} to portfolio: {e}")
            return False
    
    def _create_demo_watchlist_data(self) -> Dict:
        """Create demo watchlist data when no real data available"""
        return {
            'coins': [
                {'symbol': 'BTC', 'name': 'Bitcoin', 'bullrun_score': 0.85, 'total_score': 0.85, 
                 'bullrun_potential': 'Sehr hoch', 'current_price': 45000, 'market_cap_usd': 900000000000},
                {'symbol': 'ETH', 'name': 'Ethereum', 'bullrun_score': 0.78, 'total_score': 0.78,
                 'bullrun_potential': 'Hoch', 'current_price': 3200, 'market_cap_usd': 400000000000},
                {'symbol': 'SOL', 'name': 'Solana', 'bullrun_score': 0.72, 'total_score': 0.72,
                 'bullrun_potential': 'Hoch', 'current_price': 150, 'market_cap_usd': 60000000000},
                {'symbol': 'ADA', 'name': 'Cardano', 'bullrun_score': 0.65, 'total_score': 0.65,
                 'bullrun_potential': 'Überdurchschnittlich', 'current_price': 0.45, 'market_cap_usd': 15000000000},
                {'symbol': 'DOT', 'name': 'Polkadot', 'bullrun_score': 0.58, 'total_score': 0.58,
                 'bullrun_potential': 'Durchschnittlich', 'current_price': 7.2, 'market_cap_usd': 8000000000},
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
    
    def _create_analysis_from_cache(self, cached_data: Dict) -> Dict:
        """Create analysis result from cached coin data"""
        try:
            symbol = cached_data.get('symbol', '')
            name = cached_data.get('name', symbol)
            price = float(cached_data.get('usd_price', 0))
            market_cap = float(cached_data.get('usd_market_cap', 0))
            
            # Simple scoring based on available data
            score = min(1.0, market_cap / 1000000000) * 0.7  # Basic scoring
            
            potential_map = {
                (0.8, 1.0): "Sehr hoch",
                (0.7, 0.8): "Hoch", 
                (0.6, 0.7): "Überdurchschnittlich",
                (0.5, 0.6): "Durchschnittlich",
                (0.4, 0.5): "Unterdurchschnittlich",
                (0.0, 0.4): "Niedrig"
            }
            
            potential = "Durchschnittlich"
            for (min_score, max_score), pot in potential_map.items():
                if min_score <= score < max_score:
                    potential = pot
                    break
            
            return {
                "symbol": symbol,
                "name": name,
                "current_price_usd": price,
                "market_cap_usd": market_cap,
                "total_score": score,
                "bullrun_score": score,
                "bullrun_potential": potential,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "cache"
            }
            
        except Exception as e:
            return {"error": f"Failed to create analysis from cache: {e}"}


# services/cache_service.py  
"""
Cache Service - Data Persistence Layer
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

class CacheService:
    """Service for Cache Management Operations"""
    
    def __init__(self, cache_dirs: List[str]):
        self.cache_dirs = cache_dirs
        self.max_age_hours = 24
        self._ensure_cache_dirs()
    
    def _ensure_cache_dirs(self):
        """Ensure all cache directories exist"""
        for cache_dir in self.cache_dirs:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                print(f"📁 Cache directory ready: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not create cache directory {cache_dir}: {e}")
    
    def get_status(self) -> Dict:
        """Get cache status information"""
        status = {
            'directories': [],
            'total_files': 0,
            'total_size_mb': 0,
            'last_analysis': None
        }
        
        for cache_dir in self.cache_dirs:
            dir_info = {
                'path': cache_dir,
                'exists': os.path.exists(cache_dir),
                'files': 0,
                'size_mb': 0.0
            }
            
            if dir_info['exists']:
                try:
                    files = os.listdir(cache_dir)
                    dir_info['files'] = len(files)
                    
                    size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                              for f in files if os.path.isfile(os.path.join(cache_dir, f)))
                    dir_info['size_mb'] = size / (1024 * 1024)
                    
                except Exception as e:
                    dir_info['error'] = str(e)
            
            status['directories'].append(dir_info)
            status['total_files'] += dir_info['files']
            status['total_size_mb'] += dir_info['size_mb']
        
        return status
    
    def find_cached_file(self, filename_pattern: str, max_age_hours: int = None) -> Optional[str]:
        """Find most recent cached file matching pattern"""
        if max_age_hours is None:
            max_age_hours = self.max_age_hours
            
        best_file = None
        best_time = 0
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        for cache_dir in self.cache_dirs:
            try:
                pattern_path = os.path.join(cache_dir, filename_pattern)
                files = glob.glob(pattern_path)
                
                for file_path in files:
                    try:
                        file_time = os.path.getmtime(file_path)
                        if file_time > cutoff_time and file_time > best_time:
                            best_file = file_path
                            best_time = file_time
                    except Exception:
                        continue
            except Exception:
                continue
        
        return best_file
    
    def get_cached_coin_data(self, symbol: str) -> Optional[Dict]:
        """Get cached coin data for a symbol"""
        cache_file = self.find_cached_file(f"coin_{symbol.lower()}.csv")
        if cache_file:
            try:
                df = pd.read_csv(cache_file)
                if not df.empty:
                    data = df.iloc[0].to_dict()
                    print(f"📦 Loaded cached data for {symbol}")
                    return data
            except Exception as e:
                print(f"❌ Error loading cached data for {symbol}: {e}")
        
        return None
    
    def get_latest_analysis(self) -> Optional[Dict]:
        """Get the most recent analysis data from cache"""
        # Try loading from various analysis file patterns
        patterns = [
            "**/top200_analysis_*.csv",
            "**/crypto_report_*.csv",
            "**/top_bullrun_analysis_*.csv", 
            "analysis_results/*.csv"
        ]
        
        analysis_files = []
        for cache_dir in self.cache_dirs:
            for pattern in patterns:
                try:
                    files = glob.glob(os.path.join(cache_dir, pattern), recursive=True)
                    for file_path in files:
                        try:
                            file_time = os.path.getmtime(file_path)
                            analysis_files.append({
                                'path': file_path,
                                'modified': file_time
                            })
                        except Exception:
                            continue
                except Exception:
                    continue
        
        # Sort by modification time, newest first
        analysis_files.sort(key=lambda x: x['modified'], reverse=True)
        
        # Try to load the most recent file
        for file_info in analysis_files[:5]:  # Try top 5 most recent
            try:
                analysis_data = self._load_analysis_file(file_info['path'])
                if analysis_data:
                    print(f"📦 Loaded analysis from: {os.path.basename(file_info['path'])}")
                    return analysis_data
            except Exception as e:
                print(f"❌ Failed to load {file_info['path']}: {e}")
                continue
        
        return None
    
    def get_top200_analysis(self) -> Optional[Dict]:
        """Get Top200 analysis specifically"""
        # Look for Top200 files
        for cache_dir in self.cache_dirs:
            try:
                pattern = os.path.join(cache_dir, "**/top200_analysis_*.csv")
                files = glob.glob(pattern, recursive=True)
                
                if files:
                    # Get most recent
                    latest_file = max(files, key=os.path.getmtime)
                    analysis_data = self._load_analysis_file(latest_file)
                    if analysis_data:
                        print(f"📦 Loaded Top200 analysis from: {os.path.basename(latest_file)}")
                        return analysis_data
            except Exception:
                continue
        
        return None
    
    def _load_analysis_file(self, file_path: str) -> Optional[Dict]:
        """Load analysis file and convert to standard format"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                return self._csv_to_analysis_format(df)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if self._validate_analysis_data(data):
                        return data
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
        
        return None
    
    def _csv_to_analysis_format(self, df: pd.DataFrame) -> Optional[Dict]:
        """Convert CSV analysis to standard format"""
        try:
            coins = []
            potential_categories = {
                "very_high": [], "high": [], "above_average": [],
                "average": [], "below_average": [], "low": []
            }
            
            for _, row in df.iterrows():
                symbol = str(row.get('Symbol', row.get('symbol', '')))
                if not symbol:
                    continue
                    
                name = str(row.get('Name', row.get('name', symbol)))
                score = float(row.get('Bullrun_Score', row.get('bullrun_score', row.get('Score', 0))))
                potential = str(row.get('Bullrun_Potential', row.get('bullrun_potential', 'Unknown')))
                price = float(row.get('Price_USD', row.get('current_price', 0)))
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
                'potential_categories': potential_categories,
                'data_source': 'cache'
            }
            
        except Exception as e:
            print(f"❌ Error converting CSV to analysis format: {e}")
            return None
    
    def _validate_analysis_data(self, data: Dict) -> bool:
        """Validate analysis data structure"""
        try:
            if not isinstance(data, dict):
                return False
            
            if 'coins' not in data or not isinstance(data['coins'], list):
                return False
            
            # Check first few coins for required fields
            for coin in data['coins'][:3]:
                if not all(key in coin for key in ['symbol', 'bullrun_score']):
                    return False
            
            return True
        except:
            return False
    
    def save_analysis(self, analysis_data: Dict, filename: str = None):
        """Save analysis data to cache"""
        if not self.cache_dirs:
            return
        
        try:
            # Use first writable cache directory
            cache_dir = self.cache_dirs[0]
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_cache_{timestamp}.json"
            
            file_path = os.path.join(cache_dir, filename)
            
            with open(file_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
                
            print(f"💾 Analysis saved to cache: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save analysis to cache: {e}")
    
    def get_analysis_files(self) -> List[Dict]:
        """Get list of available analysis files"""
        analysis_files = []
        
        patterns = [
            "**/top200_analysis_*.csv",
            "**/top200_analysis_*.html", 
            "**/crypto_report_*.csv",
            "analysis_results/*.csv",
            "exports/*.csv"
        ]
        
        for cache_dir in self.cache_dirs:
            for pattern in patterns:
                try:
                    files = glob.glob(os.path.join(cache_dir, pattern), recursive=True)
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