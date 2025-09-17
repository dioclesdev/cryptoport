#!/usr/bin/env python3
"""
Cache Service for Crypto Bullrun Analyzer

This module provides caching functionality for API calls and analysis results
to reduce API usage and improve performance.

Features:
- Multiple cache directory support
- Configurable TTL (Time-To-Live)
- Automatic cleanup of stale cache files
- Cache hit/miss statistics
"""

import os
import json
import glob
import time
import shutil
import pickle
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cache_service')

class CacheService:
    """
    Cache service for storing and retrieving data from multiple cache directories
    with automatic expiration and cleanup.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the cache service with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.setup_cache_directories()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saved_api_calls': 0,
            'cleanup_count': 0,
            'last_cleanup': None,
            'cache_size_bytes': 0
        }
        
        # Run initial cleanup if enabled
        if self.config.get('auto_cleanup_on_start', False):
            self.cleanup_cache()
            
        # Calculate initial cache size
        self.update_cache_size()
        
        logger.info(f"Cache service initialized with {len(self.cache_dirs)} directories")
    
    def setup_cache_directories(self):
        """
        Set up cache directories from configuration or use defaults.
        """
        # Default directories
        default_dirs = [
            'crypto_cache',  # Local cache
            os.path.expanduser('~/.cache/crypto_analyzer'),  # User cache
        ]
        
        # Get directories from config
        config_dirs = self.config.get('cache_directories', [])
        
        # Combine and filter valid directories
        self.cache_dirs = []
        
        for cache_dir in config_dirs + default_dirs:
            if cache_dir not in self.cache_dirs:  # Avoid duplicates
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    self.cache_dirs.append(cache_dir)
                    logger.info(f"Using cache directory: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Could not create cache directory {cache_dir}: {e}")
        
        # Set primary cache directory (first valid one)
        if self.cache_dirs:
            self.primary_cache_dir = self.cache_dirs[0]
        else:
            # Fallback to local directory
            self.primary_cache_dir = 'crypto_cache'
            os.makedirs(self.primary_cache_dir, exist_ok=True)
            self.cache_dirs = [self.primary_cache_dir]
            logger.warning(f"Using fallback cache directory: {self.primary_cache_dir}")
    
    def cache_key(self, prefix: str, identifier: str) -> str:
        """
        Generate a cache key from prefix and identifier.
        
        Args:
            prefix: Cache type prefix (e.g., 'coin', 'market')
            identifier: Unique identifier (e.g., symbol)
            
        Returns:
            Formatted cache key
        """
        # Sanitize identifier to ensure it's valid for filenames
        safe_id = identifier.lower()
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
            safe_id = safe_id.replace(char, '_')
            
        return f"{prefix}_{safe_id}"
    
    def hash_key(self, key: str) -> str:
        """
        Create a hash of complex keys for caching.
        
        Args:
            key: String to hash
            
        Returns:
            MD5 hash of the key
        """
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def get_cache_path(self, key: str, extension: str = 'json') -> str:
        """
        Get full path for a cache file.
        
        Args:
            key: Cache key
            extension: File extension
            
        Returns:
            Path to cache file in primary directory
        """
        return os.path.join(self.primary_cache_dir, f"{key}.{extension}")
    
    def find_in_cache(self, key: str, extensions: List[str] = None, 
                     max_age_hours: Optional[int] = None) -> Optional[str]:
        """
        Find a cache file across all cache directories.
        
        Args:
            key: Cache key to search for
            extensions: List of file extensions to search (default: ['json', 'csv', 'pickle'])
            max_age_hours: Maximum age in hours (default: from config or 24)
            
        Returns:
            Path to cache file if found and not expired, else None
        """
        if extensions is None:
            extensions = ['json', 'csv', 'pickle']
            
        # Get max age from config or use default
        if max_age_hours is None:
            max_age_hours = self.config.get('default_ttl_hours', 24)
            
        # Calculate cutoff time
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Search all cache directories
        for cache_dir in self.cache_dirs:
            for ext in extensions:
                pattern = os.path.join(cache_dir, f"{key}.{ext}")
                matching_files = glob.glob(pattern)
                
                for file_path in matching_files:
                    try:
                        # Check file age
                        file_time = os.path.getmtime(file_path)
                        if file_time > cutoff_time:
                            self.stats['hits'] += 1
                            return file_path
                    except Exception:
                        continue
        
        self.stats['misses'] += 1
        return None
    
    def save_to_cache(self, key: str, data: Any, extension: str = 'json') -> str:
        """
        Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
            extension: File extension/format
            
        Returns:
            Path to saved cache file
        """
        cache_path = self.get_cache_path(key, extension)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Save based on extension
            if extension == 'json':
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
            elif extension == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(cache_path, index=False)
                else:
                    pd.DataFrame(data).to_csv(cache_path, index=False)
                    
            elif extension == 'pickle':
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                    
            else:
                # Generic text data
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            
            self.stats['saved_api_calls'] += 1
            self.update_cache_size()  # Update cache size after saving
            return cache_path
            
        except Exception as e:
            logger.error(f"Error saving to cache ({key}.{extension}): {e}")
            return ""
    
    def load_from_cache(self, key: str, extensions: List[str] = None, 
                       max_age_hours: Optional[int] = None) -> Tuple[Any, bool]:
        """
        Load data from cache if available and not expired.
        
        Args:
            key: Cache key
            extensions: List of file extensions to try
            max_age_hours: Maximum age in hours
            
        Returns:
            Tuple of (data, success)
        """
        if extensions is None:
            extensions = ['json', 'csv', 'pickle']
            
        cache_file = self.find_in_cache(key, extensions, max_age_hours)
        
        if cache_file:
            try:
                extension = cache_file.split('.')[-1].lower()
                
                if extension == 'json':
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f), True
                        
                elif extension == 'csv':
                    return pd.read_csv(cache_file), True
                    
                elif extension == 'pickle':
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f), True
                        
                else:
                    # Read as text
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return f.read(), True
                        
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
                return None, False
        
        return None, False
    
    def invalidate_cache(self, key: str = None, prefix: str = None):
        """
        Invalidate specific cache entries or entries with a prefix.
        
        Args:
            key: Specific cache key to invalidate
            prefix: Prefix of cache keys to invalidate
        """
        if key is None and prefix is None:
            logger.warning("Cache invalidation requires either key or prefix")
            return
            
        pattern = ""
        if key:
            pattern = f"{key}.*"
        elif prefix:
            pattern = f"{prefix}_*.*"
            
        for cache_dir in self.cache_dirs:
            matching_files = glob.glob(os.path.join(cache_dir, pattern))
            for file_path in matching_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Invalidated cache file: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing cache file {file_path}: {e}")
        
        self.update_cache_size()
    
    def cleanup_cache(self, max_age_hours: Optional[int] = None):
        """
        Clean up expired cache files.
        
        Args:
            max_age_hours: Maximum age in hours (default: from config or 168/7 days)
        """
        if max_age_hours is None:
            max_age_hours = self.config.get('cleanup_ttl_hours', 168)  # 7 days default
            
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed_count = 0
        
        for cache_dir in self.cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            for file_name in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file_name)
                if os.path.isfile(file_path):
                    try:
                        file_time = os.path.getmtime(file_path)
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            removed_count += 1
                    except Exception as e:
                        logger.error(f"Error cleaning up cache file {file_path}: {e}")
        
        self.stats['cleanup_count'] += removed_count
        self.stats['last_cleanup'] = datetime.now().isoformat()
        self.update_cache_size()
        
        logger.info(f"Cache cleanup removed {removed_count} expired files")
    
    def update_cache_size(self):
        """
        Update the total size of cache files.
        """
        total_size = 0
        
        for cache_dir in self.cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            for path, _, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(path, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except Exception:
                        pass
        
        self.stats['cache_size_bytes'] = total_size
    
    def get_cache_info(self) -> Dict:
        """
        Get information about the cache system.
        
        Returns:
            Dictionary with cache statistics and info
        """
        # Update cache size before reporting
        self.update_cache_size()
        
        # Calculate hit rate
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        cache_info = {
            'directories': self.cache_dirs,
            'primary_directory': self.primary_cache_dir,
            'hit_rate': hit_rate,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'saved_api_calls': self.stats['saved_api_calls'],
            'cleanup_count': self.stats['cleanup_count'],
            'last_cleanup': self.stats['last_cleanup'],
            'cache_size_mb': self.stats['cache_size_bytes'] / (1024 * 1024),
            'file_count': self.count_cache_files()
        }
        
        return cache_info
    
    def count_cache_files(self) -> Dict[str, int]:
        """
        Count cache files by type.
        
        Returns:
            Dictionary with file counts by extension
        """
        counts = {'total': 0}
        
        for cache_dir in self.cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            for file in os.listdir(cache_dir):
                if os.path.isfile(os.path.join(cache_dir, file)):
                    counts['total'] += 1
                    ext = file.split('.')[-1].lower()
                    if ext in counts:
                        counts[ext] += 1
                    else:
                        counts[ext] = 1
        
        return counts
    
    def clear_all_cache(self, confirmation: bool = False):
        """
        Clear all cache files (dangerous operation).
        
        Args:
            confirmation: Must be True to confirm this operation
        """
        if not confirmation:
            logger.warning("Cache clear operation requires confirmation=True")
            return False
        
        cleared_count = 0
        
        for cache_dir in self.cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            try:
                # Remove all files in directory but keep the directory itself
                for file_name in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleared_count += 1
            except Exception as e:
                logger.error(f"Error clearing cache directory {cache_dir}: {e}")
        
        # Reset stats
        self.stats['hits'] = 0
        self.stats['misses'] = 0
        self.stats['saved_api_calls'] = 0
        self.stats['cleanup_count'] += cleared_count
        self.stats['last_cleanup'] = datetime.now().isoformat()
        self.update_cache_size()
        
        logger.info(f"Cache cleared: removed {cleared_count} files")
        return True
    
    def export_cache_manifest(self, output_file: str = "cache_manifest.json"):
        """
        Export a manifest of all cache files.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to manifest file
        """
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'cache_info': self.get_cache_info(),
            'files': []
        }
        
        for cache_dir in self.cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            for path, _, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(path, file)
                    try:
                        manifest['files'].append({
                            'path': file_path,
                            'size': os.path.getsize(file_path),
                            'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                            'type': file.split('.')[-1]
                        })
                    except Exception:
                        pass
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
            
        return output_file

# Cache API for easy usage
_cache_service = None

def get_cache_service(config: Optional[Dict] = None) -> CacheService:
    """
    Get or create the global cache service instance.
    
    Args:
        config: Optional configuration to initialize cache service
        
    Returns:
        CacheService instance
    """
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService(config)
    elif config is not None:
        # Re-initialize with new config
        _cache_service = CacheService(config)
        
    return _cache_service

def cache_api_response(prefix: str, identifier: str, callback, 
                     ttl_hours: int = 24, force_refresh: bool = False):
    """
    Cache API response helper function.
    
    Args:
        prefix: Cache prefix
        identifier: Unique identifier
        callback: Function to call if cache miss
        ttl_hours: TTL in hours
        force_refresh: Force refresh cache
        
    Returns:
        API response (from cache or fresh)
    """
    cache = get_cache_service()
    cache_key = cache.cache_key(prefix, identifier)
    
    if not force_refresh:
        data, success = cache.load_from_cache(cache_key, max_age_hours=ttl_hours)
        if success:
            return data
    
    # Cache miss or force refresh
    try:
        fresh_data = callback()
        cache.save_to_cache(cache_key, fresh_data)
        return fresh_data
    except Exception as e:
        logger.error(f"Error fetching fresh data for {prefix}_{identifier}: {e}")
        
        # Last resort: try to get expired cache
        data, success = cache.load_from_cache(cache_key, max_age_hours=None)
        if success:
            logger.warning(f"Using expired cache for {prefix}_{identifier}")
            return data
            
        raise

if __name__ == "__main__":
    """Simple CLI for cache management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Analyzer Cache Service')
    parser.add_argument('--info', action='store_true', help='Show cache information')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup expired cache files')
    parser.add_argument('--clear', action='store_true', help='Clear all cache files (requires --confirm)')
    parser.add_argument('--confirm', action='store_true', help='Confirm dangerous operations')
    parser.add_argument('--export', type=str, help='Export cache manifest to file')
    parser.add_argument('--max-age', type=int, default=168, help='Max age for cleanup in hours (default: 168)')
    
    args = parser.parse_args()
    
    # Initialize cache service
    cache_service = get_cache_service()
    
    if args.info:
        info = cache_service.get_cache_info()
        print("\nCache Information:")
        print(f"Directories: {', '.join(info['directories'])}")
        print(f"Primary Directory: {info['primary_directory']}")
        print(f"Hit Rate: {info['hit_rate']:.1f}%")
        print(f"Hits/Misses: {info['hits']}/{info['misses']}")
        print(f"Saved API Calls: {info['saved_api_calls']}")
        print(f"Cache Size: {info['cache_size_mb']:.2f} MB")
        print(f"File Count: {info['file_count']['total']} files")
        
        if 'json' in info['file_count']:
            print(f"JSON Files: {info['file_count']['json']}")
        if 'csv' in info['file_count']:
            print(f"CSV Files: {info['file_count']['csv']}")
        if 'pickle' in info['file_count']:
            print(f"Pickle Files: {info['file_count']['pickle']}")
    
    if args.cleanup:
        print(f"Cleaning up cache files older than {args.max_age} hours...")
        cache_service.cleanup_cache(max_age_hours=args.max_age)
        print(f"Cleanup complete. Removed {cache_service.stats['cleanup_count']} files.")
    
    if args.clear:
        if args.confirm:
            print("Clearing all cache files...")
            cache_service.clear_all_cache(confirmation=True)
            print("Cache cleared.")
        else:
            print("Warning: Clear operation requires --confirm flag for safety.")
    
    if args.export:
        output_file = args.export
        print(f"Exporting cache manifest to {output_file}...")
        cache_service.export_cache_manifest(output_file)
        print(f"Manifest exported.")
        
    # If no arguments provided, show info
    if not (args.info or args.cleanup or args.clear or args.export):
        parser.print_help()
