import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class AutonomousTop200Analyzer:
    """
    Autonomous analyzer for the top 200 cryptocurrencies by market cap.
    Automatically fetches and analyzes the top cryptocurrencies using CoinMarketCap API.
    """
    
    def __init__(self, api_key: Optional[str] = None, max_symbols: int = 200):
        """
        Initialize the Top 200 analyzer.
        
        Args:
            api_key: Optional CoinMarketCap API key
            max_symbols: Maximum number of symbols to analyze (default: 200)
        """
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.api_key = api_key or "a24bc925-ad5b-4167-8c83-1300410519ce"
        self.max_symbols = max_symbols
        
        self.headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json"
        }
        
        # Setup cache and output directories
        self.cache_dir = "crypto_cache"
        self.output_dir = "analysis_results"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Scoring weights
        self.weights = {
            "market_cap": 0.20,
            "volume": 0.15,
            "price_change": 0.15,
            "market_dominance": 0.10,
            "volatility": 0.10,
            "liquidity_ratio": 0.10,
            "rank_stability": 0.10,
            "supply_dynamics": 0.10
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fetch_top_cryptocurrencies(self, limit: int = None) -> List[Dict]:
        """
        Fetch top cryptocurrencies by market cap from CoinMarketCap.
        
        Args:
            limit: Number of cryptocurrencies to fetch (default: self.max_symbols)
            
        Returns:
            List of cryptocurrency data dictionaries
        """
        if limit is None:
            limit = self.max_symbols
            
        url = f"{self.base_url}/cryptocurrency/listings/latest"
        params = {
            "start": 1,
            "limit": limit,
            "convert": "USD",
            "sort": "market_cap",
            "sort_dir": "desc"
        }
        
        try:
            self.logger.info(f"Fetching top {limit} cryptocurrencies...")
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            if "data" in data:
                self.logger.info(f"Successfully fetched {len(data['data'])} cryptocurrencies")
                return data["data"]
            else:
                self.logger.error("No data found in API response")
                return []
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching cryptocurrency data: {e}")
            return []
    
    def calculate_bullrun_score(self, coin_data: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Calculate bullrun score for a single cryptocurrency.
        
        Args:
            coin_data: Cryptocurrency data from CoinMarketCap
            
        Returns:
            Tuple of (total_score, individual_scores)
        """
        try:
            quote_data = coin_data.get("quote", {}).get("USD", {})
            
            # Extract basic metrics
            market_cap = quote_data.get("market_cap", 0) or 0
            volume_24h = quote_data.get("volume_24h", 0) or 0
            price = quote_data.get("price", 0) or 0
            percent_change_24h = quote_data.get("percent_change_24h", 0) or 0
            percent_change_7d = quote_data.get("percent_change_7d", 0) or 0
            percent_change_30d = quote_data.get("percent_change_30d", 0) or 0
            
            circulating_supply = coin_data.get("circulating_supply", 0) or 0
            total_supply = coin_data.get("total_supply", 0) or 0
            max_supply = coin_data.get("max_supply", 0)
            cmc_rank = coin_data.get("cmc_rank", 1000) or 1000
            
            # Calculate individual scores
            scores = {}
            
            # 1. Market Cap Score (0-1, based on ranking)
            scores["market_cap"] = max(0, min(1, (201 - cmc_rank) / 200)) if cmc_rank <= 200 else 0
            
            # 2. Volume Score (volume/market_cap ratio)
            volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
            scores["volume"] = min(1, volume_ratio * 10)  # Scale factor
            
            # 3. Price Change Score (momentum)
            price_momentum = (percent_change_7d + percent_change_30d) / 2
            scores["price_change"] = max(0, min(1, (price_momentum + 50) / 100))  # Normalize to 0-1
            
            # 4. Market Dominance (for top coins)
            scores["market_dominance"] = max(0, min(1, market_cap / 1e12))  # Scale by 1T market cap
            
            # 5. Volatility Score (based on price changes)
            volatility = abs(percent_change_24h) + abs(percent_change_7d)
            scores["volatility"] = max(0, min(1, 1 - (volatility / 100)))  # Lower volatility = higher score
            
            # 6. Liquidity Ratio
            scores["liquidity_ratio"] = min(1, volume_24h / 1e9)  # Scale by 1B volume
            
            # 7. Rank Stability (bonus for established coins)
            scores["rank_stability"] = max(0, min(1, (101 - cmc_rank) / 100)) if cmc_rank <= 100 else 0.5
            
            # 8. Supply Dynamics
            if max_supply and max_supply > 0:
                supply_ratio = circulating_supply / max_supply
                scores["supply_dynamics"] = supply_ratio  # Higher circulation = higher score
            else:
                scores["supply_dynamics"] = 0.7  # Default for coins without max supply
            
            # Calculate weighted total score
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            
            return total_score, scores
            
        except Exception as e:
            self.logger.error(f"Error calculating bullrun score: {e}")
            return 0.0, {}
    
    def categorize_potential(self, score: float) -> str:
        """
        Categorize bullrun potential based on score.
        
        Args:
            score: Bullrun score (0-1)
            
        Returns:
            Category string
        """
        if score >= 0.8:
            return "Sehr hoch"
        elif score >= 0.7:
            return "Hoch"
        elif score >= 0.6:
            return "Überdurchschnittlich"
        elif score >= 0.5:
            return "Durchschnittlich"
        elif score >= 0.4:
            return "Unterdurchschnittlich"
        else:
            return "Niedrig"
    
    def analyze_cryptocurrency(self, coin_data: Dict) -> Dict:
        """
        Analyze a single cryptocurrency.
        
        Args:
            coin_data: Cryptocurrency data from CoinMarketCap
            
        Returns:
            Analysis result dictionary
        """
        try:
            # Basic information
            symbol = coin_data.get("symbol", "")
            name = coin_data.get("name", "")
            cmc_rank = coin_data.get("cmc_rank", 0)
            
            # Quote data
            quote_data = coin_data.get("quote", {}).get("USD", {})
            price = quote_data.get("price", 0) or 0
            market_cap = quote_data.get("market_cap", 0) or 0
            volume_24h = quote_data.get("volume_24h", 0) or 0
            percent_change_24h = quote_data.get("percent_change_24h", 0) or 0
            percent_change_7d = quote_data.get("percent_change_7d", 0) or 0
            
            # Calculate bullrun score
            total_score, individual_scores = self.calculate_bullrun_score(coin_data)
            
            # Categorize potential
            potential = self.categorize_potential(total_score)
            
            return {
                "symbol": symbol,
                "name": name,
                "cmc_rank": cmc_rank,
                "current_price_usd": price,
                "market_cap_usd": market_cap,
                "volume_24h_usd": volume_24h,
                "percent_change_24h": percent_change_24h,
                "percent_change_7d": percent_change_7d,
                "bullrun_score": total_score,
                "bullrun_potential": potential,
                "individual_scores": individual_scores,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing cryptocurrency {coin_data.get('symbol', 'unknown')}: {e}")
            return {}
    
    def run_analysis(self, limit: int = None) -> Dict:
        """
        Run complete Top 200 analysis.
        
        Args:
            limit: Number of cryptocurrencies to analyze
            
        Returns:
            Analysis summary
        """
        start_time = time.time()
        
        if limit is None:
            limit = self.max_symbols
            
        self.logger.info(f"Starting Top {limit} cryptocurrency analysis...")
        
        # Fetch cryptocurrency data
        crypto_data = self.fetch_top_cryptocurrencies(limit)
        
        if not crypto_data:
            return {
                "success": False,
                "message": "Failed to fetch cryptocurrency data",
                "successful_analyses": 0,
                "failed_analyses": 0
            }
        
        # Analyze each cryptocurrency
        analysis_results = []
        successful_analyses = 0
        failed_analyses = 0
        
        for i, coin_data in enumerate(crypto_data, 1):
            try:
                self.logger.info(f"Analyzing {i}/{len(crypto_data)}: {coin_data.get('symbol', 'unknown')}")
                
                analysis = self.analyze_cryptocurrency(coin_data)
                
                if analysis:
                    analysis_results.append(analysis)
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                    
                # Small delay to avoid rate limiting
                if i % 10 == 0:
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                failed_analyses += 1
                continue
        
        # Sort results by bullrun score
        analysis_results.sort(key=lambda x: x.get("bullrun_score", 0), reverse=True)
        
        # Create categorized results
        categorized_results = self.categorize_results(analysis_results)
        
        # Save results
        self.save_analysis_results(analysis_results, categorized_results)
        
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        success_rate = (successful_analyses / len(crypto_data)) * 100 if crypto_data else 0
        
        summary = {
            "success": True,
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "total_cryptos": len(crypto_data),
            "success_rate": success_rate,
            "duration_seconds": duration,
            "top_performers": analysis_results[:10],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Analysis completed: {successful_analyses}/{len(crypto_data)} successful ({success_rate:.1f}%)")
        
        return summary
    
    def categorize_results(self, results: List[Dict]) -> Dict:
        """
        Categorize analysis results by potential.
        
        Args:
            results: List of analysis results
            
        Returns:
            Categorized results dictionary
        """
        categories = {
            "very_high": [],
            "high": [],
            "above_average": [],
            "average": [],
            "below_average": [],
            "low": []
        }
        
        for result in results:
            potential = result.get("bullrun_potential", "Niedrig")
            symbol = result.get("symbol", "")
            
            if potential == "Sehr hoch":
                categories["very_high"].append(symbol)
            elif potential == "Hoch":
                categories["high"].append(symbol)
            elif potential == "Überdurchschnittlich":
                categories["above_average"].append(symbol)
            elif potential == "Durchschnittlich":
                categories["average"].append(symbol)
            elif potential == "Unterdurchschnittlich":
                categories["below_average"].append(symbol)
            else:
                categories["low"].append(symbol)
        
        return categories
    
    def save_analysis_results(self, results: List[Dict], categories: Dict) -> None:
        """
        Save analysis results to files.
        
        Args:
            results: Analysis results
            categories: Categorized results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save as CSV
            csv_filename = os.path.join(self.output_dir, f"top200_analysis_{timestamp}.csv")
            df = pd.DataFrame(results)
            df.to_csv(csv_filename, index=False)
            self.logger.info(f"Analysis saved to CSV: {csv_filename}")
            
            # Save as JSON
            json_filename = os.path.join(self.output_dir, f"top200_analysis_{timestamp}.json")
            full_results = {
                "coins": results,
                "potential_categories": categories,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_analyzed": len(results)
            }
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Analysis saved to JSON: {json_filename}")
            
            # Save summary report
            self.save_summary_report(results, categories, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {e}")
    
    def save_summary_report(self, results: List[Dict], categories: Dict, timestamp: str) -> None:
        """
        Save a human-readable summary report.
        
        Args:
            results: Analysis results
            categories: Categorized results
            timestamp: Timestamp string
        """
        try:
            report_filename = os.path.join(self.output_dir, f"top200_summary_{timestamp}.txt")
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("CRYPTO TOP 200 BULLRUN ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Analyzed: {len(results)}\n\n")
                
                # Summary statistics
                if results:
                    avg_score = sum(r.get("bullrun_score", 0) for r in results) / len(results)
                    f.write(f"Average Bullrun Score: {avg_score:.3f}\n\n")
                
                # Category breakdown
                f.write("POTENTIAL CATEGORIES:\n")
                f.write("-" * 30 + "\n")
                for category, symbols in categories.items():
                    if symbols:
                        f.write(f"{category.replace('_', ' ').title()}: {len(symbols)} coins\n")
                        f.write(f"  {', '.join(symbols[:10])}")
                        if len(symbols) > 10:
                            f.write(f" (and {len(symbols) - 10} more)")
                        f.write("\n\n")
                
                # Top 20 performers
                f.write("TOP 20 PERFORMERS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'Rank':<5} {'Symbol':<10} {'Name':<20} {'Score':<8} {'Potential':<20}\n")
                f.write("-" * 70 + "\n")
                
                for i, result in enumerate(results[:20], 1):
                    name = result.get("name", "")[:19]
                    f.write(f"{i:<5} {result.get('symbol', ''):<10} {name:<20} "
                           f"{result.get('bullrun_score', 0):<8.3f} {result.get('bullrun_potential', ''):<20}\n")
            
            self.logger.info(f"Summary report saved: {report_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving summary report: {e}")
    
    def get_latest_analysis(self) -> Optional[Dict]:
        """
        Get the most recent analysis results.
        
        Returns:
            Latest analysis data or None
        """
        try:
            # Find the most recent JSON file
            json_files = [f for f in os.listdir(self.output_dir) if f.startswith("top200_analysis_") and f.endswith(".json")]
            
            if not json_files:
                return None
            
            # Get the most recent file
            latest_file = max(json_files, key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
            file_path = os.path.join(self.output_dir, latest_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading latest analysis: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AutonomousTop200Analyzer(max_symbols=200)
    
    print("=" * 60)
    print("CRYPTO TOP 200 BULLRUN ANALYZER")
    print("=" * 60)
    print()
    
    # Run analysis
    try:
        result = analyzer.run_analysis(limit=50)  # Start with top 50 for testing
        
        if result["success"]:
            print(f"Analysis completed successfully!")
            print(f"Successful analyses: {result['successful_analyses']}")
            print(f"Failed analyses: {result['failed_analyses']}")
            print(f"Success rate: {result['success_rate']:.1f}%")
            print(f"Duration: {result['duration_seconds']:.1f} seconds")
            print()
            
            # Show top performers
            if result["top_performers"]:
                print("TOP 10 PERFORMERS:")
                print("-" * 50)
                for i, coin in enumerate(result["top_performers"], 1):
                    print(f"{i:2}. {coin['symbol']:<8} - {coin['bullrun_score']:.3f} ({coin['bullrun_potential']})")
        else:
            print(f"Analysis failed: {result.get('message', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")