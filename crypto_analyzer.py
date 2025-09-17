import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Union
import time
import os
import json
import schedule
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class CryptoBullrunAnalyzer:
    """
    Ein Analyse-Tool zur Bewertung des Potenzials einer Kryptowährung in einem Bullrun.
    Diese Version verwendet die CoinMarketCap API mit integriertem API-Key.
    """
    
    def __init__(self, custom_api_key: Optional[str] = None):
        """
        Initialisiert den Analyzer mit dem integrierten CoinMarketCap API-Key.
        Optional kann ein eigener API-Key übergeben werden.
        
        Args:
            custom_api_key: Optionaler eigener API-Key (überschreibt den integrierten Key)
        """
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        
        # Integrierter API-Key, kann durch eigenen ersetzt werden
        self.api_key = custom_api_key or "a24bc925-ad5b-4167-8c83-1300410519ce"
        
        self.headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json"
        }
        
        # Gewichtungen für verschiedene Faktoren (anpassbar)
        self.weights = {
            "market_cap": 0.15,
            "volume": 0.10,
            "price_change": 0.15,
            "ath_distance": 0.15,
            "developer_activity": 0.15,
            "community_score": 0.10,
            "liquidity": 0.10,
            "relative_position": 0.10
        }
        
        # Cache-Directory für API-Ergebnisse erstellen (spart API-Credits)
        self.cache_dir = "crypto_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Verzeichnis für Portfolio und Watchlist
        self.data_dir = "crypto_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Standarddateien für Portfolio und Watchlist
        self.portfolio_file = os.path.join(self.data_dir, "portfolio.json")
        self.watchlist_file = os.path.join(self.data_dir, "watchlist.json")
        
        # Initialisiere Standard-Portfolio und Watchlist, falls sie nicht existieren
        self.initialize_default_files()
        
        # Global Markets Data für spätere Berechnungen
        self.global_data = None
        self.fetch_and_cache_global_data()
    
    def initialize_default_files(self):
        """
        Initialisiert Standard-Portfolio und Watchlist-Dateien, falls sie nicht existieren.
        """
        # Standardcoins für Watchlist
        default_watchlist = {
            "coins": [
                "YNE", "ZETA", "ALT", "PEAQ", "MATIC", "DUSK", "DOT", "ATOM", "API3", 
                "PIKA", "FLOW", "SUI", "SAND", "TAO", "BEAM", "CHEX", "ZK", "GRASS", 
                "AAVE", "WIF", "MEW", "XYO", "ATR", "SHIB", "RSR", "PYTH", "JUB", 
                "WEN", "LINK", "BONK", "QNT", "CRV", "LTC", "FET", "HBAR", "PRCL", 
                "NOS", "ONDO", "ZEUS"
            ],
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alert_threshold": 0.05  # 5% Änderung im Score löst Alert aus
        }
        
        # Standard-Portfolio (leer)
        default_portfolio = {
            "holdings": {},
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Watchlist-Datei erstellen, falls nicht vorhanden
        if not os.path.exists(self.watchlist_file):
            with open(self.watchlist_file, "w", encoding="utf-8") as f:
                json.dump(default_watchlist, f, indent=4)
        
        # Portfolio-Datei erstellen, falls nicht vorhanden
        if not os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, "w", encoding="utf-8") as f:
                json.dump(default_portfolio, f, indent=4)
    
    def fetch_and_cache_global_data(self):
        """
        Holt und cached globale Kryptomarkt-Daten.
        """
        cache_file = os.path.join(self.cache_dir, "global_data.csv")
        
        # Prüfe, ob Cache existiert und nicht älter als 6 Stunden ist
        if os.path.exists(cache_file):
            cache_timestamp = os.path.getmtime(cache_file)
            if (time.time() - cache_timestamp) < 21600:  # 6 Stunden in Sekunden
                try:
                    self.global_data = pd.read_csv(cache_file).to_dict(orient="records")[0]
                    print("Globale Marktdaten aus Cache geladen.")
                    return
                except Exception as e:
                    print(f"Fehler beim Laden des Cache: {e}")
        
        # Ansonsten API abfragen
        print("Lade globale Marktdaten von CoinMarketCap...")
        global_data = self.fetch_global_data()
        
        if global_data:
            # In Cache speichern
            pd.DataFrame([global_data]).to_csv(cache_file, index=False)
            self.global_data = global_data
    
    def fetch_global_data(self) -> Dict:
        """
        Holt globale Kryptomarkt-Daten von CoinMarketCap.
        
        Returns:
            Dictionary mit globalen Marktdaten
        """
        url = f"{self.base_url}/global-metrics/quotes/latest"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json().get("data", {})
            
            # Flaches Dictionary für einfache Speicherung erstellen
            flat_data = {
                "total_cryptocurrencies": data.get("total_cryptocurrencies"),
                "total_exchanges": data.get("total_exchanges"),
                "btc_dominance": data.get("btc_dominance"),
                "eth_dominance": data.get("eth_dominance"),
                "total_market_cap": data.get("quote", {}).get("USD", {}).get("total_market_cap"),
                "total_volume_24h": data.get("quote", {}).get("USD", {}).get("total_volume_24h"),
                "altcoin_volume_24h": data.get("quote", {}).get("USD", {}).get("altcoin_volume_24h"),
                "altcoin_market_cap": data.get("quote", {}).get("USD", {}).get("altcoin_market_cap")
            }
            
            return flat_data
            
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen der globalen Daten: {e}")
            # Fallback-Werte
            return {
                "total_cryptocurrencies": 10000,
                "total_exchanges": 500,
                "btc_dominance": 50,
                "eth_dominance": 20,
                "total_market_cap": 1500000000000,  # 1.5 Billionen USD
                "total_volume_24h": 100000000000,   # 100 Milliarden USD
                "altcoin_volume_24h": 70000000000,  # 70 Milliarden USD
                "altcoin_market_cap": 750000000000  # 750 Milliarden USD
            }
    
    def fetch_coin_data(self, symbol: str) -> Dict:
        """
        Holt detaillierte Daten für eine bestimmte Kryptowährung.
        
        Args:
            symbol: Das Symbol der Kryptowährung (z.B. 'BTC', 'ETH')
            
        Returns:
            Dictionary mit Coin-Daten
        """
        cache_file = os.path.join(self.cache_dir, f"coin_{symbol.lower()}.csv")
        
        # Prüfe, ob Cache existiert und nicht älter als 1 Stunde ist
        if os.path.exists(cache_file):
            cache_timestamp = os.path.getmtime(cache_file)
            if (time.time() - cache_timestamp) < 3600:  # 1 Stunde in Sekunden
                try:
                    data = pd.read_csv(cache_file)
                    return data.to_dict(orient="records")[0]
                except Exception as e:
                    print(f"Fehler beim Laden des Cache für {symbol}: {e}")
        
        # Ansonsten API abfragen
        url = f"{self.base_url}/cryptocurrency/quotes/latest"
        params = {
            "symbol": symbol,
            "convert": "USD"
        }
        
        try:
            print(f"Lade Daten für {symbol} von CoinMarketCap...")
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and symbol in data["data"]:
                coin_data = data["data"][symbol]
                
                # In flache Struktur umwandeln für CSV-Speicherung
                flat_data = self.flatten_coin_data(coin_data)
                
                # In Cache speichern
                pd.DataFrame([flat_data]).to_csv(cache_file, index=False)
                
                return flat_data
            else:
                print(f"Keine Daten für {symbol} gefunden")
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen der Coin-Daten für {symbol}: {e}")
            if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'text'):
                print(f"API Response: {e.response.text}")
            return {}
    
    def flatten_coin_data(self, coin_data: Dict) -> Dict:
        """
        Wandelt verschachtelte CoinMarketCap-Daten in eine flache Struktur um.
        
        Args:
            coin_data: Verschachtelte Coin-Daten von CoinMarketCap
            
        Returns:
            Flaches Dictionary
        """
        flat_data = {
            "id": coin_data.get("id"),
            "name": coin_data.get("name", ""),
            "symbol": coin_data.get("symbol", ""),
            "slug": coin_data.get("slug", ""),
            "cmc_rank": coin_data.get("cmc_rank"),
            "max_supply": coin_data.get("max_supply"),
            "circulating_supply": coin_data.get("circulating_supply"),
            "total_supply": coin_data.get("total_supply"),
            "date_added": coin_data.get("date_added", "")
        }
        
        # Quote-Daten extrahieren
        if "quote" in coin_data and "USD" in coin_data["quote"]:
            usd_data = coin_data["quote"]["USD"]
            for key, value in usd_data.items():
                flat_data[f"usd_{key}"] = value
        
        return flat_data
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Analysiert eine Kryptowährung anhand ihres Symbols.
        
        Args:
            symbol: Symbol der Kryptowährung
            
        Returns:
            Analyseergebnis
        """
        coin_data = self.fetch_coin_data(symbol.upper())
        
        if not coin_data:
            return {"error": f"Keine Daten gefunden für Symbol '{symbol}'"}
        
        return self.analyze_coin(coin_data)
    
    def analyze_coin(self, coin_data: Dict) -> Dict:
        """
        Führt eine umfassende Analyse einer Kryptowährung durch.
        
        Args:
            coin_data: Coin-Daten von CoinMarketCap
            
        Returns:
            Dictionary mit Analyseergebnissen und Gesamtscore
        """
        if not coin_data:
            return {"error": "Keine Coin-Daten verfügbar"}
        
        # Symbol extrahieren
        symbol = coin_data.get("symbol", "")
        
        # Market-Daten extrahieren mit sicherer Behandlung von None-Werten
        market_cap = coin_data.get("usd_market_cap", 0)
        if market_cap is None:
            market_cap = 0
        else:
            try:
                market_cap = float(market_cap)
            except (TypeError, ValueError):
                market_cap = 0
        
        volume = coin_data.get("usd_volume_24h", 0)
        if volume is None:
            volume = 0
        else:
            try:
                volume = float(volume)
            except (TypeError, ValueError):
                volume = 0
        
        current_price = coin_data.get("usd_price", 0)
        if current_price is None:
            current_price = 0
        else:
            try:
                current_price = float(current_price)
            except (TypeError, ValueError):
                current_price = 0
        
        # Simplified scoring system (placeholder)
        scores = {
            "market_cap": min(1.0, market_cap / 1000000000) if market_cap > 0 else 0.5,
            "volume": min(1.0, volume / 100000000) if volume > 0 else 0.5,
            "price_change": 0.6,  # Placeholder
            "ath_distance": 0.7,  # Placeholder
            "developer_activity": 0.5,  # Placeholder
            "community_score": 0.6,  # Placeholder
            "liquidity": min(1.0, volume / market_cap) if market_cap > 0 else 0.5,
            "relative_position": 0.5  # Placeholder
        }
        
        # Gewichteten Gesamtscore berechnen
        total_score = sum(scores[key] * self.weights[key] for key in scores)
        
        # Bullrun-Potenzial-Kategorie
        bullrun_potential = self.categorize_potential(total_score)
        
        return {
            "coin_id": coin_data.get("id", ""),
            "name": coin_data.get("name", ""),
            "symbol": symbol.upper(),
            "current_price_usd": current_price,
            "market_cap_usd": market_cap,
            "volume_24h_usd": volume,
            "high_price_usd": current_price * 1.5,  # Placeholder
            "high_distance_percent": 25.0,  # Placeholder
            "market_cap_rank": coin_data.get("cmc_rank", None),
            "scores": scores,
            "total_score": total_score,
            "bullrun_potential": bullrun_potential,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def categorize_potential(self, score: float) -> str:
        """
        Kategorisiert das Bullrun-Potenzial basierend auf dem Gesamtscore.
        
        Args:
            score: Gesamtscore zwischen 0 und 1
            
        Returns:
            Kategoriebezeichnung
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
    
    def generate_report(self, analysis_result: Dict, show_plot: bool = True) -> str:
        """
        Erstellt einen detaillierten Bericht basierend auf den Analyseergebnissen.
        
        Args:
            analysis_result: Ergebnis der Analyse
            show_plot: Ob ein Plot angezeigt werden soll
            
        Returns:
            Analysebericht als String
        """
        if "error" in analysis_result:
            return f"Fehler: {analysis_result['error']}"
        
        # Radar-Plot erstellen
        if show_plot:
            self.create_radar_plot(analysis_result)
        
        # Textbericht erstellen
        report = [
            f"# Bullrun-Potenzial Analyse: {analysis_result['name']} ({analysis_result['symbol']})",
            f"\nAnalyse-Datum: {analysis_result['analysis_date']}",
            f"\n## Gesamtbewertung",
            f"\nGesamtscore: {analysis_result['total_score']:.2f}/1.00",
            f"\nBullrun-Potenzial: {analysis_result['bullrun_potential']}",
            f"\n## Marktdaten",
            f"\n- Aktueller Preis: ${analysis_result['current_price_usd']:.4f}",
            f"\n- Marktkapitalisierung: ${analysis_result['market_cap_usd']:,.0f}",
            f"\n- 24h-Handelsvolumen: ${analysis_result['volume_24h_usd']:,.0f}",
            f"\n- Marktrang: {analysis_result['market_cap_rank'] or 'N/A'}",
            f"\n- Entfernung zum Höchstpreis: {analysis_result['high_distance_percent']:.1f}%",
            f"\n## Detaillierte Scores",
        ]
        
        # Scores im Detail
        for factor, score in analysis_result["scores"].items():
            # Faktorname für die Anzeige formatieren
            factor_name = factor.replace("_", " ").title()
            report.append(f"\n- {factor_name}: {score:.2f}/1.00")
        
        # Empfehlungen
        report.append(f"\n## Empfehlungen")
        
        if analysis_result['total_score'] >= 0.7:
            report.append(f"\n{analysis_result['symbol']} zeigt starkes Potenzial für den nächsten Bullrun.")
        elif analysis_result['total_score'] >= 0.5:
            report.append(f"\n{analysis_result['symbol']} zeigt moderates Potenzial für den nächsten Bullrun.")
        else:
            report.append(f"\n{analysis_result['symbol']} zeigt begrenztes Potenzial für den nächsten Bullrun.")
        
        return "\n".join(report)
    
    def create_radar_plot(self, analysis_result: Dict) -> None:
        """
        Erstellt einen Radar-Plot (Spider-Plot) mit den Scores der verschiedenen Faktoren.
        
        Args:
            analysis_result: Ergebnis der Analyse
        """
        # Daten vorbereiten
        categories = list(analysis_result["scores"].keys())
        categories = [c.replace("_", " ").title() for c in categories]
        
        values = list(analysis_result["scores"].values())
        
        # Ein leeres Diagramm erstellen
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Anzahl der Variablen
        N = len(categories)
        
        # Winkel der Achsen berechnen
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Den ersten Punkt wiederholen, um das Polygon zu schließen
        
        # Werte anpassen
        values += values[:1]  # Den ersten Wert wiederholen, um das Polygon zu schließen
        
        # Plot zeichnen
        ax.plot(angles, values, linewidth=2, linestyle='solid', label="Scores")
        
        # Fläche ausfüllen
        ax.fill(angles, values, alpha=0.25)
        
        # Beschriftungen und Gridlines
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Y-Achsen (0-1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.set_ylim(0, 1)
        
        # Titel hinzufügen
        plt.title(f"Bullrun-Potenzial: {analysis_result['name']} ({analysis_result['symbol']})\n"
                 f"Gesamtscore: {analysis_result['total_score']:.2f} - {analysis_result['bullrun_potential']}",
                 size=15, y=1.1)
        
        # Score-Text hinzufügen
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.text(angle, value + 0.05, f"{value:.2f}", horizontalalignment='center')
        
        # Anzeigen
        plt.tight_layout()
        plt.show()
    
    def create_watchlist_comparison_chart(self, watchlist_analysis: Dict) -> None:
        """
        Erstellt ein Balkendiagramm für Watchlist-Vergleich.
        
        Args:
            watchlist_analysis: Ergebnis der Watchlist-Analyse
        """
        if not watchlist_analysis.get("coins"):
            return
            
        # Daten vorbereiten
        symbols = [coin["symbol"] for coin in watchlist_analysis["coins"]]
        scores = [coin["bullrun_score"] for coin in watchlist_analysis["coins"]]
        
        # Farbcodierung basierend auf Score
        colors = []
        for score in scores:
            if score >= 0.8:
                colors.append('darkgreen')
            elif score >= 0.7:
                colors.append('green')
            elif score >= 0.6:
                colors.append('yellowgreen')
            elif score >= 0.5:
                colors.append('gold')
            elif score >= 0.4:
                colors.append('orange')
            else:
                colors.append('red')
        
        # Plot erstellen
        plt.figure(figsize=(14, 8))
        bars = plt.bar(symbols, scores, color=colors)
        
        # Referenzlinien
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Hohes Potenzial')
        plt.axhline(y=0.5, color='gold', linestyle='--', alpha=0.7, label='Durchschnittliches Potenzial')
        plt.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Niedriges Potenzial')
        
        # Beschriftung
        plt.xlabel('Coins')
        plt.ylabel('Bullrun-Score')
        plt.title('Bullrun-Potenzial der Coins in der Watchlist', size=15)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_portfolio_pie_chart(self, portfolio_analysis: Dict) -> None:
        """
        Erstellt ein Tortendiagramm für Portfolio-Verteilung.
        
        Args:
            portfolio_analysis: Ergebnis der Portfolio-Analyse
        """
        if not portfolio_analysis.get("coins"):
            return
            
        # Daten vorbereiten
        labels = [f"{coin['symbol']} (${coin['value']:,.0f})" for coin in portfolio_analysis["coins"]]
        values = [coin["value"] for coin in portfolio_analysis["coins"]]
        
        # Tortendiagramm erstellen
        plt.figure(figsize=(10, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Portfolio-Zusammensetzung nach aktuellem Wert', size=15)
        plt.tight_layout()
        plt.show()
    
    # Portfolio management methods
    def load_portfolio(self) -> Dict:
        """
        Lädt das Portfolio aus der JSON-Datei.
        
        Returns:
            Portfolio-Dictionary
        """
        try:
            with open(self.portfolio_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Fehler beim Laden des Portfolios: {e}")
            return {"holdings": {}, "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    def save_portfolio(self, portfolio: Dict) -> None:
        """
        Speichert das Portfolio in der JSON-Datei.
        
        Args:
            portfolio: Portfolio-Dictionary
        """
        portfolio["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.portfolio_file, "w", encoding="utf-8") as f:
                json.dump(portfolio, f, indent=4)
        except Exception as e:
            print(f"Fehler beim Speichern des Portfolios: {e}")
    
    def add_to_portfolio(self, symbol: str, amount: float, avg_price: float, mode: str = "add") -> bool:
        """
        Fügt einen Coin zum Portfolio hinzu oder aktualisiert ihn.
        
        Args:
            symbol: Symbol des Coins
            amount: Anzahl der Coins
            avg_price: Durchschnittlicher Kaufpreis in USD
            mode: "add" (hinzufügen/akkumulieren) oder "set" (überschreiben)
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        portfolio = self.load_portfolio()
        
        # Prüfen, ob der Coin existiert
        try:
            coin_data = self.fetch_coin_data(symbol.upper())
            if not coin_data:
                print(f"Coin {symbol} konnte nicht gefunden werden.")
                return False
        except Exception as e:
            print(f"Fehler beim Abrufen von Daten für {symbol}: {e}")
            return False
        
        # Zum Portfolio hinzufügen
        symbol = symbol.upper()
        
        if symbol in portfolio["holdings"] and mode == "add":
            # Akkumulieren: Vorhandenen Eintrag erweitern
            old_amount = portfolio["holdings"][symbol]["amount"]
            old_price = portfolio["holdings"][symbol]["avg_price"]
            
            # Neuer gewichteter Durchschnittspreis
            total_value = old_amount * old_price + amount * avg_price
            new_amount = old_amount + amount
            new_avg_price = total_value / new_amount if new_amount > 0 else 0
            
            portfolio["holdings"][symbol] = {
                "amount": new_amount,
                "avg_price": new_avg_price
            }
        else:
            # Setzen: Position überschreiben oder neu erstellen
            portfolio["holdings"][symbol] = {
                "amount": amount,
                "avg_price": avg_price
            }
        
        # Portfolio speichern
        self.save_portfolio(portfolio)
        return True
    
    def update_portfolio_position(self, symbol: str, new_amount: float, new_avg_price: float) -> bool:
        """
        Aktualisiert eine bestehende Portfolio-Position mit neuen Werten (überschreibt).
        
        Args:
            symbol: Symbol des Coins
            new_amount: Neue Gesamtanzahl der Coins
            new_avg_price: Neuer durchschnittlicher Preis
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        portfolio = self.load_portfolio()
        
        # Prüfen, ob der Coin existiert
        try:
            coin_data = self.fetch_coin_data(symbol.upper())
            if not coin_data:
                print(f"Coin {symbol} konnte nicht gefunden werden.")
                return False
        except Exception as e:
            print(f"Fehler beim Abrufen von Daten für {symbol}: {e}")
            return False
        
        symbol = symbol.upper()
        
        # Position direkt überschreiben
        portfolio["holdings"][symbol] = {
            "amount": new_amount,
            "avg_price": new_avg_price
        }
        
        # Portfolio speichern
        self.save_portfolio(portfolio)
        return True
    
    def modify_portfolio_amount(self, symbol: str, amount_change: float, operation: str = "set") -> bool:
        """
        Modifiziert die Anzahl der Coins in einer Position.
        
        Args:
            symbol: Symbol des Coins
            amount_change: Menge der Änderung
            operation: "set" (setzen), "add" (hinzufügen), "subtract" (abziehen)
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        portfolio = self.load_portfolio()
        symbol = symbol.upper()
        
        if symbol not in portfolio["holdings"]:
            print(f"Coin {symbol} ist nicht im Portfolio vorhanden.")
            return False
        
        current_amount = portfolio["holdings"][symbol]["amount"]
        current_price = portfolio["holdings"][symbol]["avg_price"]
        
        if operation == "set":
            new_amount = amount_change
        elif operation == "add":
            new_amount = current_amount + amount_change
        elif operation == "subtract":
            new_amount = current_amount - amount_change
        else:
            print(f"Unbekannte Operation: {operation}")
            return False
        
        # Negative Mengen verhindern
        if new_amount < 0:
            new_amount = 0
        
        # Position entfernen wenn Menge 0
        if new_amount == 0:
            del portfolio["holdings"][symbol]
        else:
            portfolio["holdings"][symbol]["amount"] = new_amount
        
        # Portfolio speichern
        self.save_portfolio(portfolio)
        return True
    
    def get_portfolio_position(self, symbol: str) -> Dict:
        """
        Gibt die Details einer Portfolio-Position zurück.
        
        Args:
            symbol: Symbol des Coins
            
        Returns:
            Dictionary mit Position-Details oder leeres Dict
        """
        portfolio = self.load_portfolio()
        symbol = symbol.upper()
        
        if symbol in portfolio["holdings"]:
            return {
                "symbol": symbol,
                "amount": portfolio["holdings"][symbol]["amount"],
                "avg_price": portfolio["holdings"][symbol]["avg_price"]
            }
        
        return {}
    
    # Watchlist management methods
    def load_watchlist(self) -> Dict:
        """
        Lädt die Watchlist aus der JSON-Datei.
        
        Returns:
            Watchlist-Dictionary
        """
        try:
            with open(self.watchlist_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Fehler beim Laden der Watchlist: {e}")
            return {"coins": [], "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "alert_threshold": 0.05}
    
    def save_watchlist(self, watchlist: Dict) -> None:
        """
        Speichert die Watchlist in der JSON-Datei.
        
        Args:
            watchlist: Watchlist-Dictionary
        """
        watchlist["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.watchlist_file, "w", encoding="utf-8") as f:
                json.dump(watchlist, f, indent=4)
        except Exception as e:
            print(f"Fehler beim Speichern der Watchlist: {e}")
    
    def add_to_watchlist(self, symbol: str) -> bool:
        """
        Fügt einen Coin zur Watchlist hinzu.
        
        Args:
            symbol: Symbol des Coins
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        watchlist = self.load_watchlist()
        
        # Prüfen, ob der Coin existiert
        try:
            coin_data = self.fetch_coin_data(symbol.upper())
            if not coin_data:
                print(f"Coin {symbol} konnte nicht gefunden werden.")
                return False
        except Exception as e:
            print(f"Fehler beim Abrufen von Daten für {symbol}: {e}")
            return False
        
        # Zur Watchlist hinzufügen
        symbol = symbol.upper()
        if symbol in watchlist["coins"]:
            print(f"Coin {symbol} ist bereits in der Watchlist.")
            return True
        
        watchlist["coins"].append(symbol)
        
        # Watchlist speichern
        self.save_watchlist(watchlist)
        return True
    
    def remove_from_watchlist(self, symbol: str) -> bool:
        """
        Entfernt einen Coin aus der Watchlist.
        
        Args:
            symbol: Symbol des Coins
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        watchlist = self.load_watchlist()
        
        symbol = symbol.upper()
        if symbol not in watchlist["coins"]:
            print(f"Coin {symbol} ist nicht in der Watchlist vorhanden.")
            return False
        
        # Coin entfernen
        watchlist["coins"].remove(symbol)
        
        # Watchlist speichern
        self.save_watchlist(watchlist)
        return True
    
    def analyze_watchlist(self) -> Dict:
        """
        Analysiert alle Coins in der Watchlist.
        
        Returns:
            Dictionary mit Analyse-Ergebnissen
        """
        watchlist = self.load_watchlist()
        
        if not watchlist["coins"]:
            return {"error": "Die Watchlist ist leer."}
        
        results = {
            "coins": [],
            "potential_categories": {
                "very_high": [],
                "high": [],
                "above_average": [],
                "average": [],
                "below_average": [],
                "low": []
            },
            "recommendations": []
        }
        
        # Jeden Coin in der Watchlist analysieren
        for symbol in watchlist["coins"]:
            try:
                analysis = self.analyze_symbol(symbol)
                
                if "error" in analysis:
                    print(f"Fehler bei der Analyse von {symbol}: {analysis['error']}")
                    continue
                
                # Coin-Details hinzufügen
                coin_result = {
                    "symbol": symbol,
                    "name": analysis["name"],
                    "current_price": analysis["current_price_usd"],
                    "market_cap": analysis["market_cap_usd"],
                    "high_distance_percent": analysis["high_distance_percent"],
                    "bullrun_score": analysis["total_score"],
                    "bullrun_potential": analysis["bullrun_potential"],
                    "scores": analysis["scores"]
                }
                
                results["coins"].append(coin_result)
                
                # Bullrun-Potenzial kategorisieren
                potential = analysis["bullrun_potential"]
                if potential == "Sehr hoch":
                    results["potential_categories"]["very_high"].append(symbol)
                elif potential == "Hoch":
                    results["potential_categories"]["high"].append(symbol)
                elif potential == "Überdurchschnittlich":
                    results["potential_categories"]["above_average"].append(symbol)
                elif potential == "Durchschnittlich":
                    results["potential_categories"]["average"].append(symbol)
                elif potential == "Unterdurchschnittlich":
                    results["potential_categories"]["below_average"].append(symbol)
                else:
                    results["potential_categories"]["low"].append(symbol)
                
            except Exception as e:
                print(f"Fehler bei der Analyse von {symbol}: {e}")
                continue
        
        # Coins nach Bullrun-Score sortieren
        results["coins"] = sorted(results["coins"], key=lambda x: x["bullrun_score"], reverse=True)
        
        return results
    
    def remove_from_portfolio(self, symbol: str, amount: Optional[float] = None) -> bool:
        """
        Entfernt einen Coin aus dem Portfolio oder reduziert die Menge.
        
        Args:
            symbol: Symbol des Coins
            amount: Optionale Anzahl der zu entfernenden Coins. Wenn None, wird der Coin komplett entfernt.
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        portfolio = self.load_portfolio()
        
        symbol = symbol.upper()
        if symbol not in portfolio["holdings"]:
            print(f"Coin {symbol} ist nicht im Portfolio vorhanden.")
            return False
        
        if amount is None or amount >= portfolio["holdings"][symbol]["amount"]:
            # Coin komplett entfernen
            del portfolio["holdings"][symbol]
        else:
            # Menge reduzieren
            portfolio["holdings"][symbol]["amount"] -= amount
        
        # Portfolio speichern
        self.save_portfolio(portfolio)
        return True
    
    def analyze_portfolio(self) -> Dict:
        """
        Analysiert das gesamte Portfolio und berechnet Performance-Metriken.
        
        Returns:
            Dictionary mit Analyse-Ergebnissen
        """
        portfolio = self.load_portfolio()
        
        if not portfolio["holdings"]:
            return {"error": "Das Portfolio ist leer."}
        
        results = {
            "total_value": 0,
            "total_investment": 0,
            "profit_loss": 0,
            "profit_loss_percent": 0,
            "coins": [],
            "bullrun_potential": {
                "very_high": [],
                "high": [],
                "above_average": [],
                "average": [],
                "below_average": [],
                "low": []
            },
            "optimization_suggestions": []
        }
        
        # Jeden Coin im Portfolio analysieren
        for symbol, data in portfolio["holdings"].items():
            try:
                analysis = self.analyze_symbol(symbol)
                
                if "error" in analysis:
                    print(f"Fehler bei der Analyse von {symbol}: {analysis['error']}")
                    continue
                
                current_price = analysis["current_price_usd"]
                amount = data["amount"]
                avg_price = data["avg_price"]
                
                # Wert und Performance berechnen
                current_value = current_price * amount
                investment = avg_price * amount
                profit_loss = current_value - investment
                profit_loss_percent = (profit_loss / investment) * 100 if investment > 0 else 0
                
                # Gesamtwerte aktualisieren
                results["total_value"] += current_value
                results["total_investment"] += investment
                results["profit_loss"] += profit_loss
                
                # Coin-Details hinzufügen
                coin_result = {
                    "symbol": symbol,
                    "name": analysis["name"],
                    "amount": amount,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "value": current_value,
                    "profit_loss": profit_loss,
                    "profit_loss_percent": profit_loss_percent,
                    "bullrun_score": analysis["total_score"],
                    "bullrun_potential": analysis["bullrun_potential"]
                }
                
                results["coins"].append(coin_result)
                
                # Bullrun-Potenzial kategorisieren
                potential = analysis["bullrun_potential"]
                if potential == "Sehr hoch":
                    results["bullrun_potential"]["very_high"].append(symbol)
                elif potential == "Hoch":
                    results["bullrun_potential"]["high"].append(symbol)
                elif potential == "Überdurchschnittlich":
                    results["bullrun_potential"]["above_average"].append(symbol)
                elif potential == "Durchschnittlich":
                    results["bullrun_potential"]["average"].append(symbol)
                elif potential == "Unterdurchschnittlich":
                    results["bullrun_potential"]["below_average"].append(symbol)
                else:
                    results["bullrun_potential"]["low"].append(symbol)
                
            except Exception as e:
                print(f"Fehler bei der Analyse von {symbol}: {e}")
                continue
        
        # Gesamtperformance berechnen
        if results["total_investment"] > 0:
            results["profit_loss_percent"] = (results["profit_loss"] / results["total_investment"]) * 100
        
        # Coins nach Bullrun-Score sortieren
        results["coins"] = sorted(results["coins"], key=lambda x: x["bullrun_score"], reverse=True)
        
        return results
    
    def save_html_report(self, analysis_result: Dict, filename: str = None) -> str:
        """
        Speichert den HTML-Report als lokale Datei.
        
        Args:
            analysis_result: Ergebnis der Watchlist-Analyse
            filename: Optionaler Dateiname
            
        Returns:
            Pfad zur gespeicherten Datei
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_report_{timestamp}.html"
        
        # Charts erstellen
        charts = {}
        if hasattr(self, 'create_watchlist_comparison_chart'):
            # Vereinfachte Chart-Erstellung für lokale HTML-Reports
            try:
                self.create_watchlist_comparison_chart(analysis_result)
                # Chart als Datei speichern
                chart_filename = filename.replace('.html', '_chart.png')
                plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
                charts['chart_path'] = chart_filename
                plt.close()
            except:
                pass
        
        # Vereinfachtes HTML-Template für lokale Anzeige
        html_content = self._create_simple_html_report(analysis_result, charts)
        
        # Datei speichern
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML-Report gespeichert als: {filename}")
        return filename
    
    def _create_simple_html_report(self, analysis_result: Dict, charts: Dict = None) -> str:
        """
        Erstellt ein vereinfachtes HTML-Template für lokale Anzeige.
        """
        current_time = datetime.now().strftime("%d.%m.%Y um %H:%M Uhr")
        
        valid_coins = [coin for coin in analysis_result.get("coins", []) 
                      if coin.get("bullrun_score") is not None]
        
        total_coins = len(valid_coins)
        avg_score = sum(float(coin["bullrun_score"]) for coin in valid_coins) / total_coins if total_coins > 0 else 0
        
        html = f"""
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Watchlist Report - {current_time}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .stats {{ display: flex; justify-content: space-around; background: #f8f9fa; padding: 20px; }}
        .stat {{ text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .score-high {{ color: #28a745; font-weight: bold; }}
        .score-medium {{ color: #ffc107; font-weight: bold; }}
        .score-low {{ color: #dc3545; font-weight: bold; }}
        .chart {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Crypto Watchlist Report</h1>
        <p>{current_time}</p>
    </div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-number">{total_coins}</div>
            <div>Analysierte Coins</div>
        </div>
        <div class="stat">
            <div class="stat-number">{avg_score:.3f}</div>
            <div>Durchschnittlicher Score</div>
        </div>
        <div class="stat">
            <div class="stat-number">{len([c for c in valid_coins if c["bullrun_score"] >= 0.7])}</div>
            <div>Top Performer (≥0.7)</div>
        </div>
    </div>
"""
        
        if charts and charts.get('chart_path'):
            html += f"""
    <div class="chart">
        <h3>Bullrun-Score Übersicht</h3>
        <img src="{charts['chart_path']}" alt="Chart" style="max-width: 100%;">
    </div>
"""
        
        html += """
    <h2>Detaillierte Coin-Analyse</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Name</th>
            <th>Score</th>
            <th>Potenzial</th>
            <th>Preis (USD)</th>
            <th>Marktkapitalisierung</th>
        </tr>
"""
        
        sorted_coins = sorted(valid_coins, key=lambda x: float(x["bullrun_score"]), reverse=True)
        
        for i, coin in enumerate(sorted_coins, 1):
            score = float(coin["bullrun_score"])
            score_class = "score-high" if score >= 0.7 else "score-medium" if score >= 0.5 else "score-low"
            
            price = coin.get("current_price", 0)
            market_cap = coin.get("market_cap", 0)
            
            html += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{coin.get("symbol", "N/A")}</strong></td>
            <td>{coin.get("name", "N/A")}</td>
            <td class="{score_class}">{score:.3f}</td>
            <td>{coin.get("bullrun_potential", "N/A")}</td>
            <td>${price:.4f}</td>
            <td>${market_cap:,.0f}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        return html
    
    def create_csv_report(self, analysis_result: Dict, filename: str = None) -> str:
        """
        Erstellt einen CSV-Report der Watchlist-Analyse.
        
        Args:
            analysis_result: Ergebnis der Watchlist-Analyse
            filename: Optionaler Dateiname
            
        Returns:
            Pfad zur gespeicherten CSV-Datei
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_report_{timestamp}.csv"
        
        valid_coins = [coin for coin in analysis_result.get("coins", []) 
                      if coin.get("bullrun_score") is not None]
        
        # DataFrame erstellen
        data = []
        for coin in valid_coins:
            data.append({
                "Symbol": coin.get("symbol", ""),
                "Name": coin.get("name", ""),
                "Bullrun_Score": coin.get("bullrun_score", 0),
                "Bullrun_Potenzial": coin.get("bullrun_potential", ""),
                "Preis_USD": coin.get("current_price", 0),
                "Marktkapitalisierung_USD": coin.get("market_cap", 0),
                "ATH_Distanz_Prozent": coin.get("high_distance_percent", 0)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values(by="Bullrun_Score", ascending=False)
        df.to_csv(filename, index=False)
        
        print(f"CSV-Report gespeichert als: {filename}")
        return filename
    
    def create_chart_attachments(self, analysis_result: Dict, output_dir: str = "charts") -> List[str]:
        """
        Erstellt Charts als PNG-Dateien für E-Mail-Anhänge.
        
        Args:
            analysis_result: Ergebnis der Watchlist-Analyse
            output_dir: Verzeichnis für die Chart-Dateien
            
        Returns:
            Liste der erstellten Dateipfade
        """
        import os
        
        # Ausgabeverzeichnis erstellen
        os.makedirs(output_dir, exist_ok=True)
        
        chart_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if "error" in analysis_result or not analysis_result.get("coins"):
            print("Keine gültigen Daten für Charts verfügbar")
            return chart_files
        
        try:
            valid_coins = [coin for coin in analysis_result["coins"] 
                          if coin.get("bullrun_score") is not None and coin.get("symbol")]
            
            if not valid_coins:
                return chart_files
            
            # 1. Bullrun-Scores Balkendiagramm
            plt.figure(figsize=(16, 10))
            
            symbols = [coin["symbol"] for coin in valid_coins]
            scores = [float(coin["bullrun_score"]) for coin in valid_coins]
            
            # Farbcodierung
            colors = []
            for score in scores:
                if score >= 0.8:
                    colors.append('#1a5490')
                elif score >= 0.7:
                    colors.append('#28a745')
                elif score >= 0.6:
                    colors.append('#fd7e14')
                elif score >= 0.5:
                    colors.append('#ffc107')
                elif score >= 0.4:
                    colors.append('#6f42c1')
                else:
                    colors.append('#dc3545')
            
            # Nach Score sortieren
            sorted_data = sorted(zip(symbols, scores, colors), key=lambda x: x[1], reverse=True)
            symbols, scores, colors = zip(*sorted_data)
            
            # Plot erstellen
            bars = plt.bar(range(len(symbols)), scores, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=0.8)
            
            # Referenzlinien
            plt.axhline(y=0.8, color='#1a5490', linestyle='--', alpha=0.7, linewidth=2, label='Sehr hoch (≥0.8)')
            plt.axhline(y=0.7, color='#28a745', linestyle='--', alpha=0.7, linewidth=2, label='Hoch (≥0.7)')
            plt.axhline(y=0.5, color='#ffc107', linestyle='--', alpha=0.7, linewidth=2, label='Durchschnitt (≥0.5)')
            
            # Score-Werte auf Balken
            for i, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(i, bar.get_height() + 0.01, f'{score:.2f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.xticks(range(len(symbols)), symbols, rotation=45, ha='right', fontsize=10)
            plt.xlabel('Cryptocurrency Symbols', fontsize=12, fontweight='bold')
            plt.ylabel('Bullrun Score', fontsize=12, fontweight='bold')
            plt.title('Bullrun-Potenzial Analyse - Watchlist Overview', fontsize=16, fontweight='bold')
            plt.ylim(0, max(1.05, max(scores) * 1.1))
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            chart1_path = os.path.join(output_dir, f"bullrun_scores_{timestamp}.png")
            plt.savefig(chart1_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            chart_files.append(chart1_path)
            
            # 2. Top 10 Horizontal Bar Chart
            plt.figure(figsize=(14, 10))
            
            top_10 = sorted(valid_coins, key=lambda x: x["bullrun_score"], reverse=True)[:10]
            symbols_top10 = [coin['symbol'] for coin in top_10]
            scores_top10 = [float(coin["bullrun_score"]) for coin in top_10]
            potential_top10 = [coin.get("bullrun_potential", "Unbekannt") for coin in top_10]
            
            colors_top10 = []
            for score in scores_top10:
                if score >= 0.8:
                    colors_top10.append('#1a5490')
                elif score >= 0.7:
                    colors_top10.append('#28a745')
                elif score >= 0.6:
                    colors_top10.append('#fd7e14')
                elif score >= 0.5:
                    colors_top10.append('#ffc107')
                else:
                    colors_top10.append('#6f42c1')
            
            y_pos = range(len(symbols_top10))
            bars = plt.barh(y_pos, scores_top10, color=colors_top10, alpha=0.8, 
                           edgecolor='black', linewidth=0.8)
            
            # Score und Potenzial anzeigen
            for i, (bar, score, potential) in enumerate(zip(bars, scores_top10, potential_top10)):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f} - {potential}',
                        ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.yticks(y_pos, symbols_top10, fontsize=11)
            plt.xlabel('Bullrun Score', fontsize=12, fontweight='bold')
            plt.title('Top 10 Coins nach Bullrun-Potenzial', fontsize=16, fontweight='bold')
            plt.xlim(0, max(1.05, max(scores_top10) * 1.2))
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            chart2_path = os.path.join(output_dir, f"top_10_coins_{timestamp}.png")
            plt.savefig(chart2_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            chart_files.append(chart2_path)
            
            # 3. Potenzial-Kategorien Tortendiagramm
            plt.figure(figsize=(12, 10))
            
            categories = ['Sehr hoch', 'Hoch', 'Überdurchschnittlich', 'Durchschnittlich', 'Unterdurchschnittlich', 'Niedrig']
            values = [
                len(analysis_result["potential_categories"]["very_high"]),
                len(analysis_result["potential_categories"]["high"]),
                len(analysis_result["potential_categories"]["above_average"]),
                len(analysis_result["potential_categories"]["average"]),
                len(analysis_result["potential_categories"]["below_average"]),
                len(analysis_result["potential_categories"]["low"])
            ]
            colors_pie = ['#1a5490', '#28a745', '#fd7e14', '#ffc107', '#6f42c1', '#dc3545']
            
            # Nur Kategorien mit Werten > 0
            non_zero_data = [(cat, val, col) for cat, val, col in zip(categories, values, colors_pie) if val > 0]
            
            if non_zero_data:
                categories, values, colors_pie = zip(*non_zero_data)
                
                wedges, texts, autotexts = plt.pie(values, labels=categories, colors=colors_pie, 
                                                  autopct='%1.1f%%', startangle=90, 
                                                  explode=[0.05] * len(values))
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(12)
                
                for text in texts:
                    text.set_fontsize(12)
                    text.set_fontweight('bold')
                
                plt.title('Verteilung nach Bullrun-Potenzial', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                chart3_path = os.path.join(output_dir, f"potential_distribution_{timestamp}.png")
                plt.savefig(chart3_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                chart_files.append(chart3_path)
            
            print(f"Charts als Dateien gespeichert: {len(chart_files)} Dateien")
            for file in chart_files:
                print(f"  - {file}")
            
        except Exception as e:
            print(f"Fehler beim Erstellen der Chart-Dateien: {e}")
            import traceback
            traceback.print_exc()
        
        return chart_files
    
    def send_email_with_chart_attachments(self, analysis_result: Dict, recipient: str, 
                                        smtp_config: Dict) -> bool:
        """
        Sendet einen E-Mail-Report mit Charts als Anhänge.
        
        Args:
            analysis_result: Ergebnis der Watchlist-Analyse
            recipient: Empfänger-E-Mail-Adresse
            smtp_config: SMTP-Konfiguration
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.base import MIMEBase
            from email import encoders
            import smtplib
            import os
            
            # Charts als Dateien erstellen
            print("Erstelle Chart-Dateien...")
            chart_files = self.create_chart_attachments(analysis_result)
            
            # Vereinfachte HTML-E-Mail (ohne eingebettete Charts)
            html_content = self._create_simple_email_html(analysis_result)
            
            # E-Mail-Nachricht erstellen
            msg = MIMEMultipart()
            msg['From'] = smtp_config['sender_email']
            msg['To'] = recipient
            msg['Subject'] = f"Crypto Watchlist Report - {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            
            # HTML-Teil hinzufügen
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # Chart-Dateien als Anhänge hinzufügen
            for chart_file in chart_files:
                if os.path.exists(chart_file):
                    with open(chart_file, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    filename = os.path.basename(chart_file)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {filename}',
                    )
                    msg.attach(part)
            
            # E-Mail senden
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(smtp_config['smtp_username'], smtp_config['smtp_password'])
                server.send_message(msg)
            
            print(f"E-Mail mit {len(chart_files)} Chart-Anhängen erfolgreich versendet!")
            return True
            
        except Exception as e:
            print(f"Fehler beim Senden der E-Mail mit Anhängen: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_simple_email_html(self, analysis_result: Dict) -> str:
        """
        Erstellt vereinfachte HTML-E-Mail ohne eingebettete Charts.
        """
        current_time = datetime.now().strftime("%d.%m.%Y um %H:%M Uhr")
        
        valid_coins = [coin for coin in analysis_result.get("coins", []) 
                      if coin.get("bullrun_score") is not None]
        
        total_coins = len(valid_coins)
        avg_score = sum(float(coin["bullrun_score"]) for coin in valid_coins) / total_coins if total_coins > 0 else 0
        top_performers = len([coin for coin in valid_coins if float(coin["bullrun_score"]) >= 0.7])
        
        html = f"""
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>Crypto Watchlist Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .stats {{ background: #f8f9fa; padding: 20px; }}
        .stat {{ display: inline-block; margin: 10px; text-align: center; }}
        .stat-number {{ font-size: 1.5em; font-weight: bold; color: #3498db; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
        .score-high {{ color: #28a745; font-weight: bold; }}
        .score-medium {{ color: #ffc107; font-weight: bold; }}
        .score-low {{ color: #dc3545; font-weight: bold; }}
        .attachment-note {{ background: #e7f3ff; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Crypto Watchlist Report</h1>
        <p>{current_time}</p>
    </div>
    
    <div class="attachment-note">
        <strong>📊 Charts als Anhänge:</strong> Die detaillierten Charts finden Sie als PNG-Dateien im Anhang dieser E-Mail!
    </div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-number">{total_coins}</div>
            <div>Analysierte Coins</div>
        </div>
        <div class="stat">
            <div class="stat-number">{avg_score:.3f}</div>
            <div>Durchschn. Score</div>
        </div>
        <div class="stat">
            <div class="stat-number">{top_performers}</div>
            <div>Top Performer</div>
        </div>
    </div>
    
    <h2>Top 15 Coins nach Bullrun-Score</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Name</th>
            <th>Score</th>
            <th>Potenzial</th>
        </tr>
"""
        
        sorted_coins = sorted(valid_coins, key=lambda x: float(x["bullrun_score"]), reverse=True)[:15]
        
        for i, coin in enumerate(sorted_coins, 1):
            score = float(coin["bullrun_score"])
            score_class = "score-high" if score >= 0.7 else "score-medium" if score >= 0.5 else "score-low"
            
            html += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{coin.get("symbol", "N/A")}</strong></td>
            <td>{coin.get("name", "N/A")}</td>
            <td class="{score_class}">{score:.3f}</td>
            <td>{coin.get("bullrun_potential", "N/A")}</td>
        </tr>
"""
        
        html += """
    </table>
    
    <p><strong>Hinweis:</strong> Die vollständigen visuellen Analysen finden Sie in den angehängten PNG-Dateien.</p>
</body>
</html>
"""
        return html
    
    def compare_coins(self, symbols: List[str]) -> pd.DataFrame:
        """
        Vergleicht mehrere Coins und erstellt eine Ranking-Tabelle.
        
        Args:
            symbols: Liste mit Coin-Symbolen
            
        Returns:
            DataFrame mit Vergleichsergebnissen
        """
        results = []
        
        for symbol in symbols:
            print(f"Analysiere {symbol}...")
            try:
                result = self.analyze_symbol(symbol)
                
                if "error" not in result:
                    results.append({
                        "Symbol": result["symbol"],
                        "Name": result["name"],
                        "Preis USD": result["current_price_usd"],
                        "Marktkapitalisierung": result["market_cap_usd"],
                        "Entfernung zum Hoch": f"{result['high_distance_percent']:.1f}%",
                        "Bullrun-Score": result["total_score"],
                        "Potenzial": result["bullrun_potential"]
                    })
                else:
                    print(f"Fehler bei {symbol}: {result['error']}")
            except Exception as e:
                print(f"Fehler bei der Analyse von {symbol}: {e}")
        
        # In DataFrame umwandeln und nach Score sortieren
        df = pd.DataFrame(results)
        if not df.empty:
            return df.sort_values(by="Bullrun-Score", ascending=False).reset_index(drop=True)
        else:
            return pd.DataFrame()


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Analyzer initialisieren (API-Key ist bereits integriert)
    analyzer = CryptoBullrunAnalyzer()
    
    print("\nCrypto Bullrun Potenzial Analyzer")
    print("=" * 35)
    print("1. Einzelnen Coin analysieren")
    print("2. Portfolio verwalten")
    print("3. Watchlist analysieren")
    print("0. Beenden")
    
    choice = input("\nWähle eine Option (0-3): ").strip()
    
    if choice == "1":
        # Einzelnen Coin analysieren
        coin_input = input("Gib das Symbol einer Kryptowährung ein (z.B. BTC): ").strip()
        
        # Analyse durchführen
        result = analyzer.analyze_symbol(coin_input)
        
        # Bericht anzeigen
        print(analyzer.generate_report(result, show_plot=False))
        
    elif choice == "2":
        # Portfolio verwalten
        symbol = input("Symbol des Coins: ").strip().upper()
        try:
            amount = float(input("Anzahl der Coins: ").strip())
            avg_price = float(input("Durchschnittlicher Kaufpreis in USD: ").strip())
            
            if analyzer.add_to_portfolio(symbol, amount, avg_price):
                print(f"{symbol} wurde zum Portfolio hinzugefügt.")
            else:
                print(f"Fehler beim Hinzufügen von {symbol} zum Portfolio.")
        except ValueError:
            print("Fehler: Bitte gib gültige Zahlen ein.")
        
    elif choice == "3":
        # Watchlist analysieren
        print("\nWatchlist wird analysiert...")
        watchlist_analysis = analyzer.analyze_watchlist()
        
        if "error" not in watchlist_analysis:
            print("\n" + "="*50)
            print("WATCHLIST ANALYSE ERGEBNISSE")
            print("="*50)
            
            total_coins = len(watchlist_analysis["coins"])
            if total_coins > 0:
                avg_score = sum(coin["bullrun_score"] for coin in watchlist_analysis["coins"]) / total_coins
                
                print(f"Analysierte Coins: {total_coins}")
                print(f"Durchschnittlicher Bullrun-Score: {avg_score:.3f}")
                
                # Top 10 Coins
                print(f"\nTop 10 Coins nach Bullrun-Score:")
                print("-" * 60)
                print(f"{'Rank':<4} {'Symbol':<8} {'Name':<15} {'Score':<6} {'Potenzial':<20}")
                print("-" * 60)
                
                top_coins = sorted(watchlist_analysis["coins"], key=lambda x: x["bullrun_score"], reverse=True)[:10]
                for i, coin in enumerate(top_coins, 1):
                    name_truncated = coin["name"][:14] if len(coin["name"]) > 14 else coin["name"]
                    print(f"{i:<4} {coin['symbol']:<8} {name_truncated:<15} {coin['bullrun_score']:<6.3f} {coin['bullrun_potential']:<20}")
            else:
                print("Keine erfolgreichen Analysen.")
        else:
            print(f"Fehler: {watchlist_analysis['error']}")
        
    elif choice == "0":
        print("Programm wird beendet.")
    
    else:
        print("Ungültige Eingabe. Programm wird beendet.")