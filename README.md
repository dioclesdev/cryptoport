# CryptoPort 🚀

A personal cryptocurrency portfolio tracker and analyzer built with Python and Flask.

## About

CryptoPort is a web-based tool I created to track and analyze my cryptocurrency investments. It provides real-time price updates, portfolio performance metrics, and helps me make better investment decisions based on technical indicators.

## Features

- **Portfolio Management**: Track your cryptocurrency holdings with purchase price and current value
- **Real-time Data**: Live price updates from CoinMarketCap API
- **Watchlist**: Monitor coins you're interested in
- **Performance Analysis**: Calculate profit/loss, percentage changes, and portfolio distribution
- **Bullrun Score**: Custom scoring algorithm to evaluate market momentum
- **Data Export**: Export portfolio data as CSV for tax purposes
- **Cache System**: Reduce API calls with intelligent caching
- **Clean UI**: Simple, responsive web interface built with Bootstrap

## Tech Stack

- **Backend**: Python 3.9+, Flask
- **Data**: pandas, NumPy for analysis
- **Visualization**: Matplotlib, Plotly
- **APIs**: CoinMarketCap, CoinGecko (optional)
- **Database**: SQLite (default) or PostgreSQL
- **Frontend**: HTML, CSS (Bootstrap), JavaScript

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- CoinMarketCap API key (free tier available)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dioclesdev/cryptoport.git
   cd cryptoport
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Activate on Linux/Mac
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   COINMARKETCAP_API_KEY=your_api_key_here
   SECRET_KEY=generate_a_random_key_here
   ```

5. **Initialize database**
   ```bash
   python setup_database.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

   Open your browser at `http://localhost:5000`

## Usage

### Web Interface

The web interface provides several pages:

- **Dashboard**: Overview of your portfolio and watchlist
- **Portfolio**: Manage your cryptocurrency holdings
- **Watchlist**: Track coins you're interested in
- **Top 200**: Analysis of top cryptocurrencies by market cap
- **Cache Status**: Monitor API cache and data freshness

### Adding Coins to Portfolio

1. Navigate to Portfolio page
2. Click "Add Position"
3. Enter coin symbol, amount, and purchase price
4. Click "Add" to save

### Managing Watchlist

1. Go to Watchlist page
2. Use the search or quick-add buttons
3. Remove coins by clicking the X button

## API Configuration

### CoinMarketCap

1. Sign up at [CoinMarketCap](https://coinmarketcap.com/api/)
2. Get your free API key
3. Add to `.env` file

The free tier includes:
- 10,000 calls/month
- 333 calls/day
- Latest market data

### Optional APIs

You can also configure:
- Binance API for trading data
- CoinGecko as alternative data source

## Project Structure

```
cryptoport/
├── app.py                 # Flask application
├── utils/
│   └── config.py         # Configuration
├── cache_service.py      # Caching logic
├── templates/            # HTML templates
├── static/              # CSS, JS, images
├── requirements.txt     # Dependencies
├── .env.example        # Environment template
└── README.md           # This file
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

The project follows PEP 8 guidelines. Format code with:
```bash
black .
```

### Contributing

This is a personal project, but suggestions and bug reports are welcome! Please open an issue to discuss changes.

## Deployment

### Local Deployment

The application runs well on a Raspberry Pi or any Linux server for personal use.

## License

This project is open source under the MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

- CoinMarketCap for the API
- Flask community for the excellent framework
- All open source contributors

## Contact

- GitHub: [@dioclesdev](https://github.com/dioclesdev)
- Project: [github.com/dioclesdev/cryptoport](https://github.com/dioclesdev/cryptoport)

Feel free to open an issue for questions or suggestions!

---

Built with ☕ and 🐍 by diocles
