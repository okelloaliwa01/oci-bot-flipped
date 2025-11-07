# Binance Futures Fast-Growth Scalping Bot

This repository contains a modular Binance Futures scalping bot implementing a 5-minute breakout strategy with volume confirmation (1.3x), dry-run/testnet support, and TP/SL management.

## Structure
- `src/` - main Python source files
- `tests/` - unit tests
- `requirements.txt` - Python dependencies
- `.env.example` - environment variable examples
- `Dockerfile` - simple image build
- `run_local.sh` - run locally script
- `run_docker.sh` - docker run helper

## Quickstart (testnet + dry-run)
1. Create a Python 3.9+ venv
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env to set keys and DRY_RUN/USE_TESTNET values
python src/bot.py
```

## Notes
- Start in DRY_RUN=True and USE_TESTNET=True for safe testing.
- Check `data/trade_history.csv` for appended simulated trades.
