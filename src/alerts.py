import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, USE_TESTNET
from logger import get_logger
logger = get_logger('alerts')

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info('Telegram not configured, skipping message: %s', msg)
        return False
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': msg}
    try:
        r = requests.post(url, json=payload, timeout=5)
        return r.ok
    except Exception as e:
        logger.warning('Failed to send telegram message: %s', e)
        return False
