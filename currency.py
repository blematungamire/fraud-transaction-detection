"""
Multi-currency Module
=====================
Provides currency metadata, reference exchange rates, and conversion helpers.

The ML model is trained on USD-denominated synthetic transactions, so every
transaction amount is normalized to USD before scoring and converted back to
the user-selected currency for display and audit logging.

Rates are static reference rates (units of currency per 1 USD) designed to
work offline. In a production deployment these should be replaced with live
rates (e.g. exchangerate.host, Open Exchange Rates) surfaced through
Streamlit secrets -- see README.
"""
from typing import Dict

BASE_CURRENCY = "USD"
HIGH_VALUE_THRESHOLD_USD = 5000.0

# Currency metadata: code -> (name, symbol)
SUPPORTED_CURRENCIES: Dict[str, Dict[str, str]] = {
    "USD": {"name": "US Dollar", "symbol": "$"},
    "EUR": {"name": "Euro", "symbol": "€"},
    "GBP": {"name": "British Pound", "symbol": "£"},
    "JPY": {"name": "Japanese Yen", "symbol": "¥"},
    "CNY": {"name": "Chinese Yuan", "symbol": "¥"},
    "INR": {"name": "Indian Rupee", "symbol": "₹"},
    "AUD": {"name": "Australian Dollar", "symbol": "A$"},
    "CAD": {"name": "Canadian Dollar", "symbol": "C$"},
    "CHF": {"name": "Swiss Franc", "symbol": "CHF "},
    "SGD": {"name": "Singapore Dollar", "symbol": "S$"},
    "AED": {"name": "UAE Dirham", "symbol": "AED "},
    "ZAR": {"name": "South African Rand", "symbol": "R"},
    "NGN": {"name": "Nigerian Naira", "symbol": "₦"},
    "BRL": {"name": "Brazilian Real", "symbol": "R$"},
    "KRW": {"name": "South Korean Won", "symbol": "₩"},
    "HKD": {"name": "Hong Kong Dollar", "symbol": "HK$"},
}

# Reference rates: units of currency per 1 USD (approximate, off-line values)
RATE_PER_USD: Dict[str, float] = {
    "USD": 1.00,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.50,
    "CNY": 7.24,
    "INR": 83.30,
    "AUD": 1.53,
    "CAD": 1.36,
    "CHF": 0.90,
    "SGD": 1.34,
    "AED": 3.67,
    "ZAR": 18.50,
    "NGN": 1450.00,
    "BRL": 5.40,
    "KRW": 1340.00,
    "HKD": 7.82,
}


def to_usd(amount: float, currency: str) -> float:
    """Convert an amount from the given currency into USD."""
    return amount / RATE_PER_USD.get(currency, 1.0)


def from_usd(amount_usd: float, currency: str) -> float:
    """Convert an amount from USD into the given currency."""
    return amount_usd * RATE_PER_USD.get(currency, 1.0)


def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert an amount between any two supported currencies."""
    return from_usd(to_usd(amount, from_currency), to_currency)


def high_value_threshold(currency: str) -> float:
    """USD 'high value' rule threshold expressed in the given currency."""
    return HIGH_VALUE_THRESHOLD_USD * RATE_PER_USD.get(currency, 1.0)


def format_amount(amount: float, currency: str) -> str:
    """Format an amount with the currency symbol (2 or 0 decimals for JPY/KRW)."""
    symbol = SUPPORTED_CURRENCIES.get(currency, {}).get("symbol", "")
    if currency in ("JPY", "KRW", "NGN"):
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


def currency_options() -> list:
    """Return selectbox labels like 'USD — US Dollar'."""
    return [f"{code} — {SUPPORTED_CURRENCIES[code]['name']}" for code in SUPPORTED_CURRENCIES]


def code_from_option(option: str) -> str:
    """Extract the currency code from a selectbox option label."""
    return option.split(" ")[0]