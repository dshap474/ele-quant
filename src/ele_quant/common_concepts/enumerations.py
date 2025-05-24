from enum import Enum

__all__ = [
    "SecurityType",
    "ExchangeMode",
    "MarketParticipantSide",
    "SellSideParticipantRole",
    "BuySideParticipantRole",
]


class SecurityType(Enum):
    """Enumeration of security types."""

    EQUITY = "EQUITY"
    ETF = "ETF"
    FUTURE = "FUTURE"
    BOND = "BOND"
    OPTION = "OPTION"
    INTEREST_RATE_SWAP = "INTEREST_RATE_SWAP"
    CREDIT_DEFAULT_SWAP = "CREDIT_DEFAULT_SWAP"


class ExchangeMode(Enum):
    """Enumeration of trading venues."""

    EXCHANGE = "EXCHANGE"
    OTC = "OTC"
    DARK_POOL = "DARK_POOL"


class MarketParticipantSide(Enum):
    """Market participant sides."""

    SELL_SIDE = "SELL_SIDE"
    BUY_SIDE = "BUY_SIDE"


class SellSideParticipantRole(Enum):
    """Roles for sell-side participants."""

    DEALER = "DEALER"
    BROKER = "BROKER"
    BROKER_DEALER = "BROKER_DEALER"


class BuySideParticipantRole(Enum):
    """Roles for buy-side participants."""

    PASSIVE_INSTITUTIONAL = "PASSIVE_INSTITUTIONAL"
    HEDGER = "HEDGER"
    INSTITUTIONAL_ACTIVE_MANAGER = "INSTITUTIONAL_ACTIVE_MANAGER"
    ASSET_ALLOCATOR = "ASSET_ALLOCATOR"
    INFORMED_TRADER = "INFORMED_TRADER"
    RETAIL_INVESTOR = "RETAIL_INVESTOR"
