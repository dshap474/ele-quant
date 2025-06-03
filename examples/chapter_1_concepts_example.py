# examples/chapter_1_concepts_example.py

"""
This script demonstrates the usage of common concepts from Chapter 1, Section C
of the Quant Elements Library.
"""

# Import necessary enums and classes
from ele_quant.common_concepts.enumerations import (
    SecurityType,
    ExchangeMode,
    MarketParticipantSide,
    SellSideParticipantRole,
    BuySideParticipantRole,
)
from ele_quant.common_concepts.participants import MarketParticipant

def demonstrate_enums():
    """Demonstrates the usage of various enumerations."""
    print("--- Demonstrating Enumerations ---")

    # SecurityType examples
    print(f"\nSecurityType examples:")
    print(f"  SecurityType.EQUITY: {SecurityType.EQUITY}")
    print(f"  SecurityType.BOND: {SecurityType.BOND}") # Changed FIXED_INCOME to BOND
    print(f"  SecurityType.FUTURE: {SecurityType.FUTURE}") # Changed DERIVATIVE to FUTURE
    print(f"  SecurityType.OPTION: {SecurityType.OPTION}")   # Added OPTION as another derivative example

    # ExchangeMode examples
    print(f"\nExchangeMode examples:")
    print(f"  ExchangeMode.EXCHANGE: {ExchangeMode.EXCHANGE}")
    print(f"  ExchangeMode.OTC: {ExchangeMode.OTC}")

    # MarketParticipantSide examples
    print(f"\nMarketParticipantSide examples:")
    print(f"  MarketParticipantSide.BUY_SIDE: {MarketParticipantSide.BUY_SIDE}")
    print(f"  MarketParticipantSide.SELL_SIDE: {MarketParticipantSide.SELL_SIDE}")

    # SellSideParticipantRole examples
    print(f"\nSellSideParticipantRole examples:")
    print(f"  SellSideParticipantRole.DEALER: {SellSideParticipantRole.DEALER}")
    print(f"  SellSideParticipantRole.BROKER_DEALER: {SellSideParticipantRole.BROKER_DEALER}") # Changed MARKET_MAKER to BROKER_DEALER
    print(f"  SellSideParticipantRole.BROKER: {SellSideParticipantRole.BROKER}")

    # BuySideParticipantRole examples
    print(f"\nBuySideParticipantRole examples:")
    print(f"  BuySideParticipantRole.INSTITUTIONAL_ACTIVE_MANAGER: {BuySideParticipantRole.INSTITUTIONAL_ACTIVE_MANAGER}")
    print(f"  BuySideParticipantRole.INFORMED_TRADER: {BuySideParticipantRole.INFORMED_TRADER}") # Changed HEDGE_FUND to INFORMED_TRADER
    print(f"  BuySideParticipantRole.RETAIL_INVESTOR: {BuySideParticipantRole.RETAIL_INVESTOR}")

def demonstrate_market_participants():
    """Demonstrates the instantiation and usage of MarketParticipant."""
    print("\n--- Demonstrating MarketParticipant ---")

    # Create a buy-side institutional active manager
    buy_side_manager = MarketParticipant(
        name="Global Alpha Managers",
        side=MarketParticipantSide.BUY_SIDE,
        role=BuySideParticipantRole.INSTITUTIONAL_ACTIVE_MANAGER
    )
    print(f"\nBuy-side participant: {buy_side_manager}")

    # Create a sell-side dealer
    sell_side_dealer = MarketParticipant(
        name="Fixed Income Dealers Inc.",
        side=MarketParticipantSide.SELL_SIDE,
        role=SellSideParticipantRole.DEALER
    )
    print(f"Sell-side participant: {sell_side_dealer}")

    # Create another buy-side participant - an informed trader
    informed_trader_participant = MarketParticipant(
        name="Alpha Seekers Trading", # Changed name
        side=MarketParticipantSide.BUY_SIDE,
        role=BuySideParticipantRole.INFORMED_TRADER # Changed HEDGE_FUND to INFORMED_TRADER
    )
    print(f"Another buy-side participant: {informed_trader_participant}") # Changed variable name

    # Create another sell-side participant - a broker-dealer
    broker_dealer_participant = MarketParticipant(
        name="Global Broker-Dealers LLC", # Changed name
        side=MarketParticipantSide.SELL_SIDE,
        role=SellSideParticipantRole.BROKER_DEALER # Changed MARKET_MAKER to BROKER_DEALER
    )
    print(f"Another sell-side participant: {broker_dealer_participant}") # Changed variable name


if __name__ == "__main__":
    print("Running Chapter 1 Concepts Example Script...")
    demonstrate_enums()
    demonstrate_market_participants()
    print("\nScript finished.")
