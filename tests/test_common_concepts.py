from quant_elements_lib.common_concepts import (
    SecurityType,
    ExchangeMode,
    MarketParticipantSide,
    SellSideParticipantRole,
    BuySideParticipantRole,
    MarketParticipant,
)
import pytest


def test_enum_members():
    assert SecurityType.EQUITY.name == "EQUITY"
    assert ExchangeMode.OTC.name == "OTC"
    assert MarketParticipantSide.SELL_SIDE.name == "SELL_SIDE"
    assert SellSideParticipantRole.BROKER.name == "BROKER"
    assert BuySideParticipantRole.RETAIL_INVESTOR.name == "RETAIL_INVESTOR"


def test_market_participant_creation():
    mp = MarketParticipant(
        MarketParticipantSide.BUY_SIDE,
        BuySideParticipantRole.RETAIL_INVESTOR,
        name="Alice",
    )
    assert mp.side is MarketParticipantSide.BUY_SIDE
    assert mp.role is BuySideParticipantRole.RETAIL_INVESTOR
    assert mp.name == "Alice"


def test_market_participant_invalid_role():
    with pytest.raises(TypeError):
        MarketParticipant(MarketParticipantSide.BUY_SIDE, "Trader")
