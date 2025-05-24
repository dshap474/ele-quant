from __future__ import annotations

from typing import Optional, Union

from .enumerations import (
    BuySideParticipantRole,
    MarketParticipantSide,
    SellSideParticipantRole,
)

__all__ = ["MarketParticipant"]


class MarketParticipant:
    """Simple representation of a market participant."""

    def __init__(
        self,
        side: MarketParticipantSide,
        role: Union[SellSideParticipantRole, BuySideParticipantRole],
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(side, MarketParticipantSide):
            raise TypeError("side must be MarketParticipantSide")
        if not isinstance(role, (SellSideParticipantRole, BuySideParticipantRole)):
            raise TypeError("role must be SellSideParticipantRole or BuySideParticipantRole")
        self.side = side
        self.role = role
        self.name = name

    def describe_role(self) -> str:
        """Return a textual description of the participant's role."""

        return self.role.name

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        name_part = f"{self.name}: " if self.name else ""
        return f"MarketParticipant({name_part}{self.side.name}, {self.role.name})"
