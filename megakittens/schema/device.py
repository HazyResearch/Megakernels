from typing import Literal

import torch
from pydantic import BaseModel, Field


class Device(BaseModel):
    type: Literal["cpu", "cuda"]
    index: int | None = Field(ge=0, le=7, default=None)

    def __str__(self) -> str:
        return f"{self.type}:{self.index}" if self.index is not None else self.type

    @classmethod
    def from_torch(cls, device: torch.device) -> "Device":
        if device.type == "cpu":
            return cls(type="cpu")
        elif device.type == "cuda":
            return cls(type=device.type, index=device.index)
        else:
            raise ValueError(f"[MegaKittens] Unsupported device type: {device.type}")

    model_config = {"frozen": True}
