"""Domain models for transformation specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TransformationSpec:
    """Represents a single transformation and parameter payload."""

    name: str
    params: Dict[str, object]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Transformation name must be provided")


@dataclass(frozen=True)
class TransformationChain:
    """Ordered collection of transformation specifications."""

    steps: Tuple[TransformationSpec, ...]

    def __post_init__(self) -> None:
        seen = set()
        for spec in self.steps:
            key = (spec.name, tuple(sorted(spec.params.items())))
            if key in seen:
                raise ValueError("Transformation chain contains duplicate identical steps")
            seen.add(key)

    @classmethod
    def empty(cls) -> "TransformationChain":
        return cls(steps=tuple())

    def append(self, spec: TransformationSpec) -> "TransformationChain":
        return TransformationChain(self.steps + (spec,))
