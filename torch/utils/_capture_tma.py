"""How a producer builds the TMA descriptors its capture bakes in.

A TMA descriptor is 128 opaque bytes encoding a base address and a global
shape. Captured into a CUDA graph it freezes both, so a system replaying that
graph at another shape has to rebuild it, and nothing about the bytes says how.
The shapes and addresses are visible to the consumer; the recipe is not.

So the producer declares the recipe, next to where the producer lives, and the
consumer holds no per-producer knowledge. `register` takes a producer key -- an
operator's qualified name for an opaque call, or a generated kernel's name for a
triton launch -- and one entry per descriptor saying which parameter it
occupies, which argument holds the tensor it was built from, and the block
shape. A consumer that can re-parameterize that argument rebuilds the descriptor
per shape; one that cannot treats the call as opaque, which is what it would
have done anyway.

Companion to `torch.utils._capture_deps`, which declares what a capture's
identity depends on rather than how to rebuild part of it, and modelled on the
same registry shape: per producer, a payload, and nothing at all when the
producer declared nothing.
"""

from collections.abc import Sequence
from dataclasses import dataclass


__all__ = ["TmaArg", "register", "lookup"]


@dataclass(frozen=True)
class TmaArg:
    """One TMA descriptor a producer builds.

    `param` is the parameter the descriptor is passed in. `source` is the
    argument holding the tensor it is built from -- that tensor's address and
    shape are what the descriptor freezes. `block_shape` is the tile the
    descriptor was built for, which does not move with the shape space.
    """

    param: str
    source: str
    block_shape: tuple[int, ...]


# producer key -> the descriptors it builds, in parameter order
_TMA: dict[str, tuple[TmaArg, ...]] = {}


def register(key: str, descriptors: Sequence[TmaArg]) -> None:
    """Declare the TMA descriptors this producer builds.

    Called where the producer is defined: an operator registers under its
    qualified name, a generated wrapper registers under the kernel's name as it
    appears in the wrapper's own namespace.
    """
    _TMA[key] = tuple(descriptors)


def lookup(key: str) -> tuple[TmaArg, ...] | None:
    """What this producer declared, or None."""
    return _TMA.get(key)
