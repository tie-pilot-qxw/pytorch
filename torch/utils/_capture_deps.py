"""What an operator's captured CUDA graph depends on, declared by the operator.

A system that captures a call once and replays it has to know when that
capture stops being valid. The shapes it was called with and the addresses it
was handed are visible to the consumer and it can track them itself. What it
cannot see is that a call also baked in something the source only names: a
functional collective bakes the communicator its group name stood for, so
rebuilding that group at another size leaves a replay quietly doing the old
thing at the old width.

So the operator declares it, next to where the operator lives, and the
consumer holds no per-operator knowledge. `register` takes the operator's
qualified name, the arguments whose values identify what was baked in, and a
resolver turning those values into an identity. A consumer folds the result
into whatever key it caches the capture under, and re-captures when it moves.

Modelled on `torch._library.simple_registry`: registration is per operator and
carries a payload, the consumer looks the operator up and does nothing when
nothing is registered. An operator that declares nothing is treated however
the consumer treats an opaque call, which is a deliberate risk rather than a
guarantee of safety.
"""

from collections.abc import Callable
from typing import Any


__all__ = ["register", "lookup"]

# op qualname ("_c10d_functional::all_reduce") -> (argument names, resolver)
_DEPS: dict[str, tuple[tuple[str, ...], Callable[..., Any]]] = {}


def register(
    qualname: str, arg_names: tuple[str, ...], resolve: Callable[..., Any]
) -> None:
    """Declare that this operator's capture is only valid while `resolve` of
    those arguments stays the same.

    `resolve` is called with the arguments' values, in the order named. It is
    evaluated once per call of the region holding the operator, so it must be
    cheap, and it must never raise.
    """
    _DEPS[qualname] = (tuple(arg_names), resolve)


def lookup(qualname: str) -> tuple[tuple[str, ...], Callable[..., Any]] | None:
    """What this operator declared, or None."""
    return _DEPS.get(qualname)
