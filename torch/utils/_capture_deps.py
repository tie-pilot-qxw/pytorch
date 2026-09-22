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
from typing import Any, NamedTuple


__all__ = [
    "Declaration",
    "Known",
    "Unavailable",
    "Unsupported",
    "register",
    "lookup",
]


class Known(NamedTuple):
    """The identity this call's capture would be tied to."""

    value: Any


class Unavailable(NamedTuple):
    """The operator cannot say right now, and might be able to later.

    The context it reads has not been installed yet, or this is a warm-up or a
    profiling run that is not a real invocation. A consumer must not capture
    under an unknown identity, and must not cache this answer: the next call
    may well be answerable.
    """

    reason: str


class Unsupported(NamedTuple):
    """This operator will not be able to say, on this path, ever.

    A backend that cannot name what it bakes in. A consumer should stop asking
    and take whatever route it has for an operator that declares nothing.
    """

    reason: str


# A resolver may return one of the three above, or a bare value, which means
# `Known(value)`. Returning None as a bare value is how the first version
# reported "I could not tell", which made four different states -- a real
# no-op, a context not yet installed, an unsupported backend, and a bug in the
# resolver -- indistinguishable, and cacheable: a capture taken during a
# profiling run became the answer for a real invocation whose lookup also
# failed. Saying which one it is costs the producer one word.
def _as_result(v: Any) -> Known | Unavailable | Unsupported:
    if isinstance(v, (Known, Unavailable, Unsupported)):
        return v
    return Known(v)


class Declaration(NamedTuple):
    """What one operator declared. `branches` is None when it named no space."""

    arg_names: tuple[str, ...]
    resolve: Callable[..., Any]
    branches: tuple[Any, ...] | None


# op qualname ("_c10d_functional::all_reduce") -> what it declared
_DEPS: dict[str, Declaration] = {}


def register(
    qualname: str,
    arg_names: tuple[str, ...],
    resolve: Callable[..., Any],
    branches: tuple[Any, ...] | None = None,
) -> None:
    """Declare that this operator's capture is only valid while `resolve` of
    those arguments stays the same.

    `resolve` is called with the arguments' values, in the order named. It is
    evaluated once per call of the region holding the operator, so it must be
    cheap, and it must never raise.

    `resolve` returns `Known(value)`, `Unavailable(reason)` or
    `Unsupported(reason)`; a bare value means `Known(value)`.

    `branches` is every value `resolve` can return, in a fixed order, for an
    operator whose identity moves between a known set rather than an open one
    -- a library that picks among kernels it could name. Equality alone tells
    a consumer that the capture went stale; the set tells it what the capture
    could have been instead, which is what lets it prepare all of them once
    rather than discover them one re-capture at a time, size a per-operator
    budget from the operator rather than from a constant, and refuse a space
    too large to prepare while it is still building rather than mid-run.

    Declaring a set is a promise about the whole set, not about the values
    seen so far: a consumer that meets a value outside it has been told
    something false, and should say so rather than pick a neighbour.
    """
    _DEPS[qualname] = Declaration(
        tuple(arg_names), resolve, None if branches is None else tuple(branches)
    )


def lookup(qualname: str) -> Declaration | None:
    """What this operator declared, or None."""
    return _DEPS.get(qualname)
