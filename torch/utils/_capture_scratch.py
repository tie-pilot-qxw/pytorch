"""The per-call buffers an operator reaches for without taking them as arguments.

An operator that a capture will replay has to be able to say where every device
pointer its kernels hold comes from, because a captured launch keeps the address
and not the storage. The addresses that arrive as arguments are already known to
whoever made the call. The ones that do not are the problem: an operator can
reach a buffer through a thread-local context, a global registry or a closure,
and nothing at the call site names it. vLLM's attention does exactly that -- its
FlashAttention scheduler metadata is built per step and fetched inside the
operator from a registry keyed by a layer name, so it appears nowhere in the
traced graph.

The declaration is a *push*, made from inside the implementation that actually
reads the buffers -- the backend, not the dispatching operator. A dispatcher can
only enumerate what its arguments and metadata object expose, which is not the
same set: an implementation reaches buffers through a wrapper object stored on
the metadata, through a module-level global, and through attributes of the layer
it was handed. Those are invisible one level up, so a dispatcher that declares
on a backend's behalf under-declares silently. The backend, by contrast, is
holding exactly the buffers it is about to read. An earlier version had the
consumer pull: the producer registered an accessor and the consumer called it
after capturing. That is the wrong shape. A consumer capturing a call does not
necessarily run inside the producer's own context -- DynaGraph re-executes a
generated wrapper to harvest, and one harvest in three found vLLM's forward
context already gone, which silently left a third of the pointers unaccounted
for. Recording at the point of use costs nothing, since the buffers are already
in hand there, and cannot come up empty.

What a record belongs to is a *scope*: the consumer says which call it is about
to capture, and every record made while that scope is open is part of that call.
The first version keyed a single pending value per operator instead, which
cannot distinguish one call from the next. Two calls of the same operator
overwrote each other, so a batch of them left only the last; two declarations
within one call overwrote each other rather than adding up; and a call that
declared nothing inherited whatever an earlier call had left behind, which reads
as fully accounted for and is not. Taking with `pop` stops a record being
consumed twice. It does not stop it being consumed by the wrong call.

Companion to `torch.utils._capture_deps`, which declares what a capture's
identity depends on, and `torch.utils._capture_tma`, which declares how to
rebuild a descriptor.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


__all__ = ["Scope", "record", "scope", "unscoped"]


class Scope:
    """One call a consumer is capturing, and what was declared inside it.

    `buffers` maps the name its producer chose to the tensor; `calls` is how
    many records landed here, which is more than one when the captured call ran
    the operator more than once -- those buffers all belong to the same capture,
    so they add up rather than replace.
    """

    __slots__ = ("key", "buffers", "calls")

    def __init__(self, key: str) -> None:
        self.key = key
        self.buffers: dict[str, Any] = {}
        self.calls = 0


# operator key -> the scopes open for it, innermost last. Empty between
# captures: a record made with nothing open belongs to a call nobody is
# capturing (a warm-up, or ordinary execution) and is dropped rather than kept
# for whoever asks next.
_OPEN: dict[str, list[Scope]] = {}
_UNSCOPED: dict[str, int] = {}


@contextmanager
def scope(key: str) -> Iterator[Scope]:
    """Attribute the records made in this block to one captured call.

    `key` is the operator as the consumer names it, e.g.
    `ops:vllm.unified_attention_with_output.default`. The block should hold the
    capture and nothing else, so a warm-up run of the same call does not land
    in it.
    """
    sc = Scope(key)
    _OPEN.setdefault(key, []).append(sc)
    try:
        yield sc
    finally:
        stack = _OPEN.get(key)
        if stack:
            # By position, not by pop(): an inner scope that leaked would
            # otherwise take this one's place.
            for at in range(len(stack) - 1, -1, -1):
                if stack[at] is sc:
                    del stack[at]
                    break
            if not stack:
                del _OPEN[key]


def record(key: str, buffers: dict[str, Any]) -> None:
    """Declare the buffers this call is about to read off its argument list.

    Called from inside the implementation that reads them, where the buffers are
    in hand. `buffers` maps a name the producer picks to the tensor. A buffer
    that is declared but not read is harmless; one that is read but not declared
    is what makes a capture of this call unaccountable.

    Names that collide within one scope are kept apart rather than replaced: two
    calls of the same operator inside one capture each declared a real buffer,
    and dropping either would leave a pointer with no source.
    """
    stack = _OPEN.get(key)
    if not stack:
        _UNSCOPED[key] = _UNSCOPED.get(key, 0) + 1
        return
    sc = stack[-1]
    sc.calls += 1
    for name, t in buffers.items():
        at, n = name, 1
        while at in sc.buffers and sc.buffers[at] is not t:
            n += 1
            at = f"{name}#{n}"
        sc.buffers[at] = t


def unscoped(key: str) -> int:
    """How many records for this operator landed outside any scope.

    All of them, for an operator nobody captures. A non-zero count for one that
    is captured means records are being made where the consumer is not looking,
    which is worth knowing before trusting that a capture is accounted for.
    """
    return _UNSCOPED.get(key, 0)
