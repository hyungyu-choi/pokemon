"""Helpers that reproduce the JavaScript semantics the Showdown engine relies on.

Pokemon Showdown is written in TypeScript; a faithful port needs a few JS
behaviours that Python does not have natively:

* ``undefined`` vs ``null``: Python ``None`` plays the role of ``undefined``
  (and of ``null`` wherever the two are not distinguished). :data:`NULL` is a
  falsy sentinel used where Showdown explicitly returns/compares ``null``
  (e.g. "silent failure" results of event handlers).
* Objects are always truthy in JS - :class:`Obj` is a ``dict`` with attribute
  access whose instances are always truthy and whose missing keys read as
  ``None`` (``undefined``).
* Integer helpers with JS semantics: ``>>> 0`` truncation, ``Math.round``,
  32-bit shifts, ...
"""
from __future__ import annotations

import copy

import math
from typing import Any


class _Null:
    """JavaScript ``null`` (distinct from ``undefined``/``None``)."""

    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self):
        return False

    def __repr__(self):
        return 'NULL'

    def __str__(self):
        return 'null'

    def __reduce__(self):
        return (_Null, ())

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


NULL = _Null()


def is_nullish(value: Any) -> bool:
    """``value == null`` in JS (null or undefined)."""
    return value is None or value is NULL


class Obj(dict):
    """A JS-like object: attribute access, missing attributes are ``None``.

    Always truthy (like every JS object, even ``{}``).
    """

    __slots__ = ()

    def __getattr__(self, name):
        if name[:2] == '__':
            raise AttributeError(name)
        return dict.get(self, name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        self.pop(name, None)

    def __missing__(self, key):
        return None

    def __bool__(self):
        return True

    def __copy__(self):
        return type(self)(self)

    def __deepcopy__(self, memo):
        # full, identity-preserving copy (used to clone whole battles); deep_clone() is the
        # JSON-style clone Showdown's code uses
        new = type(self)()
        memo[id(self)] = new
        for k, v in self.items():
            dict.__setitem__(new, k, copy.deepcopy(v, memo))
        return new

    def __hash__(self):  # identity semantics, like JS objects
        return id(self)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __repr__(self):
        return f"Obj({dict.__repr__(self)})"


def obj_hook(d: dict) -> Obj:
    return Obj(d)


def deep_clone(value: Any) -> Any:
    """Deep clone of JSON-like data (dict/list/primitives); other objects kept."""
    if isinstance(value, Obj):
        return type(value)({k: deep_clone(v) for k, v in value.items()})
    if isinstance(value, list):
        return [deep_clone(v) for v in value]
    if type(value) is dict:
        return {k: deep_clone(v) for k, v in value.items()}
    return value


def to_id(text: Any) -> str:
    """Showdown ``toID``: lowercase alphanumeric."""
    if not isinstance(text, str):
        if text is None or text is NULL or text is False:
            return ''
        if isinstance(text, bool):
            text = 'true'
        elif isinstance(text, (int, float)):
            text = js_str(text)
        else:
            text = getattr(text, 'id', None) or getattr(text, 'userid', None) or text
            if isinstance(text, (int, float)) and not isinstance(text, bool):
                text = js_str(text)
            elif not isinstance(text, str):
                return ''
    out = []
    for ch in text.lower():
        if ('a' <= ch <= 'z') or ('0' <= ch <= '9'):
            out.append(ch)
    return ''.join(out)


def trunc(num: float, bits: int = 0) -> int:
    """``num >>> 0`` (ToUint32), optionally ``% 2**bits``."""
    if isinstance(num, float):
        if num != num or num in (math.inf, -math.inf):
            n = 0
        else:
            n = int(num)  # truncates toward zero
    elif num is True:
        n = 1
    elif num is False or num is None or num is NULL:
        n = 0
    else:
        n = int(num)
    n &= 0xFFFFFFFF
    if bits:
        return n % (2 ** bits)
    return n


def to_int32(num: float) -> int:
    n = trunc(num)
    return n - 0x100000000 if n & 0x80000000 else n


def js_round(x: float) -> int:
    """``Math.round`` (round half up)."""
    return math.floor(x + 0.5)


def floor(x: float) -> int:
    return math.floor(x)


def ceil(x: float) -> int:
    return math.ceil(x)


def clamp_int_range(num: Any, min_: float | None = None, max_: float | None = None) -> int:
    """``Utils.clampIntRange``: floors the number then clamps it."""
    if not isinstance(num, (int, float)) or isinstance(num, bool):
        num = 0
    elif isinstance(num, float):
        num = 0 if num != num else math.floor(num)
    if min_ is not None and num < min_:
        num = min_
    if max_ is not None and num > max_:
        num = max_
    return num


def js_str(value: Any) -> str:
    """``String(value)`` / template-literal conversion."""
    if value is None:
        return 'undefined'
    if value is NULL:
        return 'null'
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if isinstance(value, float):
        if value != value:
            return 'NaN'
        if value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, list):
        return ','.join('' if is_nullish(v) else js_str(v) for v in value)
    return str(value)


def join_part(value: Any) -> str:
    """Conversion used by ``Array#join`` (null/undefined become '')."""
    if value is None or value is NULL:
        return ''
    return js_str(value)


def truthy(value: Any) -> bool:
    """JS truthiness (lists/dicts are truthy even when empty)."""
    if isinstance(value, (list, dict)):
        return True
    return bool(value)


def num(value: Any) -> float:
    """Loose numeric conversion used for arithmetic on possibly-undefined values."""
    if value is None or value is NULL or value is False:
        return 0
    if value is True:
        return 1
    return value


def js_typeof(value: Any) -> str:
    if value is None:
        return 'undefined'
    if value is NULL:
        return 'object'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, (int, float)):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if callable(value):
        return 'function'
    return 'object'


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
