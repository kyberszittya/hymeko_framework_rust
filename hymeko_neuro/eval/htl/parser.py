"""Recursive-descent parser for HTL formulas.

Grammar (whitespace-insensitive)::

    formula   := or_expr
    or_expr   := and_expr ("OR"  and_expr)*
    and_expr  := unary    ("AND" unary)*
    unary     := "NOT" unary
               | "G" interval? "(" formula ")"
               | "F" interval? "(" formula ")"
               | atom
    atom      := "(" formula ")"
               | predicate
    predicate := IDENT ("[" IDENT "]")? CMP NUMBER
    interval  := "[" NUMBER "," NUMBER "]"
    CMP       := "<" | "<=" | ">" | ">=" | "=="
    IDENT     := [A-Za-z_][A-Za-z_0-9]*
    NUMBER    := [-+]?[0-9]+(\\.[0-9]+)?

Default interval for ``G`` and ``F`` is ``[0, +inf)`` (open right).

Public API: :func:`parse(formula: str) -> HtlNode`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .ast import And, CmpOp, Eventually, Globally, HtlNode, Not, Or, ScalarPred


class ParseError(ValueError):
    """Raised when a formula string is malformed.

    The message includes the offending column when possible.
    """


# ---------- tokenizer ----------

_KEYWORDS = {"G", "F", "NOT", "AND", "OR", "TRUE", "FALSE"}
_CMP_OPS = ("<=", ">=", "==", "<", ">")


@dataclass(frozen=True)
class Token:
    kind: str  # 'IDENT', 'NUMBER', 'CMP', 'LPAREN', 'RPAREN', 'LBRACK', 'RBRACK',
    #          'COMMA', 'KW'
    value: str
    col: int


def _tokenize(src: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            tokens.append(Token("LPAREN", c, i))
            i += 1
            continue
        if c == ")":
            tokens.append(Token("RPAREN", c, i))
            i += 1
            continue
        if c == "[":
            tokens.append(Token("LBRACK", c, i))
            i += 1
            continue
        if c == "]":
            tokens.append(Token("RBRACK", c, i))
            i += 1
            continue
        if c == ",":
            tokens.append(Token("COMMA", c, i))
            i += 1
            continue

        matched_cmp = next((op for op in _CMP_OPS if src.startswith(op, i)), None)
        if matched_cmp is not None:
            tokens.append(Token("CMP", matched_cmp, i))
            i += len(matched_cmp)
            continue

        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            kind = "KW" if word in _KEYWORDS else "IDENT"
            tokens.append(Token(kind, word, i))
            i = j
            continue

        if c.isdigit() or (
            c in "+-" and i + 1 < n and (src[i + 1].isdigit() or src[i + 1] == ".")
        ):
            j = i
            if src[j] in "+-":
                j += 1
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            # optional exponent
            if j < n and src[j] in "eE":
                j += 1
                if j < n and src[j] in "+-":
                    j += 1
                while j < n and src[j].isdigit():
                    j += 1
            tokens.append(Token("NUMBER", src[i:j], i))
            i = j
            continue

        raise ParseError(f"unexpected character {c!r} at column {i}")
    return tokens


# ---------- parser ----------


class _Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return None
        return self.tokens[idx]

    def _consume(self, kind: str, value: Optional[str] = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise ParseError(f"unexpected end of input; expected {kind}")
        if tok.kind != kind or (value is not None and tok.value != value):
            wanted = f"{kind}({value!r})" if value else kind
            raise ParseError(
                f"expected {wanted} at column {tok.col}, got {tok.kind}({tok.value!r})"
            )
        self.pos += 1
        return tok

    def _accept(self, kind: str, value: Optional[str] = None) -> Optional[Token]:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        self.pos += 1
        return tok

    # formula := or_expr
    def parse(self) -> HtlNode:
        node = self._or_expr()
        if self.pos != len(self.tokens):
            extra = self.tokens[self.pos]
            raise ParseError(
                f"trailing input at column {extra.col}: "
                f"{extra.kind}({extra.value!r})"
            )
        return node

    def _or_expr(self) -> HtlNode:
        node = self._and_expr()
        while self._accept("KW", "OR") is not None:
            right = self._and_expr()
            node = Or(node, right)
        return node

    def _and_expr(self) -> HtlNode:
        node = self._unary()
        while self._accept("KW", "AND") is not None:
            right = self._unary()
            node = And(node, right)
        return node

    def _unary(self) -> HtlNode:
        tok = self._peek()
        if tok is None:
            raise ParseError("unexpected end of input")
        if tok.kind == "KW" and tok.value == "NOT":
            self.pos += 1
            return Not(self._unary())
        if tok.kind == "KW" and tok.value in ("G", "F"):
            self.pos += 1
            t1, t2 = self._optional_interval()
            self._consume("LPAREN")
            inner = self._or_expr()
            self._consume("RPAREN")
            return Globally(t1, t2, inner) if tok.value == "G" else Eventually(t1, t2, inner)
        return self._atom()

    def _atom(self) -> HtlNode:
        tok = self._peek()
        if tok is None:
            raise ParseError("unexpected end of input")
        if tok.kind == "LPAREN":
            self.pos += 1
            node = self._or_expr()
            self._consume("RPAREN")
            return node
        return self._predicate()

    def _optional_interval(self) -> Tuple[float, float]:
        if self._accept("LBRACK") is None:
            return 0.0, math.inf
        lo = float(self._consume("NUMBER").value)
        self._consume("COMMA")
        hi_tok = self._peek()
        if hi_tok is not None and hi_tok.kind == "IDENT" and hi_tok.value == "inf":
            self.pos += 1
            hi = math.inf
        else:
            hi = float(self._consume("NUMBER").value)
        self._consume("RBRACK")
        if hi < lo:
            raise ParseError(f"interval [{lo}, {hi}] has hi < lo")
        return lo, hi

    def _predicate(self) -> ScalarPred:
        name_tok = self._consume("IDENT")
        name = name_tok.value
        if self._accept("LBRACK") is not None:
            sub = self._consume("IDENT")
            self._consume("RBRACK")
            name = f"{name}[{sub.value}]"
        op_tok = self._consume("CMP")
        value_tok = self._consume("NUMBER")
        return ScalarPred(name=name, op=CmpOp.parse(op_tok.value), value=float(value_tok.value))


def parse(formula: str) -> HtlNode:
    """Parse ``formula`` into an HTL AST.

    Preconditions
    -------------
    - ``formula`` is a non-empty string.

    Raises
    ------
    ParseError
        On any tokenisation or grammar error.
    """

    if not isinstance(formula, str):
        raise ParseError(f"formula must be str, got {type(formula).__name__}")
    if not formula.strip():
        raise ParseError("empty formula")
    tokens = _tokenize(formula)
    if not tokens:
        raise ParseError("no tokens after stripping whitespace")
    return _Parser(tokens).parse()


__all__ = ["parse", "ParseError"]
