"""Translate the PRIMs script language into Python source.

The tokenizer and the recursive-descent parser are ports of Script.swift, so
a script means the same thing here as there.  Two features of that grammar
need care in Python:

* binary operators are RIGHT-associative: ``a - b - c`` is ``a - (b - c)``
  and ``a / b * c`` is ``a / (b * c)``; the generated code keeps that meaning
  with explicit parentheses;
* integer division truncates (``div``), a string on the left of ``+``
  concatenates (``plus``), and assigning past the end of an array pads it
  with zeros (``setitem``).

The generated function takes one argument ``m`` (a ScriptAPI) and starts with
``arg = m.arg`` (the label passed by a batch script).
"""
from __future__ import annotations

import keyword

from .dsl import ParseError

HELPERS = {"div", "plus", "setitem", "and_", "or_", "truth", "cmp", "m"}


# ===================================================================== tokenizer
def tokenize(src: str) -> list[str]:
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in " \n\t\r":
            i += 1
            continue
        if c in "()+*-/%\\{}[],":
            toks.append(c)
            i += 1
        elif c in "!<>=|&":
            j = i
            while j < n and src[j] in "!<>=|&":
                j += 1
            toks.append(src[i:j])
            i = j
        elif c.isdigit():
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            toks.append(src[i:j])
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 1
            toks.append(src[i:j + 1])
            i = j + 1
        elif c.isascii() and c.isalpha():
            j = i
            while j < n and (src[j].isascii() and (src[j].isalnum() or src[j] in "-+_")):
                j += 1
            toks.append(src[i:j])
            i = j
        else:
            raise ParseError(f"Unknown character {c!r} in script")
    return toks


# ===================================================================== AST
# Nodes are tuples:
#   ("num", text) ("str", text) ("sym", name) ("arr", [expr]) ("idx", name, expr)
#   ("call", name, [expr]) ("paren", expr) ("cmp", op, lhs, rhs|None)
#   ("expr", preop, term, op, expr|None) ("term", factor, op, term|None)
# Statements: ("assign", lhs, expr) ("call", ...) ("if", test, then, else)
#   ("while", test, body) ("for", var, start, end|None, body)

class _Parser:
    def __init__(self, toks):
        self.t = toks + ["EOF"]

    def la(self, i):
        return self.t[i + 1] if i + 1 < len(self.t) else "#"

    def nxt(self, i):
        if i + 1 >= len(self.t):
            raise ParseError("Unexpected end of script")
        return i + 1

    def statements(self):
        out, i = [], 0
        while i < len(self.t) and self.t[i] != "EOF":
            st, i = self.statement(i)
            out.append(st)
        return out

    def block(self, i):
        if self.t[i] != "{":
            raise ParseError(f"Expected '{{' near {' '.join(self.t[max(0, i - 5):i + 1])}")
        i = self.nxt(i)
        body = []
        while self.t[i] != "}":
            st, i = self.statement(i)
            body.append(st)
        return body, self.nxt(i)

    def statement(self, i):
        tok = self.t[i]
        if tok == "if":
            test, i = self.expression(self.nxt(i))
            then, i = self.block(i)
            els = []
            if self.t[i] == "else":
                els, i = self.block(self.nxt(i))
            return ("if", test, then, els), i
        if tok == "while":
            test, i = self.expression(self.nxt(i))
            body, i = self.block(i)
            return ("while", test, body), i
        if tok == "for":
            i = self.nxt(i)
            var = self.t[i]
            i = self.nxt(i)
            if self.t[i] != "in":
                raise ParseError("Expected 'in' in for loop")
            start, i = self.expression(self.nxt(i))
            end = None
            if self.t[i] == "to":
                end, i = self.expression(self.nxt(i))
            body, i = self.block(i)
            return ("for", var, start, end, body), i
        if self.la(i) == "(":
            call, i = self.funcall(i)
            return call, i
        lhs, i = self.factor(i)
        if lhs[0] not in ("sym", "idx"):
            raise ParseError("Invalid left-hand side of assignment")
        if self.t[i] != "=":
            raise ParseError(f"Expected '=' near {' '.join(self.t[max(0, i - 5):i + 1])}")
        rhs, i = self.expression(self.nxt(i))
        return ("assign", lhs, rhs), i

    def comparison(self, i):
        op = ""
        if self.t[i] == "!":
            op = "!"
            i = self.nxt(i)
        lhs, i = self.expression(i)
        rhs = None
        if self.t[i] in ("==", "!=", "<=", ">=", "=<", "=>", ">", "<", "<>"):
            op = self.t[i]
            rhs, i = self.expression(self.nxt(i))
        if self.t[i] != ")":
            raise ParseError(f"Expected ')' near {' '.join(self.t[max(0, i - 5):i + 1])}")
        return ("cmp", op, lhs, rhs), self.nxt(i)

    def expression(self, i):
        pre = ""
        if self.t[i] in ("+", "-"):
            pre = self.t[i]
            i = self.nxt(i)
        term, i = self.term(i)
        op, second = "", None
        if self.t[i] in ("+", "-", "||"):
            op = self.t[i]
            second, i = self.expression(self.nxt(i))
        return ("expr", pre, term, op, second), i

    def term(self, i):
        f, i = self.factor(i)
        op, second = "", None
        if self.t[i] in ("*", "/", "&&"):
            op = self.t[i]
            second, i = self.term(self.nxt(i))
        return ("term", f, op, second), i

    def factor(self, i):
        tok, la = self.t[i], self.la(i)
        if tok == "(":
            cmp, i = self.comparison(self.nxt(i))
            if cmp[1] == "":
                return ("paren", cmp[2]), i
            return cmp, i
        if tok.startswith('"'):
            return ("str", tok[1:-1]), self.nxt(i)
        if tok == "[":
            i = self.nxt(i)
            elems = []
            while self.t[i] != "]":
                e, i = self.expression(i)
                elems.append(e)
                if self.t[i] not in (",", "]"):
                    raise ParseError("Expected ',' in array")
                if self.t[i] == ",":
                    i = self.nxt(i)
            return ("arr", elems), self.nxt(i)
        if la == "[":
            e, i = self.expression(i + 2)
            if self.t[i] != "]":
                raise ParseError("Expected ']'")
            return ("idx", tok, e), self.nxt(i)
        if _is_int(tok) or _is_real(tok):
            return ("num", tok), self.nxt(i)
        if la == "(":
            return self.funcall(i)
        return ("sym", tok), self.nxt(i)

    def funcall(self, i):
        name = self.t[i]
        i = self.nxt(i + 1)
        args = []
        while self.t[i] != ")":
            e, i = self.expression(i)
            if self.t[i] not in (",", ")"):
                raise ParseError(f"Expected ',' in call of {name}")
            args.append(e)
            if self.t[i] == ",":
                i = self.nxt(i)
        return ("call", name, args), self.nxt(i)


def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_real(s):
    try:
        float(s)
        return s[0].isdigit()
    except ValueError:
        return False


# ===================================================================== code generation
def py_name(sym: str) -> str:
    n = sym.replace("-", "_").replace("+", "_plus_")
    if keyword.iskeyword(n) or n in HELPERS:
        n += "_"
    return n


def func_name(name: str) -> str:
    return name.replace("-", "_")


_CMP = {"==": "==", "!=": "!=", "<>": "!=", "<=": "<=", "=<": "<=", ">=": ">=", "=>": ">=",
        ">": ">", "<": "<"}
NUMERIC_FUNCS = {"random", "time", "length", "int", "str-to-int", "get-activation", "get-sji"}
HELPER_NAMES = ("div", "plus", "setitem", "and_", "or_", "truth", "cmp")


def _has_op(node) -> bool:
    return node is not None and node[0] in ("expr", "term") and node[3 if node[0] == "expr" else 2] != ""


def _core(node):
    """Strip expr/term/paren wrappers that carry no operator."""
    while True:
        if node[0] == "expr" and node[3] == "" and node[1] == "":
            node = node[2]
        elif node[0] == "term" and node[2] == "":
            node = node[1]
        elif node[0] == "paren":
            node = node[1]
        else:
            return node


def _is_flag(node) -> bool:
    """Is the value certainly 0/1 (a comparison or a negation)?"""
    return _core(node)[0] == "cmp"


class _Gen:
    """Code generator.  Keeps PRIMs script semantics exactly, and uses plain
    Python operators wherever a simple type inference shows they mean the same."""

    def __init__(self, stmts):
        self.numeric = self._infer(stmts)

    # ---------------------------------------------------------------- type inference
    def _infer(self, stmts):
        assigns, loop_num, loop_other = {}, set(), set()

        def walk(sts):
            for st in sts:
                if st[0] == "assign" and st[1][0] == "sym":
                    assigns.setdefault(st[1][1], []).append(st[2])
                elif st[0] == "if":
                    walk(st[2]); walk(st[3])
                elif st[0] == "while":
                    walk(st[2])
                elif st[0] == "for":
                    (loop_num if st[3] is not None else loop_other).add(st[1])
                    walk(st[4])
        walk(stmts)
        env = {v: True for v in assigns}
        env.update({v: True for v in loop_num})
        env.update({v: False for v in loop_other})
        env["arg"] = False
        changed = True
        while changed:
            changed = False
            for v, rhss in assigns.items():
                val = env[v] and v not in loop_other and all(self._num(r, env) for r in rhss)
                if val != env[v]:
                    env[v] = val
                    changed = True
        return env

    def _num(self, node, env=None):
        env = self.numeric if env is None else env
        k = node[0]
        if k == "num" or k == "cmp":
            return True
        if k in ("str", "arr", "idx"):
            return False
        if k == "call":
            return node[1] in NUMERIC_FUNCS
        if k == "sym":
            return env.get(node[1], False)
        if k == "paren":
            return self._num(node[1], env)
        if k == "expr":
            if node[3] in ("-", "||") or node[1] == "-":
                return True
            return self._num(node[2], env)        # '+' or none: the left operand decides
        if k == "term":
            return True if node[2] != "" else self._num(node[1], env)
        return False

    # ---------------------------------------------------------------- expressions
    def expr(self, node) -> str:
        k = node[0]
        if k == "expr":
            _, pre, term, op, second = node
            t = self.expr(term)
            if pre == "-":
                t = f"-{t}" if not _has_op(term) else f"-({t})"
            if op == "":
                return t
            s = self.expr(second)
            if op == "||":
                return f"or_({t}, {s})"
            if _has_op(second) or second[1] == "-":
                s = f"({s})"
            if op == "+" and not self._num(term):
                return f"plus({t}, {s})"
            return f"{t} {op} {s}"
        if k == "term":
            _, f, op, second = node
            a = self.expr(f)
            if op == "":
                return a
            b = self.expr(second)
            if op == "/":
                return f"div({a}, {b})"
            if op == "&&":
                return f"and_({a}, {b})"
            if _has_op(second):
                b = f"({b})"
            return f"{a} * {b}"
        if k == "num":
            return node[1]
        if k == "str":
            return '"' + node[1].replace("\\", "\\\\").replace('"', '\\"') + '"'
        if k == "sym":
            return py_name(node[1])
        if k == "arr":
            return "[" + ", ".join(self.expr(e) for e in node[1]) + "]"
        if k == "idx":
            return f"{py_name(node[1])}[{self.expr(node[2])}]"
        if k == "call":
            return f"m.{func_name(node[1])}(" + ", ".join(self.expr(a) for a in node[2]) + ")"
        if k == "paren":
            inner = self.expr(node[1])
            return f"({inner})" if _has_op(node[1]) else inner
        if k == "cmp":
            _, op, lhs, rhs = node
            if op == "!":
                return f"(not {self.expr(lhs)})"
            a, b = self.expr(lhs), self.expr(rhs)
            if op in ("==", "!=", "<>") or (self._num(lhs) and self._num(rhs)):
                return f"({a} {_CMP[op]} {b})"
            return f'cmp({a}, "{_CMP[op]}", {b})'      # Swift: < and > are false for strings
        raise ValueError(node)

    def test(self, node) -> str:
        """Condition of an if: Swift takes any non-integer value as true."""
        code = _strip(self.expr(node))
        if _is_flag(node) or _core(node)[0] == "num":
            return code
        c = _core(node)
        if c[0] == "term" and c[2] == "&&" or c[0] == "expr" and c[3] == "||":
            return code                              # and_/or_ return integers
        return f"truth({code})"

    # ---------------------------------------------------------------- statements
    def statements(self, stmts, indent: int) -> list[str]:
        pad = "    " * indent
        lines = []
        for st in stmts:
            kind = st[0]
            if kind == "assign":
                _, lhs, rhs = st
                if lhs[0] == "idx":
                    lines.append(f"{pad}setitem({py_name(lhs[1])}, {self.expr(lhs[2])}, {self.expr(rhs)})")
                else:
                    lines.append(f"{pad}{py_name(lhs[1])} = {self.expr(rhs)}")
            elif kind == "call":
                lines.append(pad + self.expr(st))
            elif kind == "if":
                _, test, then, els = st
                lines.append(f"{pad}if {self.test(test)}:")
                lines += self.statements(then, indent + 1) or [pad + "    pass"]
                if els:
                    lines.append(f"{pad}else:")
                    lines += self.statements(els, indent + 1)
            elif kind == "while":
                _, test, body = st
                body_lines = self.statements(body, indent + 1) or [pad + "    pass"]
                if _is_flag(test):
                    lines.append(f"{pad}while {_strip(self.expr(test))}:")
                    lines += body_lines
                else:
                    # Swift: enter if the test is non-zero, repeat only while it equals 1
                    t = _strip(self.expr(test))
                    lines.append(f"{pad}if {self.test(test)}:")
                    lines.append(f"{pad}    while True:")
                    lines += ["    " + ln for ln in body_lines]
                    lines.append(f"{pad}        if ({t}) != 1:")
                    lines.append(f"{pad}            break")
            elif kind == "for":
                _, var, start, end, body = st
                v = py_name(var)
                body_lines = self.statements(body, indent + 1)
                if end is None:
                    arr = self.expr(start)
                    if _assigns_into(body, start):
                        # Swift fixes the number of iterations when the loop starts
                        tmp = f"_items_{v}"
                        lines.append(f"{pad}{tmp} = {arr}")
                        lines.append(f"{pad}for _k_{v} in range(len({tmp})):")
                        lines.append(f"{pad}    {v} = {tmp}[_k_{v}]")
                    else:
                        lines.append(f"{pad}for {v} in {arr}:")
                    lines += body_lines or [pad + "    pass"]
                else:
                    # Swift re-tests i <= end every pass and honours changes to i
                    lines.append(f"{pad}{v} = {self.expr(start)}")
                    e = self.expr(end)
                    lines.append(f"{pad}while {v} <= {e if not _has_op(end) else '(' + e + ')'}:")
                    lines += body_lines
                    lines.append(f"{pad}    {v} = {v} + 1")
        return lines


def _assigns_into(body, arr_node) -> bool:
    c = _core(arr_node)
    if c[0] != "sym":
        return False
    name = c[1]

    def walk(sts):
        for st in sts:
            if st[0] == "assign" and st[1][0] == "idx" and st[1][1] == name:
                return True
            for sub in st[2:]:
                if isinstance(sub, list) and walk(sub):
                    return True
        return False
    return walk(body)


def _strip(s: str) -> str:
    if s.startswith("(") and s.endswith(")"):
        depth = 0
        for k, ch in enumerate(s):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and k < len(s) - 1:
                return s
        return s[1:-1]
    return s


def translate_source(src: str, name: str = "script") -> str:
    """PRIMs script text -> Python source of ``def name(m): ...``."""
    from .dsl import strip_comments
    src = strip_comments(src)
    stmts = _Parser(tokenize(src)).statements()
    body = _Gen(stmts).statements(stmts, 1)
    uses_arg = any("arg" == t for t in tokenize(src))
    head = [f"def {name}(m):"]
    if uses_arg:
        head.append("    arg = m.arg")
    return "\n".join(head + (body or ["    pass"])) + "\n"


def translate_script(src: str, name: str = "script"):
    """Return (function, python source)."""
    from . import script_api
    code = translate_source(src, name)
    ns = {h: getattr(script_api, h) for h in HELPER_NAMES}
    exec(compile(code, f"<prims {name}>", "exec"), ns)
    return ns[name], code
