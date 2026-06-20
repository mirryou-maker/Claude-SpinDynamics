"""mx3 — a mumax3-compatible ``.mx3`` script runner for micromag.

Parses and executes a practical subset of the mumax3 scripting language on
top of the micromag engine (GPU when ``cuda_available()``, else CPU). The goal
is drop-in execution of the common standard-problem / example scripts, not full
language coverage — unsupported statements emit a warning and are skipped so a
partially-supported script still runs as far as it can.

Supported subset
----------------
Geometry / grid
    SetGridSize(nx, ny, nz)
    SetCellSize(cx, cy, cz)
    SetPBC(px, py, pz)                 # periodic image counts (non-zero -> periodic)

Material parameters (mumax3 settable quantities)
    Msat = <expr>                      # A/m
    Aex  = <expr>                      # J/m
    alpha = <expr>
    Ku1  = <expr>                      # J/m^3   (uniaxial)
    anisU = vector(x, y, z)            # easy axis
    Dind = <expr>                      # interfacial DMI  [J/m^2]
    Dbulk = <expr>                     # bulk DMI         [J/m^2]
    B_ext = vector(bx, by, bz)         # Tesla -> H = B/mu0
    EnableDemag = true|false

Magnetization initial state
    m = uniform(x, y, z)
    m = vortex(circ, pol)
    m = random()
    m = twodomain(ax,ay,az, wx,wy,wz, bx,by,bz)   # approx (left/right halves)

Solver controls
    MaxErr = <expr>;  MaxDt = <expr>;  MinDt = <expr>
    SetSolver(n)                       # accepted; 5/6 -> adaptive RK45, else RK4

Geometry (CPU backend only — the GPU solver lacks empty-cell handling)
    setgeom(ellipse(dx, dy) | circle(d) | rect(lx, ly) | square(s) |
            cylinder(d, h) | cuboid(lx, ly, lz) | ellipsoid(dx, dy, dz))

Control flow (Go-style braces)
    for init; cond; post { ... }       # e.g. for i := 0; i < N; i++ { ... }
    if cond { ... } else { ... }
    operators: < > <= >= == != && || !  and  ++ -- += -= *= /=

Run / relax
    relax()
    minimize()
    run(t)
    steps(n)

Output
    save(m)                            # -> <basename>NNNNNN.ovf
    saveas(m, "name")
    tableSave()                        # append averaged m to <basename>.txt
    autosave(m, interval)              # periodic save during run()
    tableautosave(interval)
    print(...)

Expressions: numbers (1e-9 etc.), + - * /, unary -, parentheses, vector(...),
math functions sqrt/sin/cos/tan/exp/abs/pow, constants pi and mu0, and
user variables declared with ``name := expr``.

CLI:  python -m micromag.mx3 script.mx3
"""

from __future__ import annotations

import math
import os
import re
import sys

import micromag as mm

MU0 = 4.0e-7 * math.pi


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"""
      (?P<NUMBER>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
    | (?P<IDENT>[A-Za-z_]\w*)
    | (?P<STRING>"[^"]*")
    | (?P<OP>:=|\+\+|--|\+=|-=|\*=|/=|==|!=|<=|>=|&&|\|\||[-+*/(){},.<>=!;])
    | (?P<WS>[ \t\r]+)
""", re.VERBOSE)


class Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"{self.kind}:{self.value!r}"


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)   # block comments
    src = re.sub(r"//[^\n]*", "", src)                       # line comments
    src = re.sub(r"#[^\n]*", "", src)                        # '#' comments (tolerated)
    return src


def tokenize_line(line: str):
    """Tokenize a single logical line. Identifiers are lower-cased (mumax3 is
    case-insensitive); string contents are preserved."""
    toks = []
    pos = 0
    while pos < len(line):
        m = _TOKEN_RE.match(line, pos)
        if not m:
            raise SyntaxError(f"cannot tokenize near: {line[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        val = m.group()
        if kind == "NUMBER":
            toks.append(Token("NUMBER", float(val)))
        elif kind == "IDENT":
            toks.append(Token("IDENT", val.lower()))
        elif kind == "STRING":
            toks.append(Token("STRING", val[1:-1]))
        else:  # OP
            toks.append(Token("OP", val))
    return toks


def tokenize(src):
    """Tokenize a whole source into one flat token list, emitting NEWLINE
    statement terminators (suppressed inside unbalanced parentheses so
    multi-line calls work) and a trailing EOF token."""
    src = _strip_comments(src)
    out, depth = [], 0
    for line in src.split("\n"):
        lt = tokenize_line(line)
        for t in lt:
            if t.kind == "OP" and t.value == "(":
                depth += 1
            elif t.kind == "OP" and t.value == ")":
                depth -= 1
            out.append(t)
        if depth <= 0 and lt:
            out.append(Token("NEWLINE", "\n"))
    out.append(Token("EOF", None))
    return out


# ---------------------------------------------------------------------------
# Values produced by expression evaluation
# ---------------------------------------------------------------------------
class Vec(tuple):
    """A 3-vector value (subclass of tuple so it prints nicely)."""
    def __new__(cls, x, y, z):
        return super().__new__(cls, (float(x), float(y), float(z)))


class Config:
    """A magnetization initial-state config, e.g. uniform / vortex / random."""
    def __init__(self, kind, args):
        self.kind = kind
        self.args = args


# ---------------------------------------------------------------------------
# Expression parser / evaluator (recursive descent, eager)
# ---------------------------------------------------------------------------
class _ExprParser:
    def __init__(self, toks, env):
        self.toks = toks
        self.i = 0
        self.env = env          # user variable dict (name -> value)

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _accept(self, kind, value=None):
        t = self._peek()
        if t and t.kind == kind and (value is None or t.value == value):
            self.i += 1
            return t
        return None

    def parse(self):
        v = self._or()
        if self.i != len(self.toks):
            raise SyntaxError(f"trailing tokens in expression: {self.toks[self.i:]}")
        return v

    def _or(self):
        v = self._and()
        while self._peek() and self._peek().value == "||":
            self._next(); r = self._and()
            v = 1.0 if (_truth(v) or _truth(r)) else 0.0
        return v

    def _and(self):
        v = self._equality()
        while self._peek() and self._peek().value == "&&":
            self._next(); r = self._equality()
            v = 1.0 if (_truth(v) and _truth(r)) else 0.0
        return v

    def _equality(self):
        v = self._relational()
        while self._peek() and self._peek().value in ("==", "!="):
            op = self._next().value; r = self._relational()
            v = 1.0 if ((v == r) == (op == "==")) else 0.0
        return v

    def _relational(self):
        v = self._expr()
        while self._peek() and self._peek().value in ("<", ">", "<=", ">="):
            op = self._next().value; r = self._expr()
            v = 1.0 if _cmp(op, v, r) else 0.0
        return v

    def _expr(self):           # additive
        v = self._term()
        while True:
            t = self._peek()
            if t and t.kind == "OP" and t.value in "+-":
                self._next()
                r = self._term()
                v = _binop(t.value, v, r)
            else:
                return v

    def _term(self):           # multiplicative
        v = self._unary()
        while True:
            t = self._peek()
            if t and t.kind == "OP" and t.value in "*/":
                self._next()
                r = self._unary()
                v = _binop(t.value, v, r)
            else:
                return v

    def _unary(self):
        t = self._peek()
        if t and t.kind == "OP" and t.value == "-":
            self._next()
            return _neg(self._unary())
        if t and t.kind == "OP" and t.value == "+":
            self._next()
            return self._unary()
        if t and t.kind == "OP" and t.value == "!":
            self._next()
            return 0.0 if _truth(self._unary()) else 1.0
        return self._atom()

    def _atom(self):
        t = self._next()
        if t.kind == "NUMBER":
            return t.value
        if t.kind == "STRING":
            return t.value
        if t.kind == "OP" and t.value == "(":
            v = self._expr()
            if not self._accept("OP", ")"):
                raise SyntaxError("missing ')'")
            return v
        if t.kind == "IDENT":
            name = t.value
            if self._accept("OP", "("):       # function call
                args = self._arglist()
                return _call_func(name, args)
            # bare identifier: constant or user variable
            if name in _CONSTS:
                return _CONSTS[name]
            if name in self.env:
                return self.env[name]
            raise NameError(f"unknown identifier '{name}'")
        raise SyntaxError(f"unexpected token {t}")

    def _arglist(self):
        args = []
        if self._accept("OP", ")"):
            return args
        while True:
            args.append(self._expr())
            if self._accept("OP", ","):
                continue
            if self._accept("OP", ")"):
                break
            raise SyntaxError("expected ',' or ')' in argument list")
        return args


_CONSTS = {"pi": math.pi, "mu0": MU0, "true": 1.0, "false": 0.0, "inf": math.inf}


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _binop(op, a, b):
    if _is_num(a) and _is_num(b):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
    raise TypeError(f"cannot apply '{op}' to {a!r}, {b!r}")


def _neg(a):
    if _is_num(a):
        return -a
    raise TypeError(f"cannot negate {a!r}")


def _truth(v):
    return bool(v) if not _is_num(v) else v != 0.0


def _cmp(op, a, b):
    return {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]


def _call_func(name, args):
    if name == "vector":
        if len(args) != 3:
            raise TypeError("vector() takes 3 arguments")
        return Vec(*args)
    if name in ("uniform", "vortex", "random", "randommag", "twodomain",
                "neelskyrmion", "blochskyrmion"):
        return Config(name, args)
    if name in ("ellipse", "circle", "rect", "cylinder", "square",
                "cuboid", "ellipsoid"):
        return Config("shape:" + name, args)
    fn = _MATH_FUNCS.get(name)
    if fn is not None:
        return fn(*args)
    raise NameError(f"unknown function '{name}()'")


_MATH_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "abs": abs, "pow": pow, "log": math.log, "floor": math.floor,
    "ceil": math.ceil,
}


def eval_expr(toks, env):
    return _ExprParser(toks, env).parse()


# ---------------------------------------------------------------------------
# Engine — backend abstraction (GPU preferred, CPU fallback)
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, gpu=None, outdir="."):
        self.gpu = mm.cuda_available() if gpu is None else gpu
        self.outdir = outdir
        self.grid = None
        self.gridsize = None        # (nx, ny, nz)
        self.cellsize = None        # (cx, cy, cz)
        self.pbc = (0, 0, 0)
        self.mat = mm.Material()
        self.mat.alpha = 0.5
        self.m = None               # CPU master VectorField3D

        # field parameters
        self.aex = 0.0
        self.ku1 = 0.0
        self.anisU = (0.0, 0.0, 1.0)
        self.dind = 0.0
        self.dbulk = 0.0
        self.bext = (0.0, 0.0, 0.0)
        self.enable_demag = True

        # solver options
        self.maxerr = 1e-5
        self.maxdt = 0.0
        self.mindt = 0.0
        self.solver = 5             # 5/6 -> adaptive RK45; else RK4

        # built objects (lazy)
        self._fields_built = False
        self._demag = self._exch = self._aniso = self._dmi = self._zeeman = None
        self._dmi_bulk = None
        self.geom_mask = None       # optional GeomMask (setgeom)

        # time + outputs
        self.t = 0.0
        self.table_path = None
        self.table_started = False
        self.save_counter = 0
        self.autosaves = []         # list of [kind, interval, next_t]
        self.basename = "m"

    # -- grid --------------------------------------------------------------
    def _maybe_build_grid(self):
        if self.gridsize and self.cellsize and self.grid is None:
            nx, ny, nz = self.gridsize
            cx, cy, cz = self.cellsize
            self.grid = mm.StructuredGrid(nx, ny, nz, cx, cy, cz)
            self.m = mm.VectorField3D(self.grid)
            self.m.set_uniform(mm.Vec3(1, 0, 0))

    def set_gridsize(self, nx, ny, nz):
        self.gridsize = (int(nx), int(ny), int(nz))
        self._maybe_build_grid()

    def set_cellsize(self, cx, cy, cz):
        self.cellsize = (float(cx), float(cy), float(cz))
        self._maybe_build_grid()

    def _require_grid(self):
        if self.grid is None:
            raise RuntimeError("grid not defined — call SetGridSize and SetCellSize first")

    # -- geometry ----------------------------------------------------------
    def set_geom(self, cfg):
        self._require_grid()
        mask = self._build_mask(cfg)
        if self.gpu:
            # The GPU solver normalises every cell, so empty (m=0) cells blow up
            # to NaN. Geometry needs the CPU backend (its normalize() skips them).
            _warn("setgeom: geometry is only supported on the CPU backend; "
                  "ignoring geometry — re-run with gpu=False for masked shapes")
            return
        self.geom_mask = mask
        if self.m is not None:
            self._apply_geom()

    def _build_mask(self, cfg):
        g = self.grid
        if not (isinstance(cfg, Config) and cfg.kind.startswith("shape:")):
            raise TypeError("setgeom expects a shape, e.g. ellipse(dx, dy)")
        shape = cfg.kind.split(":", 1)[1]
        a = cfg.args
        # mumax3 shapes take full diameters/lengths; micromag ellipse/circle/
        # cylinder use semi-axes/radii, so halve those.
        if shape == "ellipse":
            return mm.ellipse(g, a[0] / 2, a[1] / 2)
        if shape == "circle":
            return mm.circle(g, a[0] / 2)
        if shape == "cylinder":
            return mm.cylinder(g, a[0] / 2, a[1])
        if shape == "ellipsoid":
            return mm.ellipsoid(g, a[0] / 2, a[1] / 2, a[2] / 2)
        if shape == "rect":
            return mm.rect(g, a[0], a[1])
        if shape == "square":
            return mm.square(g, a[0])
        if shape == "cuboid":
            return mm.cuboid(g, a[0], a[1], a[2])
        raise ValueError(f"unsupported shape '{shape}'")

    def _apply_geom(self):
        if self.geom_mask is not None:
            self.m.apply_mask(self.geom_mask)

    # -- fields ------------------------------------------------------------
    def _build_fields(self):
        if self._fields_built:
            return
        self._require_grid()
        g = self.grid
        periodic = any(self.pbc)
        if self.gpu:
            self._demag = (mm.DemagFieldPeriodicGPU(g) if periodic
                           else mm.DemagFieldGPU(g))
            self._exch = mm.ExchangeFieldGPU(g)
            self._aniso = mm.UniaxialAnisotropyFieldGPU(g)
            self._zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 0))
            self._dmi = mm.InterfacialDMIFieldGPU(g, 0.0)
            self._dmi_bulk = mm.BulkDMIFieldGPU(g, 0.0)
        else:
            # CPU fields are stateless w.r.t. the grid (read it from the field
            # at accumulate time); only demag needs the grid for FFT setup.
            self._demag = (mm.DemagFieldPeriodic(g) if periodic
                           else mm.DemagField(g))
            self._exch = mm.ExchangeField()
            self._aniso = mm.UniaxialAnisotropyField()
            self._zeeman = mm.ZeemanField(mm.Vec3(0, 0, 0))
            self._dmi = mm.InterfacialDMIField(0.0)
            self._dmi_bulk = mm.BulkDMIField(0.0)
        self._fields_built = True

    def _sync_params(self):
        """Push current scalar/vector parameters into Material + field objects."""
        self.mat.A_exchange = self.aex
        self.mat.K_uniaxial = self.ku1
        self.mat.easy_axis = mm.Vec3(*self.anisU)
        # Zeeman: B [T] -> H [A/m]  (H_ext is a read/write property)
        H = mm.Vec3(self.bext[0] / MU0, self.bext[1] / MU0, self.bext[2] / MU0)
        _set(self._zeeman, "H_ext", "set_H_ext", H)
        # DMI strength (D is a read/write property)
        _set(self._dmi, "D", "set_D", self.dind)
        _set(self._dmi_bulk, "D", "set_D", self.dbulk)

    def _active_fields(self):
        """Return (demag_or_None, field_sum_or_effsum) with current params synced."""
        self._build_fields()
        self._sync_params()
        if self.geom_mask is not None:
            _call_method(self._exch, "set_mask", self.geom_mask)   # CPU only; GPU no-op
        use_dmi = abs(self.dind) > 0
        use_bulk = abs(self.dbulk) > 0
        use_ani = abs(self.ku1) > 0
        if self.gpu:
            fs = mm.FieldSumGPU()
            fs.add(self._exch)
            if use_ani:
                fs.add(self._aniso)
            if use_dmi:
                fs.add(self._dmi)
            if use_bulk:
                fs.add(self._dmi_bulk)
            fs.add(self._zeeman)
            demag = self._demag if self.enable_demag else None
            return demag, fs
        else:
            fs = mm.EffectiveFieldSum()
            if self.enable_demag:
                fs.add(self._demag)
            fs.add(self._exch)
            if use_ani:
                fs.add(self._aniso)
            if use_dmi:
                fs.add(self._dmi)
            if use_bulk:
                fs.add(self._dmi_bulk)
            fs.add(self._zeeman)
            return None, fs

    # -- magnetization init ------------------------------------------------
    def set_m(self, cfg):
        self._require_grid()
        if isinstance(cfg, Vec):
            cfg = Config("uniform", list(cfg))
        if not isinstance(cfg, Config):
            raise TypeError(f"cannot assign {cfg!r} to m")
        k = cfg.kind
        if k == "uniform":
            self.m.set_uniform(mm.Vec3(*cfg.args))
            self.m.normalize()
        elif k == "vortex":
            cx = self.gridsize[0] * self.cellsize[0] * 0.5
            cy = self.gridsize[1] * self.cellsize[1] * 0.5
            core = 3.0 * min(self.cellsize[0], self.cellsize[1])
            self.m.set_vortex(cx, cy, core)
        elif k in ("random", "randommag"):
            self._set_random()
        elif k == "twodomain":
            self._set_twodomain(cfg.args)
        elif k in ("neelskyrmion", "blochskyrmion"):
            self.m.set_uniform(mm.Vec3(0, 0, 1))
            self.m.normalize()
            _warn(f"m = {k}(...) approximated as uniform +z (use micromag skyrmion helpers for exact)")
        else:
            _warn(f"unsupported m initialiser '{k}', leaving m unchanged")
        self._apply_geom()

    def _set_random(self):
        import numpy as np
        nx, ny, nz = self.gridsize
        v = np.random.normal(size=(nz, ny, nx, 3))
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        mm.from_numpy(self.m, v)

    def _set_twodomain(self, args):
        # twodomain(ax,ay,az, wx,wy,wz, bx,by,bz): left half -> a, right half -> b
        a = mm.Vec3(*args[0:3]); b = mm.Vec3(*args[6:9])
        nx, ny, nz = self.gridsize
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    self.m[i + nx * (j + ny * k)] = a if i < nx // 2 else b
        self.m.normalize()

    # -- solvers -----------------------------------------------------------
    def relax(self):
        demag, fs = self._active_fields()
        if self.gpu and demag is not None and not any(self.pbc):
            relax = mm.RelaxGPU(self.grid)
            opts = mm.RelaxGPUOptions()
            relax.upload(self.m)
            relax.run(self.mat, demag, fs, opts)
            relax.download(self.m)
        else:
            self._relax_via_stepping()

    def minimize(self):
        demag, fs = self._active_fields()
        if self.gpu and demag is not None and not any(self.pbc):
            mini = mm.MinimizeGPU(self.grid)
            opts = mm.MinimizeGPUOptions()
            mini.upload(self.m)
            mini.run(self.mat, demag, fs, opts)
            mini.download(self.m)
        else:
            self._relax_via_stepping()

    def _relax_via_stepping(self):
        """Backend-agnostic relax: high-damping RK4 until <m> stops moving."""
        save_alpha = self.mat.alpha
        self.mat.alpha = 1.0
        dt = self._default_dt()
        integ, stepper = self._make_rk4(dt)
        prev = None
        for it in range(400):
            for _ in range(25):
                stepper()
            self._download(integ)
            cur = self._avg_m()
            if prev is not None and max(abs(cur[i] - prev[i]) for i in range(3)) < 1e-5:
                break
            prev = cur
        self._download(integ)
        self.mat.alpha = save_alpha

    def run(self, t):
        demag, fs = self._active_fields()
        if self.gpu:
            self._run_gpu(demag, fs, t)
        else:
            self._run_cpu(fs, t)
        self.t += t

    def steps(self, n):
        demag, fs = self._active_fields()
        dt = self._default_dt()
        integ, stepper = self._make_rk4(dt)
        for _ in range(int(n)):
            stepper()
            self.t += dt
        self._download(integ)

    def _run_gpu(self, demag, fs, t):
        adaptive = self.solver in (5, 6) and demag is not None
        if adaptive:
            opts = mm.RK45IntegratorGPUOptions() if hasattr(mm, "RK45IntegratorGPUOptions") else None
            integ = mm.RK45IntegratorGPU(self.grid) if opts is None else mm.RK45IntegratorGPU(self.grid, opts)
            integ.upload(self.m)
            elapsed = 0.0
            while elapsed < t * (1 - 1e-9):
                dt = integ.step(self.mat, demag, fs)
                elapsed += dt
                self._handle_autosave(integ, elapsed)
            integ.download(self.m)
        else:
            dt = self._default_dt()
            integ = mm.RK4IntegratorGPU(self.grid, dt)
            integ.upload(self.m)
            n = max(1, int(round(t / dt)))
            for k in range(n):
                integ.step(self.mat, demag, fs)
                self._handle_autosave(integ, (k + 1) * dt)
            integ.download(self.m)

    def _run_cpu(self, fs, t):
        dt = self._default_dt()
        integ = mm.RK45Integrator()       # adaptive DOPRI5 (default tolerances)
        elapsed = 0.0
        guard = 0
        while elapsed < t * (1 - 1e-9):
            adv = integ.step(self.m, self.mat, fs)
            elapsed += (adv if adv else dt)
            guard += 1
            if guard > 10_000_000:
                break

    # -- integrator helpers (for stepping-based relax/steps) ---------------
    def _make_rk4(self, dt):
        demag, fs = self._active_fields()
        if self.gpu:
            integ = mm.RK4IntegratorGPU(self.grid, dt)
            integ.upload(self.m)
            stepper = lambda: integ.step(self.mat, demag, fs)
        else:
            integ = mm.RK4Integrator(dt)
            stepper = lambda: integ.step(self.m, self.mat, fs)
        return integ, stepper

    def _download(self, integ):
        if self.gpu:
            integ.download(self.m)

    def _max_angle(self, integ):
        if self.gpu and hasattr(integ, "max_angle_gpu"):
            return integ.max_angle_gpu()
        return None

    def _default_dt(self):
        if self.maxdt and self.maxdt > 0:
            return self.maxdt
        return 5e-14

    # -- outputs -----------------------------------------------------------
    def _avg_m(self):
        return mm.mean_magnetization(self.m)

    def save(self, basename=None):
        name = basename or self.basename
        path = os.path.join(self.outdir, f"{name}{self.save_counter:06d}.ovf")
        mm.save_ovf(path, self.m)
        self.save_counter += 1
        print(f"  saved {path}")

    def table_save(self):
        if self.table_path is None:
            self.table_path = os.path.join(self.outdir, f"{self.basename}.txt")
        new = not self.table_started
        with open(self.table_path, "a", encoding="utf-8") as fh:
            if new:
                fh.write("# t (s)\tmx\tmy\tmz\n")
                self.table_started = True
            mx, my, mz = self._avg_m()
            fh.write(f"{self.t:.6e}\t{mx:.6e}\t{my:.6e}\t{mz:.6e}\n")

    def add_autosave(self, interval):
        self.autosaves.append(["save", float(interval), float(interval)])

    def add_tableautosave(self, interval):
        self.autosaves.append(["table", float(interval), float(interval)])

    def _handle_autosave(self, integ, elapsed):
        if not self.autosaves:
            return
        cur = self.t + elapsed
        for entry in self.autosaves:
            kind, interval, nxt = entry
            if cur + 1e-18 >= nxt:
                if self.gpu:
                    integ.download(self.m)
                if kind == "save":
                    self.save()
                else:
                    self.table_save()
                entry[2] = nxt + interval


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _call_method(obj, method, *args):
    """Call obj.method(*args) if it exists; return False if absent/failed."""
    fn = getattr(obj, method, None)
    if callable(fn):
        try:
            fn(*args)
            return True
        except Exception:
            return False
    return False


def _set(obj, prop, method, val):
    """Set a value via a pybind property (preferred) or a setter method."""
    try:
        setattr(obj, prop, val)
        return True
    except Exception:
        pass
    fn = getattr(obj, method, None)
    if callable(fn):
        try:
            fn(val)
            return True
        except Exception:
            return False
    return False


def _warn(msg):
    print(f"[mx3 warning] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Interpreter — dispatch parsed statements onto the Engine
# ---------------------------------------------------------------------------
_OUTPUT_FUNCS = {"save", "saveas", "autosave", "tableadd", "tableaddvar", "print"}


def _split_args(toks):
    """Split a flat token list by top-level commas (respecting nested parens)."""
    groups, cur, depth = [], [], 0
    for t in toks:
        if t.kind == "OP" and t.value == "(":
            depth += 1; cur.append(t)
        elif t.kind == "OP" and t.value == ")":
            depth -= 1; cur.append(t)
        elif t.kind == "OP" and t.value == "," and depth == 0:
            groups.append(cur); cur = []
        else:
            cur.append(t)
    if cur:
        groups.append(cur)
    return groups


class Interpreter:
    def __init__(self, engine=None, outdir="."):
        self.eng = engine or Engine(outdir=outdir)
        self.env = {}            # user variables (name := expr)

    def run_source(self, src):
        try:
            nodes = _Parser(tokenize(src)).parse_program()
        except Exception as e:
            _warn(f"parse error: {type(e).__name__}: {e}")
            return
        self._exec_block(nodes)

    def _exec_block(self, nodes):
        for node in nodes:
            try:
                self._exec_node(node)
            except Exception as e:
                _warn(f"{type(e).__name__}: {e}  [{_node_str(node)}]")

    def _exec_node(self, node):
        kind = node[0]
        if kind == "assign":
            self._do_assign(node[1], node[2], node[3])
        elif kind == "call":
            self._do_call(node[1], node[2])
        elif kind == "for":
            self._exec_for(node)
        elif kind == "if":
            self._exec_if(node)
        elif kind == "method":
            _warn(f"method call '{node[1]}.{node[2]}(...)' not supported, skipped")
        elif kind == "nop":
            pass
        else:
            _warn(f"unrecognised statement: {_node_str(node)}")

    def _exec_for(self, node):
        _, init, cond, post, body = node
        self._exec_node(init)
        guard = 0
        while _truth(eval_expr(cond, self.env)):
            self._exec_block(body)
            self._exec_node(post)
            guard += 1
            if guard > 1_000_000:
                _warn("for loop exceeded 1e6 iterations, aborting")
                break

    def _exec_if(self, node):
        _, cond, then_body, else_body = node
        if _truth(eval_expr(cond, self.env)):
            self._exec_block(then_body)
        else:
            self._exec_block(else_body)

    def _get_var(self, name):
        if name in self.env:
            return self.env[name]
        e = self.eng
        return {"msat": e.mat.Ms, "aex": e.aex, "alpha": e.mat.alpha,
                "ku1": e.ku1, "dind": e.dind, "dbulk": e.dbulk,
                "maxerr": e.maxerr, "maxdt": e.maxdt, "mindt": e.mindt}.get(name, 0.0)

    def _do_assign(self, name, op, rhs_toks):
        if op in ("++", "--"):
            self.env[name] = self._get_var(name) + (1.0 if op == "++" else -1.0)
            return
        val = eval_expr(rhs_toks, self.env)
        if op in ("+=", "-=", "*=", "/="):
            cur = self._get_var(name)
            if op == "+=":
                val = cur + val
            elif op == "-=":
                val = cur - val
            elif op == "*=":
                val = cur * val
            else:
                val = cur / val
        e = self.eng
        if name == "msat":
            e.mat.Ms = float(val)
        elif name == "aex":
            e.aex = float(val)
        elif name == "alpha":
            e.mat.alpha = float(val)
        elif name in ("ku1", "k1"):
            e.ku1 = float(val)
        elif name == "anisu":
            e.anisU = tuple(val) if isinstance(val, Vec) else (0, 0, 1)
        elif name == "dind":
            e.dind = float(val)
        elif name == "dbulk":
            e.dbulk = float(val)
        elif name == "b_ext":
            e.bext = tuple(val) if isinstance(val, Vec) else (0, 0, 0)
        elif name == "enabledemag":
            e.enable_demag = bool(val)
        elif name == "maxerr":
            e.maxerr = float(val)
        elif name == "maxdt":
            e.maxdt = float(val)
        elif name == "mindt":
            e.mindt = float(val)
        elif name == "m":
            e.set_m(val)
        else:
            self.env[name] = val      # user variable / unknown global

    def _do_call(self, name, inner_toks):
        e = self.eng
        if name in _OUTPUT_FUNCS:
            self._do_output(name, _split_args(inner_toks))
            return
        args = [eval_expr(g, self.env) for g in _split_args(inner_toks)] if inner_toks else []
        if name == "setgridsize":
            e.set_gridsize(*args)
        elif name == "setcellsize":
            e.set_cellsize(*args)
        elif name == "setpbc":
            e.pbc = tuple(int(a) for a in args)
        elif name == "setsolver":
            e.solver = int(args[0])
        elif name == "relax":
            print("relax()..."); e.relax(); self._report()
        elif name == "minimize":
            print("minimize()..."); e.minimize(); self._report()
        elif name == "run":
            print(f"run({args[0]:.3e})..."); e.run(float(args[0])); self._report()
        elif name == "steps":
            print(f"steps({int(args[0])})..."); e.steps(int(args[0])); self._report()
        elif name == "tablesave":
            e.table_save()
        elif name == "tableautosave":
            e.add_tableautosave(float(args[0]))
        elif name == "setgeom":
            e.set_geom(args[0])
        elif name in ("defregion", "setregion", "edgesmooth", "tablesaveafter"):
            _warn(f"'{name}(...)' not supported, skipped")
        else:
            _warn(f"unknown command '{name}(...)', skipped")

    def _do_output(self, name, arg_groups):
        e = self.eng
        if name == "print":
            vals = []
            for g in arg_groups:
                try:
                    vals.append(eval_expr(g, self.env))
                except Exception:
                    vals.append(" ".join(str(t.value) for t in g))
            print("print:", *vals)
        elif name == "save":
            e.save()
        elif name == "saveas":
            sname = _first_string(arg_groups[1:]) or e.basename
            e.save(sname)
        elif name == "autosave":
            interval = eval_expr(arg_groups[-1], self.env)
            e.add_autosave(float(interval))
        elif name in ("tableadd", "tableaddvar"):
            pass  # we always log <mx,my,mz>

    def _report(self):
        mx, my, mz = self.eng._avg_m()
        print(f"  t={self.eng.t:.4e}s  <m>=({mx:+.4f}, {my:+.4f}, {mz:+.4f})")


def _matched_inner(toks, open_idx):
    """Return tokens strictly inside the parens that open at toks[open_idx]."""
    depth = 0
    for k in range(open_idx, len(toks)):
        if toks[k].kind == "OP" and toks[k].value == "(":
            depth += 1
        elif toks[k].kind == "OP" and toks[k].value == ")":
            depth -= 1
            if depth == 0:
                return toks[open_idx + 1:k]
    raise SyntaxError("unbalanced parentheses")


def _first_string(arg_groups):
    for g in arg_groups:
        for t in g:
            if t.kind == "STRING":
                return t.value
    return None


# ---------------------------------------------------------------------------
# Statement parser — token stream -> statement AST (supports for / if / else)
# ---------------------------------------------------------------------------
class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def _peek(self):
        return self.toks[self.i]

    def _next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _skip_terms(self):
        while self._peek().kind == "NEWLINE" or \
                (self._peek().kind == "OP" and self._peek().value == ";"):
            self.i += 1

    def parse_program(self):
        stmts = []
        self._skip_terms()
        while self._peek().kind != "EOF":
            stmts.append(self._statement())
            self._skip_terms()
        return stmts

    def _block(self):
        if not (self._peek().kind == "OP" and self._peek().value == "{"):
            raise SyntaxError("expected '{'")
        self._next()
        stmts = []
        self._skip_terms()
        while not (self._peek().kind == "OP" and self._peek().value == "}"):
            if self._peek().kind == "EOF":
                raise SyntaxError("unclosed '{'")
            stmts.append(self._statement())
            self._skip_terms()
        self._next()                       # consume '}'
        return stmts

    def _statement(self):
        t = self._peek()
        if t.kind == "IDENT" and t.value == "for":
            return self._for()
        if t.kind == "IDENT" and t.value == "if":
            return self._if()
        return _parse_simple(self._collect())

    def _collect(self):
        """Collect tokens until a top-level NEWLINE / ';' / '}' terminator."""
        toks, depth = [], 0
        while True:
            t = self._peek()
            if t.kind == "EOF":
                break
            if depth == 0 and (t.kind == "NEWLINE" or
                               (t.kind == "OP" and t.value in (";", "}"))):
                break
            if t.kind == "OP" and t.value == "(":
                depth += 1
            elif t.kind == "OP" and t.value == ")":
                depth -= 1
            toks.append(self._next())
        return toks

    def _header(self):
        """Collect tokens up to the top-level '{' that opens a block."""
        toks, depth = [], 0
        while True:
            t = self._peek()
            if t.kind == "EOF":
                raise SyntaxError("expected '{'")
            if depth == 0 and t.kind == "OP" and t.value == "{":
                break
            if t.kind == "OP" and t.value == "(":
                depth += 1
            elif t.kind == "OP" and t.value == ")":
                depth -= 1
            toks.append(self._next())
        return toks

    def _for(self):
        self._next()                       # 'for'
        header = self._header()
        body = self._block()
        parts = _split_semicolons(header)
        if len(parts) == 3:
            return ("for", _parse_simple(parts[0]), parts[1],
                    _parse_simple(parts[2]), body)
        if len(parts) == 1:                # for <cond> { ... }
            return ("for", ("nop",), parts[0], ("nop",), body)
        raise SyntaxError("malformed 'for' header")

    def _if(self):
        self._next()                       # 'if'
        cond = self._header()
        then_body = self._block()
        else_body = []
        save = self.i
        self._skip_terms()
        if self._peek().kind == "IDENT" and self._peek().value == "else":
            self._next()
            self._skip_terms()
            if self._peek().kind == "IDENT" and self._peek().value == "if":
                else_body = [self._if()]
            else:
                else_body = self._block()
        else:
            self.i = save                  # no else — don't swallow terminators
        return ("if", cond, then_body, else_body)


def _split_semicolons(toks):
    groups, cur, depth = [], [], 0
    for t in toks:
        if t.kind == "OP" and t.value == "(":
            depth += 1; cur.append(t)
        elif t.kind == "OP" and t.value == ")":
            depth -= 1; cur.append(t)
        elif t.kind == "OP" and t.value == ";" and depth == 0:
            groups.append(cur); cur = []
        else:
            cur.append(t)
    groups.append(cur)
    return groups


def _parse_simple(toks):
    if not toks:
        return ("nop",)
    if len(toks) == 2 and toks[0].kind == "IDENT" and toks[1].value in ("++", "--"):
        return ("assign", toks[0].value, toks[1].value, [])
    if len(toks) >= 2 and toks[0].kind == "IDENT" and toks[1].kind == "OP" \
            and toks[1].value in ("=", ":=", "+=", "-=", "*=", "/="):
        return ("assign", toks[0].value, toks[1].value, toks[2:])
    if len(toks) >= 2 and toks[0].kind == "IDENT" and toks[1].value == "(":
        return ("call", toks[0].value, _matched_inner(toks, 1))
    if len(toks) >= 4 and toks[0].kind == "IDENT" and toks[1].value == "." \
            and toks[2].kind == "IDENT" and toks[3].value == "(":
        return ("method", toks[0].value, toks[2].value, _matched_inner(toks, 3))
    return ("unknown", toks)


def _node_str(node):
    return f"<{node[0]} {node[1] if len(node) > 1 else ''}>"


def _logical_lines(src):
    """Yield (lineno, statement), joining physical lines while parens are
    unbalanced so multi-line calls work, then splitting on ';'."""
    src = _strip_comments(src)
    buf, depth, start = "", 0, 0
    for n, phys in enumerate(src.split("\n"), 1):
        if not buf:
            start = n
        buf += (" " if buf else "") + phys.strip()
        depth += phys.count("(") - phys.count(")")
        if depth <= 0 and buf.strip():
            for part in buf.split(";"):
                if part.strip():
                    yield start, part.strip()
            buf, depth = "", 0
    if buf.strip():
        yield start, buf.strip()


# ---------------------------------------------------------------------------
# Public entry point + CLI
# ---------------------------------------------------------------------------
def run_mx3(path, outdir=None, gpu=None):
    """Parse and execute a .mx3 script file. Returns the Engine (with final m)."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(outdir, exist_ok=True)
    eng = Engine(gpu=gpu, outdir=outdir)
    backend = "GPU" if eng.gpu else "CPU"
    print(f"=== mx3: running {path}  (backend: {backend}) ===")
    Interpreter(eng).run_source(src)
    print("=== mx3: done ===")
    return eng


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m micromag.mx3 <script.mx3> [outdir]", file=sys.stderr)
        return 2
    run_mx3(argv[0], outdir=(argv[1] if len(argv) > 1 else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
