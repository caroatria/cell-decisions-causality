"""Boolean GRN models with known driver / marker roles for benchmarking driver-gene methods.

Design principle
----------------
Every model contains
  * a progenitor program  (p1 -> p2 -> p3) that times differentiation but does not choose a fate,
  * one or more fate switches (mutually inhibiting, self-activating TFs)  -> ground-truth DRIVERS,
  * many downstream readouts of each fate TF (one and two regulatory layers deep) -> MARKERS.
Markers are as fate-specific as drivers at steady state (often more strongly DE), but they
sit downstream and have no feedback onto the switch. A method that ranks by fate association
alone cannot separate them from drivers; a method that captures direction/precedence can.

Roles written to var['role']: driver | marker | progenitor | mediator | null
var['decision'] says which decision a driver/marker belongs to (e.g. 'D1').
Null genes are added later (simulate.py) as fate-independent noise.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Model:
    name: str
    rules: dict[str, str]                 # gene -> Boolean rule (BoolODE syntax)
    roles: dict[str, str]                 # gene -> role
    decision: dict[str, str]              # gene -> decision id ('' if none)
    fate_genes: dict[str, list[str]]      # decision -> switch genes whose final state defines the fate
    initial_on: list[str] = field(default_factory=lambda: ["p1", "c1"])
    kind: dict[str, str] = field(default_factory=dict)   # finer label: fast/slow/layer2/repressed
    parents: dict[str, tuple[str, str]] = field(default_factory=dict)  # decision -> (parent decision, parent fate)
    strengths: list[tuple[str, str, float]] = field(default_factory=list)
    simulation_time: float = 8.0

    def write(self, model_dir: Path) -> dict[str, str]:
        model_dir.mkdir(parents=True, exist_ok=True)
        f_rules = model_dir / f"{self.name}.txt"
        with open(f_rules, "w") as fh:
            fh.write("Gene\tRule\n")
            for g, r in self.rules.items():
                fh.write(f"{g}\t{r}\n")
        f_ics = model_dir / f"{self.name}_ics.txt"
        with open(f_ics, "w") as fh:
            fh.write("Genes\tValues\n")
            fh.write(f"{self.initial_on!r}\t{[1] * len(self.initial_on)!r}\n")
        out = {"model_definition": f_rules.name, "model_initial_conditions": f_ics.name}
        if self.strengths:
            f_str = model_dir / f"{self.name}_strengths.txt"
            with open(f_str, "w") as fh:
                fh.write("Gene1\tGene2\tStrength\n")
                for a, b, s in self.strengths:
                    fh.write(f"{a}\t{b}\t{s}\n")
            out["interaction_strengths"] = f_str.name
        return out


HARD = True        # module-level switch: False reproduces the original "easy" benchmark
N_REPRESSED = 2    # genes repressed by each switch TF (fast, ON in progenitors) when HARD


def _add_markers(rules, roles, decision, tf, dec, n_layer1, n_layer2, prefix, kind, strengths):
    """n_layer1 genes activated directly by tf (in HARD mode the first half respond FAST: strong
    interaction = low activation threshold); n_layer2 genes activated by layer-1 markers;
    in HARD mode also N_REPRESSED fast genes repressed by tf."""
    l1 = []
    for i in range(n_layer1):
        g = f"{prefix}{i + 1}"
        rules[g] = tf
        roles[g], decision[g] = "marker", dec
        fast = HARD and i < (n_layer1 + 1) // 2
        kind[g] = "fast" if fast else "slow"
        if fast:
            strengths.append((g, tf, 20))
        l1.append(g)
    for i in range(n_layer2):
        g = f"{prefix}{n_layer1 + i + 1}"
        rules[g] = l1[i % len(l1)]
        roles[g], decision[g], kind[g] = "marker", dec, "layer2"
    if HARD:
        for i in range(N_REPRESSED):
            g = f"r{prefix[1:]}{i + 1}"
            rules[g] = f"not({tf})"
            roles[g], decision[g], kind[g] = "marker", dec, "repressed"
            strengths.append((g, tf, 20))
    return l1


def _nuisance(rules, roles, decision, kind):
    """Independent repressilator + two readouts: fate-unrelated structured variation (cell-cycle-like)."""
    if not HARD:
        return
    rules.update({"c1": "not(c3)", "c2": "not(c1)", "c3": "not(c2)", "c4": "c1", "c5": "c2"})
    for g in ("c1", "c2", "c3", "c4", "c5"):
        roles[g], decision[g], kind[g] = "nuisance", "", "nuisance"


def _progenitor(rules, roles, decision, inhibitors: list[str]):
    rules["p1"] = f"not({' or '.join(inhibitors)})"
    rules["p2"] = "p1"
    rules["p3"] = "p2"
    for g in ("p1", "p2", "p3"):
        roles[g], decision[g] = "progenitor", ""


def _switch_strengths(switch: list[str]) -> list[tuple[str, str, float]]:
    s = []
    for a in switch:
        s.append((a, a, 10))
        s.append(("p1", a, 10))
        for b in switch:
            if a != b:
                s.append((a, b, 10))
    return s


def bifurcation(n_layer1: int = 6, n_layer2: int = 4) -> Model:
    """p -> {A | B}.  Drivers: A, B."""
    rules, roles, dec, kind = {}, {}, {}, {}
    st = _switch_strengths(["A", "B"])
    _progenitor(rules, roles, dec, ["A", "B"])
    rules["A"] = "(p3 or A) and not(B)"
    rules["B"] = "(p3 or B) and not(A)"
    for g in ("A", "B"):
        roles[g], dec[g] = "driver", "D1"
    _add_markers(rules, roles, dec, "A", "D1", n_layer1, n_layer2, "mA", kind, st)
    _add_markers(rules, roles, dec, "B", "D1", n_layer1, n_layer2, "mB", kind, st)
    _nuisance(rules, roles, dec, kind)
    return Model("BF", rules, roles, dec, {"D1": ["A", "B"]}, strengths=st, kind=kind)


def trifurcation(n_layer1: int = 5, n_layer2: int = 3) -> Model:
    """p -> {A | B | C}.  Drivers: A, B, C."""
    rules, roles, dec, kind = {}, {}, {}, {}
    sw = ["A", "B", "C"]
    st = _switch_strengths(sw)
    _progenitor(rules, roles, dec, sw)
    for g in sw:
        others = [o for o in sw if o != g]
        rules[g] = f"(p3 or {g}) and not({' or '.join(others)})"
        roles[g], dec[g] = "driver", "D1"
        _add_markers(rules, roles, dec, g, "D1", n_layer1, n_layer2, f"m{g}", kind, st)
    _nuisance(rules, roles, dec, kind)
    return Model("TF", rules, roles, dec, {"D1": sw}, strengths=st, kind=kind, simulation_time=9.0)


def consecutive(n_layer1: int = 5, n_layer2: int = 3) -> Model:
    """p -> {A | B};  A -> q1 -> q2 -> {C | D}.
    Decision D1 drivers: A, B.  Decision D2 drivers: C, D (only within the A lineage).
    q1, q2 are mediators (timer downstream of A, upstream of the second switch)."""
    rules, roles, dec, kind = {}, {}, {}, {}
    _progenitor(rules, roles, dec, ["A", "B"])
    rules["A"] = "(p3 or A) and not(B)"
    rules["B"] = "(p3 or B) and not(A)"
    for g in ("A", "B"):
        roles[g], dec[g] = "driver", "D1"
    rules["q1"] = "A"
    rules["q2"] = "q1"
    for g in ("q1", "q2"):
        roles[g], dec[g] = "mediator", "D1"
    rules["C"] = "(q2 or C) and not(D)"
    rules["D"] = "(q2 or D) and not(C)"
    for g in ("C", "D"):
        roles[g], dec[g] = "driver", "D2"
    st = _switch_strengths(["A", "B"]) + [("C", "C", 10), ("D", "D", 10), ("C", "D", 10), ("D", "C", 10)]
    _add_markers(rules, roles, dec, "B", "D1", n_layer1, n_layer2, "mB", kind, st)
    _add_markers(rules, roles, dec, "C", "D2", n_layer1, n_layer2, "mC", kind, st)
    _add_markers(rules, roles, dec, "D", "D2", n_layer1, n_layer2, "mD", kind, st)
    _nuisance(rules, roles, dec, kind)
    return Model("CBF", rules, roles, dec, {"D1": ["A", "B"], "D2": ["C", "D"]},
                 strengths=st, kind=kind, simulation_time=12.0, parents={"D2": ("D1", "A")})


def hsc_curated(n_layer1: int = 4, n_layer2: int = 2) -> Model:
    """Curated hematopoietic stem cell Boolean model (Krumsiek et al. 2011, as shipped with BEELINE /
    BoolODE data/HSC.txt), wiring unchanged. Three nested decisions with literature drivers:
      D1  MEP vs GMP        : Gata1 vs Pu1
      D2  Ery vs Mk  (Gata1 lineage) : Eklf vs Fli1
      D3  Mono vs Gran (Pu1 lineage) : EgrNab vs Gfi1
    Other network genes (Gata2, Fog1, Scl, Cebpa, cJun) have mixed upstream/feedback roles and are
    labelled 'network' (excluded from the driver-vs-marker metric). Synthetic readout markers are
    attached to each lineage TF (same scheme as the synthetic networks) + the nuisance oscillator."""
    rules = {
        "Gata2": "Gata2 and (not(Pu1 or (Gata1 and Fog1)))",
        "Gata1": "(Gata1 or Gata2 or Fli1) and (not(Pu1))",
        "Fog1": "Gata1",
        "Eklf": "Gata1 and (not(Fli1))",
        "Fli1": "Gata1 and (not(Eklf))",
        "Scl": "Gata1 and (not(Pu1))",
        "Cebpa": "Cebpa and (not(Scl or (Fog1 and Gata1)))",
        "Pu1": "(Cebpa or Pu1) and (not(Gata1 or Gata2))",
        "cJun": "Pu1 and (not(Gfi1))",
        "EgrNab": "(Pu1 and cJun) and (not(Gfi1))",
        "Gfi1": "Cebpa and (not(EgrNab))",
    }
    roles = {g: "network" for g in rules}
    dec = {g: "" for g in rules}
    kind = {g: "network" for g in rules}
    st: list = []
    drivers = {"D1": ["Gata1", "Pu1"], "D2": ["Eklf", "Fli1"], "D3": ["EgrNab", "Gfi1"]}
    for d, tfs in drivers.items():
        for tf in tfs:
            roles[tf], dec[tf], kind[tf] = "driver", d, "driver"
            _add_markers(rules, roles, dec, tf, d, n_layer1, n_layer2, f"m{tf}", kind, st)
    _nuisance(rules, roles, dec, kind)
    return Model("HSC", rules, roles, dec, drivers, initial_on=["Cebpa", "Pu1", "Gata2", "c1"],
                 strengths=st, kind=kind, simulation_time=8.0,
                 parents={"D2": ("D1", "Gata1"), "D3": ("D1", "Pu1")})


MODELS = {"BF": bifurcation, "TF": trifurcation, "CBF": consecutive, "HSC": hsc_curated}
