"""Random nested lineage trees for the depth experiment (17 Nov 2026).

Tree growth: the root progenitor always splits into two fates (depth 1). Every new fate then splits again with
probability p, up to a maximum depth of 4. The depth of a network is the number of splits on its deepest path:
root -> split = depth 1; root -> split -> one child splits again = depth 2, etc.
A network of depth D is drawn by growing a tree with split probability p, capped at depth D, and keeping it only if
at least one path reaches depth D (rejection sampling). This gives the same number of networks at every depth;
p sets how bushy they are (p = 0.8: most fates split again; p = 0.5: about half do).

Wiring of every split is the same as in the consecutive-bifurcation model (networks.consecutive):
  * the root split is driven by the progenitor timer p1 -> p2 -> p3, as in BF;
  * a nested split below fate X is driven by a mediator timer X -> qX1 -> qX2, as q1 -> q2 in CBF;
  * each split has two mutually inhibiting, self-activating switch TFs = ground-truth DRIVERS of that decision;
  * every switch TF has its own downstream MARKERS (layer 1 fast + slow, layer 2, repressed), as in all models.
Fates are named by their path: root children Xa, Xb; children of Xa are Xaa, Xab; ...
"""
from __future__ import annotations
import numpy as np
from networks import Model, _add_markers, _nuisance, _progenitor, _switch_strengths

MAX_DEPTH = 4


def sample_topology(p: float, rng: np.random.Generator, max_depth: int = MAX_DEPTH) -> list[str]:
    """Return the list of fates that split (internal nodes), root = 'X'. Root always splits."""
    internal, frontier = ["X"], ["Xa", "Xb"]
    while frontier:
        f = frontier.pop(0)
        depth = len(f) - 1                       # 'Xa' has depth 1 (it is a child of the first split)
        if depth < max_depth and rng.random() < p:
            internal.append(f)
            frontier += [f + "a", f + "b"]
    return internal


def tree_depth(internal: list[str]) -> int:
    return max(len(f) for f in internal)        # 'X' -> 1 split, 'Xa' splitting -> 2, ...


def draw_tree(p: float, depth: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    while True:
        t = sample_topology(p, rng, max_depth=depth)    # grow with p, capped at the requested depth ...
        if tree_depth(t) == depth:                        # ... and keep it only if some path reaches that depth
            return t


def tree_model(internal: list[str], name: str, n_layer1: int = 4, n_layer2: int = 2) -> Model:
    rules, roles, dec, kind = {}, {}, {}, {}
    st: list = []
    fate_genes, parents = {}, {}
    internal = sorted(internal, key=lambda f: (len(f), f))          # parents before children (simulate.py needs this)
    dname = {f: f"D{i + 1}" for i, f in enumerate(internal)}
    for f in internal:
        d, a, b = dname[f], f + "a", f + "b"
        if f == "X":
            _progenitor(rules, roles, dec, [a, b])
            trig = "p3"
            st += _switch_strengths([a, b])
        else:
            q1, q2 = "q" + f + "1", "q" + f + "2"
            rules[q1], rules[q2] = f, q1
            pdec = dname[f[:-1]]
            for g in (q1, q2):
                roles[g], dec[g], kind[g] = "mediator", pdec, "mediator"
            trig = q2
            st += [(a, a, 10), (b, b, 10), (a, b, 10), (b, a, 10)]
            parents[d] = (pdec, f)
        rules[a] = f"({trig} or {a}) and not({b})"
        rules[b] = f"({trig} or {b}) and not({a})"
        for g in (a, b):
            roles[g], dec[g], kind[g] = "driver", d, "driver"
            _add_markers(rules, roles, dec, g, d, n_layer1, n_layer2, f"m{g}", kind, st)
        fate_genes[d] = [a, b]
    _nuisance(rules, roles, dec, kind)
    depth = tree_depth(internal)
    return Model(name, rules, roles, dec, fate_genes, strengths=st, kind=kind,
                 simulation_time=8.0 + 4.0 * (depth - 1), parents=parents)


if __name__ == "__main__":
    for p in (0.8, 0.5):
        for d in range(1, 5):
            t = draw_tree(p, d, seed=d)
            m = tree_model(t, f"T{d}")
            print(p, d, "splits:", len(t), "genes:", len(m.rules), t)
