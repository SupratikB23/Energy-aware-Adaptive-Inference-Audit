"""Read-only loaders for the Phase 3 result JSONs. Nothing here creates or edits numbers.

Every value plotted comes straight from a result file; derived quantities (ratios to the
full model, propagated intervals) are computed here and nowhere else.
"""
import json
import math
import os

# Display names; keys are the policy ids written by src/e2e_policy.py.
CIFAR_POLICIES = [
    ("conf-0.97", r"Conf. $\tau$=0.97"),
    ("conf-0.99", r"Conf. $\tau$=0.99 (LTT)"),
    ("perhead-0.999/0.995/0.95", "Per-head P1"),
    ("perhead-0.99/0.98/0.7", "Per-head P2"),
    ("perhead-0.93/0.85/0.6", "Per-head P3"),
]
HAR_POLICIES = [
    ("conf-0.97", r"Conf. $\tau$=0.97"),
    ("conf-0.99", r"Conf. $\tau$=0.99"),
    ("perhead-0.6/0.6", "Per-head 0.6 (LTT)"),
]
SHORT = {
    "conf-0.97": r"$\tau$.97", "conf-0.99": r"$\tau$.99",
    "perhead-0.999/0.995/0.95": "P1", "perhead-0.99/0.98/0.7": "P2",
    "perhead-0.93/0.85/0.6": "P3", "perhead-0.6/0.6": "LTT",
}


class Results:
    def __init__(self, root):
        self.root = root
        self.p3 = os.path.join(root, "p3")
        self.unlocked = os.path.join(root, "p3_unlocked")
        if not os.path.isdir(self.p3):
            raise FileNotFoundError("expected Phase 3 results at %s" % self.p3)

    def _load(self, *parts):
        with open(os.path.join(*parts)) as f:
            return json.load(f)

    # ---- end to end ----
    def e2e(self, dataset, session):
        name = "e2e_cifar10_resnet14_s42_%s.json" if dataset == "cifar10" else "e2e_har_harcnn_s42_%s.json"
        return self._load(self.p3, name % session)["rows"]

    @staticmethod
    def row(rows, policy, batch, mode="compact"):
        for r in rows:
            if r["policy"] == policy and r["batch"] == batch and r["mode"] == mode:
                return r
        return None

    def full(self, rows, batch):
        """Full-model row at this batch size (always measured in compaction mode)."""
        return self.row(rows, "full", batch, "compact")

    @staticmethod
    def ratio_ci(r, f):
        """Measured energy relative to the full model, with a propagated 95% half-width
        (independent windows; first-order delta method on the two Student-t intervals)."""
        q = r["j_per_sample"] / f["j_per_sample"]
        rel = math.hypot(r["j_ci95_halfwidth"] / r["j_per_sample"],
                         f["j_ci95_halfwidth"] / f["j_per_sample"])
        return q, q * rel

    # ---- per exit ----
    def eval(self, dataset, seed=42, unlocked=False):
        tag = "cifar10_resnet14" if dataset == "cifar10" else "har_harcnn"
        if unlocked:
            return self._load(self.unlocked, "eval_%s_s%d_U.json" % (tag, seed))
        return self._load(self.p3, "eval_%s_s%d_A.json" % (tag, seed))

    @staticmethod
    def batch_row(ev, batch):
        for b in ev["batch_rows"]:
            if b["batch"] == batch:
                return b
        return None

    @staticmethod
    def windows_by_exit(energy):
        """Per-window marginal J/sample grouped by exit index."""
        out = {}
        for w in energy["windows"]:
            out.setdefault(w["exit"], []).append(w["per_sample_marginal_j"])
        return out

    # ---- LTT and audit ----
    def ltt(self, dataset, seed):
        tag = "cifar10_resnet14" if dataset == "cifar10" else "har_harcnn"
        return self._load(self.p3, "ltt_%s_s%d.json" % (tag, seed))

    def audit_compare(self):
        return self._load(self.p3, "audit_compare.json")
