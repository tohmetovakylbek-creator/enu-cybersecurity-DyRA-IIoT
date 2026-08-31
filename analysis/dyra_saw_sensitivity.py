#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dyra_saw_sensitivity.py -- sensitivity of DyRA-IIoT asset prioritisation to the
fuzzy SAW criterion weights.

Answers Reviewer 2, comment "The chosen weights, communication = 0.4,
hardware = 0.3, software = 0.3 ... strongly affect which assets can ever trigger
alerts. A sensitivity analysis over alternative weight profiles would show
whether the prioritization behavior is robust or mostly an artifact of the
selected weights."

The analysis is deterministic and requires no retraining. For each weight
profile it recomputes

    Impact(A) = sum_j w_j * c_Aj                                   (Eq. 4)

from the expert-defuzzified criterion scores of Table 2, and then derives the
quantities that actually govern alerting:

    gamma*(A)      = tau / Impact(A)        per-asset activation threshold
    R_max(A)       = Impact(A) * gamma_max  maximum attainable risk (at P -> 1)
    activation     per ISA-95 phase: can this asset raise an alert at all?

Outputs
-------
  * impact table per weight profile (with node ranking)
  * activation threshold table
  * phase x asset activation matrix per profile
  * a robustness verdict: which conclusions of the paper hold under every
    profile, and which node is the boundary case

Usage
-----
    python dyra_saw_sensitivity.py
    python dyra_saw_sensitivity.py --tau 0.5 --outdir tables/
    python dyra_saw_sensitivity.py --config config_saw.json --write-config config_saw.json

Author: DyRA-IIoT authors. Released under the licence of the parent repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict

# --------------------------------------------------------------------------- #
# Table 2 of the manuscript: expert-defuzzified criterion scores
# --------------------------------------------------------------------------- #

NODES = OrderedDict([
    #                        hardware  software  communication
    ("N1 Edge Gateway",     dict(hw=0.80, sw=0.70, comm=0.90)),
    ("N2 SCADA Server",     dict(hw=0.90, sw=0.90, comm=0.90)),
    ("N3 HMI Terminal",     dict(hw=0.50, sw=0.50, comm=0.70)),
    ("N4 Isolated Sensor",  dict(hw=0.20, sw=0.10, comm=0.30)),
    ("N5 Actuator PLC",     dict(hw=0.80, sw=0.70, comm=0.90)),
])

# Weight profiles: the published one plus four alternatives spanning the simplex
PROFILES = OrderedDict([
    ("published (0.3/0.3/0.4)",      dict(hw=0.3,  sw=0.3,  comm=0.4)),
    ("equal (1/3 each)",             dict(hw=1/3., sw=1/3., comm=1/3.)),
    ("communication-dominant",       dict(hw=0.25, sw=0.25, comm=0.50)),
    ("hardware-dominant",            dict(hw=0.50, sw=0.25, comm=0.25)),
    ("software-dominant",            dict(hw=0.25, sw=0.50, comm=0.25)),
])

# Table 3 of the manuscript
PHASES = OrderedDict([
    ("Scheduled maintenance", 0.4),
    ("Standby / Ready",       0.6),
    ("Standard production",   1.0),
    ("Shift change-over",     1.2),
    ("Critical window",       1.5),
])

DEFAULT_CONFIG = OrderedDict([("nodes", NODES), ("profiles", PROFILES), ("phases", PHASES)])


# --------------------------------------------------------------------------- #

def impact(scores, weights):
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1 (got %.6f)" % total_w)
    return sum(weights[k] * scores[k] for k in ("hw", "sw", "comm"))


def analyse(nodes, profiles, phases, tau):
    gamma_max = max(phases.values())
    impacts, thresholds, activation = OrderedDict(), OrderedDict(), OrderedDict()

    for pname, w in profiles.items():
        imp = OrderedDict((n, impact(s, w)) for n, s in nodes.items())
        impacts[pname] = imp
        thresholds[pname] = OrderedDict(
            (n, (tau / v if v > 0 else float("inf"))) for n, v in imp.items())
        activation[pname] = OrderedDict(
            (n, OrderedDict((ph, (imp[n] * g) > tau) for ph, g in phases.items()))
            for n in nodes
        )
    return impacts, thresholds, activation, gamma_max


def ranking(imp):
    return [n for n, _ in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)]


def fmt_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--config", help="JSON with nodes / profiles / phases")
    ap.add_argument("--write-config", metavar="PATH", help="write a template config and exit")
    ap.add_argument("--outdir", default=".", help="directory for the CSV outputs")
    a = ap.parse_args()

    if a.write_config:
        with open(a.write_config, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print("template written to %s" % a.write_config)
        return

    nodes, profiles, phases = NODES, PROFILES, PHASES
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f, object_pairs_hook=OrderedDict)
        nodes = cfg.get("nodes", nodes)
        profiles = cfg.get("profiles", profiles)
        phases = cfg.get("phases", phases)

    impacts, thresholds, activation, gamma_max = analyse(nodes, profiles, phases, a.tau)
    node_names = list(nodes.keys())

    # ---- Impact per profile -------------------------------------------------
    header = ["Weight profile"] + node_names + ["Ranking (high -> low)"]
    rows = []
    for p in profiles:
        imp = impacts[p]
        rows.append([p] + ["%.3f" % imp[n] for n in node_names]
                    + [" > ".join(n.split()[0] for n in ranking(imp))])
    print("\nImpact(A) under alternative fuzzy SAW weight profiles (tau = %.2f)\n" % a.tau)
    print(fmt_table(header, rows))
    write_csv(os.path.join(a.outdir, "tableS16a_impact_by_profile.csv"), header, rows)

    # ---- activation thresholds ---------------------------------------------
    header2 = ["Weight profile"] + ["gamma*(%s)" % n.split()[0] for n in node_names]
    rows2 = []
    for p in profiles:
        th = thresholds[p]
        rows2.append([p] + ["%.2f" % th[n] for n in node_names])
    print("\nPer-asset activation threshold gamma*(A) = tau / Impact(A). "
          "An asset can never alert if gamma*(A) > gamma_max = %.1f.\n" % gamma_max)
    print(fmt_table(header2, rows2))
    write_csv(os.path.join(a.outdir, "tableS16b_activation_thresholds.csv"), header2, rows2)

    # ---- activation matrix --------------------------------------------------
    header3 = ["Weight profile", "Asset"] + list(phases.keys())
    rows3 = []
    for p in profiles:
        for n in node_names:
            rows3.append([p, n] + ["yes" if activation[p][n][ph] else "no" for ph in phases])
    print("\nCan the asset raise an alert in this ISA-95 phase (at P(t) -> 1)?\n")
    print(fmt_table(header3, rows3))
    write_csv(os.path.join(a.outdir, "tableS16c_phase_activation.csv"), header3, rows3)

    # ---- robustness verdict -------------------------------------------------
    print("\nRobustness summary\n")
    never = [n for n in node_names
             if all(not any(activation[p][n].values()) for p in profiles)]
    always = [n for n in node_names
              if all(activation[p][n]["Standard production"] for p in profiles)]
    boundary = [n for n in node_names
                if len({tuple(activation[p][n].values()) for p in profiles}) > 1]
    rank_set = {tuple(ranking(impacts[p])) for p in profiles}

    print("  suppressed in every phase under every profile : %s"
          % (", ".join(never) if never else "none"))
    print("  alert-enabled at standard production, all profiles : %s"
          % (", ".join(always) if always else "none"))
    print("  boundary case(s) whose activation phase varies : %s"
          % (", ".join(boundary) if boundary else "none"))
    print("  distinct node rankings across profiles : %d" % len(rank_set))
    for r in rank_set:
        print("      " + " > ".join(x.split()[0] for x in r))
    print("\nCSV outputs written to %s\n" % os.path.abspath(a.outdir))


if __name__ == "__main__":
    main()
