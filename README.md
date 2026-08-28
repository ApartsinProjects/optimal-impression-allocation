# Optimal Impression Allocation

A technical report on a two-stage architecture for revenue-optimal ad impression allocation in operator-run mobile advertising: offline predictive modeling and linear-programming allocation planning, online table-driven decisioning with pacing, and a closed feedback loop.

- **Read the paper (HTML):** https://apartsinprojects.github.io/optimal-impression-allocation/
- **PDF:** https://apartsinprojects.github.io/optimal-impression-allocation/paper.pdf

## Contents

| File | Description |
|---|---|
| `index.html` | The paper (GitHub Pages index) |
| `paper.pdf` | PDF rendering of the paper |
| `alloc_sim.py` | Monte-Carlo simulation comparing greedy serving with LP-planned allocation (Section 7 of the paper) |

## Reproducing the simulation

```bash
pip install numpy scipy
python alloc_sim.py
```

Twenty replications of a synthetic market (60 audience segments, 25 campaigns, ~200,000 impressions per run). The planned policy achieves a mean 6.5% revenue lift and 33% expected-click lift over greedy serving on identical event streams, capturing 94% of the clairvoyant offline optimum. The harness verifies both policies against the clairvoyant bound in every replication and includes a degenerate uniform-probability control whose measured lift is zero.

## Real-log replay (Section 8)

`h2_replay.py` reconstructs an allocation market from the public Taobao display-ad dataset (Tianchi dataset 56 / Kaggle mirror) and replays greedy, LP-planned, and dual-price pacing policies under walk-forward dual-model scoring; `h2_errorbars.py` and `h2_wf_analyze.py` compute the closed-form error bars and fold-combined statistics reported in the paper.
