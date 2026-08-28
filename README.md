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
