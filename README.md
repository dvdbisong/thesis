# Learning Automata for Adaptive Kelp Forest Monitoring

**PhD Thesis Research | University of Victoria**

---

## The Problem

British Columbia's kelp forests are ecological and cultural treasures facing unprecedented threats from climate change, marine heatwaves, and sea urchin predation. Satellite remote sensing offers the spatial coverage needed for coastline-scale monitoring, but **translating imagery into reliable kelp maps remains fundamentally difficult**.

### Why Current Methods Fail

1. **Environmental Heterogeneity**: A detection method that works in calm summer waters fails in turbid winter conditions. Bull kelp in exposed outer coasts behaves differently than in protected inland waters. No single algorithm works everywhere.

2. **Sparse Validation**: Ground-truth surveys are expensive and logistically challenging in remote coastal areas. Labels arrive irregularly—sometimes days, sometimes months after image acquisition. Traditional ML assumes dense, contemporaneous labels that don't exist in operational monitoring.

3. **Non-Stationarity**: The optimal detection strategy changes with seasons (kelp phenology) and years (climate regime shifts). Fixed models cannot adapt.

**The core tension**: Methods need feedback to improve, but feedback arrives sparsely and unpredictably. The environment keeps changing, but retraining is impractical.

---

## The Solution: Learning Automata Framework

I propose a **Learning Automata (LA) framework** that fundamentally reimagines kelp forest monitoring as an adaptive decision-making problem.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   INPUT TILE ──► CONTEXT ──► LA SELECTS ──► HEURISTIC ──► PREDICTION   │
│       ↑          FEATURES     HEURISTIC      EXECUTES         │        │
│       │                          ↑                            │        │
│       │                          │                            ▼        │
│       │              ┌───────────┴────────────┐         BUFFER         │
│       │              │   PROBABILITY UPDATE   │◄──── (await label)     │
│       │              │   (when label arrives) │                        │
│       │              └────────────────────────┘                        │
│       │                                                                │
│       └────────────────── NEXT TILE ◄──────────────────────────────────┘
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three Integrated Components

| Component | What It Does | Research Question |
|-----------|--------------|-------------------|
| **Global Policy Learning** | Maintains probability distribution over detection heuristics; updates based on sparse rewards | RQ1, RQ2 |
| **Context-Aware Adaptation** | Extracts environmental features; uses memory-based similarity to blend global and local preferences | RQ2, RQ4 |
| **Sparse Label Handling** | Buffers predictions; propagates delayed rewards when validation arrives | RQ3 |

---

## Novel Scientific Contributions

### 1. Dual-Layer Non-Stationarity Model

**The Insight**: Kelp monitoring exhibits TWO distinct sources of non-stationarity that classical LA theory treats separately:

```
LAYER 1: Environment Dynamics
├── PSE (Periodic): Seasonal variation in which heuristic is optimal
└── MSE (Markovian): Climate regime shifts (El Niño, marine heatwaves)

LAYER 2: Feedback Dynamics
├── PSE (Periodic): Seasonal fieldwork calendar
└── MSE (Markovian): Operational variability (funding, logistics)
```

**Why Novel**: Classical LA work on Non-Stationary Environments (MSE/PSE) focuses on environment dynamics alone. I extend this to model feedback timing as a SEPARATE MSE/PSE process, enabling analysis of learning under both changing optimality AND irregular observation.

### 2. Context-Aware Heuristic Selection via Memory Bank

**The Insight**: Instead of learning a single global policy, use k-NN retrieval over past contexts to blend global and local heuristic preferences.

```python
p_final(c) = α(c) · p_global + (1 - α(c)) · p_local(c)
```

When memory bank has few similar contexts → trust global policy
When many similar contexts with consistent performance → trust local adaptation

**Why Novel**: Generalized Learning Automata (GLA) provide theoretical basis for context-dependent action selection, but practical implementations for environmental monitoring are limited. This bridges LA theory with modern retrieval-augmented approaches.

### 3. Sparse Label Learning for Environmental Monitoring

**The Insight**: Design the system from the ground up for the reality that most predictions will never receive validation.

- Prediction buffer with O(1) lookup when labels arrive
- Temporal decay for stale predictions
- PSE+MSE simulation of realistic label arrival patterns

**Why Novel**: Most ML for remote sensing assumes dense labels. This framework treats sparsity as the default condition, not an edge case.

---

## Research Questions

### Adaptation & Generalization

**RQ1**: How can we adapt kelp segmentation to diverse environmental conditions without extensive labeled data per region?

*Hypothesis*: LA learns from sparse feedback with orders of magnitude fewer labels than conventional approaches.

**RQ2**: Can adaptive selection of detection methods enable effective monitoring across heterogeneous regions where single models fail?

*Hypothesis*: Different heuristics exhibit complementary strengths; adaptive selection exploits this complementarity.

### Learning from Sparse Feedback

**RQ3**: How can a monitoring system improve its performance during deployment using sparse, asynchronously-arriving validation labels?

*Hypothesis*: Properly designed sparse label handling enables continuous improvement despite irregular feedback.

### Data Fusion

**RQ4**: How can heterogeneous data sources (10m-5km) be fused for context-aware adaptation despite resolution mismatches?

### Knowledge Integration

**RQ5**: How can Traditional Ecological Knowledge from coastal First Nations be integrated into automated monitoring?

*Future work—requires appropriate protocols respecting indigenous data sovereignty.*

---

## Why This Matters

### Scientific Impact

- **Extends LA Theory**: Dual-layer MSE/PSE model for simultaneous environment and feedback non-stationarity
- **Bridges Communities**: Connects Learning Automata (control theory) with environmental remote sensing (Earth science)
- **Generalizable Framework**: Applicable beyond kelp to any monitoring task with sparse, delayed feedback

### Ecological Impact

- **Scale**: BC has ~25,000 km of coastline; satellite monitoring is the only feasible approach
- **Timeliness**: Adaptive methods can track changes faster than traditional survey cycles
- **Resilience**: Framework adapts to novel conditions (marine heatwaves, range shifts) without retraining

### Cultural Impact

- **First Nations Stewardship**: Kelp forests have sustained coastal communities for millennia
- **Foundation for TEK Integration**: Framework creates pathways for indigenous knowledge (RQ5)

---

## Experimental Design

### Datasets

| Dataset | Images | Sites | Labels | Purpose |
|---------|--------|-------|--------|---------|
| BC Sentinel-2 (Labeled) | 7 | 7 | Yes | Primary training/validation |
| BC Sentinel-2 (Multi-temporal) | ~100 | 7 | No | Layer 1 testing, sparse simulation |
| Figshare Landsat | 432 | California | Yes | Cross-sensor transfer |

### Heuristic Pool

```
H = {h_UWS, h_UXR, h_UXQ, h_UYQ, h_UCU, h_UXS, h_UCA, h_UWT, h_UDU, h_UUU, h_RGB}
     └──────────────── 10 site-specific 12-band models ────────────────┘  └─ cross-sensor ─┘
```

### LA Variants

- LR-I (Linear Reward-Inaction)
- LR-P (Linear Reward-Penalty)
- VSLA (Variable Structure LA)
- Pursuit
- Discretized Pursuit
- SERI (Stochastic Estimator Reward-Inaction)

### Staged Ablation Protocol

| Stage | Focus | Experiments |
|-------|-------|-------------|
| 1 | Best LA variants | 6 variants × 10 sites = 60 |
| 2 | Component contributions | 3 variants × 4 combos × 10 sites = 120 |
| 3 | Sparsity tolerance | 4 sparsity × 5 delay × 10 sites = 200 |
| 4 | Auxiliary data value | 4 configs × 10 sites = 40 |
| 5 | Cross-sensor transfer | Focused set |

---

## Project Structure

```
thesis/
├── data/
│   ├── bc_sentinel2/                 # Labeled (Mohsen preprocessed)
│   ├── bc_sentinel2_multitemporal/   # Unlabeled seasonal
│   └── figshare_landsat/             # Cross-sensor
├── models/
│   ├── site_specific/                # 10 LOO models
│   └── rgb/                          # Cross-sensor model
├── src/
│   ├── la_framework/                 # Core LA algorithms
│   ├── heuristics/                   # Model wrappers
│   ├── preprocessing/                # Data pipelines
│   ├── context/                      # Feature extraction
│   ├── experiments/                  # Experiment runners
│   └── analysis/                     # Results analysis
├── results/
│   ├── raw/                          # JSON experiment outputs
│   └── papers/                       # Publication figures
├── CLAUDE.md                         # Full research context
├── plan.md                           # Implementation plan
└── README.md                         # This file
```

---

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Complete thesis grounding (problem, objectives, methodology) |
| `plan.md` | Detailed implementation plan with code specifications |
| `papers.md` | Publication strategy |

---

## Quick Reference: The Elevator Pitch

> **Problem**: Kelp forest monitoring using satellite imagery fails because (1) environmental heterogeneity defeats single-method approaches, (2) validation data is sparse and delayed, and (3) optimal detection strategies change over time.
>
> **Solution**: A Learning Automata framework that adaptively selects among diverse detection heuristics based on environmental context, learns from sparse asynchronous feedback, and handles dual non-stationarity in both environment dynamics and feedback timing.
>
> **Contribution**: Novel extension of LA theory to model simultaneous environment and feedback non-stationarity; practical framework for adaptive environmental monitoring under realistic operational constraints.
>
> **Impact**: Enables scalable, adaptive kelp forest monitoring for BC's 25,000 km coastline, supporting conservation efforts and First Nations stewardship.

---

## References

Key literature grounding this work:

- Narendra & Thathachar (1989). *Learning Automata: An Introduction*
- Thathachar & Sastry (2002). *Varieties of Learning Automata*
- Schroeder et al. (2020). Image quality scoring for kelp detection
- Starko et al. (2024). BC kelp forest heterogeneity
- Mora-Soto et al. (2024). Global kelp detection challenges

---

*Last Updated: March 14, 2026*
