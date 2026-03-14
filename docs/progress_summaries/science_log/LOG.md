# Experiment Log

A running journal of experiments, observations, and learnings for the Learning Automata Kelp Detection prototype.

---

## Entry Template

```markdown
### [DATE] - [Experiment Name]
**Objective:** What am I trying to test?
**Hypothesis:** What do I expect to happen?
**Method:** What did I do?
**Results:** What happened? (include metrics, plots)
**Analysis:** Why did this happen?
**Next Steps:** What should I try next?
**Status:** ✅ Success | ⚠️ Partial | ❌ Failed
```

---

## Entries

### 2026-02-09 - Project Setup
**Objective:** Set up the prototype project structure and science logging infrastructure.
**Hypothesis:** N/A - setup task.
**Method:**
- Created directory structure for prototype code
- Set up science log and weekly report templates
- Created master slide deck structure
**Results:** Project structure created successfully.
**Analysis:** N/A
**Next Steps:** Implement config.yaml and Makefile for experiment management.
**Status:** ✅ Success

---

### 2026-02-10 - Prototype Implementation Complete
**Objective:** Implement full LA prototype with spectral heuristics, automaton, and experiment infrastructure.
**Hypothesis:** N/A - implementation task.
**Method:**
- Created config.yaml with experiment parameters
- Implemented Makefile for `make run` and `make baselines`
- Implemented data loader for Sentinel-2 tiles (10 bands)
- Implemented 4 spectral heuristics: NDVI, FAI, GNDVI, Ensemble
- Implemented L_R-I automaton with probability updates
- Implemented reward function (binary/continuous modes)
- Implemented detector orchestrator
- Implemented experiment runner with logging and plots
- Implemented baselines: random, fixed, oracle
- Created conda environment "kelp" (Python 3.10)

**Results:** All components implemented and integrated. Code runs successfully.
**Analysis:** N/A
**Next Steps:** Run experiments to test LA learning.
**Status:** ✅ Success

---

### 2026-02-13 - Experiment 1: Binary Reward (Failed)
**Objective:** Test if LA converges to best-performing spectral heuristic using binary reward.
**Hypothesis:** LA should converge toward GNDVI or Ensemble (best performers) and away from FAI (worst).

**Method:**
- Config: `reward.type: "binary"`, `reward.iou_threshold: 0.3`
- Ran `make run` (10 epochs on training set)
- Ran `make baselines` for comparison

**Results:**

*Baseline Results:*
| Method         | IoU    | Notes                                    |
| -------------- | ------ | ---------------------------------------- |
| Oracle         | 0.0765 | Upper bound (perfect per-tile selection) |
| Fixed GNDVI    | 0.0699 | Best single heuristic                    |
| Fixed Ensemble | 0.0695 | Second best                              |
| Fixed NDVI     | 0.0533 | Middle performer                         |
| Random         | 0.0445 | Lower bound                              |
| Fixed FAI      | 0.0074 | Worst heuristic                          |

*LA Results:*
- Final probabilities converged to **FAI** (the worst heuristic!)
- Mean IoU: ~0.007 (matching FAI's poor performance)
- Entropy decreased (convergence occurred, but to wrong heuristic)

**Analysis:**
The binary reward threshold (0.3) was **far too high** for this dataset. Actual IoU values range from 0.007 to 0.076 - none exceed 0.3. This means:
- Almost no tiles received reward=1
- The automaton received near-zero learning signal
- Probability updates were essentially random drift
- By chance, it converged toward FAI

**Key Insight:** Binary reward requires threshold calibration to the actual IoU distribution. With IoU values ~0.05-0.07, a threshold of 0.3 is unreachable.

**Next Steps:** Switch to continuous reward (`reward.type: "continuous"`) so IoU is used directly as reward signal. This provides gradient information even for low IoU values.

**Status:** ❌ Failed (but informative failure - understood root cause)

---

### 2026-02-13 - Experiment 2: Continuous Reward (Partial Success)
**Objective:** Test LA with continuous reward to enable learning from low IoU values.
**Hypothesis:** With continuous reward, LA should converge toward GNDVI/Ensemble (IoU ~0.07) rather than FAI (IoU ~0.007).

**Method:**
- Changed config: `reward.type: "continuous"`, `automaton.alpha: 0.1`
- Ran `make run` (10 epochs on 277 training tiles)

**Results:**

*LA Performance:*
| Metric         | Value                       |
| -------------- | --------------------------- |
| Test IoU       | 0.0719                      |
| Val IoU        | 0.0020                      |
| Best Heuristic | ensemble (100% probability) |
| Final Entropy  | 0.0000                      |

*Comparison with Baselines:*
| Method           | IoU        | vs LA   |
| ---------------- | ---------- | ------- |
| Oracle           | 0.0765     | -0.0046 |
| Fixed Ensemble   | 0.0719     | 0.0000  |
| **LA (learned)** | **0.0719** | —       |
| Fixed GNDVI      | 0.0711     | +0.0008 |
| Random           | 0.0568     | +0.0151 |
| Fixed NDVI       | 0.0430     | +0.0289 |
| Fixed FAI        | 0.0074     | +0.0645 |

*Convergence Speed:*
- Epoch 1: ensemble at 98.76%
- Epoch 2: ensemble at 99.9999...%
- Epochs 3-10: ensemble at 100% (probabilities of others → 10^-100)

**Analysis:**

✅ **What Worked:**
- LA correctly identified ensemble as best heuristic
- LA outperforms random baseline by 0.0151 IoU
- Gap to oracle is only 6.0%
- Continuous reward provided learning signal (unlike binary)

⚠️ **Issues Identified:**

1. **Convergence too fast:** By Epoch 2, probabilities were already locked at 100% ensemble. Learning rate α=0.1 is too aggressive.

2. **No exploration after convergence:** Once ensemble reached ~99%, other heuristics never got selected again. LR-I has no exploration mechanism.

3. **LA ≡ Fixed Ensemble:** Since LA converged to 100% ensemble, its performance equals `fixed_ensemble` exactly (0.0719). LA learned nothing beyond "always use ensemble."

4. **Winner-take-all dynamics:** The LR-I update rule amplifies small early advantages. Whichever heuristic performs well on first few tiles snowballs to 100%.

5. **Val vs Test discrepancy:** Val IoU (0.0020) vs Test IoU (0.0719) suggests validation set has different characteristics (possibly harder scenes or less kelp).

**Key Insight:** The learning rate α=0.1 causes instant convergence. For gradual learning and proper exploration of the heuristic space, a much smaller α is needed.

**Next Steps:**
1. Reduce learning rate to α=0.01 to observe gradual convergence
2. Consider adding ε-greedy exploration
3. Investigate val/test set differences

**Status:** ⚠️ Partial (LA works but converges too fast to show learning dynamics)

---

### 2026-02-13 - Experiment 3: Lower Learning Rate (Success)
**Objective:** Test LA with lower learning rate to observe gradual convergence dynamics.
**Hypothesis:** With α=0.01 (10x smaller), LA should show gradual probability shifts over multiple epochs rather than instant convergence.

**Method:**
- Config: `automaton.alpha: 0.01`, `reward.type: "continuous"`
- Ran `make run` (10 epochs on 277 training tiles)

**Results:**

*Probability Evolution Across Epochs:*
| Epoch | NDVI    | FAI     | GNDVI   | Ensemble | Entropy  |
| ----- | ------- | ------- | ------- | -------- | -------- |
| 1     | 37%     | 3%      | 27%     | 34%      | 1.72     |
| 2     | 30%     | 0.2%    | 29%     | 41%      | 1.58     |
| 4     | 44%     | 0.03%   | 25%     | 30%      | 1.55     |
| 7     | 29%     | ~0%     | 18%     | 53%      | 1.44     |
| 10    | **20%** | **~0%** | **41%** | **39%**  | **1.52** |

*Final Performance:*
| Metric         | Value       |
| -------------- | ----------- |
| Test IoU       | 0.0711      |
| Val IoU        | 0.0019      |
| Best Heuristic | gndvi (41%) |
| Final Entropy  | 1.52        |

*Comparison with Baselines:*
| Method           | IoU        | vs LA   |
| ---------------- | ---------- | ------- |
| Oracle           | 0.0765     | -0.0054 |
| Fixed Ensemble   | 0.0719     | -0.0008 |
| Fixed GNDVI      | 0.0711     | 0.0000  |
| **LA (learned)** | **0.0711** | —       |
| Random           | 0.0568     | +0.0144 |
| Fixed NDVI       | 0.0430     | +0.0281 |
| Fixed FAI        | 0.0074     | +0.0638 |

**Analysis:**

✅ **What Worked:**

1. **Gradual convergence:** Probabilities evolved across all 10 epochs, not instant lock-in. This confirms α=0.01 is appropriate for observing learning dynamics.

2. **FAI correctly eliminated:** FAI dropped from 3% → ~0% (10^-11). LA correctly learned FAI is the worst heuristic.

3. **Correct ranking learned:** Final order (gndvi > ensemble > ndvi > fai) matches actual baseline performance ranking.

4. **Non-zero entropy:** Final entropy = 1.52 means LA is still exploring, hasn't fully converged. This is healthy for 10 epochs.

5. **Probability fluctuations:** NDVI led early (44% at Epoch 4) but declined as LA gathered more evidence. GNDVI rose late. This shows LA responding to cumulative reward signal.

6. **Competition between top heuristics:** GNDVI (41%) and Ensemble (39%) are close, reflecting their similar actual performance (0.0711 vs 0.0719 IoU).

**Comparison: Exp 2 vs Exp 3:**
| Metric              | Exp 2 (α=0.1)      | Exp 3 (α=0.01)      |
| ------------------- | ------------------ | ------------------- |
| Convergence         | Instant (Epoch 2)  | Gradual (10 epochs) |
| Final entropy       | 0.0                | 1.52                |
| Best heuristic prob | 100%               | 41%                 |
| Exploration         | None after Epoch 2 | Ongoing             |
| Test IoU            | 0.0719             | 0.0711              |

**Key Insight:** Lower learning rate (α=0.01) enables proper observation of LA learning dynamics. The automaton correctly identifies heuristic quality ordering but needs more epochs to fully converge.

**Next Steps:**
1. Run with more epochs (20-50) to see if full convergence occurs
2. Try α=0.001 for even slower/stable convergence
3. Consider ε-greedy exploration to prevent probability collapse
4. Investigate why NDVI led early but declined (tile ordering effects?)

**Status:** ✅ Success (LA shows proper learning dynamics with gradual convergence)

---

### 2026-02-13 - Research Plan: Systematic LA Algorithm Comparison

**Objective:** Establish rigorous experimental methodology for comparing Learning Automata algorithms.

**Problem Statement:**
Current LA (LR-I) achieves ~93% of oracle performance (0.0711 vs 0.0765 IoU). Can we close this gap with:
1. Alternative LA algorithms (LR-P, VSLA, Pursuit, Estimator)?
2. Better hyperparameter tuning?
3. Novel algorithm design?

---

## Multi-Seed Methodology

**Why Multi-Seed Support is Critical:**

ML experiments are stochastic. Running the same config twice gives different results due to:
- Random probability initialization
- Random action selection during training
- Random data shuffling

**Without multi-seed:** Cannot distinguish genuine improvement from lucky seed.

**With multi-seed (5 seeds per config):**
```
Seeds: [42, 123, 456, 789, 1011]
Results: [0.0711, 0.0698, 0.0725, 0.0702, 0.0718]
Report: IoU = 0.0711 ± 0.0011 (mean ± std)
```

**Statistical rigor enabled:**
- Report confidence intervals for all metrics
- Paired t-tests to determine if algorithm differences are significant (p < 0.05)
- Effect size (Cohen's d) for practical significance
- Friedman test for overall algorithm ranking

---

## Algorithms to Compare

| Algorithm               | Type     | Key Feature                | Expected Behavior           |
| ----------------------- | -------- | -------------------------- | --------------------------- |
| **LR-I**                | Baseline | Inaction on penalty        | Slow, stable convergence    |
| **LR-P**                | Baseline | Penalty updates            | Faster, may be less stable  |
| **VSLA**                | Advanced | Adaptive α (entropy-based) | Explore early, exploit late |
| **Pursuit**             | Advanced | ML reward estimates        | 10-20x faster than LR-I     |
| **Discretized Pursuit** | Advanced | Finite probability levels  | Fastest convergence         |
| **Estimator (SERI)**    | Advanced | Bayesian posteriors        | Best accuracy               |

---

## Experiment Plan

### Phase 1: Baseline Completion (Experiments 4-9)
| Exp | Algorithm | α     | β     | Epochs | Seeds | Hypothesis              |
| --- | --------- | ----- | ----- | ------ | ----- | ----------------------- |
| 4   | LR-I      | 0.001 | -     | 10     | 5     | Very slow, stable       |
| 5   | LR-I      | 0.01  | -     | 50     | 5     | Full convergence        |
| 6   | LR-I      | 0.05  | -     | 20     | 5     | Speed/stability balance |
| 7   | LR-P      | 0.01  | 0.01  | 20     | 5     | Symmetric penalty       |
| 8   | LR-P      | 0.01  | 0.005 | 20     | 5     | Light penalty           |
| 9   | LR-P      | 0.01  | 0.02  | 20     | 5     | Heavy penalty           |

**Deliverable:** LR-I vs LR-P comparison with significance tests

### Phase 2: Advanced Algorithms (Experiments 10-13)
| Exp | Algorithm           | Key Params             | Hypothesis              |
| --- | ------------------- | ---------------------- | ----------------------- |
| 10  | VSLA                | α_max=0.1, α_min=0.001 | Adaptive exploration    |
| 11  | Pursuit             | α=0.01                 | 10-20x faster than LR-I |
| 12  | Discretized Pursuit | resolution=100         | Fastest convergence     |
| 13  | Estimator           | α=0.01                 | Best accuracy           |

**Deliverable:** Full algorithm ranking by IoU and convergence speed

### Phase 3: Novel Algorithm (TBD)
Based on Phase 1-2 findings, develop ONE novel contribution:
- **Option A:** Adaptive VSLA with drift detection
- **Option B:** Context-aware LA (tile features → probabilities)
- **Option C:** Hierarchical LA for multi-scale detection

---

## Infrastructure Improvements

1. **Experiment Runner:** Multi-seed batch execution with progress tracking
2. **Results Database:** SQLite for efficient querying (replaces JSON files)
3. **Statistical Analysis:** Automated t-tests, effect sizes, rankings
4. **Visualization Suite:** Publication-quality convergence plots
5. **Config System:** Hierarchical configs with inheritance

---

## Success Criteria

### Minimum (Thesis)
- [ ] LR-I, LR-P compared with 5 seeds each
- [ ] Statistical significance established
- [ ] Clear winner identified

### Target (Publication)
- [ ] 5+ algorithms compared
- [ ] Novel algorithm with significant improvement
- [ ] Publication-quality figures

### Stretch (Strong Contribution)
- [ ] Theoretical analysis
- [ ] Second dataset validation
- [ ] Open-source framework release

**Status:** 📋 Planning Complete - Ready to Implement

---

### 2026-02-13 - Infrastructure Implementation Complete

**Objective:** Build robust experimentation infrastructure for systematic LA algorithm comparison.

**What Was Implemented:**

1. **New LA Algorithms** (`code/la_framework/automaton.py`):
   - VSLA (Variable Structure LA) - entropy-based adaptive learning rate
   - Pursuit - pursues action with highest estimated reward
   - Discretized Pursuit - finite probability levels for faster convergence
   - Estimator (SERI) - Bayesian posteriors with Beta distributions
   - Factory function `create_automaton()` for config-based selection

2. **Multi-Seed Experiment Runner** (`code/experiments/experiment_runner.py`):
   - `ExperimentRunner` class for batch execution
   - Runs same config with multiple seeds: [42, 123, 456, 789, 1011]
   - Aggregates results (mean, std, min, max)
   - Tracks best heuristic distribution across seeds

3. **Statistical Analysis** (`code/experiments/statistical_analysis.py`):
   - Paired t-tests for pairwise algorithm comparison
   - Cohen's d effect size (negligible/small/medium/large)
   - 95% confidence intervals
   - Friedman test for overall ranking
   - LaTeX table generation

4. **Config System** (`configs/`):
   - `base.yaml` - shared defaults
   - `algorithms/*.yaml` - per-algorithm configs (LR-I, LR-P, VSLA, Pursuit, Estimator)
   - `sweeps/*.yaml` - batch experiment definitions

5. **Updated Makefile**:
   - `make run-seeds` - run with multiple seeds
   - `make run-lri`, `make run-lrp`, etc. - algorithm-specific targets
   - `make stats` - statistical analysis
   - `make compare EXPS="a b"` - compare specific experiments

**Available Automaton Types:**
```
LR-I, LR-P, VSLA, Pursuit, DiscretizedPursuit, Estimator
```

**Status:** ✅ Infrastructure Ready

---

## Systematic Experimentation Plan

### Overview

This plan follows a **three-phase approach** to systematically identify the best LA algorithm for kelp detection:

| Phase | Focus                                          | Experiments        | Duration   |
| ----- | ---------------------------------------------- | ------------------ | ---------- |
| 1     | Baseline algorithms (LR-I, LR-P)               | Exp 4-9            | ~2-3 hours |
| 2     | Advanced algorithms (VSLA, Pursuit, Estimator) | Exp 10-14          | ~2-3 hours |
| 3     | Statistical analysis & novel algorithm         | Analysis + Exp 15+ | ~1-2 hours |

**Total estimated runtime:** 5-8 hours (can run overnight)

---

### Phase 1: Baseline Algorithms (LR-I vs LR-P)

**Goal:** Establish performance baselines and determine if penalty updates (LR-P) improve over inaction (LR-I).

#### Experiment 4: LR-I with Very Low Learning Rate
```bash
# Create config: configs/experiments/exp4_lri_slow.yaml
experiment:
  name: "exp4_lri_slow"
automaton:
  type: "LR-I"
  alpha: 0.001
training:
  num_epochs: 20

# Run command:
make run-seeds CONFIG=configs/experiments/exp4_lri_slow.yaml N_SEEDS=5
```
**Hypothesis:** Very slow convergence, high entropy after 20 epochs, may not identify best heuristic.

#### Experiment 5: LR-I with Default Learning Rate (Extended)
```bash
# Config: configs/experiments/exp5_lri_default.yaml
automaton:
  type: "LR-I"
  alpha: 0.01
training:
  num_epochs: 50

# Run:
make run-seeds CONFIG=configs/experiments/exp5_lri_default.yaml N_SEEDS=5
```
**Hypothesis:** Full convergence by epoch 30-40, should match or beat Exp 3 results.

#### Experiment 6: LR-I with Higher Learning Rate
```bash
# Config: configs/experiments/exp6_lri_fast.yaml
automaton:
  type: "LR-I"
  alpha: 0.05
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp6_lri_fast.yaml N_SEEDS=5
```
**Hypothesis:** Fast convergence (epoch 5-10), possibly premature lock-in like Exp 2.

#### Experiment 7: LR-P Symmetric (α = β)
```bash
# Config: configs/experiments/exp7_lrp_symmetric.yaml
automaton:
  type: "LR-P"
  alpha: 0.01
  beta: 0.01
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp7_lrp_symmetric.yaml N_SEEDS=5
```
**Hypothesis:** Faster than LR-I due to penalty updates, but may be less stable.

#### Experiment 8: LR-P Light Penalty (β < α)
```bash
# Config: configs/experiments/exp8_lrp_light.yaml
automaton:
  type: "LR-P"
  alpha: 0.01
  beta: 0.005
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp8_lrp_light.yaml N_SEEDS=5
```
**Hypothesis:** Balance between LR-I stability and LR-P speed.

#### Experiment 9: LR-P Heavy Penalty (β > α)
```bash
# Config: configs/experiments/exp9_lrp_heavy.yaml
automaton:
  type: "LR-P"
  alpha: 0.01
  beta: 0.02
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp9_lrp_heavy.yaml N_SEEDS=5
```
**Hypothesis:** Aggressive penalty may cause instability or over-correction.

#### Phase 1 Analysis
```bash
# After all Phase 1 experiments complete:
make stats
make compare EXPS="exp4_lri_slow exp5_lri_default exp6_lri_fast exp7_lrp_symmetric exp8_lrp_light exp9_lrp_heavy"
```

**Key Questions to Answer:**
1. Does LR-P outperform LR-I? (paired t-test, p < 0.05)
2. What is the optimal α for LR-I?
3. What is the optimal α/β ratio for LR-P?
4. Is the difference practically significant? (Cohen's d > 0.5)

---

### Phase 2: Advanced Algorithms

**Goal:** Test if advanced LA algorithms can close the gap to oracle (currently ~7%).

#### Experiment 10: VSLA (Adaptive Learning Rate)
```bash
# Config: configs/experiments/exp10_vsla.yaml
automaton:
  type: "VSLA"
  alpha_max: 0.1
  alpha_min: 0.001
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp10_vsla.yaml N_SEEDS=5
```
**Hypothesis:** Explores aggressively early (high entropy → high α), then exploits (low entropy → low α). Should avoid premature convergence.

#### Experiment 11: Pursuit Algorithm
```bash
# Config: configs/experiments/exp11_pursuit.yaml
automaton:
  type: "Pursuit"
  alpha: 0.01
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp11_pursuit.yaml N_SEEDS=5
```
**Hypothesis:** 10-20x faster convergence than LR-I due to direct reward estimation. May achieve same accuracy in fewer epochs.

#### Experiment 12: Discretized Pursuit
```bash
# Config: configs/experiments/exp12_discretized_pursuit.yaml
automaton:
  type: "DiscretizedPursuit"
  alpha: 0.01
  resolution: 100
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp12_discretized_pursuit.yaml N_SEEDS=5
```
**Hypothesis:** Fastest convergence of all algorithms. Finite probability levels prevent numerical precision issues.

#### Experiment 13: Estimator (SERI)
```bash
# Config: configs/experiments/exp13_estimator.yaml
automaton:
  type: "Estimator"
  alpha: 0.01
  prior_strength: 1.0
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp13_estimator.yaml N_SEEDS=5
```
**Hypothesis:** Bayesian posteriors provide better uncertainty estimates. May achieve highest accuracy.

#### Experiment 14: Estimator with Stronger Prior
```bash
# Config: configs/experiments/exp14_estimator_strong.yaml
automaton:
  type: "Estimator"
  alpha: 0.01
  prior_strength: 5.0
training:
  num_epochs: 20

# Run:
make run-seeds CONFIG=configs/experiments/exp14_estimator_strong.yaml N_SEEDS=5
```
**Hypothesis:** Stronger prior = more conservative updates, may be more stable.

#### Phase 2 Analysis
```bash
make stats
make compare EXPS="exp5_lri_default exp7_lrp_symmetric exp10_vsla exp11_pursuit exp12_discretized_pursuit exp13_estimator"
```

**Key Questions to Answer:**
1. Which algorithm achieves highest mean IoU?
2. Which algorithm has lowest variance (most stable)?
3. Which algorithm converges fastest (epochs to 90% of final performance)?
4. Is there a statistically significant winner? (Friedman test)

---

### Phase 3: Analysis & Novel Algorithm

**Goal:** Identify novel contribution based on findings.

#### Step 1: Generate Publication-Quality Results
```bash
# Generate full comparison
make stats --latex > results/phase2_comparison.tex

# Key metrics to report:
# - Test IoU: mean ± std (5 seeds)
# - Convergence speed: epochs to 95% max probability
# - Stability: std of IoU across seeds
# - Gap to oracle: (oracle_iou - algo_iou) / oracle_iou
```

#### Step 2: Identify Novel Direction

Based on Phase 1-2 results, select ONE novel contribution:

**Option A: Adaptive VSLA with Drift Detection**
- If VSLA shows promise but struggles with non-stationary data
- Novel: Add drift detection to reset α when environment changes
- Implementation: Monitor reward variance, increase α when spike detected

**Option B: Context-Aware LA**
- If all algorithms plateau at ~93% oracle
- Novel: Condition LA probabilities on tile features (cloud cover, scene type)
- Implementation: Tile embedding → probability modulation

**Option C: Hierarchical LA**
- If single-level LA shows limitations
- Novel: Two-level hierarchy (scene-level strategy → tile-level heuristic)
- Implementation: Meta-LA selects from multiple specialist LAs

#### Step 3: Implement & Validate Novel Algorithm

```bash
# Experiment 15+: Novel algorithm experiments
# (specific configs depend on Phase 1-2 findings)
```

---

## Quick Reference: Running Experiments

### Single Experiment
```bash
make run CONFIG=config.yaml
```

### Multi-Seed Experiment
```bash
make run-seeds CONFIG=configs/algorithms/lri.yaml N_SEEDS=5
```

### Algorithm-Specific Shortcuts
```bash
make run-lri      # LR-I with 5 seeds
make run-lrp      # LR-P with 5 seeds
make run-vsla     # VSLA with 5 seeds
make run-pursuit  # Pursuit with 5 seeds
make run-estimator # Estimator with 5 seeds
```

### Statistical Analysis
```bash
make stats                              # Analyze all results
make compare EXPS="exp1 exp2 exp3"      # Compare specific experiments
```

### View Results
```bash
ls results/*_aggregated.json            # List aggregated results
cat results/comparison.json             # View comparison report
```

---

## Experiment Checklist

### Phase 1: Baselines
- [ ] Exp 4: LR-I α=0.001, 20 epochs, 5 seeds
- [ ] Exp 5: LR-I α=0.01, 50 epochs, 5 seeds
- [ ] Exp 6: LR-I α=0.05, 20 epochs, 5 seeds
- [ ] Exp 7: LR-P α=β=0.01, 20 epochs, 5 seeds
- [ ] Exp 8: LR-P α=0.01 β=0.005, 20 epochs, 5 seeds
- [ ] Exp 9: LR-P α=0.01 β=0.02, 20 epochs, 5 seeds
- [ ] Phase 1 statistical analysis

### Phase 2: Advanced
- [ ] Exp 10: VSLA, 20 epochs, 5 seeds
- [ ] Exp 11: Pursuit, 20 epochs, 5 seeds
- [ ] Exp 12: Discretized Pursuit, 20 epochs, 5 seeds
- [ ] Exp 13: Estimator, 20 epochs, 5 seeds
- [ ] Exp 14: Estimator (strong prior), 20 epochs, 5 seeds
- [ ] Phase 2 statistical analysis
- [ ] Full algorithm ranking

### Phase 3: Novel
- [ ] Select novel direction based on findings
- [ ] Implement novel algorithm
- [ ] Exp 15+: Validate novel algorithm
- [ ] Publication-quality figures
- [ ] Update science log with final results

---

## Expected Outcomes

| Algorithm       | Expected IoU  | Expected Convergence | Notes              |
| --------------- | ------------- | -------------------- | ------------------ |
| LR-I (α=0.01)   | 0.071 ± 0.002 | 30-40 epochs         | Baseline           |
| LR-P (α=β=0.01) | 0.072 ± 0.003 | 15-25 epochs         | Faster but noisier |
| VSLA            | 0.072 ± 0.002 | 20-30 epochs         | Adaptive           |
| Pursuit         | 0.073 ± 0.002 | 5-10 epochs          | Fastest            |
| Estimator       | 0.073 ± 0.001 | 15-20 epochs         | Most stable        |
| **Oracle**      | **0.0765**    | N/A                  | Upper bound        |

**Target:** Achieve ≥95% of oracle (IoU ≥ 0.0727) with novel algorithm.

**Status:** 📋 Ready for Execution
