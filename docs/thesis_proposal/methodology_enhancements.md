# Methodology Enhancements Log

## Overview
This document tracks opportunities to enhance the proposed methodology identified during literature review.

---

## Enhancement Categories

### Component 1: Global Policy Learning (LELA)
- Potential improvements to learning rate adaptation
- Alternative reinforcement schemes (pursuit, estimator)
- Convergence acceleration techniques

### Component 2: Context-Aware Local Adaptation
- Alternative similarity metrics
- Learned feature representations
- Attention mechanisms for neighbor weighting

### Component 3: Sparse Label Handling
- Experience replay strategies
- Importance sampling for delayed rewards
- Active learning integration

### Heuristic Pool Design
- Additional spectral indices
- Transformer architectures
- Dynamic pool modification strategies

### Evaluation Metrics
- Ecological relevance metrics
- Temporal consistency measures
- Uncertainty calibration

---

## Enhancements Identified

### Enhancement 1: Pursuit Algorithms for Faster Convergence
**Source:** Narendra & Thathachar (1989), Thathachar & Sastry (1986)
**Component Affected:** Component 1 (Global Policy Learning)
**Current Limitation:** L_R-I scheme converges slowly, especially with many actions
**Proposed Integration:** Implement pursuit variant that estimates reward probabilities and moves action probability vector toward estimated optimal action
**Expected Improvement:** Significantly faster convergence (order of magnitude improvement reported in literature)
**Implementation Complexity:** Medium - requires maintaining reward estimates for each heuristic

### Enhancement 2: Estimator Algorithms for Non-stationary Environments
**Source:** Narendra & Thathachar (1989), Chapter 7
**Component Affected:** Component 1 (Global Policy Learning)
**Current Limitation:** Basic L_R-I assumes stationary environment; kelp conditions vary seasonally
**Proposed Integration:** Maintain running estimates of heuristic performance; adapt learning rate based on detected non-stationarity
**Expected Improvement:** Better tracking of seasonal changes in optimal heuristic selection
**Implementation Complexity:** Medium - requires change detection mechanism

### Enhancement 3: Context Vector Learning (Validates Component 2)
**Source:** Narendra & Thathachar (1989), Chapter 7
**Component Affected:** Component 2 (Context-Aware Local Adaptation)
**Current Limitation:** Standard LA lacks context awareness
**Proposed Integration:** Already incorporated - memory bank + k-NN approach implements context-aware learning
**Expected Improvement:** N/A - validates existing design
**Implementation Complexity:** Already designed

### Enhancement 4: Hierarchical Learning Automata
**Source:** Narendra & Thathachar (1989), Section 7.6
**Component Affected:** Potential new component
**Current Limitation:** Single-level heuristic selection may not capture multi-scale patterns
**Proposed Integration:** Consider two-level hierarchy: (1) select heuristic tier, (2) select specific heuristic within tier
**Expected Improvement:** Faster convergence by first eliminating poor-performing tiers
**Implementation Complexity:** High - significant architectural change

### Enhancement 5: Generalized Learning Automata (GLA) Formulation
**Source:** Thathachar & Sastry (2002)
**Component Affected:** Component 2 (Context-Aware Local Adaptation)
**Current Limitation:** Current k-NN blending is heuristic; lacks formal LA convergence guarantees
**Proposed Integration:** Formalize Component 2 as GLA where action probabilities are g(x, y_i, u) - dependent on both context features x and internal state u
**Expected Improvement:** Formal convergence guarantees; unified LA framework across all components
**Implementation Complexity:** Medium - requires reformulating probability generating function

### Enhancement 6: Modules of LA for Parallel Processing
**Source:** Thathachar & Sastry (2002), Section VI
**Component Affected:** Component 1 (Global Policy Learning)
**Current Limitation:** Sequential processing of tiles may be slow for real-time monitoring
**Proposed Integration:** Use n parallel LA with shared action probability vector; fuser combines responses; speed scales linearly with module size
**Expected Improvement:** n-fold speedup without sacrificing accuracy (λ̃ = λ/n controls accuracy)
**Implementation Complexity:** Medium - requires parallel processing infrastructure

### Enhancement 7: PLA for Global Optimization
**Source:** Thathachar & Sastry (2002), Section III
**Component Affected:** Component 1 (Global Policy Learning)
**Current Limitation:** L_R-I may converge to local optima in heuristic selection
**Proposed Integration:** Use PLA with probability generating function p_ij = exp(u_ij)/Σ exp(u_il) and Langevin-type perturbations
**Expected Improvement:** Guaranteed convergence to global maximum of expected reinforcement
**Implementation Complexity:** High - requires significant algorithm redesign

### Enhancement 8: LRP Scheme Instead of L_R-I
**Source:** Savargiv et al. (2021)
**Component Affected:** Component 1 (Global Policy Learning)
**Current Limitation:** L_R-I uses only rewards (b=0); may not adapt quickly to changing conditions
**Proposed Integration:** Consider LRP scheme (a=b) where actions are both rewarded and penalized; Savargiv found a=0.5, b=0.5 optimal
**Expected Improvement:** Friedman test showed LRP (a=0.5, b=0.5) ranked #1 vs L_R-I ranked #12-13
**Implementation Complexity:** Low - only requires setting b=a in existing update equations
**Trade-off:** L_R-I has absorbing states (converges permanently); LRP remains adaptive but may oscillate

### Enhancement 9: LA-Based Ensemble Weighting
**Source:** Savargiv et al. (2021)
**Component Affected:** Heuristic Pool Design
**Current Limitation:** Single heuristic selection may discard useful information from other heuristics
**Proposed Integration:** Use LA to learn weights for ensemble combination of multiple heuristics instead of selecting one
**Expected Improvement:** Combines strengths of multiple heuristics; more robust to individual heuristic failures
**Implementation Complexity:** Medium - requires output fusion mechanism

### Enhancement 10: Noise Tolerance Evaluation
**Source:** Savargiv et al. (2021), Cuevas et al. (2011)
**Component Affected:** Component 3 (Sparse Label Handling)
**Current Limitation:** Unknown robustness to label noise in kelp monitoring
**Proposed Integration:** Evaluate LELA performance with 20%, 40% label noise injection
**Expected Improvement:** LA demonstrated 20% noise robustness (Savargiv); CARLA worked with 40% label noise (Cuevas)
**Implementation Complexity:** Low - evaluation only

### Enhancement 11: Multi-Driver Context Features for BC Heterogeneity
**Source:** Starko et al. (2024)
**Component Affected:** Component 2 (Context-Aware Local Adaptation)
**Current Limitation:** Current context features may not capture full range of BC kelp drivers
**Proposed Integration:** Expand 32D context vector to include:
- Sea surface temperature (SST) from LiveOcean/MUR products
- Current velocity (from oceanographic models)
- Fetch/wave exposure (DFO provincial model)
- Trophic indicators (sea otter presence, urchin density proxies)
- Climatic oscillator indices (PDO, ENSO, ONI)
**Expected Improvement:** Better adaptation to spatially-varying drivers (temperature in south, urchins in north)
**Implementation Complexity:** Medium - requires environmental data integration

### Enhancement 12: Regime Shift Detection
**Source:** Starko et al. (2024), Filbee-Dexter & Wernberg (2018)
**Component Affected:** Heuristic Pool Design + Reward Function
**Current Limitation:** Current metrics focus on kelp detection accuracy, not state transitions
**Proposed Integration:** Add regime shift indicators:
- Urchin barren detection heuristics (coralline algae spectral signatures)
- State transition probability estimation
- Multi-year persistence metrics instead of single-timepoint accuracy
**Expected Improvement:** System can detect and flag ecosystem state changes, not just kelp presence
**Implementation Complexity:** High - requires new heuristic types and reward formulations

### Enhancement 13: Heterogeneity-Aware Evaluation Protocol
**Source:** Starko et al. (2024)
**Component Affected:** Experimental Design (Section 3.5)
**Current Limitation:** Standard evaluation may not reveal performance across BC's diverse conditions
**Proposed Integration:** Stratified evaluation across:
- Temperature regimes (warm inner seas vs cool outer coasts)
- Trophic contexts (sea otter present/absent areas)
- Fetch levels (exposed vs sheltered coastlines)
- Driver dominance (temperature-driven vs urchin-driven decline areas)
**Expected Improvement:** More nuanced understanding of where adaptive system provides value
**Implementation Complexity:** Medium - requires spatial stratification of test data

### Template for Each Enhancement:
```
## Enhancement: [Name]
**Source:** [Paper citation]
**Component Affected:** [Component 1/2/3 or Heuristic Pool]
**Current Limitation:** [What limitation does this address?]
**Proposed Integration:** [How to incorporate into methodology]
**Expected Improvement:** [Quantitative or qualitative benefit]
**Implementation Complexity:** [Low/Medium/High]
```
