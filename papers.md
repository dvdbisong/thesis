# PhD Publication Plan

**Goal:** Produce high-quality publications from thesis research
**Last Updated:** March 13, 2026

---

## Publication Overview

| # | Paper | Focus | Target Venue | Priority |
|---|-------|-------|--------------|----------|
| 1 | Core Method | LA framework for adaptive model selection | Remote Sensing of Environment / IEEE TGRS | High |
| 2 | Sparse Learning | Learning from delayed, async feedback | ICML / NeurIPS | Very High |
| 3 | Cross-Sensor | Multi-sensor fusion & domain adaptation | CVPR Workshop / Remote Sensing | High |
| 4 | BC Kelp Application | Operational kelp monitoring system | Methods in Ecology and Evolution | High |
| 5 | Synthesis | Comprehensive framework for ecosystem monitoring | Nature Communications | Aspirational |

---

## Paper 1: Core Method

**Title:** *Learning Automata for Adaptive Model Selection in Environmental Monitoring*

### Research Questions
- RQ1: Adapt without extensive labeled data
- RQ2: Adaptive selection across heterogeneous regions

### Key Contributions
1. Novel LA framework for adaptive heuristic selection in remote sensing
2. Context-aware local adaptation using memory-based similarity matching
3. Global-local blending for generalization across sites

### Components Used
- Component 1: Global Policy Learning
- Component 2: Context-Aware Local Adaptation

### Key Results Needed
- [ ] LA outperforms random selection across all sites
- [ ] LA-selected models outperform any single fixed model
- [ ] Context-aware adaptation improves over global-only
- [ ] Convergence analysis for all 6 LA variants

### Data
- BC Sentinel-2 (10 sites, ~1000 tiles)
- LOSO evaluation

### Target Venues
| Venue | Impact Factor | Type |
|-------|---------------|------|
| Remote Sensing of Environment | 13.5 | Journal |
| IEEE TGRS | 8.2 | Journal |
| ISPRS Journal | 12.7 | Journal |

### Timeline
- Ready after: Phase 3 (Context-Aware Adaptation)
- Estimated: Week 10

---

## Paper 2: Sparse Learning

**Title:** *Learning from Sparse, Asynchronous Labels in Operational Monitoring Systems*

### Research Questions
- RQ3: Improve with sparse, asynchronously-arriving validation labels

### Key Contributions
1. Prediction buffer with delayed reward propagation
2. Analysis of label density vs. system performance
3. Temporal decay strategies for stale feedback
4. Theoretical analysis of convergence under sparse feedback

### Components Used
- Component 3: Sparse Label Handling

### Key Results Needed
- [ ] System maintains >80% performance with 10% labels
- [ ] Characterize minimum viable label density
- [ ] Compare delay distribution strategies
- [ ] Convergence guarantees under sparse feedback

### Data
- BC Sentinel-2 with simulated sparsity
- Multiple delay distributions (uniform, exponential, realistic)

### Target Venues
| Venue | Impact Factor | Type |
|-------|---------------|------|
| ICML | Top ML | Conference |
| NeurIPS | Top ML | Conference |
| AAAI | Top AI | Conference |
| JMLR | 6.0 | Journal |

### Timeline
- Ready after: Phase 4 (Sparse Label Handling)
- Estimated: Week 11

### Notes
- **Highest potential impact** - generalizable beyond remote sensing
- Frame as general ML contribution, not domain-specific
- Emphasize operational monitoring scenarios (any domain)

---

## Paper 3: Cross-Sensor

**Title:** *Adaptive Multi-Sensor Fusion for Scalable Environmental Monitoring*

### Research Questions
- RQ4: Fuse heterogeneous data sources (10m-5km resolution)

### Key Contributions
1. LA-based domain adaptation across sensors without retraining
2. Cross-sensor context features that enable transfer
3. RGB-only baseline for fair cross-sensor comparison
4. Multi-resolution auxiliary data fusion

### Components Used
- Component 2: Context-Aware Local Adaptation
- Fusion modules (multi-resolution, cross-sensor)

### Key Results Needed
- [ ] Quantify Sentinel-2 to Landsat domain gap
- [ ] LA adapts selection policy across sensors
- [ ] Identify which context features transfer best
- [ ] Compare early vs. late fusion strategies

### Data
- BC Sentinel-2 (primary)
- Figshare Landsat (cross-sensor)
- Auxiliary: Bathymetry, Substrate

### Target Venues
| Venue | Impact Factor | Type |
|-------|---------------|------|
| CVPR Workshop | Top CV | Workshop |
| Remote Sensing | 5.0 | Journal |
| ISPRS Journal | 12.7 | Journal |

### Timeline
- Ready after: Phase 5 (Cross-sensor evaluation)
- Estimated: Week 13

---

## Paper 4: BC Kelp Application

**Title:** *Adaptive Kelp Forest Monitoring Across British Columbia's Heterogeneous Coastline*

### Research Questions
- RQ1, RQ2: Operational monitoring across diverse conditions

### Key Contributions
1. First adaptive monitoring system for BC bull kelp
2. Site-specific analysis of 10 BC coastal regions
3. Practical recommendations for operational deployment
4. Comparison with existing monitoring approaches

### Components Used
- All components integrated

### Key Results Needed
- [ ] Per-site performance analysis
- [ ] Ecological interpretation of site differences
- [ ] Comparison with manual monitoring efforts
- [ ] Operational deployment recommendations

### Data
- BC Sentinel-2 (full dataset)
- Ground truth from Mohsen's labels

### Target Venues
| Venue | Impact Factor | Type |
|-------|---------------|------|
| Methods in Ecology and Evolution | 6.6 | Journal |
| Ecological Applications | 5.0 | Journal |
| Remote Sensing in Ecology and Conservation | 5.5 | Journal |

### Timeline
- Ready after: Phase 5 (Full integration)
- Estimated: Week 13

### Notes
- **Easiest to publish** - clear application focus
- Good for establishing domain credibility
- Can be written in parallel with method papers

---

## Paper 5: Synthesis

**Title:** *Adaptive AI for Global Ecosystem Monitoring: From Sparse Feedback to Scalable Conservation*

### Research Questions
- All RQs synthesized

### Key Contributions
1. Comprehensive framework for adaptive ecosystem monitoring
2. Cross-region validation (BC + California + ideally more)
3. Sparse feedback learning at operational scale
4. Vision for global monitoring infrastructure

### Components Used
- All components
- Extended validation across multiple regions

### Key Results Needed
- [ ] Strong results from Papers 1-4
- [ ] Cross-region generalization (BC → California)
- [ ] Compelling narrative for conservation impact
- [ ] Additional datasets if possible (Falklands, Hakai)

### Data
- BC Sentinel-2
- Figshare Landsat
- Falkland Islands (if available)
- Hakai Institute (if available)

### Target Venues
| Venue | Impact Factor | Type |
|-------|---------------|------|
| Nature Communications | 17.7 | Journal |
| Nature Sustainability | 27.2 | Journal |
| Science Advances | 14.1 | Journal |

### Timeline
- Ready after: Post-thesis or extended validation
- Estimated: 6+ months after other papers

### Notes
- **Aspirational** - requires strong results and additional data
- Consider collaborators for broader impact
- May evolve based on results from Papers 1-4

---

## Publication Strategy

### Recommended Order

```
1. Paper 4 (BC Kelp Application)
   └── Write first - establishes domain credibility
   └── Easiest path to publication
   └── Builds relationships with ecology community

2. Paper 1 (Core Method)
   └── Main thesis contribution
   └── Foundation for other papers
   └── Target RS/CS venues

3. Paper 2 (Sparse Learning)
   └── Standalone ML contribution
   └── Highest potential impact
   └── Target top ML venues

4. Paper 3 (Cross-Sensor)
   └── Depends on Figshare results
   └── Important for scalability argument

5. Paper 5 (Synthesis)
   └── After Papers 1-4 published/submitted
   └── Requires additional data for strong impact
```

### Parallel Writing Strategy

```
Weeks 10-11: Draft Paper 4 (Application) while running experiments
Weeks 12-13: Draft Paper 1 (Core Method) with full results
Week 14:     Draft Paper 2 (Sparse Learning)
Week 15:     Draft Paper 3 (Cross-Sensor) if results support
Post-thesis: Paper 5 (Synthesis) with extended validation
```

---

## Research Output Mapping

| Thesis Component | Papers |
|------------------|--------|
| Component 1: Global Policy Learning | 1, 4, 5 |
| Component 2: Context-Aware Adaptation | 1, 3, 4, 5 |
| Component 3: Sparse Label Handling | 2, 4, 5 |
| RQ4: Data Fusion | 3, 5 |
| BC Kelp Application | 4, 5 |

---

## Phase → Paper Mapping

| Phase | Primary Paper | Also Feeds |
|-------|---------------|------------|
| Phase 1: Heuristics | Paper 4 (BC app) | Paper 1 |
| Phase 2: Global LA | Paper 1 (core) | Paper 4 |
| Phase 3: Context | Paper 1 (core) | Paper 3 |
| Phase 4: Sparse | Paper 2 (sparse) | - |
| Phase 5: Cross-Sensor | Paper 3 (cross) | - |
| Phase 6: Integration | Paper 4 (BC app) | Paper 5 |

---

## Folder Structure

```
docs/papers/
├── 1_core_method/
│   ├── paper/           # LaTeX/Word documents
│   ├── figures/         # Publication figures
│   └── code/            # Paper-specific analysis
├── 2_sparse_learning/
│   ├── paper/
│   ├── figures/
│   └── code/
├── 3_cross_sensor/
│   ├── paper/
│   ├── figures/
│   └── code/
├── 4_bc_kelp_application/
│   ├── paper/
│   ├── figures/
│   └── code/
└── 5_synthesis/
    ├── paper/
    ├── figures/
    └── code/
```

---

## Success Metrics

- [ ] At least 3 papers submitted by thesis defense
- [ ] At least 1 paper in top venue (ICML/NeurIPS/Nature Comms)
- [ ] At least 1 paper in domain venue (RSE/MEE)
- [ ] Citation potential: Methods papers enable future work
