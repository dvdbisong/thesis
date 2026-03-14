# Paper 1: Core Method

**Title:** Learning Automata for Adaptive Model Selection in Environmental Monitoring

**Status:** Not Started

---

## Target Venues

| Venue | Impact Factor | Deadline |
|-------|---------------|----------|
| Remote Sensing of Environment | 13.5 | Rolling |
| IEEE TGRS | 8.2 | Rolling |
| ISPRS Journal | 12.7 | Rolling |

---

## Research Questions

- RQ1: How can we adapt kelp segmentation to diverse environmental conditions without extensive labeled data per region?
- RQ2: Can adaptive selection of detection methods enable effective monitoring across heterogeneous regions?

---

## Key Contributions

1. Novel LA framework for adaptive heuristic selection in remote sensing
2. Context-aware local adaptation using memory-based similarity matching
3. Global-local blending for generalization across sites
4. Empirical comparison of 6 LA variants on real-world monitoring task

---

## Components Used

- Component 1: Global Policy Learning (all 6 LA variants)
- Component 2: Context-Aware Local Adaptation (memory bank, k-NN)

---

## Key Results Needed

- [ ] LA convergence analysis (all 6 variants)
- [ ] LA vs baselines (random, oracle, fixed-best)
- [ ] Context-aware vs global-only comparison
- [ ] LOSO evaluation across 10 BC sites
- [ ] Heuristic probability evolution visualization

---

## Figures Planned

1. System architecture diagram
2. LA convergence curves (all variants)
3. LOSO performance heatmap
4. Context embedding visualization (t-SNE/UMAP)
5. Heuristic selection patterns per site

---

## Timeline

- Experiments ready: Week 10 (after Phase 3)
- First draft: Week 12
- Submission target: TBD

---

## Notes

- Main thesis contribution
- Foundation for Papers 2-5
