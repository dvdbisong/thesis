# Paper 3: Cross-Sensor Adaptation

**Title:** Adaptive Multi-Sensor Fusion for Scalable Environmental Monitoring

**Status:** Not Started

---

## Target Venues

| Venue | Impact Factor | Deadline |
|-------|---------------|----------|
| CVPR EarthVision Workshop | Top CV | ~March |
| Remote Sensing | 5.0 | Rolling |
| ISPRS Journal | 12.7 | Rolling |

---

## Research Questions

- RQ4: How can heterogeneous data sources (10m-5km) be fused for context-aware adaptation?

---

## Key Contributions

1. LA-based domain adaptation across sensors without model retraining
2. Cross-sensor context features enabling transfer
3. RGB-only baseline methodology for fair cross-sensor comparison
4. Multi-resolution auxiliary data fusion strategy
5. Quantification of sensor-specific vs. generalizable features

---

## Components Used

- Component 2: Context-Aware Local Adaptation
- Fusion modules: multi-resolution, cross-sensor

---

## Key Results Needed

- [ ] Sentinel-2 to Landsat domain gap quantification
- [ ] Cross-sensor transfer performance (BC → Figshare)
- [ ] Context feature transferability analysis
- [ ] Early vs late fusion comparison
- [ ] Auxiliary data impact (Bathymetry, Substrate)

---

## Data Required

| Dataset | Sensor | Resolution | Status |
|---------|--------|------------|--------|
| BC Sentinel-2 | Sentinel-2 | 10m, 12 bands | Available |
| Figshare Landsat | Landsat | 30m, RGB | Available |
| Bathymetry | Auxiliary | ~30m | Available |
| Substrate | Auxiliary | ~30m | Available |

---

## Figures Planned

1. Multi-sensor architecture diagram
2. Domain gap visualization (Sentinel-2 vs Landsat features)
3. Cross-sensor transfer performance matrix
4. Context feature importance analysis
5. Fusion strategy comparison

---

## Timeline

- Experiments ready: Week 13 (after Phase 5)
- First draft: Week 14
- Submission target: CVPR 2027 EarthVision or journal

---

## Notes

- Depends on Figshare experiment results
- Critical for scalability argument (monitoring shouldn't require sensor-specific models)
- RGB-only baseline is key for fair comparison
