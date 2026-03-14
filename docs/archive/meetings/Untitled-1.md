
  Methodology Structure (4 Phases, 17 Steps)

  Phase 1: Data Preparation

  | Step | Task                                                     | Output                 |
  | ---- | -------------------------------------------------------- | ---------------------- |
  | 1.1  | Sentinel-2 image acquisition via GEE                     | /data/sentinel2/tiles/ |
  | 1.2  | Ground truth compilation (ShoreZone, WorldView, in-situ) | /data/labels/          |
  | 1.3  | Environmental context feature extraction (~40 dims)      | /data/context/         |
  | 1.4  | Dataset stratification and train/val/test splits         | /data/splits/          |

  Phase 2: Heuristic Pool Development

  | Step | Task                                                     | Output                         |
  | ---- | -------------------------------------------------------- | ------------------------------ |
  | 2.1  | Tier 1: Spectral index heuristics (H1–H4)                | /code/heuristics/spectral/     |
  | 2.2  | Tier 2: ML heuristics (H5–H8: RF, XGBoost, SVM, K-means) | /code/heuristics/ml/           |
  | 2.3  | Tier 3: DL heuristics (H9–H12: U-Net variants, DeepLab)  | /code/heuristics/dl/           |
  | 2.4  | Heuristic pool validation and complementarity analysis   | /results/heuristic_benchmarks/ |

  Phase 3: LA Framework Implementation

  | Step | Task                                          | Output                |
  | ---- | --------------------------------------------- | --------------------- |
  | 3.1  | Problem formalization                         | Documentation         |
  | 3.2  | Component 1: Global policy (L_R-I update)     | global_policy.py      |
  | 3.3  | Component 2: Context-aware adaptation (FAISS) | context_adaptation.py |
  | 3.4  | Inference pipeline integration                | detector.py           |

  Phase 4: Experimental Validation

  | Step | Task                                            | Output               |
  | ---- | ----------------------------------------------- | -------------------- |
  | 4.1  | Baseline implementations (7 methods)            | /code/baselines/     |
  | 4.2  | Protocol A: Temporal adaptation (2017–2024)     | /results/protocol_a/ |
  | 4.3  | Protocol B: Spatial adaptation (South→North BC) | /results/protocol_b/ |
  | 4.4  | Ablation studies (8 ablations)                  | /results/ablations/  |
  | 4.5  | Statistical analysis and figures                | /figures/            |

  The appendix now includes:
  - Appendix A: Implementation checklist with checkboxes
  - Appendix B: Proposed code directory structure
  - Appendix C: Figures/tables checklist
  - Appendix D: Writing progress tracker

  The PDF compiled successfully (28 pages). You can begin with Phase 1 (Data Preparation) immediately.