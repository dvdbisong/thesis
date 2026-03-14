# PhD Thesis Project: Learning Automata for Adaptive Kelp Forest Monitoring

## Important Files
- **plan.md** - Implementation plan and task tracking
- **papers.md** - Literature references and paper notes

---

## Grounding Information on PhD Research

### Goal of Thesis - LA for Adaptive Kelp Monitoring
1. Adapting detection methods to diverse environmental conditions without extensive labeled data
2. Enabling effective monitoring across heterogeneous regions where single detection models have poor or significantly reduced performances
3. Improving system performance using sparse, asynchronously-arriving validation labels

### How it will work
- Global policy learning mechanism using LA schemes (the most powerful) to select optimal detection heuristics
- Context-aware local adaptation system using memory-based similarity matching
- Sparse label handling module for delayed reward propagation

### How we will evaluate
We will evaluate the framework on kelp Sentinel-2 satellite imagery. The evaluation will test adaptive monitoring performances across temporal, spatial, and environmental domains.

---

## Problem Statement

Tackling the problem of developing automated approaches to kelp forest monitoring is pertinent due to the scale and heterogeneity of British Columbia's kelp forest ecosystems, and the various limitations faced by the current methods. As an example, satellite remote sensing provides the spatial coverage that is necessary for coastline-scale assessment, with sensors such as Sentinel-2 providing 10-meter resolution imagery every 5 days (Gendall et al., 2023). However, difficulty arises when translating this imagery into reliable kelp extent maps for several interconnected reasons.

**Firstly**, the spectral signature of floating kelp canopy varies with differing environmental conditions. Factors like water turbidity, sun glint, atmospheric conditions, tide height, and kelp physiological state all influence the detectability of kelp in satellite imagery (Schroeder et al., 2020). Whereby, methods that work well under ideal conditions such as calm water, low tide, and minimal cloud cover may fail to generalize when conditions deviate. The image quality scoring framework developed by Schroeder et al. (2020) show that a significant fraction of available imagery falls below reliability thresholds, thereby limiting the temporal frequency of monitoring.

**Secondly**, given the diversity of kelp forest environments along BC's coast, it is improbable that a single detection algorithm will perform optimally everywhere (Starko et al., 2024; Mora-Soto et al., 2024). For example, bull kelp in exposed outer coastal environments will behave differently from bull kelp in protected inland waters (Schroeder et al., 2019). Further, kelp beds that are adjacent to sandy substrates have different spectral characteristics than those over rocky reef (Timmer et al., 2022). And the influence of adjacent land, river plumes, and industrial activities create site-specific detection challenges (Schroeder et al., 2020). Current approaches to kelp forest detection typically either: (a) apply a single method uniformly, accepting suboptimal performance in many areas; or (b) require extensive site-specific calibration that limits scalability (Timmer et al., 2024).

**Thirdly**, validation data for kelp detection is inherently sparse and asynchronously distributed (Cavanaugh et al., 2021b). Ground-truth surveys are expensive and logistically challenging in remote coastal areas (Gendall, 2022; Schroeder et al., 2019), and cannot possibly keep up with the pace of satellite image acquisition. This issue creates a fundamental tension where methods need feedback to improve, but feedback arrives irregularly and with significant temporal lag relative to the imagery being validated. Traditional supervised learning approaches assume dense, contemporaneous labels that are unavailable in operational monitoring contexts.

These challenges motivate the need for a fundamentally different approach to kelp forest monitoring. And one that can adapt its detection strategy based on environmental context, learn from sparse, delayed feedback and continuously improve its detection accuracy without requiring extensive retraining or manual intervention.

---

## Research Objectives

The core objective of this research is to design a Learning Automata framework for adaptive kelp forest monitoring that will address the challenges of environmental heterogeneity and sparse validation feedback. The core objective will be expressed in the following "sub"-objectives:

1. **Designing a learning automata-based heuristic selection mechanism** that maintains a probability distribution across various detection methods (i.e. heuristics). It will update this distribution based on the performance it receives from the "Environment" thus enabling the system to converge toward optimal heuristic selection for given environmental conditions.

2. **Developing a context-aware local adaptation component** that extracts environmental features from input imagery and uses memory-based similarity matching to blend global and local heuristic preferences, thereby enabling the system to generalize across the diverse environmental conditions present along BC's coastline.

3. **Implementing a sparse label handling mechanism** that maintains a buffer of predictions yet to be validated. When the ground-truth labels eventually arrive, the delayed rewards are then calculated. This process enables the system to learn from irregular feedback patterns that are characteristic of operational monitoring.

4. **Evaluating the framework** on the BC Bull Kelp dataset using Sentinel-2 imagery. The framework will assess the ability to monitor kelp forest ecosystems across temporal, spatial, and environmental domain shifts, with particular attention to performance in previously unseen conditions.

5. **Exploring integration regimes** that incorporate Traditional Ecological Knowledge (TEK) from coastal First Nations communities into the monitoring framework. This integration must institute protocols that respect indigenous data sovereignty.

---

## Research Questions

### 1.4.1 Adaptation and Generalization

**RQ1:** How can we adapt kelp segmentation to diverse environmental conditions without extensive labeled data per region?
- This question addresses the fundamental tension between the need for site-specific optimization and the difficulty of obtaining sufficient labeled data for every environmental context along BC's coastline.
- **Hypothesis:** A learning automata approach, which adapts based on sparse feedback rather than requiring dense supervision, can achieve effective adaptation with orders of magnitude fewer labels than conventional approaches.

**RQ2:** Can adaptive selection of detection methods enable effective monitoring across heterogeneous regions where single models fail?
- This question examines whether dynamic selection among a pool of diverse heuristics outperforms any fixed approach when applied across the range of environmental conditions present in BC waters.
- **Hypothesis:** Different heuristics will exhibit complementary strengths across conditions, and adaptive selection can exploit this complementarity to achieve robust performance.

### 1.4.2 Learning from Sparse Feedback

**RQ3:** How can a monitoring system improve its performance during deployment using sparse, asynchronously-arriving validation labels?
- This question addresses the operational reality that ground-truth validation arrives irregularly, perhaps within days or even months after image acquisition.
- **Hypothesis:** A properly designed sparse label handling mechanism can propagate delayed rewards effectively, thereby enabling continuous improvement despite irregular feedback.

### 1.4.3 Data Fusion

**RQ4:** How can heterogeneous data sources (10m-5km) be fused for context-aware adaptation despite resolution mismatches and missing data?

### 1.4.4 Knowledge Integration

**RQ5:** How can Traditional Ecological Knowledge from coastal First Nations be integrated into automated kelp forest monitoring?
- This question acknowledges that quantitative satellite-based monitoring represents only one way of knowing about kelp forest dynamics.
- **Hypothesis:** Indigenous knowledge systems offer complementary temporal depth, spatial resolution, and contextual understanding that can enhance automated monitoring when integrated through appropriate protocols that respect indigenous data sovereignty (Reid et al., 2021; Proulx et al., 2021).

---

## Research Gaps

1. **Environmental monitoring needs adaptation:** BC kelp forests experience heterogeneous and non-stationary conditions that pose a challenge to fixed detection methods (Starko et al., 2024; Mora-Soto et al., 2024; Hollarsmith et al., 2022). Existing approaches either tolerate suboptimal performance, or require impractical amounts of site-specific calibration. Scale-specific drivers have different impacts on kelp forest communities at different spatial scales (Lamy et al., 2018) and require adaptive approaches.

2. **Learning automata provide principled adaptation:** LA theory provides a mathematically grounded framework for learning optimal actions based on sparse feedback, with convergence guarantees under well-specified conditions (Narendra and Thathachar, 1989; Thathachar and Sastry, 2002). Yet LA has not been used in the case of remote sensing-based environmental monitoring, despite successful applications found in other adaptive systems (Ben-Zvi, 2018; Nicopolitidis et al., 2011).

3. **Sparse feedback is the operational reality:** Ground-truth validation for kelp detection comes irregularly and with delay, but most ML approaches assume dense, contemporaneous labels (Ma et al., 2019). Methods that are explicitly built for sparse feedback learning are needed. Experience replay and delayed reward propagation methods from reinforcement learning provide possible solutions (Sutton and Barto, 2018).

4. **Context-aware generalization is critical:** Because of the diversity of environmental conditions along BC's coast, detection methods must generalize across contexts (Beas-Luna et al., 2020; Schroeder et al., 2020). Global studies show that processes that are important in structuring kelp assemblages at small scales may not be generalizable to larger scales because of intrinsic spatial and temporal variability (Marzinelli et al., 2015; Underwood and Petraitis, 1993). Kelp detection challenges vary dramatically across geographic contexts such as:
   - Arctic kelp communities responding to Atlantification and UV stress (Bischof et al., 2019)
   - Australian deep-water kelp forests constrained by temperature and substrate availability (Marzinelli et al., 2015; Connell and Irving, 2008)
   - Atlantic kelp forests experiencing poleward range contractions (Voerman et al., 2013; Wernberg et al., 2012)

   These regional differences in kelp species composition, depth distributions and environmental drivers suggest that there is no universal detection model that will work optimally in all contexts. Generalized Learning Automata (GLA) provide a theoretical basis for the context-dependent action selection, but practical implementations are still limited.

5. **Indigenous knowledge underutilized:** TEK provides complementary perspectives and temporal depth that could improve automated monitoring, yet frameworks for respectful integration are still evolving. Archeological evidence reveals millennia of human-kelp interactions (Steneck et al., 2002; Jackson et al., 2001).

**This thesis addresses the gap at the intersection of these needs:** the development of a learning automata framework for adaptive kelp forest monitoring that deals with sparse feedback, is generalizable across environmental contexts, and that creates pathways for the integration of indigenous knowledge.

---

## Proposed Methodology

The proposed framework conceptualizes adaptive kelp forest monitoring as a learning automata problem in which the system has to choose from a variety of heuristics for detection based on performance feedback. The framework discusses the important research questions through three interlinked components:

1. **Global Policy Learning:** Maintains and updates a probability distribution over detection heuristics by using Learning Automata updates, which allows for convergence towards optimal heuristic selection (addresses RQ1, RQ2).

2. **Context-Aware Local Adaptation:** Extracts features of the environmental context and applies memory-based similarity matching to fuse global and local heuristic preferences to generalize in heterogeneous regions (addresses RQ2).

3. **Sparse Label Handling:** Preserves prediction buffers and backpropagates delayed rewards on receipt of validation labels, facilitating learning from the sparse feedback nature of operational monitoring (addresses RQ3).

### Formal Problem Definition

**Input space:** Let `x_t ∈ R^(B×H×W)` be a tile of a satellite image at time t, where B is the number of spectral bands, and H × W is the spatial resolution. Associated with each tile is a context vector `c_t ∈ R^D` encoding environmental conditions (e.g. sea surface temperature (SST), turbidity, solar geometry).

**Heuristic pool:** A pool of r detection heuristics `H = {h_1, h_2, ..., h_r}`, where each heuristic `h_i : x → {0, 1}^(H×W)` generates a binary segmentation map of kelp/non-kelp.

**Ground truth:** Validation labels `y_t ∈ {0, 1}^(H×W)` with label for time t arriving eventually at time `t + δ` where δ varies from days to months.

**Performance metric:** A reward function `R : (ŷ, y) → [0, 1]` that measures the quality of detection, such as Intersection over Union (IoU) or F1-score.

**Objective:** Learn a policy `π : (x, c) → Δ^(r−1)` that takes inputs and context and outputs a probability distribution over heuristics such that the expected reward is maximized:
```
π* = arg max_π E_(x,c,y)∼D [ Σ_{i=1}^r π(x, c)_i · R(h_i(x), y) ]
```

### Component 1: Global Policy Learning

The global policy maintains a probability vector `p = [p_1, p_2, ..., p_r]` over the r heuristics, which is updated by a modified Linear Reward-Inaction scheme.

- **Initialization:** Uniform distribution `p_i(0) = 1/r` for all i.
- **Action selection:** At each time step, sample heuristic `h_i` with probability `p_i` (potentially modified by context-aware blending).
- **Reward computation:** When validation label `y_t` arrives for a prediction made at time t using heuristic `h_i`:
  ```
  R_t = IoU(h_i(x_t), y_t) = |h_i(x_t) ∩ y_t| / |h_i(x_t) ∪ y_t|
  ```

**Multi-objective extension:** Beyond segmentation accuracy, the reward can incorporate additional objectives:
```
R_total = w_1 · R_IoU + w_2 · R_speed + w_3 · R_uncertainty + w_4 · R_carbon
```
where `R_speed` rewards computational efficiency, `R_uncertainty` rewards well-calibrated confidence, and `R_carbon` penalizes computational carbon footprint. Weights `w_i` can be set based on application priorities.

### Component 2: Context-Aware Local Adaptation

To enable generalization across the heterogeneous environmental conditions of BC's coastline, Component 2 implements context-dependent heuristic selection based on the principles of Generalized Learning Automata (GLA).

**Context feature extraction:** From each input tile and associated metadata, extract a D-dimensional context vector c encoding:
- Spectral statistics (mean, variance of each band)
- Spatial texture features (entropy, homogeneity)
- Temporal context (day of year, time since last valid observation)
- Environmental conditions (SST from auxiliary data, tide height, solar zenith angle)
- Geographic location (latitude, longitude, distance to shore)

**Memory bank:** Maintain a memory bank `M = {(c_j, p_j, R_j)}_{j=1}^M` of past context vectors, associated local probability distributions, and observed rewards.

**k-NN retrieval:** For new context vector c, fetch k most similar entries from M using FAISS (Facebook AI Similarity Search) for efficient approximate nearest neighbor search:
```
N_k(c) = arg-k-min_{c_j∈M} ||c − c_j||_2
```

**Local probability estimation:** Compute local heuristic preferences as the reward-weighted average of retrieved entries:
```
p_local(c) = Σ_{j∈N_k(c)} w_j · R_j · p_j / Σ_{j∈N_k(c)} w_j · R_j
```
where `w_j = exp(−||c − c_j||_2^2 / σ^2)` are distance-based weights.

**Global-local blending:** The final selection probability is a combination of the global and local estimates:
```
p_final(c) = α(c) · p_global + (1 − α(c)) · p_local(c)
```
where `α(c) ∈ [0, 1]` is an adaptive blending coefficient. When the memory bank contains few similar entries (low confidence in local estimate), α → 1 favors the global policy; when many similar entries exist with consistent performance, α → 0 favors local adaptation.

### Component 3: Sparse Label Handling

To deal with the asynchronous arrival of validation labels, Component 3 has a buffering and delayed reward propagation mechanism.

**Prediction buffer:** Maintain a FIFO buffer `B = {(t_j, x_j, c_j, i_j, ŷ_j)}` for storing timestamps, inputs, contexts, selected heuristic indices, and predictions for tiles waiting to be validated.

**Hash map lookup:** Use a hash map `L : (location, time) → buffer_index` for O(1) lookup when validation labels arrive.

**Label arrival processing:** When validation label y arrives for location ℓ and time t:
1. Look up entry in buffer using `L(ℓ,t)`
2. Compute reward `R = IoU(ŷ, y)`
3. Update Component 1 global policy with `(i, R)`
4. Update Component 2 memory bank with `(c, p_final, R)`
5. Remove entry from buffer

**Buffer management:** Entries older than a threshold `T_max` (e.g., 6 months) are removed without update, under the assumption that conditions have changed too much for the reward to be informative.

**Temporal decay (optional):** Rewards for older predictions can be discounted:
```
R_effective = R · γ^δ
```
where `δ = t_label − t_prediction` is the delay in days and `γ < 1` is a decay factor.
