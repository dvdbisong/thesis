# PDF Summaries for Thesis Proposal

## Overview
This document contains summaries of all PDFs analyzed for the thesis proposal:
"Learning Automata Framework for Adaptive Kelp Forest Monitoring in British Columbia Coastal Waters"

---

## Summary Format
For each PDF:
- **Citation:** Author(s) (Year). Title. Journal/Venue.
- **Key Findings:** Main results and conclusions
- **Methodology:** Methods used
- **Relevance:** How it relates to thesis RQs
- **Notable Citations:** Secondary references worth following

---

## Learning Automata Papers

### 1. Narendra & Thathachar (1989) - Learning Automata: An Introduction [FOUNDATIONAL]

**Citation:** Narendra, K.S. and Thathachar, M.A.L. (1989). *Learning Automata: An Introduction*. Prentice Hall, Englewood Cliffs, New Jersey.

**Key Concepts:**
- **Learning Automaton Definition:** Stochastic automaton that interacts with an environment and updates its action probabilities based on feedback
- **Environment Model:** Random environment characterized by penalty probabilities c_i for each action α_i
- **Reinforcement Schemes:**
  - **L_R-P (Linear Reward-Penalty):** Updates probabilities on both favorable and unfavorable responses
  - **L_R-I (Linear Reward-Inaction):** Only updates on favorable responses; has absorbing states
  - **L_R-εP (Linear Reward-ε-Penalty):** Hybrid approach with small penalty
- **Multi-action L_R-I Update Rules (Eq. 4.27 with b=0):**
  - When action α_i selected and β=0 (reward): p_i(n+1) = p_i(n) + a[1-p_i(n)]
  - When action α_i selected and β=0: p_j(n+1) = (1-a)p_j(n) for j≠i
  - When β=1 (penalty): p(n+1) = p(n) (no change - inaction)
- **Convergence:** L_R-I converges to absorbing states (unit vectors) with probability 1
- **ε-optimality:** Probability of converging to optimal action → 1 as learning rate a → 0

**Advanced Topics:**
- **Pursuit algorithms:** Faster convergence by moving toward estimated optimal action (Thathachar & Sastry, 1986)
- **Estimator algorithms:** Estimate environment parameters for improved performance
- **Nonstationary environments:** Periodic, slowly-varying, and state-dependent
- **Context-aware learning:** Different optimal actions for different regions in context space
- **Q- and S-models:** Generalized environments with continuous reward values

**Relevance to Thesis:**
- **RQ1 (Adaptation):** L_R-I scheme provides foundation for adaptive heuristic selection
- **RQ2 (Heterogeneous regions):** Context space learning directly supports Component 2
- **RQ3 (Sparse labels):** Inaction property handles missing feedback naturally
- **Methodology:** Equations 1-6 in thesis based on Chapter 4 formulations

**Methodology Enhancements Identified:**
1. **Pursuit algorithms** - Could accelerate convergence in stable conditions
2. **Context vector approach** - Validates Component 2 design
3. **Estimator algorithms** - Could improve accuracy with sufficient data
4. **Hierarchical structures** - Potential for multi-scale adaptation

**Notable Secondary References:**
- Bush & Mosteller (1958) - Mathematical psychology foundations
- Norman (1968) - L_R-I in psychology
- Shapiro & Narendra (1969) - L_R-I engineering introduction
- Thathachar & Sastry (1986) - Pursuit algorithms
- Thathachar & Oommen (1979), Oommen (1986) - Discretized schemes

---

### 2. Thathachar & Sastry (2002) - Varieties of Learning Automata: An Overview

**Citation:** Thathachar, M.A.L. and Sastry, P.S. (2002). Varieties of Learning Automata: An Overview. *IEEE Transactions on Systems, Man, and Cybernetics—Part B: Cybernetics*, 32(6), 711-722.

**LA Types Described:**

1. **FALA (Finite Action-Set LA):** Traditional model with discrete actions
   - ε-optimal algorithms: L_R-I, estimator algorithms, pursuit algorithms
   - Teams of FALA for common payoff games (converge to Nash equilibrium/modes)
   - Networks of FALA for pattern recognition (feedforward architecture)

2. **PLA (Parameterized LA):** Overcomes local maximum limitation
   - Parameterizes action probabilities with real numbers
   - Uses probability generating function: p_ij = exp(u_ij) / Σ_l exp(u_il)
   - Simulated annealing-type perturbations for global optimization
   - Converges to global maximum via Langevin equation analysis

3. **GLA (Generalized LA):** **CRITICAL FOR THESIS COMPONENT 2**
   - Handles context vector input directly
   - Action probabilities depend on BOTH state u AND context vector x
   - Prob[α(k) = y_i | u, x] = g(x, y_i, u)
   - Ideal for associative reinforcement learning problems
   - Different optimal actions for different context regions

4. **CALA (Continuous Action-Set LA):**
   - Action-set is the real line (continuous parameters)
   - Uses normal distribution N(μ(k), σ(k))
   - Updates both mean and standard deviation
   - No need to discretize parameter space

5. **Modules of LA:** Parallel operation for faster convergence
   - n LA operating in parallel with shared action probability vector
   - Speed increases linearly with module size n
   - Accuracy controlled by λ̃ = λ/n (normalized learning parameter)

**Key Theorems/Results:**
- L_R-I in common payoff game converges to mode of reward probability matrix
- Estimator algorithms can achieve global maximum but with memory overhead
- Network of GLA using REINFORCE-type algorithm converges to local maximum
- Modules achieve faster convergence without sacrificing accuracy

**Relevance to Thesis:**
- **GLA validates Component 2 design** - context-dependent action selection
- **Modules** could enhance convergence speed (methodology enhancement)
- **Teams/Networks** provide framework for multi-heuristic selection
- **Pattern recognition application** directly analogous to kelp segmentation

**Methodology Enhancements:**
1. **GLA formulation** - Could formalize Component 2 as GLA instead of k-NN blending
2. **Modules for parallel operation** - Faster convergence for real-time monitoring
3. **PLA for global optimization** - Escape local optima in heuristic selection

**Notable Secondary References:**
- Barto & Anandan (1985) - Pattern-recognizing stochastic LA
- Williams (1992) - REINFORCE algorithm
- Sutton & Barto (1998) - Reinforcement Learning textbook

---

### 3. Cuevas et al. (2011) - LA for Multi-Threshold Image Segmentation [DIRECTLY RELEVANT]

**Citation:** Cuevas, E., Zaldivar, D., and Pérez-Cisneros, M. (2011). Seeking multi-thresholds for image segmentation with Learning Automata. *Machine Vision and Applications*, 22(5), 805-818.

**Problem:** Automatic multi-threshold image segmentation using Gaussian mixture model (GMM) approximation of image histograms.

**Method - CARLA (Continuous Action Reinforcement LA):**
- Approximates 1-D histogram with K Gaussian functions: p(x) = Σ P_i · N(μ_i, σ_i)
- Each Gaussian represents one pixel class → one threshold point
- Parameters to optimize: P_i (probability), μ_i (mean), σ_i (std) for each class
- Uses **team of 12 automata** in parallel (3 parameters × 4 classes)
- Automata coupled ONLY through environment (objective function)

**CARLA Update Rule:**
- Continuous probability density function f(x,n) replaces discrete probabilities
- Update: f(x, n+1) = α · [f(x,n) + β(n) · H(x,r)]
- H(x,r) = Gaussian neighborhood function centered on selected action r
- β(n) = max{0, (J_med - J(n))/(J_med - J_min)} - reinforcement based on median/min costs

**Key Advantages Over EM and Levenberg-Marquardt:**
1. **Insensitive to initial conditions** - searches in probability space, not parameter space
2. **Avoids local minima** - stochastic search allows jumps to distant regions
3. **Fast convergence** - ~1000 iterations vs 1855 (EM) or 985 (LM)
4. **Handles singularities** - works with near-zero variance and overlapped Gaussians
5. **Lower computational cost** - 1.48s vs 2.73s (EM) or 4.98s (LM)

**Experimental Results:**
- 4-class segmentation on benchmark images
- LA produces consistent results regardless of initialization
- EM fails to converge with 20% label noise; LA works with 40% noise
- LA better able to reach global minimum

**Relevance to Thesis:**
- **DIRECT PRECEDENT** for LA in image segmentation
- Team of automata approach mirrors heuristic pool design
- Histogram-based thresholding analogous to spectral index thresholds
- Robustness to noise relevant to sparse/delayed labels
- Could apply CARLA for continuous spectral threshold optimization

**Methodology Enhancements:**
1. **CARLA for spectral indices** - Optimize continuous threshold values (NIR, NDVI, FAI)
2. **Parallel automata architecture** - One automaton per heuristic parameter
3. **Histogram-based kelp detection** - GMM on spectral histograms for kelp/non-kelp separation

**Notable Secondary References:**
- Najim & Poznyak (1994) - LA theory and applications textbook
- Howell & Gordon (2001) - CARLA for adaptive digital filter design
- Zeng et al. (2005) - LA for continuous complex function optimization

---

### 4. Savargiv et al. (2021) - Random Forest Algorithm Based on Learning Automata

**Citation:** Savargiv, M., Masoumi, B., and Keyvanpour, M.R. (2021). A New Random Forest Algorithm Based on Learning Automata. *Computational Intelligence and Neuroscience*, 2021, Article ID 5572781, 19 pages.

**Problem:** Classical random forest lacks adaptability to dynamic environments where data exhibits different behaviors in different domains.

**Method - LA-Enhanced Random Forest:**
- Each decision tree (base learner) becomes an LA action
- LA selects which base learner to use for each input sample
- Environment = pool of trained decision trees
- Feedback: correct prediction → reward; incorrect → penalty
- Initial action probabilities: p(DT_r) = 1/R (uniform)
- Process continues until convergence or iteration limit

**LA Update Rules (standard formulation):**
- Reward (β=0): p_i(n+1) = p_i(n) + a[1-p_i(n)]; p_j(n+1) = (1-a)p_j(n)
- Penalty (β=1): p_j(n+1) = (1-b)p_i(n); p_j(n+1) = b/(r-1) + (1-b)p_j(n)

**Modes Evaluated:**
1. **LRI (b=0):** Only rewards, no penalties - absorbing states
2. **LRεP (b << a):** Small penalty, larger reward
3. **LRP (a=b):** Equal reward and penalty - **BEST PERFORMANCE**

**Key Results:**
- **LRP with a=0.5, b=0.5 achieved highest accuracy** (Mean Rank 19.17 in Friedman test)
- Outperformed: Averaging, Majority Voting, standard Random Forest
- **Noise Robustness:** Maintains performance with 20% noise injection
- Convergence: ~400-600 iterations with optimal parameters
- Evaluated on 18 datasets across text, healthcare, physical, and sound domains

**Advantages Claimed:**
1. Adaptability to dynamic problem conditions
2. Domain independence
3. Global optima guaranteed through LA search
4. Simple feedback requirements
5. Low computational complexity

**Relevance to Thesis:**
- **DIRECTLY ANALOGOUS** to heuristic selection in LELA framework
- LA actions = base learners ↔ LA actions = detection heuristics
- Validates LA for classifier/algorithm selection
- **Important finding:** LRP (equal a,b) outperforms L_R-I
- Noise robustness (20%) relevant to sparse/noisy labels in kelp monitoring
- Dynamic environment handling → seasonal kelp variations

**Methodology Enhancements:**
1. **Consider LRP scheme instead of L_R-I** - evidence suggests balanced reward/penalty may outperform inaction-only
2. **Ensemble weighting via LA** - could weight multiple heuristic outputs rather than selecting one
3. **Noise tolerance validation** - evaluate LELA robustness to label noise (20-40%)

**Notable Secondary References:**
- Rezvanian et al. (2018) - Recent Advances in Learning Automata (textbook)
- Goodwin & Yazidi (2020) - Distributed LA for classification with pursuit scheme

---

### 5. Ben-Zvi (2018) - Learning Automata for Sensor Placement [ENVIRONMENTAL APPLICATION]

**Citation:** Ben-Zvi, T. (2018). Learning automata decision analysis for sensor placement. *Journal of the Operational Research Society*, 69(9), 1396-1405.

**Problem:** Design sensor systems that respond to environmental changes. Detect moving targets (intruders, marine mammals) where environmental factors (currents) affect target appearance probability.

**Key Innovation - Dual-Sensor System:**
1. **Environmental sensors:** Detect elements in environment (current speeds)
2. **Target sensors:** Detect the actual target
3. Use environmental sensor data to position target sensors dynamically

**LA Framework:**
- **States:** Possible sensor positions
- **Actions:** Permitted movements of each sensor
- **Reward function:** Feedback from environment (higher current speed → lower intruder probability → higher reward for positioning elsewhere)
- **Multiple LA simultaneously:** Extended Gosavi's (2014) algorithm to run multiple automata in parallel

**Algorithm (modified Gosavi 2014):**
- Initialize: P(i, u_i) = 1/|A_i| (uniform over actions)
- Compute: β(i) = R(i)/T(i) (reward/time ratio)
- Update if same action: P(i, u_i) ← P(i, u_i) + β(i)[1-P(i, u_i)]
- Update if different action: P(i, u_i) ← P(i, u_i) - β(i)[1-P(i, u_i)]/(|A_i|-1)
- Terminate when probabilities converge to 0 or 1

**Key Results:**
- **28% improvement:** LA achieved 0.8179 detection probability vs 0.6362 for evenly spread sensors
- **Consistent gap:** ~18 percentage points across 1-20 sensors
- **Dynamic adaptation:** Superior performance over 24-hour varying conditions
- **Statistical significance:** 95% confidence level
- Application: Hudson River maritime monitoring using NYHOPS real environmental data

**Key Contributions:**
1. Extended LA to allow >2 actions per state
2. Execute LA on multiple automata simultaneously
3. Employ real environmental data for maritime sensor optimization

**Relevance to Thesis:**
- **DIRECTLY APPLICABLE** to adaptive kelp monitoring architecture
- **Dual-sensor concept:** Environmental context (ocean conditions, turbidity, cloud cover) + detection (kelp heuristics)
- **Multiple simultaneous LA:** One per region/tile in BC coastal waters
- **Environmental correlates:** Use easily-observed conditions as proxy for detection probability
- **Dynamic coastal adaptation:** Currents, tides, seasonal changes affect kelp detection

**Methodology Enhancements:**
1. **Dual-sensing architecture** - separate environmental context features from detection heuristics
2. **Simultaneous LA execution** - extends single-automaton to multi-region parallel processing
3. **Environmental proxy learning** - learn correlations between observable conditions and detection success

**Notable Secondary References:**
- Gosavi (2014) - Simulation-based optimization (original algorithm)
- Wheeler & Narendra (1986) - ε-optimality convergence proofs
- Thathachar & Sastry (2004) - Networks of learning automata

---

### 6. Nicopolitidis et al. (2011) - Adaptive Wireless Networks Using Learning Automata [SURVEY]

**Citation:** Nicopolitidis, P., Papadimitriou, G.I., Pomportsis, A.S., Sarigiannidis, P., and Obaidat, M.S. (2011). Adaptive Wireless Networks Using Learning Automata. *IEEE Wireless Communications*, 18(2), 75-81.

**Overview:** Comprehensive survey of LA applications in wireless networking, providing excellent background on LA schemes and parameters.

**Environment Models:**
- **P-model:** Binary response (β = 0 reward, β = 1 penalty)
- **Q-model:** Finite discrete values in [0,1] - more nuanced feedback
- **S-model:** Continuous values in [0,1] - **most relevant for thesis** (continuous F1-score feedback)

**Reinforcement Schemes Detailed:**
1. **LR-P:** p_i(n+1) = p_i(n) + β(n)[L/(M-1) - L(p_i(n)-a)] - [1-β(n)]L(p_i(n)-a)
2. **LR-I:** p_i(n+1) = p_i(n) - L(1-β(n))(p_i(n)-a) if a(n)≠a_i; h_i always = 0
3. **Pursuit LA:** Maintain running estimates; always reward action with minimum penalty estimate → faster convergence

**Critical Parameters:**
- **L (learning rate):** Controls speed vs accuracy trade-off
  - Large L: Fast convergence, lower accuracy
  - Small L: Slow convergence, higher accuracy
- **a (adaptivity parameter):** Prevents probabilities from reaching zero
  - When p_i → 0, action never selected even if environment changes
  - Non-zero a maintains exploration capability
  - **Critical for nonstationary environments like kelp monitoring**

**Applications Surveyed:**
| Layer/Area | Application | LA Type |
|------------|-------------|---------|
| Physical | Power control, rate adaptation | LR-P, Pursuit |
| MAC | Polling (LEAP), ad-hoc (AHLAP) | PLR-P |
| Network | Multicast routing | LR-P |
| Transport | TCP congestion control | LR-I |
| Sensor Networks | Clustering, data aggregation, scheduling | Various |
| Cognitive Radio | Spectrum access | Priority-based LA |

**Key Results from Cited Works:**
- LEAP outperforms RAP/GRAP under bursty traffic
- AHLAP outperforms IEEE 802.11 DCF for bursty traffic
- LA-based sensor scheduling reduces energy consumption
- Adaptive broadcast significantly improves response time

**Relevance to Thesis:**
- **S-model environment** appropriate for continuous reward feedback (F1-score)
- **Parameter a** ensures adaptivity - critical for seasonal kelp variations
- **Pursuit LA** for faster convergence when computational resources allow
- Sensor network applications (clustering, scheduling) directly relevant
- Validates LA for dynamic, unknown environments

**Methodology Enhancements:**
1. **S-model LA** - use continuous reward values instead of binary P-model
2. **Non-zero adaptivity parameter a** - prevent probabilities from converging to zero
3. **Pursuit scheme** - maintain running estimates for faster convergence

**Notable Secondary References:**
- Esnaashari & Meybodi (2010) - LA scheduling for sensor networks
- Torkestani & Meybodi (2010) - LA for wireless ad-hoc clustering
- Misra et al. (2009) - LACAS congestion avoidance in healthcare WSN

---

## BC Kelp Papers (Critical for Geographic Context)

### 7. Mora-Soto et al. (2024) - Kelp Dynamics in Southern Salish Sea, BC [LOCAL STUDY]

**Citation:** Mora-Soto, A., Schroeder, S., Gendall, L., Wachmann, A., Narayan, G.R., Read, S., Pearsall, I., Rubidge, E., Lessard, J., Martell, K., Wills, P., and Costa, M. (2024). Kelp dynamics and environmental drivers in the southern Salish Sea, British Columbia, Canada. *Frontiers in Marine Science*, 11:1323448.

**Study Area:** Southern Vancouver Island and Gulf Islands, Salish Sea, BC, Canada (south of 49°N)

**Species Focus:** Bull kelp (*Nereocystis luetkeana*) floating canopy

**Time Period:**
- High-resolution imagery: 2005-2022
- Sentinel site (Ella Beach): 1972-2022 using Landsat

**Methods:**

1. **Environmental Clustering (K-means):**
   - Classified coastline into 4 clusters based on:
     - SST climatology (spring/summer from Landsat)
     - Fetch (wave exposure)
     - Tidal amplitude current
     - Wind power density
     - Total suspended matter (TSM)
   - Points 1000m apart, 300m offshore

2. **Temporal Organization:**
   - Periods based on ONI (Oceanic Niño Index) and PDO (Pacific Decadal Oscillation)
   - Optimal periods: negative ONI+PDO
   - Suboptimal periods: positive ONI+PDO
   - 2014-2019 "Blob" period = Suboptimal 2

3. **Kelp Mapping:**
   - Multi-resolution approach: <3m for small nearshore beds, 3-10m for medium, >10m for offshore
   - Annual summer (July-August) images at low tide
   - Overall accuracy >70%
   - Percentage cover per segment (~1000m alongshore)

**Four Cluster Characteristics:**
| Cluster | SST Spring | SST Summer | Fetch | Wind | Kelp Area |
|---------|------------|------------|-------|------|-----------|
| 1 (Cold/Exposed) | 9.8°C | 12.0°C | 549km | 133.8 W/m² | 1116.4 ha |
| 2 (Semi-sheltered) | 11.3°C | 14.4°C | 195km | - | 444.5 ha |
| 3 (Strait of Georgia) | 12.8°C | 17.4°C | - | - | 83.6 ha |
| 4 (Warm/Protected) | 13.2°C | 16.8°C | 132km | - | 10.5 ha |

**Key Results:**

1. **Marine Heatwaves:** 115 MHWs total (2002-2022); 58 during 2014-2019 Blob
2. **Temperature Trends:** Summer SST increasing despite negative ONI+PDO periods
3. **Wind Trends:** Extreme wind frequency increased during Optimal 2 (2020-2022)

4. **Kelp Resilience by Cluster:**
   - **Cluster 1:** Constant presence (80% segments), high resilience, favorable conditions
   - **Cluster 2:** Increased from 37% to 61% segment presence
   - **Cluster 3:** Fluctuated 38% to 33%, lowest resilience
   - **Cluster 4:** Increased from 31% to 96%, but limited data

5. **50-year Sentinel Site (Ella Beach):**
   - No significant trend (Mann-Kendall tau = -0.14, p = 0.16)
   - Range: 13.6% to 57.5% cover
   - Persistent presence throughout entire series

**Conclusions:**
- Bull kelp generally resilient in study area during 2014-2019 warming
- Different clusters show resilience for different reasons:
  - Cold/exposed: favorable thermal conditions
  - Protected: potentially benefited from wind-wave forcing
  - Strait of Georgia: moderate resilience, may benefit from Fraser River nutrients
- Local geographic conditions critical for understanding kelp dynamics

**Relevance to Thesis:**

1. **Geographic Context:** Directly within BC coastal waters study area
2. **Environmental Variables:** SST, MHWs, wind, fetch - context features for LELA Component 2
3. **Spatial Heterogeneity:** 4 clusters demonstrate need for adaptive monitoring
4. **Temporal Dynamics:** Multi-scale (seasonal, interannual, decadal) variation
5. **Multi-resolution Imagery:** Precedent for combining Landsat + high-res satellites
6. **Resilience Framework:** Reference bounds approach for anomaly detection

**Methodology Enhancements:**
1. **Environmental clustering** - Could use K-means clustering to define LELA context regions
2. **Global-local driver integration** - Use ONI/PDO as global context, local SST as fine-scale
3. **Multi-resolution fusion** - Adaptive sensor selection based on kelp bed size

**Notable Secondary References:**
- Schroeder et al. (2020) - Kelp persistence on BC west coast
- Gendall et al. (2023) - Multi-satellite kelp mapping framework
- Starko et al. (2022) - Microclimate and kelp extinction
- Cavanaugh et al. (2019) - Giant kelp resistance and resilience

### 8. Schroeder et al. (2020) - Kelp Persistence in BC Using Satellite Remote Sensing [METHODOLOGY]

**Citation:** Schroeder, S.B., Boyer, L., Juanes, F., and Costa, M. (2020). Spatial and temporal persistence of nearshore kelp beds on the west coast of British Columbia, Canada using satellite remote sensing. *Remote Sensing in Ecology and Conservation*, 6(3), 327-343.

**Study Area:** Cowichan Bay and Sansum Narrows, east coast of Vancouver Island, Salish Sea, BC (~50 km coastline)

**Species:** Bull kelp (*Nereocystis luetkeana*) - annual species, dominant canopy-forming kelp

**Time Period:** 2004-2017 using high-resolution satellite imagery

**Sensors Used:**
- QuickBird (2.6 m) - 2004
- WorldView-2 (1.8 m) - 2012, 2015
- WorldView-3 (1.2 m) - 2016, 2017

**Methods:**

1. **Image Quality Scoring (Critical for thesis):**
   | Criterion | Ideal (3) | Medium (2) | Poor (1) |
   |-----------|-----------|------------|----------|
   | Season | Late July-early Sept | Mid-June to late July | Before June/after Sept |
   | Tide height | ≤1.2 m | 1.3-2 m | >2 m |
   | Glint | None | Some | High throughout |
   | Water Surface Roughness | Smooth calm | Some texture | Breaking waves |
   - Images scoring <7 (of 12) deemed unreliable

2. **Image Processing:**
   - Geometric correction (RMSE < 0.06 m)
   - Atmospheric correction (FLAASH in ENVI v5.5)
   - Pseudo-invariant feature normalization
   - Land mask (object-based segmentation + 4m buffer)
   - Deep water mask (30m isobath)

3. **Classification Indices:**
   - **NDVI** = (NIR - Red) / (NIR + Red) - best for kelp/water separation
   - **GNDVI** = (NIR - Green) / (NIR + Green) - sensitive to chlorophyll
   - **PC1** - separates kelp from shadow over water
   - Jefferies-Matusita distance for feature selection

4. **Classification:**
   - ISODATA unsupervised classification
   - 8-12 classes produced per image
   - Classes assigned to kelp based on spectral comparison
   - Filter to remove isolated pixels (>8 pixels separation)

5. **Persistence Analysis:**
   - 100m shoreline bins (448 units = 48.8 km)
   - Persistence = % of years with kelp present in bin
   - Accounts for bed size variability and detection uncertainty

**Accuracy Results:**
- Total accuracy: **86.9%**
- User's accuracy: 81.9%
- Producer's accuracy: 93.1%
- Higher errors in single bulbs/small clusters (water dampens NIR)

**Spatial Drivers of Persistence:**
| Factor | High Persistence | Low Persistence |
|--------|------------------|-----------------|
| Substrate | Rocky | Sand/Gravel |
| RMS Tide | 0.3-0.6 (medium-high mixing) | <0.1 (low mixing) |
| Stratification | Low (well-mixed) | High (stratified) |
| Temperature | Cooler (<17°C) | Warmer (>17°C) |

**Key Results:**

1. **Spatial Patterns:**
   - 37.1% of 448 units had kelp in ≥1 year
   - West Sansum: 36.7% of units had kelp in ALL years
   - North of Sansum Point: 0% kelp (warmer, turbid Fraser River influence)

2. **Temporal Changes:**
   - Highest kelp: 2015 (89.2% of kelp units)
   - Lowest kelp: 2017 (45.7%)
   - Decline from 2015-2017 of 48.6%
   - South Cowichan: 91.9% (2015) → 12.5% (2017)

3. **Lag Effects:**
   - Warm SSTs in 2015-2016 may explain decline in 2016-2017
   - 1-year lag is best predictor of kelp growth vs SST
   - Sansum Narrows: 3°C cooler than Strait of Georgia

**Conclusions:**
- Limited evidence of overall decline (may be natural variability)
- Local drivers (substrate, currents, mixing) critical
- Persistent populations in Sansum Narrows may be important spore source
- Longer time series needed to distinguish trends from variability

**Relevance to Thesis:**

1. **Image Quality Scoring** - Framework for adaptive image selection in LELA
2. **NDVI/GNDVI Classification** - Heuristics for LELA heuristic pool (Tier 1)
3. **Persistence Analysis** - Alternative to raw area for change detection
4. **Spatial Drivers** - Context features for Component 2 (substrate, current, temperature)
5. **Lag Effects** - Important for temporal modeling in sparse label scenarios
6. **Multi-sensor Approach** - Precedent for combining different resolution imagery

**Methodology Enhancements:**
1. **Image quality as LA action selection criterion** - Weight heuristic selection by image reliability
2. **Persistence-based reward** - Use multi-year persistence instead of single-year accuracy
3. **Environmental context integration** - Substrate type, RMS tide as context features

**Notable Secondary References:**
- Pfister et al. (2018) - Kelp dynamics in NE Pacific
- Cavanaugh et al. (2011) - Giant kelp environmental controls
- Springer et al. (2007) - Bull kelp ecology and management

---

### 9. Starko et al. (2024) - Local and Regional Variation in Kelp Loss and Stability Across Coastal BC [CRITICAL FOR THESIS]

**Citation:** Starko, S., Timmer, B., Reshitnyk, L., Csordas, M., McHenry, J., Schroeder, S., Hessing-Lewis, M., Costa, M., Zielinksi, A., Zielinksi, R., Cook, S., Underhill, R., Boyer, L., Fretwell, C., Yakimishyn, J., Heath, W.A., Gruman, C., Hingmire, D., Baum, J.K., and Neufeld, C.J. (2024). Local and regional variation in kelp loss and stability across coastal British Columbia. *Marine Ecology Progress Series*, 733, 1-26.

**Study Design:**
- Compared kelp (Macrocystis pyrifera, Nereocystis luetkeana) distributions from 1.5-3 decades ago (1994-2007) to recent (2017-2021)
- 11 subregions spanning Southern, Central, and Northern BC
- Snapshot analyses (2 timepoints) + 7 timeseries datasets
- ~26,000 km of coastline (more than CA, OR, WA combined)

**Key Findings:**

1. **Variable Patterns of Change:**
   - 6/11 subregions had more losses than gains
   - 2/11 had more gains than losses
   - 3/11 had roughly no change (<10% net)
   - **Extreme losses:** Valdes/Gabriola (74%), Dundas Island (62%), Barkley Sound (43%), Laredo Sound (31%)

2. **Southern Region - Temperature-Driven Declines:**
   - Kelp persistence negatively correlated with summer SST
   - Inner Salish Sea experienced largest declines (SST > 18-20°C thresholds)
   - Best model: SST + Current speed
   - Current velocity in narrow passages correlated with local persistence

3. **Northern Region - Urchin-Driven Declines:**
   - Declines despite cool temperatures (<16-17°C)
   - Transitions to urchin barrens visible in aerial imagery
   - Sea Star Wasting Disease (SSWD) caused Pycnopodia extinction
   - Ecological release of sea urchins (Strongylocentrotus, Mesocentrotus)
   - Losses concentrated on low-fetch (sheltered) coastlines

4. **Central Region - Sea Otter Mediated Stability:**
   - Subregions with sea otters (Nootka Sound, Quatsino Sound, South Central Coast) showed stability or increases
   - Functional redundancy in urchin predation

5. **2014-2016 Marine Heatwave ("The Blob"):**
   - Most declines occurred during or after this event
   - Combined effects: physiological stress + SSWD-induced trophic cascade
   - Some recovery in moderate areas (Mayne Island), persistent declines elsewhere

**Methods Used:**
| Method | Resolution | Subregions |
|--------|------------|------------|
| Oblique shoreline photography (ShoreZone) | Very high (<100m altitude) | Most subregions |
| WorldView satellite imagery | 1.2-2.6 m | Cowichan Bay |
| Landsat timeseries | 30 m | South Central Coast |
| RPAS (drone) imagery | Sub-meter | Calvert Island |
| In situ surveys (kayak, SCUBA) | Direct observation | Mayne Island, Central SoG |

**Environmental Variables Analyzed:**
- SST from LiveOcean Model (500-1500m grid) and MUR SST (1km)
- Current velocity from LiveOcean Model
- Fetch from DFO provincial model
- Temperature anomalies from Lighthouse stations (1982-2012 baseline)

**Statistical Approaches:**
- Fisher's exact tests for subregion variation
- Spatial generalized linear mixed models (spaMM package)
- Conditional AIC (cAIC) for model comparison with spatial autocorrelation
- Dickey-Fuller test for stationarity in long timeseries

**Relevance to Thesis:**

1. **Geographic Context:** Most comprehensive assessment of BC kelp forests - directly relevant to thesis study area

2. **Spatial Heterogeneity Evidence:**
   - Different drivers dominate in different regions
   - Fine-scale variation within subregions
   - Site-level differences in stability and persistence
   - **Validates need for adaptive, context-aware monitoring (Component 2)**

3. **Multiple Driver Framework:**
   - Temperature (bottom-up physiological)
   - Trophic dynamics (top-down grazing)
   - Wave exposure (physical refugia)
   - Sea otter presence (predator control)
   - **Context features for LA framework**

4. **Multi-source Data Integration:**
   - Oblique photography, satellite imagery, in situ surveys
   - Different resolutions for different questions
   - **Validates multi-heuristic approach**

5. **Conservation Urgency:**
   - "Kelp forests should be a conservation concern in this province"
   - Losses are spatially clustered, not uniform
   - **Motivates development of monitoring systems**

**Methodology Enhancements:**

1. **Multi-driver context features** - Include SST, fetch, current speed, and trophic indicators in Component 2 context vector

2. **Regime shift detection** - LA could detect transitions between kelp forest and urchin barren states

3. **Heterogeneity-aware evaluation** - Test system across different driver contexts (warm/cool, high/low fetch, otter/no-otter)

4. **Presence-absence vs. abundance** - Presence-absence more stable; abundance more sensitive to interannual variation

5. **Multi-temporal validation** - System should detect both acute (MHW) and chronic (decadal) changes

**Notable Secondary References:**
- Cavanaugh et al. (2019) - Giant kelp resistance and resilience to MHW
- Hamilton et al. (2021) - SSWD-driven Pycnopodia extirpation
- Burt et al. (2018) - Mesopredator collapse and trophic cascades
- Mora-Soto et al. (2024) - Southern Salish Sea kelp dynamics
- Schroeder et al. (2020) - Satellite remote sensing of BC kelp

---

## Kelp Ecology Papers

### 10. Steneck et al. (2002) - Kelp Forest Ecosystems: Biodiversity, Stability, Resilience and Future [FOUNDATIONAL]

**Citation:** Steneck, R.S., Graham, M.H., Bourque, B.J., Corbett, D., Erlandson, J.M., Estes, J.A., and Tegner, M.J. (2002). Kelp forest ecosystems: biodiversity, stability, resilience and future. *Environmental Conservation*, 29(4), 436-459.

**Key Concepts:**

1. **Kelp Forest Productivity:**
   - Net primary production: 500 to >1,000 g C m⁻² year⁻¹ in dense stands
   - Among most productive ecosystems on Earth, rivaling tropical rainforests
   - Support complex food webs through direct herbivory and detrital pathways

2. **Three Morphological Guilds:**
   - **Canopy kelps:** Floating canopies (Macrocystis up to 45m, Nereocystis ~10m)
   - **Stipitate kelps:** Rigid stipes holding fronds above benthos (Laminaria, Ecklonia)
   - **Prostrate kelps:** Cover benthos with fronds (most Laminaria species)

3. **Global Distribution Constraints:**
   - High latitudes: Limited by light availability
   - Low latitudes: Limited by nutrients, warm temperatures, and competition from fucoids
   - Mid-latitudes (40-60°): Most threatened by sea urchin herbivory

4. **Trophic Cascades and Phase Shifts:**
   - Overfishing of apex predators → sea urchin population increase → kelp deforestation
   - Phase shift from kelp forest to "urchin barrens" (coralline-dominated)
   - Hysteresis: once urchin barrens established, difficult to reverse
   - Recent phenomenon: widespread urchin-induced deforestation since 1960s-1980s

5. **Biodiversity and Stability:**
   - High diversity systems (e.g., Southern California) more resistant to phase shifts
   - Functional redundancy among predators and herbivores buffers against collapse
   - Low diversity systems (e.g., Gulf of Maine, Alaska) more vulnerable to widespread deforestation

6. **Three North American Case Studies:**
   - **Alaska:** Sea otter extirpation → urchin increase → kelp loss (reversed with otter recovery)
   - **Western North Atlantic:** Cod extirpation → urchin increase → widespread deforestation
   - **Southern California:** High diversity buffers system; patchy, short-duration deforestations

7. **Climate Impacts:**
   - El Niño events cause thermal stress, nutrient limitation, kelp mortality
   - Regime shifts cause decadal-scale temperature fluctuations
   - Storm intensity increasing with global warming

8. **Archaeological Evidence:**
   - Indigenous peoples exploited kelp forest organisms for thousands of years
   - Localized losses of apex predators and small-scale deforestations occurred prehistorically
   - Commercial exploitation for export (past 2 centuries) caused large-scale changes

**Relevance to Thesis:**

1. **Ecological context:** Establishes kelp forests as critical ecosystems worth monitoring
2. **Phase shift detection:** Adaptive monitoring should detect transitions between kelp forest and urchin barren states
3. **Spatial heterogeneity:** Different regions have different vulnerabilities - validates need for adaptive, context-aware monitoring
4. **Trophic complexity:** Multiple drivers (temperature, herbivory, predation) require multi-factor monitoring approaches
5. **Temporal scales:** Changes occur at multiple scales (seasonal, interannual, decadal) - system must adapt across scales

**Key Secondary Citations to Add:**

- Mann (1973) - Seaweeds: productivity and strategy for growth
- Dayton (1985a) - Ecology of kelp communities (Annual Review)
- Dayton (1985b) - South American kelp communities
- Kain (1979) - View of genus Laminaria
- Schiel & Foster (1986) - Structure of subtidal algal stands
- Jackson et al. (2001) - Historical overfishing and coastal ecosystems collapse
- Estes & Duggins (1995) - Sea otters and kelp forests in Alaska
- Lawrence (1975) - Relationships between marine plants and sea urchins
- Duggins et al. (1989) - Magnification of secondary production by kelp detritus
- Tegner & Dayton (1987) - El Niño effects on southern California kelp forests
- Harrold & Reed (1985) - Food availability, sea urchin grazing, kelp structure
- Scheibling (1986) - Macroalgal abundance following sea urchin mass mortalities

---

### 11. Krumhansl et al. (2016) - Global Patterns of Kelp Forest Change [CRITICAL GLOBAL PERSPECTIVE]

**Citation:** Krumhansl, K.A., Okamoto, D.K., Rassweiler, A., Novak, M., Bolton, J.J., Cavanaugh, K.C., Connell, S.D., Johnson, C.R., Konar, B., Ling, S.D., Micheli, F., Norderhaug, K.M., Pérez-Matus, A., Sousa-Pinto, I., Reed, D.C., Salomon, A.K., Shears, N.T., Wernberg, T., ... Byrnes, J.E.K. (2016). Global patterns of kelp forest change over the past half-century. *Proceedings of the National Academy of Sciences*, 113(48), 13785-13790.

**Study Design:**
- First comprehensive global analysis of kelp forest change
- Database of 1,454 sites across 34 of 99 kelp-containing ecoregions
- Time series spanning ~50 years (1950s-2013)
- Bayesian hierarchical analysis

**Key Results:**

1. **Global Pattern:**
   - Small global average decline (instantaneous rate = -0.018 y⁻¹)
   - BUT: Regional variability FAR exceeds global average
   - Local/regional drivers dominate kelp dynamics

2. **Regional Trajectories:**
   - **Declines (38% of ecoregions):** -0.015 to -0.18 y⁻¹
     - Central Chile (-0.150)
     - Aleutian Islands (-0.071)
     - South Australian Gulfs (-0.059)
     - North Sea (-0.024)
     - North-Central California (-0.019)
   - **Increases (27% of ecoregions):** 0.015 to 0.11 y⁻¹
   - **No detectable change (35% of ecoregions)**

3. **Driver Categories:**
   - Climate change (temperature, storm frequency)
   - Overfishing (trophic cascades)
   - Direct harvest
   - Pollution and sedimentation
   - Invasive species

**Key Conclusions:**
- "Local stressors and regional variation in the effects of these drivers dominate kelp dynamics"
- "In contrast to many other marine and terrestrial foundation species"
- "Increased monitoring aimed at understanding regional kelp forest dynamics is likely to prove most effective for adaptive management"

**Relevance to Thesis:**

1. **Validates adaptive approach:** Regional variation requires context-aware monitoring
2. **Spatial heterogeneity evidence:** Same stressor → different outcomes in different regions
3. **Management recommendation:** Regional monitoring systems needed
4. **BC context:** Includes data from BC (Vancouver Island, Puget Trough/Georgia Basin)
5. **Foundation species framing:** Kelp as ecosystem engineer worth monitoring

**Key Secondary Citations to Add:**

- Jackson et al. (2001) - Historical overfishing and coastal ecosystem collapse
- Wernberg et al. (2011) - Seaweed communities in retreat from climate change
- Ling et al. (2009) - Overgrazing and urchin barrens in Tasmania
- Estes & Palmisano (1974) - Sea otters as keystone predators
- Worm et al. (2006) - Impacts of biodiversity loss on ocean ecosystem services
- Ellison et al. (2005) - Foundation species and ecosystem function

---

### 12. Rogers-Bennett & Catton (2019) - Marine Heat Wave Tips Kelp to Urchin Barrens [CRITICAL CASE STUDY]

**Citation:** Rogers-Bennett, L. and Catton, C.A. (2019). Marine heat wave and multiple stressors tip bull kelp forest to sea urchin barrens. *Scientific Reports*, 9, 15050.

**Study Area:** Northern California (>350 km coastline from San Francisco to Oregon border)

**Species:** Bull kelp (*Nereocystis luetkeana*) - same species dominant in BC

**Time Period:** Long-term monitoring 1999-2018

**The Catastrophic Shift:**

1. **Kelp Canopy Collapse:**
   - Historic maximum: >50 km² (1999-2008)
   - 2014-2016: declined to <2 km²
   - **>90% loss within one year**
   - No appreciable recovery through 2019

2. **Multiple Stressor Cascade:**
   | Year | Event |
   |------|-------|
   | 2013 | Sea Star Wasting Syndrome (SSWS) begins |
   | 2014 | Marine Heat Wave (MHW) reaches California |
   | 2014-2017 | Persistent warm water (>12°C for extended periods) |
   | 2014-2015 | Purple urchin population begins increasing |
   | 2015+ | Urchin barrens established |
   | 2017 | Mass abalone mortality (80%) |

3. **Sea Star Wasting Disease:**
   - Sunflower star (*Pycnopodia helianthoides*): key urchin predator
   - Pre-2013: 0.01-0.12 stars/m²
   - Post-2013: Functionally extinct (1 observed in 2014-2015)
   - No observations 2016-2019: locally extinct

4. **Purple Urchin Explosion:**
   - Pre-2014: 0-1.7 urchins/m²
   - 2015: 8.2-12.9 urchins/m² (60-fold increase)
   - 2018: 9.2-24.1 urchins/m²
   - Shifted to aggressive grazing behavior in barren conditions

5. **Economic Impacts:**
   - Recreational abalone fishery closed (2018): worth $44M/year
   - Commercial red sea urchin fishery collapsed (2015-): worth $3M/year
   - Tourism impacts from lost diving, kayaking opportunities

**Key Mechanism - Phase Shift:**
- Warm water: thermal stress + nutrient limitation → reduced kelp growth
- SSWS: removed urchin predator → trophic cascade
- Urchin increase: intensive grazing pressure
- Limited spore production: recovery impaired
- **Hysteresis:** Urchin barrens may persist as alternative stable state

**Relevance to Thesis:**

1. **Urgency of monitoring:** Catastrophic shifts can occur rapidly (<1 year)
2. **Multiple stressor interactions:** Single-driver monitoring insufficient
3. **Phase shift detection:** Adaptive system should detect transitions
4. **BC relevance:** Same species (bull kelp), same predator (Pycnopodia), similar vulnerability
5. **Trophic cascade monitoring:** Need to track predator-prey dynamics, not just kelp
6. **Economic motivation:** Monitoring supports fishery management

**Key Secondary Citations to Add:**

- Hamilton et al. (2021) - SSWD-driven Pycnopodia extirpation
- Filbee-Dexter & Wernberg (2018) - Climate change and kelp forests
- Harvell et al. (2019) - Disease epidemic and marine heatwave
- Ling et al. (2015) - Sea urchin barrens as alternative stable states
- Bond et al. (2015) - "The Blob" marine heatwave
- Di Lorenzo & Mantua (2016) - Multi-year persistence of North Pacific conditions

---

## Remote Sensing / ML Papers

### 13. Bennion et al. (2019) - Remote Sensing of Kelp: Monitoring Tools and Implications [COMPREHENSIVE REVIEW]

**Citation:** Bennion, M., Fisher, J., Yesson, C., and Brodie, J. (2019). Remote sensing of kelp (Laminariales, Ochrophyta): monitoring tools and implications for wild harvesting. *Reviews in Fisheries Science & Aquaculture*, 27(4), 395-434.

**Scope:** Comprehensive review of remote sensing technologies for kelp monitoring

**Key Remote Sensing Technologies Reviewed:**

1. **Satellite-Based Sensors:**
   - Landsat (30m): Most widely used, 40+ year archive
   - Sentinel-2 (10-20m): Free, better resolution than Landsat
   - WorldView (0.5-1.8m): High resolution but expensive
   - SPOT, RapidEye, PlanetScope

2. **Aerial Photography:**
   - Traditional aerial surveys
   - UAV/drone imagery (sub-meter resolution)
   - Oblique shoreline photography (ShoreZone)

3. **Acoustic Methods:**
   - Multibeam sonar for submerged kelp
   - Side-scan sonar
   - Single-beam echosounder

4. **Other Methods:**
   - LiDAR (bathymetric)
   - Hyperspectral sensors
   - Thermal imaging

**Key Limitations Identified:**
- Turbid temperate waters reduce optical detection
- Tide effects not always accounted for
- Submerged kelp difficult to detect optically
- Temporal coverage gaps in archived imagery
- Lack of standardized monitoring protocols

**Spectral Indices for Kelp Detection:**
| Index | Formula | Application |
|-------|---------|-------------|
| NDVI | (NIR - Red)/(NIR + Red) | Floating canopy detection |
| GNDVI | (NIR - Green)/(NIR + Green) | Chlorophyll sensitivity |
| FAI | NIR - (Red + (SWIR - Red) × λ_ratio) | Floating algae index |
| NDWI | (Green - NIR)/(Green + NIR) | Water vs vegetation |

**Conceptual Framework for Monitoring:**
- Need baseline information before monitoring
- Multi-scale approach: local (field), regional (aerial), global (satellite)
- Seasonal timing critical for annual species
- Integration of multiple data sources recommended

**Relevance to Thesis:**

1. **Heuristic pool design:** Review provides comprehensive list of detection methods for LELA framework
2. **Multi-sensor validation:** Supports multi-resolution approach in Component 2
3. **Index selection:** NDVI, GNDVI, FAI as Tier 1 spectral indices
4. **Limitation awareness:** Turbidity, tide effects as context features
5. **Standardization need:** LELA could provide adaptive standardization

**Key Secondary Citations:**
- Dayton (1985) - Ecology of kelp communities
- Cavanaugh et al. (2010) - Scaling kelp measurements with satellite
- Yesson et al. (2015) - Kelp distribution modeling
- Smale et al. (2013) - Kelp and climate change

---

### 14. Gendall et al. (2023) - Multi-Satellite Mapping Framework for Floating Kelp [CRITICAL METHODOLOGY]

**Citation:** Gendall, L., Schroeder, S.B., Wills, P., Hessing-Lewis, M., and Costa, M. (2023). A Multi-Satellite Mapping Framework for Floating Kelp Forests. *Remote Sensing*, 15(5), 1276.

**Study Area:** Haida Gwaii, BC (Cumshewa Inlet, ~800 km²)

**Key Innovation:** Object-based image analysis (OBIA) framework for combining multi-resolution satellite imagery

**Four-Step Workflow:**

1. **Imagery Compilation & Quality Assessment:**
   - 20 satellite sensors (1973-2021)
   - Resolution range: 0.5-60m
   - Sources: Open data, private agreements, commercial acquisition
   - Quality factors: cloud cover, glint, tide, waves

2. **Preprocessing:**
   - Atmospheric correction
   - Land/water masking
   - Geometric correction

3. **Object-Oriented Classification:**
   - Segmentation into homogeneous objects
   - Rule-based classification
   - Spectral indices (NDVI, GNDVI, FAI)

4. **Accuracy Assessment:**
   - Ground truth validation
   - Global accuracy: 88-94%
   - Cross-resolution comparison

**Multi-Sensor Analysis:**
| Sensor Type | Resolution | Detection Capability |
|-------------|------------|---------------------|
| Very High (WorldView, Pleiades) | <3m | Small fringing beds |
| High (Sentinel-2, RapidEye) | 3-10m | Medium nearshore |
| Medium (Landsat, SPOT) | 10-60m | Large offshore forests |

**Key Finding - Resolution Impact:**
- Lower resolutions unreliable for small kelp forests in high-slope areas
- Recommend removing high-slope areas (11.4%) from time series analysis
- Error of up to 7% when comparing different resolutions in low/mid slope areas

**Challenges on BC Coast:**
- Small fringing kelp forests (vs. large offshore California)
- High topographic complexity
- Increasing cloud cover north to south
- Large tidal amplitude
- Strong currents

**Relevance to Thesis:**

1. **BC-specific methodology:** Directly applicable to thesis study area
2. **Multi-resolution validation:** Supports adaptive resolution selection in LELA
3. **OBIA approach:** Could inform heuristic design (object vs. pixel methods)
4. **Quality assessment criteria:** Framework for image selection in Component 2
5. **Accuracy benchmarks:** 88-94% as target for LELA system

**Key Secondary Citations:**
- Mora-Soto et al. (2020) - High-resolution global kelp map
- Schroeder et al. (2020) - Kelp persistence in BC
- Cavanaugh et al. (2021) - Remote sensing for kelp management
- Bell et al. (2020) - Three decades of variability in California

---

### 15. Bell et al. (2023) - Kelpwatch: Visualization and Analysis Tool [OPERATIONAL SYSTEM]

**Citation:** Bell, T.W., Cavanaugh, K.C., Saccomanno, V.R., Cavanaugh, K.C., Houskeeper, H.F., Eddy, N., Schuetzenmeister, F., Rindlaub, N., and Gleason, M. (2023). Kelpwatch: A new visualization and analysis tool to explore kelp canopy dynamics reveals variable response to and recovery from marine heatwaves. *PLoS ONE*, 18(3), e0271477.

**System Overview:** Web-based visualization and analysis tool for Landsat-derived kelp canopy data

**Technical Implementation:**
- Cloud-based backend microservice with API
- Tiling service for web map visualization
- JavaScript frontend user interface
- Data: 1984-2021, 30m resolution, seasonal means
- Coverage: Oregon to Baja California Sur

**Classification Method:**
- Binary decision tree classifier trained on band-normalized Landsat
- K-means clustering (15 clusters) for training data
- MESMA (Multiple Endmember Spectral Mixture Analysis) for subpixel abundance
- 30 seawater endmembers to account for varying conditions
- Single kelp canopy endmember

**Key Findings:**

1. **Long-term Trends (1984-2021):**
   - 18.6% of regional sites showed significant trends
   - High spatial variability in trajectories

2. **Marine Heatwave Response (2014-2016):**
   - Latitudinal response pattern for each species
   - Bull kelp: severe decline in Northern California
   - Giant kelp: widespread but variable decline

3. **Recovery Patterns:**
   - Highly variable across space
   - Some areas (Bahía Tortugas): high recovery
   - Monterey Peninsula: continued slow decline
   - Northern California: persistent decline (urchin barrens)

**Analysis Framework:**
- 10 × 10 km cells for regional analysis
- 1 × 1 km cells for local analysis
- Annual maximum canopy area time series
- Response = minimum during heatwave / pre-heatwave baseline
- Recovery = post-heatwave mean / pre-heatwave baseline

**Relevance to Thesis:**

1. **Operational precedent:** Web-based monitoring system as deployment target
2. **Landsat methodology:** Established classification approach for LELA comparison
3. **Multi-scale analysis:** Regional (10km) vs. local (1km) scale framework
4. **MHW analysis framework:** Response/recovery metrics as evaluation criteria
5. **Data accessibility:** Demonstrates value of accessible monitoring tools

**Key Secondary Citations:**
- Cavanaugh et al. (2011) - Environmental controls of giant kelp
- Bell et al. (2020) - Three decades of California kelp variability
- Houskeeper et al. (2022) - Automated remote sensing at Falkland Islands
- Hamilton et al. (2020) - Long-term kelp datasets

### 16. Ma et al. (2019) - Deep Learning in Remote Sensing: Meta-Analysis and Review [FOUNDATIONAL DL REVIEW]

**Citation:** Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., and Johnson, B.A. (2019). Deep learning in remote sensing applications: A meta-analysis and review. *ISPRS Journal of Photogrammetry and Remote Sensing*, 152, 166-177.

**Scope:** Comprehensive meta-analysis of 200+ DL publications in remote sensing (2014-2018)

**Key DL Architectures Reviewed:**

1. **Convolutional Neural Networks (CNNs):**
   - AlexNet, VGG, ResNet, GoogleNet/Inception
   - Fully Convolutional Networks (FCN)
   - Most extensively used for remote sensing

2. **Recurrent Neural Networks (RNNs):**
   - Long Short-Term Memory (LSTM)
   - Gated Recurrent Units (GRU)
   - Good for temporal/sequential data

3. **Autoencoders (AEs):**
   - Stacked autoencoders for feature learning
   - Useful for dimensionality reduction
   - Spectral-spatial feature extraction

4. **Deep Belief Networks (DBNs):**
   - Restricted Boltzmann Machines
   - Unsupervised pretraining

5. **Generative Adversarial Networks (GANs):**
   - Image synthesis and augmentation
   - Super-resolution enhancement

**Remote Sensing Applications:**
| Application | Common DL Model | Key Challenge |
|-------------|-----------------|---------------|
| Image fusion | CNN, AE | Multi-sensor integration |
| Image registration | CNN | Geometric alignment |
| Scene classification | CNN, DBN | Scale variation |
| Object detection | CNN (YOLO, SSD, R-CNN) | Small objects |
| LULC classification | CNN, RNN | Class imbalance |
| Semantic segmentation | FCN, U-Net | Boundary precision |
| OBIA | CNN | Object delineation |

**Key Findings:**
- DL outperforms traditional ML (SVM, RF) in most tasks
- CNN most popular architecture for remote sensing
- Transfer learning from ImageNet improves performance
- Data augmentation critical for limited training samples
- Challenges: interpretability, computational requirements, training data needs

**Relevance to Thesis:**

1. **Tier 3 heuristic design:** CNN architectures for kelp detection
2. **Transfer learning:** Pretrained models for limited kelp training data
3. **Data augmentation:** Address sparse label problem
4. **Architecture selection:** CNN for spatial, RNN for temporal features
5. **Comparison baseline:** Traditional ML vs. DL for LELA evaluation

---

### 17. Marquez et al. (2022) - CNN for Giant Kelp Forest Mapping [DIRECT PRECEDENT]

**Citation:** Marquez, L., Fragkopoulou, E., Cavanaugh, K.C., Houskeeper, H.F., and Assis, J. (2022). Artificial intelligence convolutional neural networks map giant kelp forests from satellite imagery. *Scientific Reports*, 12, 22196.

**Innovation:** First application of Mask R-CNN for kelp forest detection from satellite imagery

**Method - Mask R-CNN:**
- Instance segmentation (detection + boundary delineation)
- Combines Faster R-CNN (target identification) + FCN (mask prediction)
- Transfer learning from COCO dataset
- Python with Keras/TensorFlow

**Data:**
- Landsat 5 and 8 (30m resolution)
- Pseudo-RGB composites (NIR, Red, Green bands)
- Study area: Southern California and Baja California
- 421 tiles, 3345 kelp polygons annotated
- Training: 75%, Testing: 17.5%, Validation: 7.5%

**Hyperparameter Tuning:**
- Learning rates: 0.001, 0.0001
- Anchor sizes: 32, 64, 128, 256, 512
- Data augmentation: rotation (90° steps), flipping, rescaling

**Performance Results:**
- **Jaccard Index:** 0.87 ± 0.07
- **Dice Coefficient:** 0.93 ± 0.04
- **Overprediction:** 0.06 (very low)
- Data augmentation significantly improved performance

**32-Year Time Series (Baja California):**
- Reconstructed kelp coverage 1989-2021
- Captured El Niño event impacts (1997-98, 2015-16)
- Consistent with known ecological patterns

**Key Advantages Over Traditional Methods:**
1. Automatic detection (reduced human effort)
2. Instance segmentation (individual forest delineation)
3. Cost-efficient long-term monitoring
4. High performance with limited training data

**Relevance to Thesis:**

1. **Tier 3 heuristic candidate:** Mask R-CNN for kelp detection
2. **Performance benchmark:** Jaccard 0.87, Dice 0.93 as targets
3. **Transfer learning validation:** COCO pretraining works for kelp
4. **Data augmentation strategy:** Address LELA sparse label challenge
5. **BC applicability:** Could extend to Nereocystis detection

**Key Secondary Citations:**
- He et al. (2017) - Mask R-CNN original paper
- Long et al. (2015) - Fully convolutional networks
- Krizhevsky et al. (2012) - AlexNet

---

## Other Papers

### 18. Bisong (2018) - On Designing Adaptive Data Structures with Adaptive Data Sub-Structures [FOUNDATIONAL - AUTHOR'S MASTERS THESIS]

**Citation:** Bisong, E. (2018). *On Designing Adaptive Data Structures with Adaptive Data "Sub"-Structures*. Master's Thesis, Carleton University, Ottawa, Canada. Supervisor: Prof. B. John Oommen.

**Problem:** Optimizing data structure access costs in Non-stationary Environments (NSEs) where query access probabilities vary with time and exhibit "locality of reference" (probabilistic dependence between consecutive queries).

**Core Theoretical Framework - Learning Automata:**

1. **LA Model Components:**
   - **Environment:** E = {α, c, β} where α = action set, c = penalty probabilities, β = feedback (reward/penalty)
   - **Automaton:** A = {Φ, α, β, F(.,.), G(.,.)} where Φ = states, F = transition function, G = output mapping
   - **P-model:** Binary feedback β ∈ {0, 1}
   - **Q-model:** Finite discrete values in [0,1]
   - **S-model:** Continuous values in [0,1] - **most relevant for PhD thesis** (continuous F1-score feedback)

2. **Performance Criteria:**
   - **Expedient:** lim E[M(n)] < M₀ (better than pure chance)
   - **Optimal:** lim E[M(n)] = c_ℓ (converges to minimum penalty)
   - **ε-optimal:** lim E[M(n)] < c_ℓ + ε (arbitrarily close to optimal)
   - **Absolutely Expedient:** E[M(n+1)|P(n)] < M(n) (continually decreasing)

3. **Fixed Structure Stochastic Automata (FSSA):**
   - Tsetlin L₂N,₂: Memory-based automaton with N states per action
   - Krinsky: ε-optimal variant (jumps to innermost state on reward)
   - Krylov: ε-optimal variant (random walk on penalty)
   - Key limitation: ε-optimality requires min(cᵢ) < 0.5

4. **Variable Structure Stochastic Automata (VSSA):**
   - **L_R-P (Linear Reward-Penalty):** Updates on both reward and penalty; expedient when a=b
   - **L_R-I (Linear Reward-Inaction):** Updates only on reward; ε-optimal; absorbing Markov chain
   - **L_I-P (Linear Inaction-Penalty):** Updates only on penalty; expedient but never ε-optimal
   - Update equations (when action αᵢ selected):
     - Reward (β=0): pᵢ(n+1) = pᵢ(n) + a[1-pᵢ(n)]; pⱼ(n+1) = (1-a)pⱼ(n)
     - Penalty (β=1): pᵢ(n+1) = (1-b)pᵢ(n); pⱼ(n+1) = b/(r-1) + (1-b)pⱼ(n)

5. **Estimator and Pursuit Schemes:**
   - Use Maximum Likelihood estimates of reward probabilities
   - Pursuit: "Pursue" the currently estimated best action
   - Faster convergence when estimates are reliable
   - Discretized versions are ε-optimal and outperform continuous variants

**Non-Stationary Environments (NSEs):**

1. **Markovian Switching Environment (MSE):**
   - Environment states {E₁, E₂, ..., E_d} form a Markov chain
   - Penalty probabilities depend on current environment state
   - Average penalty: M(n) = (1/T)Σᵢpᵢ(n)[Σⱼqⱼ(n)cⱼᵢ]

2. **Periodic Switching Environment (PSE):**
   - Environment switches periodically with period T
   - When T known: Deploy T automata, each operating once per period
   - When T unknown: Two-level hierarchy of automata

**Object Partitioning Problem (OPP):**

- Partition W elements into R disjoint groups such that frequently co-accessed elements are grouped together
- NP-hard problem with applications in: cloud computing, distributed databases, image retrieval, cryptanalysis
- **Object Migration Automaton (OMA):** LA approach where:
  - Actions = groups (R actions, N states per action)
  - Reward: Query pair in same group → move toward innermost state
  - Penalty: Query pair in different groups → move toward boundary; if at boundary, migrate object

**OMA Family of Reinforcement Schemes:**

1. **OMA (Object Migration Automaton):** Base scheme
2. **EOMA (Enhanced OMA):** Mitigates deadlock scenarios in OMA
3. **PEOMA (Pursuit Enhanced OMA):** Incorporates pursuit paradigm; order of magnitude better than EOMA
4. **TPEOMA (Transitivity Pursuit Enhanced OMA):** Uses transitivity relationships; superior in MSEs but unstable in PSEs

**Adaptive Data Structures - Hierarchical Lists-on-Lists (LOL):**

- Outer-list context + sub-list context
- Elements likely accessed together grouped in same sub-list
- Sub-lists moved "en masse" toward head using reorganization rules
- Reorganization strategies: MTF-MTF, MTF-TR, TR-MTF, TR-TR

**Key Results:**

1. EOMA-augmented hierarchical schemes outperform OMA-augmented and stand-alone MTF/TR
2. PEOMA schemes are order of magnitude superior to EOMA in MSEs
3. TPEOMA excellent in MSEs but unsuitable for PSEs (transitivity causes instability)
4. "Periodic" variants (with period knowledge) superior to "UnPeriodic" in PSEs
5. MTF outer-context generally outperforms TR outer-context

**Critical Relevance to PhD Thesis:**

1. **Theoretical Foundation:** Provides rigorous mathematical framework for LA that underpins the LELA framework
2. **S-model Environment:** Continuous reward values (like F1-score) supported by LA theory
3. **NSE Models:** MSE and PSE directly applicable to kelp monitoring (seasonal, interannual variation)
4. **OPP Connection:** Heuristic selection analogous to object partitioning - grouping similar "contexts" together
5. **Pursuit Schemes:** PEOMA's faster convergence relevant for operational monitoring with limited feedback
6. **Hierarchy Design:** LOL concept parallels Component 2's context-aware local adaptation
7. **Estimator Integration:** Running estimates of performance can improve heuristic selection
8. **Period Knowledge:** Systems can be designed to exploit known seasonal patterns

**Methodology Enhancements for PhD:**

1. **Apply S-model LA:** Use continuous IoU/F1 rewards instead of binary P-model
2. **Consider PEOMA-style pursuit:** Maintain running estimates of heuristic performance
3. **Exploit seasonal periodicity:** Design "Periodic" variants that leverage known kelp seasonality
4. **Avoid transitivity in PSE-like settings:** TPEOMA instability warns against transitive inference in seasonal environments
5. **Hierarchical context design:** Apply LOL principles to global-local adaptation architecture

**Key Equations from Thesis:**

- Average penalty: M(n) = Σᵢ cᵢpᵢ(n)
- Pure-chance baseline: M₀ = (1/r)Σᵢcᵢ
- L_R-I update (reward): pᵢ(n+1) = pᵢ(n) + a[1-pᵢ(n)]
- L_R-I update (other actions): pⱼ(n+1) = (1-a)pⱼ(n)

---
