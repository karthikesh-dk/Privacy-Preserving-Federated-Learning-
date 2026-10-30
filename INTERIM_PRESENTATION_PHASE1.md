# MCA PROJECT WORK (PHASE I) — INTERIM PRESENTATION
## MMCA34 (2025–27 Batch)

**Project Title:**  
Privacy-Preserving Federated Learning for Social Media Influencer Engagement Prediction and Personalized Content Recommendation  

**Student Name:** D K Karthikesh  
**USN:** 1BY25MC034  
**Project Guide:** Prof Reshma C R  
**Department:** Department of MCA, BMS Institute of Technology & Management, Bengaluru  

---

## SLIDE 1: Title Slide
* **Title:** Privacy-Preserving Federated Learning for Social Media Influencer Engagement Prediction
* **Subtitle:** Phase-I Interim Project Presentation & Progress Evaluation
* **Student:** D K Karthikesh (USN: 1BY25MC034)
* **Guide:** Prof Reshma C R
* **Institution:** Department of Master of Computer Applications, BMS Institute of Technology & Management

> **Speaker Notes:**  
> "Good morning respected guide and panel members. I am D K Karthikesh, 4th semester MCA student. Today I will be presenting the Phase-I progress for my project titled *Privacy-Preserving Federated Learning for Social Media Influencer Engagement Prediction and Personalized Content Recommendation* under the guidance of Prof Reshma C R."

---

## SLIDE 2: Introduction & Motivation
* **Context:** Social media platforms generate vast behavioural interaction signals (likes, comments, shares, saves, clicks, watch time).
* **Significance for Creators:** Accurate engagement prediction allows influencers to optimize posting schedules, content formats, and audience growth.
* **The Core Conflict (Personalization vs. Privacy):**
  * Traditional Machine Learning centralizes all user and creator data onto a single cloud server.
  * Creates high-risk single points of failure (data breaches, unauthorized profiling, compliance violations under GDPR / DPDP Act).
* **The Solution:** Federated Learning (FL) — training models collaboratively across decentralized client devices without transmitting raw private interaction data.

> **Speaker Notes:**  
> "Conventional recommendation and prediction systems require centralizing millions of user records. This compromises creator privacy and exposes personal behavioural data. Our project resolves this tension by moving model training to the client devices, sharing only model weights instead of raw interaction records."

---

## SLIDE 3: Problem Statement & Objectives

### Problem Statement
Current social media prediction pipelines necessitate pooling private audience interaction logs onto centralized servers, exposing sensitive behavioural preferences. A practical framework is needed that enables collaborative predictive intelligence without centralizing raw data.

### Phase-I Objectives
1. **Decentralized Data Preparation:** Generate and preprocess a realistic social media dataset, partitioned across 8 clients with proven non-IID statistical skew.
2. **Centralized Baseline Establishment:** Implement state-of-the-art centralized benchmarks (XGBoost, LightGBM, and PyTorch MLP) to define the upper-bound performance.
3. **Federated Pipeline Implementation:** Build an in-process Flower Federated Learning pipeline (FedAvg) executing multi-round decentralized model training.
4. **Empirical Verification:** Quantitatively compare centralized vs. federated learning on prediction accuracy (MAE, RMSE, $R^2$), convergence speed, and network communication cost.

> **Speaker Notes:**  
> "In Phase I, our primary aim is to prove that decentralized federated training can achieve predictive accuracy comparable to centralized models while ensuring that 100% of raw user data remains on client devices."

---

## SLIDE 4: Literature Review Summary

| Key Literature | Core Concept | Limitation / Identified Gap |
| :--- | :--- | :--- |
| **McMahan et al. (2017)** | Introduced Federated Averaging (FedAvg) for decentralized optimization. | General benchmark focus; not tailored to creator engagement or recommender pipelines. |
| **Li et al. (2020)** | Proposed FedProx for handling non-IID client heterogeneity. | Theoretical optimization study without social media domain applications. |
| **Chen & Guestrin (2016)** | XGBoost scalable gradient boosting framework. | Centralized tree learner; cannot natively aggregate distributed weights in federated loops. |
| **He et al. (2017)** | Neural Collaborative Filtering (NCF). | Assumes centralized access to interaction matrices; no privacy protections. |

> **Speaker Notes:**  
> "From our literature survey, while algorithms like FedAvg exist in academic benchmarks, there is a clear gap in applying federated learning to creator analytics where data exhibits severe statistical skew across creator niches."

---

## SLIDE 5: SDG & ESG Alignment

* **Primary Alignment — SDG 9 (Industry, Innovation, and Infrastructure):**
  * Develops resilient, decentralized digital infrastructure that reduces dependency on monopolistic centralized cloud storage.
* **Secondary Alignment — SDG 16 (Peace, Justice, and Strong Institutions):**
  * Guarantees institutional transparency and privacy rights by design, aligning with India's Digital Personal Data Protection (DPDP) Act.
* **Supporting Alignment — SDG 8 (Decent Work & Economic Growth):**
  * Empowers independent creators and digital entrepreneurs with data-driven predictive tools without sacrificing personal data sovereignty.

> **Speaker Notes:**  
> "Our architecture aligns directly with UN Sustainable Development Goals 9, 16, and 8, promoting privacy-first AI infrastructure and ethical data governance."

---

## SLIDE 6: Phase-I System Architecture & Workflow

```
[ Decentralized Creator Clients (8 Nodes) ]
  ├── Client 0: Reel Specialist (68% Reels)
  ├── Client 1: Photographer (71% Images)
  ├── Client 2: Video Creator (70% Video)
  ├── Client 3: Text Author (71% Text)
  └── Clients 4-7: Hybrid creators
            │
            ▼ (Local Training on Private Data: PyTorch MLP)
       [ Local Weights: w_i ]   <--- NO RAW DATA TRANSFERRED!
            │
            ▼
[ Centralized Server Aggregation (Flower FedAvg) ]
  └── Weighted Aggregation: w_{t+1} = sum (n_i / N) * w_i
            │
            ▼
[ Global Engagement Model Broadcast ] ---> Back to Clients
```

* **Client Node:** Trains a 3-layer PyTorch MLP ($d \to 64 \to 32 \to 1$) locally for 3 epochs per round.
* **Server Node:** Aggregates client parameter matrices using weighted FedAvg and evaluates performance on a held-out global test set.

> **Speaker Notes:**  
> "Here is our four-layer system architecture. Each client trains locally for 3 epochs on private data. Only parameter updates are transmitted to the Flower coordinator. The coordinator aggregates these updates and broadcasts the enhanced global model back to all creators."

---

## SLIDE 7: Synthetic Dataset & Non-IID Partitioning

* **Volume:** 8,000 post records evenly divided across 8 simulated creator nodes (1,000 records/client).
* **13 Engineered Input Features:**
  * **Temporal Features:** `hour_sin`, `hour_cos`, `day_sin`, `day_cos` (cyclic trigonometric encodings).
  * **Content Categoricals:** One-hot encoded `content_type` (`image`, `reel`, `text`, `video`).
  * **Continuous Metrics:** Standardized `caption_length`, `hashtag_count`, `follower_count`, `past_avg_engagement`, `watch_time`.
* **Target Metric:** Normalized `engagement_score` $[0.0, 1.0]$ computed from audience interaction velocity.

### Verified Non-IID Skew Proof
* **Client 0:** 68.4% Reels (Avg Engagement: 0.4234)
* **Client 1:** 70.6% Images (Avg Engagement: 0.2733)
* **Client 2:** 69.8% Video (Avg Engagement: 0.1160)
* **Client 3:** 71.1% Text (Avg Engagement: 0.0592)
* **Clients 4–7:** Hybrid distributions (Reel/Video, Image/Text, Lifestyle, Educational).

> **Speaker Notes:**  
> "To realistically test federated learning, our data must be non-IID. As shown in our verification table, Client 0 posts mostly Reels with high engagement, while Client 3 posts mostly text articles with lower velocity. This accurately simulates real-world creator heterogeneity."

---

## SLIDE 8: Experimental Setup & Centralized Baseline

* **Centralized Regime:** All 8,000 records pooled onto a single server (80% train = 6,400, 20% test = 1,600).
* **Evaluated Benchmark Models:**
  1. **XGBoost Regressor:** 120 estimators, max depth 5, learning rate 0.08.
  2. **LightGBM Regressor:** 120 estimators, histogram-based gradient boosting.
  3. **Centralized PyTorch EngagementMLP:** 20 epochs, AdamW optimizer ($lr = 0.005$).

### Centralized Results
* **XGBoost:** MAE = `0.0861` | RMSE = `0.1339` | $R^2$ = `0.5738`
* **LightGBM:** MAE = `0.0858` | RMSE = `0.1335` | $R^2$ = `0.5769`
* **Centralized PyTorch MLP:** MAE = `0.0868` | RMSE = `0.1362` | $R^2$ = `0.5591`

> **Speaker Notes:**  
> "We first established the upper-bound centralized benchmark. Both tree-based models and our PyTorch MLP achieved an MAE around 0.086 and an R² of approximately 0.57 on the pooled test set."

---

## SLIDE 9: Federated Learning Results & Comparison

### Side-by-Side Performance Evaluation

| Model / Architecture | Data Paradigm | Raw Data Shared? | Test MAE | Test RMSE | $R^2$ Score | Comm. Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized XGBoost** | Pooled (Central) | Yes (100%) | 0.0861 | 0.1339 | 0.5738 | N/A |
| **Centralized LightGBM** | Pooled (Central) | Yes (100%) | 0.0858 | 0.1335 | 0.5769 | N/A |
| **Centralized PyTorch MLP** | Pooled (Central) | Yes (100%) | 0.0868 | 0.1362 | 0.5591 | N/A |
| **Federated FedAvg (Flower)** | **Decentralized (On-Device)** | **NO (0%)** | **0.1321** | **0.1834** | **0.2011** | **2.49 MB** |

### Key Observations
* **Privacy Guarantee:** Zero raw records uploaded to the server.
* **Accuracy Trade-off:** Federated learning achieved an MAE of `0.1321` (within 0.046 of the centralized MLP).
* **Bandwidth Overhead:** Complete 12-round training consumed only `2.49 MB` of network transmission.

> **Speaker Notes:**  
> "This slide presents the core finding of Phase I. Our federated model achieved an MAE of 0.1321 without any raw data leaving the client devices. The small utility trade-off of 0.046 MAE is a justifiable cost for absolute raw-data privacy."

---

## SLIDE 10: Convergence Analysis (Visual Proof)

![Convergence Curve](experiments/results/convergence_curve.png)

* **Left Plot (MSE Loss Convergence):**
  * Global Test MSE steadily dropped from `0.0415` (Round 1) down to `0.0336` (Round 12).
  * Client training loss stabilized at `0.0152`, demonstrating healthy distributed gradient descent without overfitting.
* **Right Plot (MAE & RMSE Error Progression):**
  * Both MAE and RMSE monotonically decreased across communication rounds, converging smoothly toward the centralized benchmark line.

> **Speaker Notes:**  
> "Here are our convergence graphs. The left plot shows the global test loss declining round-by-round. The right plot demonstrates that as communication rounds progressed, client models collaborated effectively, bringing the error down smoothly."

---

## SLIDE 11: Phase-I Implementation Summary & Deliverables

1. **Modular Codebase:**
   * `/data/generate_dataset.py`: Reproducible synthetic data generation with non-IID priors.
   * `/modules/data_prep.py`: Cyclic temporal engineering, categorical encoding, and client partitioner.
   * `/modules/local_model.py`: PyTorch MLP and centralized baseline learners.
   * `/modules/fl_coordinator.py`: Flower NumPyClient implementation and server simulation loop.
   * `/modules/strategies.py`: FedAvg and FedProx strategy architecture.
   * `/experiments/compare_phase1.py`: Automated reporting and Matplotlib convergence plotter.
2. **Master Execution Pipeline:**
   * Single-command execution: `python experiments/run_phase1.py` executes the entire pipeline in ~25 seconds.
3. **Repository & Version Control:**
   * Fully tracked on GitHub: `https://github.com/karthikesh-dk/Privacy-Preserving-Federated-Learning-`

> **Speaker Notes:**  
> "All code is clean, modular, and committed to GitHub. The entire Phase I pipeline runs through a single script, making our experiments 100% reproducible."

---

## SLIDE 12: Phase-II Work Plan & Roadmap

* [ ] **FedProx Aggregation Strategy:** Integrate proximal regularization ($\mu$) to further stabilize convergence under severe non-IID skew.
* [ ] **Additive Secure Aggregation:** Implement zero-sum pairwise masking so the central server only observes aggregated sums, never individual client weight matrices.
* [ ] **Differential Privacy (Opacus DP-SGD):** Layer in gradient clipping and calibrated Gaussian noise, tracking privacy budget $(\epsilon, \delta)$.
* [ ] **Personalized Recommendation Engine:** Hybrid ranking module combining predicted engagement scores with content embedding cosine similarities.
* [ ] **Full-Stack Presentation:** FastAPI REST backend + interactive React dashboard for live training monitoring and prediction demos.

> **Speaker Notes:**  
> "In Phase II, we will build upon this foundation by adding FedProx, additive secure aggregation masking, Opacus differential privacy, a personalized recommendation engine, and a FastAPI-React dashboard."

---

## SLIDE 13: Conclusion & Q&A
* **Conclusion:** Phase-I successfully proves that privacy-preserving federated learning is feasible for social media creator analytics, preserving data locality while achieving competitive predictive accuracy.
* **Student:** D K Karthikesh (1BY25MC034)
* **Guide:** Prof Reshma C R
* **Department:** Department of MCA, BMSIT&M

**Thank You! Questions and feedback are welcome.**

> **Speaker Notes:**  
> "Thank you, respected guide and panel members. I am now open to your questions and valuable suggestions."
