# Privacy-Preserving Federated Learning for Social Media Influencer

> **MCA Project Work — Phase I & II**  
> **Institution:** BMS Institute of Technology & Management (Autonomous under VTU), Bengaluru  
> **Department:** Department of MCA  
> **Student Name:** D K Karthikesh | **USN:** 1BY25MC034  
> **Project Guide:** Prof Reshma C R  

---

## 1. Project Overview & Aim

Social media content creators and influencers face unprecedented volumes of online hostility, abusive comments, vulgarity, and coordinated harassment across platforms such as YouTube, Instagram, X (Twitter), and TikTok. Repeated exposure to toxicity severely impacts creator mental wellbeing and degrades community discourse.

Conventional centralized moderation architectures require transmitting and pooling raw user comments, direct messages, and engagement logs onto central servers, introducing severe privacy violations, data leakage risks, and regulatory non-compliance under GDPR and DPDP.

**Aim:** To design, develop, and evaluate a **privacy-preserving federated learning and behavioral profiling framework** for social media influencers. The system classifies abusive versus positive comments, models user behavioral interaction histories, identifies repeat and escalating abusers, and recommends tiered moderation actions—while guaranteeing that raw comment texts never leave the creator's local device.

```mermaid
flowchart TD
    subgraph ClientCommunities ["Decentralized Influencer Client Nodes (Private Communities)"]
        direction TB
        C0["Influencer 0: Gaming (YouTube)"]
        C1["Influencer 1: Beauty & Fashion (Instagram)"]
        C2["Influencer 2: Politics (X / Twitter)"]
        C3["Influencer 3: Tech & Gadgets (YouTube)"]
        C4["Influencer 4: Fitness (Instagram)"]
        C5["Influencer 5: Comedy Skits (Hi-En)"]
        C6["Influencer 6: Tech & Startups (Kn-En)"]
        C7["Influencer 7: Lifestyle Vlogs (TikTok)"]
    end

    subgraph LocalAnalysis ["On-Device Local Intelligence (Per Client)"]
        RawComments["Private Raw Comments (Never Transmitted)"]
        Model["Local CommentClassifierMLP (V->64->32->4)"]
        BehAnalyzer["User Behaviour & Repeat Abuser Profiler"]
        ModEngine["Local Tiered Moderation Engine"]
    end

    subgraph FLServer ["Central Federated Learning Server"]
        Agg["Federated Aggregation (FedAvg / FedProx)"]
        GlobalModel["Global Abuse Classifier Parameters"]
    end

    ClientCommunities --> LocalAnalysis
    LocalAnalysis -->|Transmits Only Model Weights (Zero Raw Text)| Agg
    Agg --> GlobalModel
    GlobalModel -.->|Distributes Updated Global Weights| ClientCommunities
```

---

## 2. Core Modules & Key Features

1. **Synthetic Abuse & Behavioural Dataset Generator (`data/generate_synthetic_data.py`):**
   - 8 Influencer client nodes across diverse niches and platforms (YouTube, Instagram, X, TikTok).
   - 2,000 users mapped to 7 distinct behavioral archetypes: `normal_positive`, `occasional_critic`, `frequent_commenter`, `repeat_abuser`, `escalating_abuser`, `burst_spammer`, `lurker`.
   - Real-world linguistic guards: code-mixed slang (`hi-en`, `kn-en`), sarcastic criticism, positive profanity for emphasis, and l33tspeak obfuscations (`@` for `a`, `$` for `s`).
   - Auditable lexicons (`data/lexicons/`) focusing strictly on profanity and insults with zero slurs against protected groups.

2. **Disjoint Client Partitioning & Feature Extraction (`modules/data_prep.py`):**
   - Pure NumPy implementation of TF-IDF vectorization (1,200 vocabulary features).
   - Disjoint user-ID and template-ID train/test splits, preventing phrasing memorization and verifying genuine generalization on unseen users.

3. **Neural Comment Classifier (`modules/local_model.py`):**
   - 4-Class Multi-Layer Perceptron ($V \to 64 \to 32 \to 4$) with Softmax, Cross-Entropy loss, Adam optimizer, and FedProx proximal regularization.
   - Outputs calibrated probabilities across 4 categories: `Positive`, `Neutral`, `Criticism` (negative-but-not-abusive), and `Abusive-Vulgar`.

4. **Federated Optimization Engine (`modules/fl_coordinator.py`, `modules/strategies.py`):**
   - Flower-compatible federated coordinator implementing **FedAvg** and **FedProx**.
   - Aggregates client parameter updates over 12 communication rounds with zero raw text leakage.

5. **User Behaviour Analyzer & Risk Profiler (`modules/behaviour_analyzer.py`):**
   - Aggregates user interaction timelines across posts and communities.
   - Detects comment bursts ($<5$ minute clusters) and severity escalation trends (regression slope over time).
   - Computes composite `user_risk_score` ($0.0$ to $1.0$) and recommends automated tiered moderation actions (`allow`, `flag_for_review`, `rate_limit_cooldown`, `shadow_filter`, `temporary_mute`, `permanent_block_referral`).

---

## 3. Experimental Evaluation & Results

### Side-by-Side Model Utility Benchmark

Evaluated on held-out test splits with unseen users and templates:

| Training Paradigm | Model Architecture | Macro-F1 | Accuracy | Abuse-Class F1 | Benign User FPR | Raw Text Leaked? | Comm. Payload |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | CommentClassifierMLP | 0.9381 | 0.9424 | 0.9635 | 1.01% | **YES (100% Leaked)** | N/A |
| **Federated FedAvg (12 Rnds)** | **CommentClassifierMLP** | **0.9381** | **0.9424** | **0.9635** | **1.01%** | **NO (0% Leaked)** | **115.76 MB** |

### User Behaviour & Abuser Profiling Benchmark

Evaluated against ground-truth simulation archetypes across 1,899 active users:

| Metric | Result | Interpretation |
| :--- | :---: | :--- |
| **Abuser Identification Precision** | **93.15%** | When a user is flagged as an abuser, 93.15% are true repeat/escalating abusers. |
| **Abuser Identification Recall** | **80.84%** | Captures over 80% of all hostile and abusive users automatically. |
| **Abuser Detection F1-Score** | **86.56%** | Robust harmonic balance between precision and coverage. |
| **Overall Classification Accuracy** | **94.00%** | Highly accurate behavioral profiling across all 7 archetypes. |
| **Benign User False Alarm (FPR)** | **1.87%** | Only 1.87% of non-abusive users receive scrutiny, protecting free speech and constructive criticism. |

### Moderation Actions Recommended

- **Allow (Clean / Normal):** 1,154 users (60.8%)
- **Flag for Review (Constructive Criticism / Borderline):** 351 users (18.5%)
- **Temporary Mute (Repeat Abusers):** 224 users (11.8%)
- **Permanent Block Referral (Severe Chronic Toxicity):** 85 users (4.5%)
- **Shadow Filter (Escalating Abusers):** 71 users (3.7%)
- **Rate Limit / Slow Mode (Burst Spammers):** 14 users (0.7%)

### Loss & F1 Convergence Plots

![Convergence Curve](experiments/results/convergence_curve.png)

---

## 4. Quickstart: Reproducing Experiments

### Prerequisites
- Python 3.10+ (tested on Windows 11 with Python 3.14 / 3.12).
- Zero external C++ DLL dependencies: runs seamlessly without Windows Smart App Control (SAC) blocks.

### Single-Command Phase I Pipeline
Run the master runner script to execute dataset generation, centralized benchmark, 12 rounds of federated training, user behavior profiling, and chart generation:

```powershell
.\python.bat experiments/run_phase1.py
```

### Individual Execution Commands
- **Generate Synthetic Dataset:**
  ```powershell
  .\python.bat data/generate_synthetic_data.py
  ```
- **Train Centralized Baseline:**
  ```powershell
  .\python.bat experiments/run_centralized.py
  ```
- **Run Federated Simulation (FedAvg):**
  ```powershell
  .\python.bat experiments/run_federated.py
  ```
- **Execute Behaviour Analyzer & Risk Profiler:**
  ```powershell
  .\python.bat modules/behaviour_analyzer.py
  ```
- **View Comparison Table & Plot Charts:**
  ```powershell
  .\python.bat experiments/compare_phase1.py
  ```

---

## 5. Repository Structure

```
├── AUDIT_AND_REFACTOR_PLAN.md      # Initial audit and migration documentation
├── README.md                       # Complete project documentation and results
├── requirements.txt                # Python environment requirements
├── python.bat                      # Native launcher script
├── archive/
│   └── legacy_engagement/          # Archived regression experiments
├── data/
│   ├── generate_synthetic_data.py  # Synthetic generator for influencers, users & posts
│   ├── lexicons/                   # Insult, slang, positive, criticism lexicons
│   ├── synthetic_comments.csv      # 5,954 annotated comments
│   ├── synthetic_posts.csv         # 120 posts across 6 types
│   ├── synthetic_likes_shares.csv  # 3,658 user interaction events
│   ├── synthetic_influencers.json  # 8 influencer community metadata
│   └── synthetic_users.json        # 2,000 user ground-truth archetypes
├── modules/
│   ├── data_prep.py                # Pure NumPy TF-IDF & disjoint train/test splitter
│   ├── local_model.py              # Differentiable 4-class CommentClassifierMLP
│   ├── fl_coordinator.py           # Flower-compatible federated simulation coordinator
│   ├── strategies.py               # FedAvg and FedProx aggregation algorithms
│   └── behaviour_analyzer.py       # Repeat abuser detection & tiered moderation profiler
└── experiments/
    ├── run_centralized.py          # Centralized MLP baseline
    ├── run_federated.py            # Federated FedAvg simulation
    ├── run_phase1.py               # Phase I end-to-end master runner
    ├── compare_phase1.py           # Comparison table & convergence visualizer
    └── results/
        ├── centralized_results.json
        ├── fedavg_results.json
        ├── fedavg_log.csv
        ├── user_behaviour_profiles.csv
        ├── behaviour_benchmark.json
        ├── global_fedavg_model.npz
        └── convergence_curve.png
```

---

## 6. Academic & Privacy Contributions

1. **Complete Decentralization of Sensitive Text:** Social media comments often contain sensitive personal communications, private opinions, or identifying context. By shifting training entirely onto the influencer's local client node, zero raw comments are ever transmitted or aggregated centrally.
2. **Disjoint Template & User Generalization:** Unlike standard random splits that memorize repeated synthetic phrases, our split ensures the test set contains zero users or text templates seen during training, proving true linguistic generalization.
3. **Multi-Faceted Abuse Profiling:** Rather than treating toxicity as isolated text predictions, the framework evaluates temporal comment frequency, activity bursts, and escalation velocity, reliably distinguishing chronic harassers from occasional critics with a low 1.87% false alarm rate.
