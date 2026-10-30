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

## 7. Phase III: EchoSafe Live Social Media Demo Platform

**EchoSafe** is an interactive, full-stack demonstration social media application that brings the trained privacy-preserving federated models to life in a realistic Instagram/YouTube-style creator platform.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             ECHOSAFE WEB PLATFORM                                │
├──────────────────────────┬───────────────────────────┬───────────────────────────┤
│ 1. Follower Feed         │ 2. Creator Studio         │ 3. Safety & FL Console    │
│ • 6 Post formats         │ • Community Health Gauge  │ • Explainable Queue       │
│ • Live comment stream    │ • Toxicity velocity alert │ • Graded interventions    │
│ • Masked text preview    │ • Sentiment breakdown     │ • Non-clinical wellbeing  │
│ • Fair appeal workflow   │ • Post risk analysis      │ • Flower FL & SecAgg mesh │
└──────────────────────────┴───────────────────────────┴───────────────────────────┘
```

### 7.1 One-Command Setup & Launch

1. **Launch Everything (Backend + Built React Frontend):**
   Double-click `run_website.bat` or run:
   ```powershell
   .\python.bat -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
   Open your browser to: **`http://127.0.0.1:8000/`**

2. **Frontend Development Server (Optional for Live HMR):**
   ```powershell
   cd frontend
   npm run dev
   ```

### 7.2 Pre-Configured Demo Accounts & Instant Role Switcher

The top navbar includes a **1-Click Role Switcher** dropdown allowing you to switch identities without manually typing credentials:

| Role / Persona | Username | Password | Purpose & Special Permissions |
| :--- | :--- | :--- | :--- |
| **Follower (Positive)** | `user_positive` | `demo123` | High-sentiment audience member posting encouraging support |
| **Follower (Critic)** | `user_critic` | `demo123` | Skeptical commenter demonstrating non-censored constructive critique |
| **Follower (Abuser)** | `user_abuser` | `demo123` | Repeat hostile offender testing safety triggers & graded cooldowns |
| **Influencer (Gaming)** | `creator_0_competitive` | `demo123` | Creator with high debate/opinion toxicity velocity |
| **Influencer (Comedy)** | `creator_5_regional` | `demo123` | Creator with diverse cultural format comments |
| **Moderator** | `moderator_sarah` | `demo123` | Safety team member: queue triage, unmasking, user behavior inspection |
| **Admin** | `admin` | `demo123` | FL coordinator: execute federated training rounds & inspect privacy proofs |

---

### 7.3 5-Minute Interactive Demo Walkthrough Script

Follow these steps in order to demonstrate the complete live pipeline:

1. **Explore the Feed:**
   - Open `http://127.0.0.1:8000/`. You are logged in as a normal user.
   - Filter by post types: *All*, *Opinions*, *Reels*, *Images*, *Videos*, *Live Debates*.
   - Click the Heart button to like a post. Expand comments to see the live feed.

2. **Test Benign Praise & Free Speech Protection:**
   - On the top **Simulator Bar**, click **"✨ 1. Praise"**: The comment is analyzed by the on-device model and published immediately.
   - Click **"💬 2. Fair Critique"**: The constructive criticism is classified as non-abusive. The policy engine protects free speech and publishes it without censorship.

3. **Trigger Hostility Interception & Safety Masking:**
   - Click **"📈 4. Escalating Toxicity"**: The simulator switches to `user_abuser` and posts targeted harassment.
   - Notice that the comment text is **blurred and masked** (`***`). Normal users are never exposed to raw toxic comments.
   - Click **"🚨 5. Burst Attack"**: Rapid burst comments trigger dynamic rate limiting and temporary slow-mode cooldown.

4. **Moderator Console & Explainable AI Triage:**
   - In the navbar, click **"Moderator Console"** (or switch role to *Moderator* in the dropdown).
   - In the **Flagged Queue**, observe the flagged comment.
   - Click **"Click to Reveal (Moderator Only)"** to safely unmask the text.
   - Notice the **Trigger Words** chips highlighting explanatory terms (e.g., `["garbage", "clown", "trash"]`).
   - Click **"Inspect Behavior Profile"**: View the user's comment frequency, escalation slope, burst count, and the **Wellbeing Signal** (*Low/Medium/High*).
   - Note the prominent ethical notice: *"Not a clinical diagnosis. For human safety team review only."*
   - Choose a graded action: `Approve`, `Warn`, `15m Slow Mode`, `Shadow Filter`, or `48h Mute`.

5. **Creator Safety & Analytics Studio:**
   - Click **"Creator Studio"** in the navbar.
   - Switch between creator channels (e.g. Gaming vs Comedy).
   - Review the **Community Health Gauge** (0–100), **Toxicity Velocity**, and the **Sentiment Distribution Doughnut Chart**.
   - Review **"Which Post Types Attract Negativity?"** to see format-level controversy breakdowns.
   - Review the **Top Repeat Targeters** table showing active restrictions.

6. **Federated Learning Control Center (Admin):**
   - Click **"FL Console"** in the navbar (or switch role to *Admin*).
   - Observe the **8 Decentralized Influencer Nodes**, each showing isolated local comment partitions and **0 Bytes raw text leaked**.
   - Configure the round: Toggle between **FedAvg** and **FedProx**, enable **Secure Aggregation (SecAgg)**, and toggle **Differential Privacy (DP-FL)** with the noise slider ($\sigma$).
   - Click **"Run Federated Round Now"**: Watch the live step aggregation update and see the new round plotted on the **Convergence Curve**.
   - Inspect the **Cryptographic Data Locality & Privacy Audit** panel proving exact zero-sum mask cancellation and bounded $(\epsilon, \delta)$ budgets.

7. **How Privacy Works Explainer:**
   - Click **"How Privacy Works"** to review the visual architectural comparison between legacy centralized platforms and EchoSafe's decentralized federated design.

---

### 7.4 Verification & Automated Testing

All pipelines, threshold engines, and auth flows are verified via automated tests:

```powershell
# 1. Run all pytest test suites (9 automated tests covering pipeline, policy & auth)
.\python.bat -m pytest tests/ -v

# 2. Run the end-to-end demo replay script (replays all 9 scenarios)
.\python.bat demo_replay.py

# 3. Verify production frontend build
cd frontend
npm run build
```

