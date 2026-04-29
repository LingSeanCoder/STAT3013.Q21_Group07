# 🏋️ TrainHyp AI — AI-Driven Muscle Hypertrophy Prediction & Optimization System

**STAT3013.Q21 · Group 07** · `v1.0` · Repository: `STAT3013.Q21_Group07`

📊 [Dataset (OSF)](https://osf.io/6z3xu/overview) · 🎬 [Demo Video](https://drive.google.com/drive/folders/1t5nOGhR7J-a1V3LxqmBxCuO7EX1gw5cj?usp=drive_link)

> ⚠️ **Disclaimer:** TrainHyp AI is an academic research system. Outputs are not medical advice and should not replace professional coaching or clinical guidance.

---

## Overview

Hypertrophy research is scattered across hundreds of studies with inconsistent protocols — making evidence-based practical guidance extremely hard to extract.

**TrainHyp AI** solves this by building an integrated statistical and AI pipeline on top of peer-reviewed RCT meta-analyses:

```
Raw Literature → Data Cleaning → Feature Engineering → Statistics
→ Multi-Model AI → Prediction + Explainability + Safety → Dashboard
```

**Core goal:** Given a user's training profile, predict their hypertrophy outcome (Hedges' *g*) and recommend the evidence-based optimal weekly training volume.

**Why it matters:** TrainHyp AI bridges fragmented hypertrophy science into a deployable, evidence-based optimization system — translating constrained academic literature into actionable, explainable predictions.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [OSF Meta-analytic Hypertrophy Dataset](https://osf.io/6z3xu/overview) |
| Size | ~69 RCTs · ~198–199 observations |
| Target variable | **Hedges' *g*** — standardized hypertrophy effect size |
| Why this source | Peer-reviewed RCTs → higher scientific credibility than generic fitness datasets |

**Features used:** weekly sets, training frequency, nutrition status, failure percentage, age, sex, training experience level.

**Challenges encountered:** paywalled studies, missing raw data, incomplete variable reporting, strict inclusion/exclusion filtering criteria.

> **On sample size:** The final ~199 observations reflect strict scientific filtering for variable completeness and inter-study comparability. Methodological quality was prioritized over dataset size — a deliberate tradeoff, not a shortcoming.

---

## Model System

This project uses **6 models**, each with a specific role in the pipeline — not interchangeable, not redundant.

| Model | Role | Why this model |
|---|---|---|
| **NGBoost** | Primary prediction engine | Outputs a full probability distribution, not just a point estimate |
| **EBM** (Explainable Boosting Machine) | Explainability + volume threshold | Shape functions reveal exactly how each feature shifts the prediction |
| **CatBoost** | Responder segmentation | Classifies user into Low / Medium / High responder tier before prediction |
| **GPR** (Gaussian Process Regression) | Uncertainty quantification + safety | Provides calibrated confidence intervals; detects out-of-distribution inputs |
| **TabNet** | Deep learning benchmark | Validates whether gradient boosting outperforms DL on this tabular dataset |
| **SEM** (Structural Equation Modeling) | Scientific structural logic | Models indirect causal pathways (e.g. volume → fatigue → hypertrophy) |

**Full inference flow at runtime:**

```
User Input
    │
    ▼
[CatBoost] — Segment: Low / Medium / High responder?
    │
    ▼
[NGBoost] — Predict hypertrophy effect size (Hedges' g) + full distribution
    │
    ▼
[EBM] — Explain: which features drive this prediction and by how much?
    │
    ▼
[GPR] — Safety check: is this input in-distribution? What is the uncertainty?
    │
    ▼
Personalized volume recommendation with confidence bounds
```

---

## Notebook Structure

| Notebook | Content |
|---|---|
| `Data_Cleaning.ipynb` | Missing value handling, outlier removal, feature engineering, encoding |
| `NB01 — EBM/GAM` | Shape function analysis, optimal volume thresholds, marginal effects |
| `NB02 — NGBoost` | Probabilistic regression, hyperparameter tuning, distributional output |
| `NB03 — TabNet` | Deep learning benchmark, attention mechanism analysis |
| `NB04 — CatBoost/GPR` | Responder segmentation, GPR uncertainty estimation, OOD detection |
| `NB05 — SEM` | Structural paths, indirect effects, mediation analysis |

---

## Dashboard Pages

The dashboard serves as the **translational deployment layer** — converting model outputs into accessible, interactive visualizations. It is not the scientific engine itself; the notebooks and models are.

| Page | What it does |
|---|---|
| `/overview` | Dataset exploration, descriptive statistics, subgroup breakdowns |
| `/volume` | Interactive volume–hypertrophy dose-response visualization |
| `/optimizer` | Input your profile → get prediction + recommendation + explanation |
| `/case-study` | Walk through real RCT examples with model output overlaid |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + TypeScript + Tailwind CSS |
| Backend | FastAPI + Python 3.10+ |
| ML | NGBoost · EBM · CatBoost · GPR · TabNet · scikit-learn |

---

## Project Structure

```
STAT3013.Q21_Group07/
│
├── start.bat                  # One-click launcher (Windows)
│
├── src/                       # Frontend (React + Vite)
│   ├── pages/
│   └── components/
│
├── AI_ML/                     # Backend + ML
│   ├── main_fastapi.py        # FastAPI entry point
│   ├── models/                # Trained model files
│   ├── notebooks/             # NB01–NB05 + Data_Cleaning
│   └── requirements.txt
│
└── README.md
```

---

## Installation & Usage

**Requirements:** Python 3.10+ · Node.js 18+

### Clone the Repository

```bash
git clone https://github.com/your-org/STAT3013.Q21_Group07.git
cd STAT3013.Q21_Group07
```

### Quick Start (Windows)

```bash
start.bat
```

### Manual — Frontend

```bash
npm install
npm run dev
```

### Manual — Backend

```bash
cd AI_ML
pip install -r requirements.txt
uvicorn main_fastapi:app --reload
```

### Local URLs

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

### Environment Variables

Create a `.env` file in the project root:

```env
VITE_API_URL=http://localhost:8000
GEMINI_API_KEY=your_key_here
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Server health check |
| `GET` | `/api/v1/model-info` | Model versions and metadata |
| `POST` | `/api/v1/predict` | Submit profile, receive hypertrophy prediction + recommendation |

Interactive docs with full request/response schemas: **http://localhost:8000/docs**

---

## Safety Layer

All predictions pass through a dedicated safety mechanism before being returned to the user:

- **OOD Detection** — flags inputs that fall outside the training data distribution
- **Confidence Intervals** — GPR provides calibrated uncertainty bounds on every prediction
- **Volume Cap** — recommendations are hard-capped at ~32 weekly sets based on literature evidence
- **Extrapolation Warnings** — explicit alerts when the model is operating beyond safe interpolation range

This layer ensures the system communicates uncertainty rather than projecting false confidence.

---

## Key Findings

- **Training volume** is the strongest single predictor of hypertrophy outcomes
- More weekly sets improve results, but **diminishing returns** emerge progressively at higher volumes — practical hypertrophy gains appear to plateau around **~32 weekly sets**, the signature threshold identified by this system
- **Nutritional surplus** consistently and significantly supports hypertrophy across subgroups
- **Training to failure** — effects are context-dependent; not universally beneficial across all populations and experience levels

---

## Strengths & Limitations

**Strengths**
- Built on peer-reviewed RCT literature, not generic fitness app data
- Fully explainable predictions via EBM shape functions — not a black box
- Probabilistic output — the system communicates uncertainty, not false confidence
- Safety-aware: OOD detection and confidence bounds on all recommendations
- End-to-end deployable: from raw literature to live interactive dashboard

**Limitations**
- Small effective sample (~199 observations after strict filtering)
- Indirect effect sizes from meta-analysis, not individual-level raw data
- Potential publication bias in the source literature pool
- External validity requires caution for highly specific or clinical populations

> TrainHyp AI is an **evidence-constrained optimization system**, not a universal prescription. Results should be interpreted within the boundaries of the underlying literature.

---

## Future Work

- Incorporate larger, more diverse RCT datasets
- Add biomarker inputs (testosterone, cortisol, creatine kinase)
- Integrate wearable device data streams for longitudinal tracking
- Explore reinforcement learning for dynamic plan adaptation over time

---

## License

Academic / Educational use only — **STAT3013.Q21 · Group 07**

Not intended for commercial use, clinical application, or medical advice.
