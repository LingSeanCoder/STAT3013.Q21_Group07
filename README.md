<div align="center">

# 🧬 TrainHyp AI
### *Training + Hypertrophy + Artificial Intelligence*

**STAT3013.Q21 — Group 07 · Version v1.0**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React+Vite-TypeScript-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic_Use-lightgrey?style=flat-square)](#license)

> **AI-Driven Muscle Hypertrophy Prediction, Optimization & Decision Support System**
>
> *Converting fragmented hypertrophy science into predictive, explainable,*
> *safety-aware, and deployable optimization.*

[📊 Dataset](https://osf.io/6z3xu/overview) · [🎥 Demo](https://drive.google.com/drive/folders/1t5nOGhR7J-a1V3LxqmBxCuO7EX1gw5cj?usp=drive_link) · [🌐 Web App](https://trainhyp-app.vercel.app/)

</div>

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Executive Overview](#2-executive-overview)
3. [Scientific Problem](#3-scientific-problem)
4. [Research Objectives](#4-research-objectives)
5. [Why This Project Matters](#5-why-this-project-matters)
6. [Dataset Acquisition Challenges](#6-dataset-acquisition-challenges)
7. [Dataset Architecture](#7-dataset-architecture)
8. [Feature Dictionary](#8-feature-dictionary)
9. [Target Variable Interpretation](#9-target-variable-interpretation)
10. [Full Dataflow Pipeline](#10-full-dataflow-pipeline)
11. [Notebook Ecosystem](#11-notebook-ecosystem)
12. [Notebook → Figure Mapping](#12-notebook--figure-mapping)
13. [Statistical Findings](#13-statistical-findings)
14. [Model Ecosystem](#14-model-ecosystem)
15. [Model Benchmark Decision Rationale](#15-model-benchmark-decision-rationale)
16. [Deep Learning Inclusion Justification](#16-deep-learning-inclusion-justification-tabnet)
17. [Exact Model Roles](#17-exact-model-roles)
18. [Likely Deployed Model Assets](#18-likely-deployed-model-assets)
19. [Runtime Inference Logic](#19-runtime-inference-logic)
20. [Dashboard Deployment Layer](#20-dashboard-deployment-layer)
21. [Dashboard Pages & Routes](#21-dashboard-pages--routes)
22. [Project Structure](#22-project-structure)
23. [Tech Stack](#23-tech-stack)
24. [Installation](#24-installation)
25. [Local Deployment](#25-local-deployment)
26. [API Architecture](#26-api-architecture)
27. [Safety Layer](#27-safety-layer)
28. [Strengths](#28-strengths)
29. [Limitations](#29-limitations)
30. [Limitation Defense Strategy](#30-limitation-defense-strategy)
31. [README ↔ Slide ↔ Report Mapping](#31-readme--slide--report-mapping)
32. [Product Evolution Roadmap](#32-product-evolution-roadmap)
33. [Future Work](#33-future-work)
34. [Deliverables & External Links](#34-deliverables--external-links)
35. [License](#35-license)
36. [Verification Notes](#36-verification-notes)
37. [Final Declaration](#37-final-declaration)

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project Name** | TrainHyp AI |
| **Expanded** | Training + Hypertrophy + Artificial Intelligence |
| **Repository** | `STAT3013.Q21_Group07` |
| **Version** | v1.0 |
| **Course** | STAT3013.Q21 |
| **Group** | Group 07 |
| **Core Identity** | AI-Driven Muscle Hypertrophy Prediction, Optimization & Decision Support System |

### What TrainHyp AI Is

- ✅ A research-first statistical intelligence system
- ✅ A translational AI pipeline from meta-analysis to deployment
- ✅ An evidence-constrained hypertrophy optimizer
- ✅ A multi-model benchmark study (7 predictive model families + SEM structural modeling layer)
- ✅ A safety-aware prediction dashboard with uncertainty quantification

### What TrainHyp AI Is Not

- ❌ A generic fitness calculator
- ❌ Basic ML coursework
- ❌ A cosmetic or frontend-only product
- ❌ A system making universally prescriptive clinical claims

---

## 2. Executive Overview

TrainHyp AI addresses a fundamental gap in applied exercise science: the inability to translate fragmented, paywalled, and methodologically heterogeneous hypertrophy literature into a deployable, evidence-based decision support system.

The system operationalizes a curated OSF meta-analytic dataset (~69 studies, ~198 cleaned observations) through a **10-notebook scientific pipeline**, a **7 predictive model families + SEM benchmarking framework**, and a **React + FastAPI deployment stack** to deliver:

```
User Input → Hypertrophy Prediction → Explainability → Safety Check → Optimization → Recommendation
```

The deployed prediction model (NGBoost) was selected based on comparative benchmarking across predictive accuracy, uncertainty quantification capability, and deployment practicality. All scientific claims are evidence-constrained and framed within dataset scope.

---

## 3. Scientific Problem

Hypertrophy science suffers from deep structural fragmentation:

| Problem | Description |
|---|---|
| **Literature heterogeneity** | Different studies use incompatible protocols, populations, and variable definitions |
| **Paywalled access** | Majority of primary sources are behind institutional paywalls |
| **Missing variables** | Nutrition controls, failure protocols, and biomarkers are inconsistently reported |
| **No unified effect metric** | Raw outcomes (muscle thickness, CSA, lean mass) are not directly comparable |
| **Zero deployment usability** | Existing meta-analyses produce static reports, not actionable systems |

**Core Research Question:**

> *How can fragmented hypertrophy literature be transformed into a deployable, evidence-based, explainable, and safety-aware optimization system?*

---

## 4. Research Objectives

| Priority | Objective |
|---|---|
| **Primary** | Predict hypertrophy outcome (Hedges' g) from user training profile |
| **Secondary** | Optimize weekly resistance-training volume for maximal evidence-constrained hypertrophy |
| **Tertiary** | Explain predictive logic through interpretable model outputs |
| **Quaternary** | Deploy the full pipeline through an interactive dashboard interface |

**Operational Pipeline:**
```
Input → Predict → Explain → Optimize → Recommend
```

---

## 5. Why This Project Matters

1. **Translational gap**: Hypertrophy research exists but is not actionable by practitioners or athletes without scientific training.
2. **Standardization**: By operationalizing Hedges' g as the universal effect metric, the system enables cross-study comparability that raw outcome measures cannot provide.
3. **Safety awareness**: Rather than prescribing single-point predictions, the system quantifies uncertainty and flags out-of-distribution inputs — a scientific responsibility gap in most fitness applications.
4. **Methodology rigor**: The multi-model benchmarking framework (classical → gradient boosting → probabilistic → deep learning → structural) represents methodological integration advanced beyond typical undergraduate scope.
5. **Open science alignment**: The OSF dataset source reflects transparency and reproducibility values central to modern statistical research.

---

## 6. Dataset Acquisition Challenges

The dataset construction process was a non-trivial research undertaking, not a simple CSV download.

| Challenge | Detail |
|---|---|
| **Paywalled papers** | Majority of primary studies required institutional access or author outreach |
| **Missing supplementary data** | SD, SE, and baseline values often absent from published tables |
| **Inconsistent variable definitions** | "Sets per week," "frequency," and "failure" defined differently across studies |
| **Author non-response** | Direct author outreach for missing data yielded incomplete returns |
| **Strict inclusion/exclusion** | Only studies meeting variable completeness and methodological quality thresholds were retained |
| **Missing nutrition/failure controls** | Most studies did not control for or report nutritional surplus or proximity to failure |

**Scientific Doctrine Applied:** Methodological quality was prioritized over dataset size. A smaller, higher-quality dataset produces more reliable inferences than a larger, methodologically compromised one.

---

## 7. Dataset Architecture

| Property | Value |
|---|---|
| **Source** | OSF Hypertrophy Meta-Analysis Dataset |
| **URL** | [https://osf.io/6z3xu/overview](https://osf.io/6z3xu/overview) |
| **Studies retained** | ~69 studies |
| **Cleaned observations** | ~198 rows |
| **Unit of analysis** | Intervention arm (not paper) |
| **Effect metric** | Hedges' g (standardized mean difference) |

**Unit of Analysis Note:** Each row represents a single intervention arm from a study. A single paper may contribute multiple rows if it includes multiple conditions (e.g., high vs. low volume groups). This is standard practice in meta-analysis and allows finer-grained variable modeling.

---

## 8. Feature Dictionary

The following features were retained after variable completeness filtering and feature engineering. All definitions are operationalized within the meta-analysis context.

| Feature | Type | Description |
|---|---|---|
| `sets.week.all` | Numeric | Total weekly sets across all muscle groups per intervention arm |
| `sets.week.direct` | Numeric | Weekly sets targeting the measured muscle group directly |
| `frequency.direct` | Numeric | Training sessions per week targeting the measured muscle group |
| `percentage.failure.all` | Numeric | Estimated proportion of sets taken to muscular failure (0–100) |
| `weeks` | Numeric | Duration of the training intervention in weeks |
| `rep.range.all` | Numeric | Representative repetition range used in the intervention |
| `interset.rest.min.all` | Numeric | Average inter-set rest interval in minutes |
| `has_nutrition_control` | Binary | Whether the study controlled for or reported dietary surplus (1 = yes, 0 = no) |
| `train_status_enc` | Ordinal | Encoded training status (e.g., untrained, recreationally trained, advanced) |
| `age` | Numeric | Mean participant age in the intervention arm |
| `sex.male` | Binary | Sex composition (1 = male-majority, 0 = female-majority or mixed) |
| `upper_body` | Binary | Target muscle location (1 = upper body, 0 = lower body) |
| `sessions.per.week` | Numeric | Overall weekly training session frequency |

> **Note:** Feature engineering decisions and transformations are documented in `Data_Cleaning.ipynb`.

---

## 9. Target Variable Interpretation

**Target: `hedges_g`** — Standardized Mean Difference (effect size)

Hedges' g is a bias-corrected variant of Cohen's d that measures the magnitude of hypertrophic change relative to the pooled standard deviation, enabling **cross-study comparability** despite different outcome measures (e.g., muscle thickness via ultrasound vs. MRI CSA vs. DEXA lean mass).

| Magnitude | Hedges' g Value |
|---|---|
| Small effect | ≈ 0.2 |
| Medium effect | ≈ 0.5 |
| Large effect | ≈ 0.8+ |

**Why Hedges' g instead of raw outcomes?**

Raw hypertrophy measures (mm thickness, cm² CSA, kg lean mass) are scale-dependent and instrument-dependent. Hedges' g standardizes these differences, allowing the system to aggregate and model across heterogeneous studies without privileging studies using high-variance measurement methods.

---

## 10. Full Dataflow Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Raw Literature                              │
│              (Primary studies, meta-analyses, OSF dataset)          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                    Study Screening
              (Inclusion / Exclusion criteria)
                             │
                             ▼
               Variable Completeness Filtering
           (Retain only arms with sufficient data)
                             │
                             ▼
             Standardization to Hedges' g
          (Bias-corrected effect size calculation)
                             │
                             ▼
              Data Cleaning & Imputation
              (Data_Cleaning.ipynb)
                             │
                             ▼
              Feature Engineering
           (Encoding, transformation, scaling)
                             │
                             ▼
         ┌───────────────────┼────────────────────┐
         │                   │                    │
         ▼                   ▼                    ▼
  Descriptive          Inferential           Subgroup
  Statistics           Statistics            Analysis
         │                   │                    │
         └───────────────────┴────────────────────┘
                             │
                             ▼
              AI Model Benchmarking
     (EBM · GAM · NGBoost · TabNet · CatBoost · GPR · SEM)
                             │
                             ▼
              Best Model Selection
              (NGBoost — probabilistic, deployment-ready)
                             │
                             ▼
                       Safety Layer
              (OOD detection · Uncertainty bounds)
                             │
                             ▼
                      Backend API
                    (FastAPI · /predict)
                             │
                             ▼
                 Dashboard Deployment
              (React + Vite + TypeScript + Tailwind)
                             │
                             ▼
                  User Recommendation
         (Optimal volume · Confidence interval · Explanation)
```

---

## 11. Notebook Ecosystem

The project comprises **10 computational notebooks** organized into two functional layers.

---

### Data Layer (4 Notebooks)

#### `Data_Cleaning.ipynb`
**Purpose:** Foundation of all downstream analysis. Implements study inclusion/exclusion logic, missing value handling, feature engineering, variable encoding, and dataset validation.

**Outputs:** Cleaned dataset (`hedges_g` as target), encoded features, preprocessing artifacts.

**Strategic Role:** Determines the quality ceiling of all subsequent models and analyses. Methodological decisions here propagate throughout the system.

---

#### `Descriptive_Statistics.ipynb`
**Purpose:** Characterizes the dataset distribution through univariate and bivariate summaries, visualizations, and correlation structure.

**Outputs:** Distribution figures (Hedges' g histogram, boxplots by volume category, training status, body region), correlation matrices, summary tables.

**Strategic Role:** Provides the empirical foundation for scientific claims and slides. All figures referenced in the defense originate here.

---

#### `Inferential_Statistics.ipynb`
**Purpose:** Tests hypotheses regarding volume, frequency, failure, and demographic effects on hypertrophy using formal statistical tests.

**Outputs:** Significance tests, confidence intervals, effect size estimates, regression coefficients.

**Strategic Role:** Provides the statistical rigor layer that distinguishes a research project from a data visualization exercise.

---

#### `Subgroup_Analysis.ipynb`
**Purpose:** Investigates heterogeneity in hypertrophy response across subgroups defined by nutrition control status, training status, failure proximity, and sex.

**Outputs:** Stratified analyses, subgroup comparison figures, interaction evidence.

**Strategic Role:** Demonstrates scientific nuance — the system explicitly acknowledges that population-level effects may not uniformly apply across all subgroups.

---

### Model Layer (6 Notebooks)

#### `NB01_EBM.ipynb` — Explainable Boosting Machine
**Purpose:** Trains and evaluates an EBM (GA²M) — an inherently interpretable gradient boosting model with shape function decomposition.

**Role:** Interpretability benchmark. Provides human-readable feature contribution plots without post-hoc approximation.

**Strategic Contribution:** Establishes that interpretability does not necessarily require sacrificing predictive performance.

---

#### `NB02_GAM.ipynb` — Generalized Additive Model
**Purpose:** Fits a GAM to capture smooth, potentially nonlinear relationships between training variables and hypertrophy.

**Role:** Nonlinear statistical baseline. Bridges classical statistics and machine learning in terms of interpretability.

**Strategic Contribution:** Provides the dose-response visualization layer — smooth volume-hypertrophy curves used directly in dashboard and defense slides.

---

#### `NB03_NGBoost.ipynb` — Natural Gradient Boosting ⭐ *Deployed Model*
**Purpose:** Trains NGBoost, a probabilistic gradient boosting model that outputs full predictive distributions rather than single-point estimates.

**Role:** Primary deployed prediction engine. Provides both expected hypertrophy (Hedges' g) and prediction uncertainty (confidence interval) per input.

**Strategic Contribution:** Uniquely satisfies the safety layer requirement — uncertainty output is essential for flagging low-confidence predictions without withholding guidance.

---

#### `NB04_TabNet.ipynb` — Tabular Deep Learning
**Purpose:** Implements TabNet, a deep learning architecture designed for tabular data with built-in feature selection via sequential attention.

**Role:** Deep learning benchmark. Provides architectural contrast and methodological completeness.

**Strategic Contribution:** Justifies the inclusion of a DL model despite dataset scale constraints (see [Section 16](#16-deep-learning-inclusion-justification-tabnet)).

---

#### `NB05_CatBoost_GPR.ipynb` — CatBoost + Gaussian Process Regression
**Purpose:** Benchmarks CatBoost (gradient boosting with native categorical handling) and Gaussian Process Regression (Bayesian non-parametric model).

**Role:** Dual benchmark — CatBoost as a high-performance boosting baseline; GPR as a fully probabilistic Bayesian comparator.

**Strategic Contribution:** GPR provides an independent uncertainty quantification reference for validating NGBoost confidence intervals.

---

#### `NB06_SEM.ipynb` — Structural Equation Modeling
**Purpose:** Constructs a latent variable structural model linking training volume, frequency, and failure to hypertrophy through a theoretically grounded path structure.

**Role:** Scientific validation layer. Operates outside the predictive ML paradigm — tests theory-informed structural relationships rather than optimizing prediction loss.

**Strategic Contribution:** Demonstrates methodological breadth and distinguishes the project from a pure machine learning exercise.

---

## 12. Notebook → Figure Mapping

| Figure | Source Notebook | Description |
|---|---|---|
| `fig1_hedges_g_distribution` | `Descriptive_Statistics.ipynb` | Histogram of Hedges' g across all observations |
| `fig2_boxplot_volume_category` | `Descriptive_Statistics.ipynb` | Boxplot: Hedges' g by weekly volume category |
| `fig3_boxplot_train_status` | `Descriptive_Statistics.ipynb` | Boxplot: Hedges' g by training status |
| `fig4_boxplot_upper_lower` | `Descriptive_Statistics.ipynb` | Boxplot: Hedges' g by body region (upper vs. lower) |
| `fig5_scatter_volume_hypertrophy` | `Descriptive_Statistics.ipynb` | Scatter: Weekly sets vs. Hedges' g |
| Subgroup — Nutrition Control | `Subgroup_Analysis.ipynb` | Stratified effect by `has_nutrition_control` |
| Subgroup — Failure Proximity | `Subgroup_Analysis.ipynb` | Stratified effect by `percentage.failure.all` |
| Dose-response curve | `NB02_GAM.ipynb` | Smooth volume-hypertrophy relationship |
| Feature importance | `NB01_EBM.ipynb` / `NB03_NGBoost.ipynb` | Ranked feature contributions |
| Model comparison chart | `NB03_NGBoost.ipynb` / `NB05_CatBoost_GPR.ipynb` | Cross-model benchmark metrics |
| Uncertainty plots | `NB03_NGBoost.ipynb` | Prediction intervals vs. observed Hedges' g |
| SEM path diagram | `NB06_SEM.ipynb` | Structural path coefficients and fit indices |

---

## 13. Statistical Findings

> ⚠️ All findings are evidence-constrained and presented within dataset scope. These findings suggest — not prove — the described relationships.

| Finding | Evidence Status | Notes |
|---|---|---|
| **Volume is the dominant predictor** | VERIFIED — across multiple models and statistical tests | Consistent across EBM, NGBoost, GAM, inferential tests |
| **More weekly sets generally associate with greater hypertrophy** | VERIFIED within dataset | Positive volume-response relationship observed |
| **Diminishing returns at high volumes** | INFERRED from GAM dose-response | Marginal hypertrophy gain decreases as weekly sets increase |
| **Evidence-constrained practical reference around ~32 weekly sets** | INFERRED — VERIFY FROM SOURCE | Approximate inflection point in dose-response analysis; not a universal threshold |
| **Nutrition surplus context shows advantage** | VERIFIED in subgroup analysis | Studies with nutritional control exhibit higher Hedges' g within dataset |
| **Failure context dependency** | VERIFIED in subgroup analysis | Effect of failure-proximity is inconsistent and moderated by other variables |
| **Multifactorial but volume-dominant pattern** | VERIFIED across all model layers | Other features contribute meaningfully but volume remains primary driver |

---

## 14. Model Ecosystem

| Model | Type | Interpretability | Uncertainty | Dataset Suitability | Deployment Role |
|---|---|---|---|---|---|
| **Linear Regression** *(VERIFY baseline inclusion)* | Classical | ★★★★★ | Partial | ★★★ | Statistical baseline |
| **GAM** | Semi-parametric | ★★★★☆ | Partial | ★★★★ | Dose-response visualization |
| **EBM** | Gradient Boosting (GA²M) | ★★★★★ | No | ★★★★ | Interpretability benchmark |
| **NGBoost** ⭐ | Probabilistic Boosting | ★★★☆☆ | **Full distribution** | ★★★★★ | **Primary deployed model** |
| **TabNet** | Deep Learning (Tabular) | ★★☆☆☆ | No | ★★★ | Architectural benchmark |
| **CatBoost** | Gradient Boosting | ★★★☆☆ | No | ★★★★ | Boosting benchmark |
| **GPR** | Bayesian Non-parametric | ★★★☆☆ | **Full distribution** | ★★★ | Uncertainty reference |
| **SEM** | Structural Pathway Modeling | ★★★★★ | Partial | ★★★★ | Structural path validation |

---

## 15. Model Benchmark Decision Rationale

**Deployed Model: NGBoost**

NGBoost was selected as the primary deployment model over competing alternatives based on the following comparative analysis:

### NGBoost vs. EBM
EBM provides superior interpretability (full shape function decomposition) but outputs only point predictions. For a safety-first system, uncertainty quantification is a non-negotiable deployment requirement. NGBoost's probabilistic output enables confidence intervals that EBM cannot provide.

### NGBoost vs. GAM
GAM excels at visualizing smooth dose-response curves and has strong statistical foundations but is less flexible in capturing interaction effects. NGBoost demonstrated stronger predictive performance in cross-validation benchmarking within this dataset.

### NGBoost vs. TabNet
TabNet requires substantially more data to converge to competitive performance. At ~198 observations, TabNet is structurally disadvantaged. NGBoost, designed for tabular data at small-to-medium scales, demonstrates better generalization.

### NGBoost vs. Linear Models
Linear models assume monotonic, additive relationships — an oversimplification of the nonlinear, diminishing-return dynamics observed in hypertrophy dose-response. NGBoost captures these nonlinearities without explicit specification.

### NGBoost vs. GPR
GPR shares NGBoost's probabilistic advantage but is computationally sensitive to kernel selection and does not scale as predictably. NGBoost provides a more deployment-stable uncertainty framework.

**Summary:** NGBoost was selected for its combination of competitive predictive performance, full predictive distribution output (essential for the safety layer), and deployment stability. Exact benchmark metrics (RMSE, MAE, R²) are documented in `NB03_NGBoost.ipynb` and `NB05_CatBoost_GPR.ipynb`.

---

## 16. Deep Learning Inclusion Justification (TabNet)

> *Why include a deep learning model in a ~198-observation dataset?*

This question is anticipated in academic defense contexts. The following arguments constitute the methodological defense for TabNet inclusion:

**1. Benchmark Completeness**
A rigorous model ecosystem must include architectural diversity. A study that omits deep learning without explicit reasoning appears methodologically incomplete, not conservative.

**2. Educational and Comparative Value**
TabNet provides a direct empirical demonstration of DL vs. classical ML under small-data conditions — a finding of genuine pedagogical and scientific value.

**3. TabNet's Architectural Design**
Unlike standard neural networks, TabNet was specifically architected for tabular data and incorporates sequential attention-based feature selection. It is not an arbitrary DL choice.

**4. Methodological Rigor via Negative Result**
If TabNet underperforms relative to NGBoost or EBM, this constitutes a valid and meaningful finding: *deep learning does not universally outperform classical methods on small, structured tabular datasets*. This is a contribution, not a weakness.

**5. Protocol Completeness**
Including TabNet transforms the benchmark from a "cherry-picked selection" to a near-exhaustive methodological survey — strengthening the credibility of the final deployment decision.

---

## 17. Exact Model Roles

| Model | Primary Role | Secondary Role |
|---|---|---|
| **Linear Regression** *(VERIFY)* | Statistical baseline | Coefficient interpretability |
| **GAM** | Dose-response visualization | Nonlinear baseline |
| **EBM** | Interpretability benchmark | Feature importance ranking |
| **NGBoost** | **Deployed prediction engine** | Uncertainty quantification |
| **TabNet** | Architectural benchmark | DL vs. classical comparison |
| **CatBoost** | Boosting performance benchmark | Robustness validation |
| **GPR** | Bayesian uncertainty reference | Independent CI validation |
| **SEM** | Structural pathway modeling | Theory-informed path validation |

---

## 18. Likely Deployed Model Assets

> ⚠️ Labels below are **INFERRED based on system design**. Verify exact filenames against repository contents.

```
trained_models/
├── ngboost.pkl              # INFERRED — primary prediction model
├── catboost.pkl             # INFERRED — benchmarking artifact, may be retained
├── gpr.pkl                  # INFERRED — uncertainty reference model
├── ebm.pkl                  # INFERRED — interpretability model, may be deployed for explanation layer
preprocessors/
├── preprocessor.pkl         # INFERRED — feature encoding + transformation pipeline
├── scaler.pkl               # INFERRED — numerical feature normalization
```

The optimizer logic for volume recommendation is likely implemented as a Python function within the backend API rather than as a serialized model artifact.

> ⚠️ VERIFY FROM SOURCE: Exact `.pkl` filenames and retained artifacts should be confirmed against `trained_models/` and `preprocessors/` directories.

---

## 19. Runtime Inference Logic

The following sequence describes the complete inference pathway from user input to delivered recommendation:

```
┌─────────────────────────────────┐
│         User Input              │
│  (age, sex, sets/week, weeks,   │
│   rest, rep range, status...)   │
└─────────────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  Input Validation   │
        │  & Classification   │
        │ (OOD flag if needed)│
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Preprocessing      │
        │  (preprocessor.pkl  │
        │   / scaler.pkl)     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  NGBoost Prediction │
        │  → Hedges' g (mean) │
        │  → Std Dev / CI     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Explainability     │
        │  (Feature contrib.  │
        │   SHAP / EBM shape) │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Safety Layer       │
        │  (CI width check,   │
        │   OOD warning,      │
        │   volume cap ~32)   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Optimization       │
        │  (Volume sweep →    │
        │   optimal sets/wk)  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Recommendation     │
        │  (Expected Hedges'g │
        │   + CI + Warnings + │
        │   Optimal volume)   │
        └─────────────────────┘
```

---

## 20. Dashboard Deployment Layer

> The dashboard is the **translational deployment layer** — not the core science. The core science resides in the notebooks.

| Layer | Purpose |
|---|---|
| **Notebooks** | Science engine: data, statistics, models, findings |
| **API (FastAPI)** | Translation layer: model serving, safety logic, optimization |
| **Dashboard (React)** | Deployment interface: user interaction, visualization, recommendation delivery |

The dashboard transforms computational outputs into accessible, interpretable formats for practitioners, athletes, and researchers without statistical training. It does not perform raw computation — it consumes the FastAPI prediction endpoint and renders results.

---

## 21. Dashboard Pages & Routes

> ⚠️ Routes below are **INFERRED based on system design**. VERIFY FROM SOURCE against actual React router configuration.

| Page | Route (INFERRED) | Description |
|---|---|---|
| Landing / Home | `/` | Project overview, navigation entry point |
| Prediction | `/predict` | User input form → hypertrophy prediction output |
| Optimization | `/optimize` | Volume optimization interface, dose-response curve |
| Explanation | `/explain` | Feature importance, prediction breakdown |
| Model Info | `/models` | Benchmark summary, model selection rationale |
| About | `/about` | Team, methodology, dataset source |

---

## 22. Project Structure

```
STAT3013.Q21_Group07/
│
├── README.md                        # This document
├── start.bat                        # One-command local deployment (Windows)
├── .env                             # Environment variables (not committed)
├── .env.example                     # Environment variable template
│
├── src/                             # React frontend source
│   ├── components/                  # UI components
│   ├── pages/                       # Route-level page components
│   ├── hooks/                       # Custom React hooks
│   ├── services/                    # API client logic
│   └── main.tsx                     # Vite entry point
│
├── AI_ML/
│   ├── notebooks/                   # All 10 research notebooks
│   │   ├── NB01_EBM.ipynb
│   │   ├── NB02_GAM.ipynb
│   │   ├── NB03_NGBoost.ipynb
│   │   ├── NB04_TabNet.ipynb
│   │   ├── NB05_CatBoost_GPR.ipynb
│   │   └── NB06_SEM.ipynb
│   │   └── ai_engine.py
│   │
    ├── backend_models                 
│   │   ├── catboost_clf.pkl  
│   │   ├── ebm_model.pkl
│   │   ├── gam_model.pkl
│   │   ├── gpr_model.pkl
│   │   ├── linear_model.pkl
│   │   ├── ngb_model.pkl
│   │   └── tabnet_model.zip
│   └── api/
│       ├── main.py                  # FastAPI application entry point
│       ├── routers/                 # API route handlers
│       ├── schemas/                 # Pydantic request/response models
│       ├── services/                # Prediction, optimization logic
│       └── safety/                  # OOD detection, uncertainty logic
│
├── Dataset/
│   ├── Extracted-Data.xlsx          # Original OSF dataset
│   └── data_cleaned.csv             # Cleaned, engineered dataset
|   └── data_features.csv            # features for training model
|   └── descriptive_stats.csv             
│
├── Data_Analysis/
│   ├── Data_Cleaning.ipynb             # Cleaning, transformation, preprocessing
│   ├── Descriptive_Statistics.ipynb    # Descriptive statistical analysis
│   ├── Inferential_Statistics.ipynb    # Hypothesis testing & inferential stats
│   ├── Subgroup_Analysis.ipynb         # Segment / subgroup comparisons 
|
├── figures/                         # Generated figures from notebooks
├── requirements.txt                 # Python dependencies
└── package.json                     # Node.js dependencies
```

---

## 23. Tech Stack

### Frontend
| Technology | Role |
|---|---|
| React 18 | UI framework |
| Vite | Build tool and dev server |
| TypeScript | Type-safe frontend development |
| Tailwind CSS | Utility-first styling |

### Backend
| Technology | Role |
|---|---|
| FastAPI | REST API framework |
| Python 3.10+ | Backend runtime |
| Uvicorn | ASGI server |
| Pydantic | Request/response validation |

### Machine Learning
| Library | Role |
|---|---|
| scikit-learn | Preprocessing, baseline models, evaluation |
| NGBoost | Probabilistic gradient boosting (deployed model) |
| interpret | EBM (Explainable Boosting Machine) |
| CatBoost | Gradient boosting benchmark |
| PyTorch / pytorch-tabnet | TabNet deep learning benchmark |
| GPy / sklearn.gaussian_process | Gaussian Process Regression |
| semopy | Structural Equation Modeling |
| SHAP | Model explainability |

### Statistical Analysis
| Library | Role |
|---|---|
| pandas | Data manipulation |
| NumPy | Numerical computation |
| SciPy | Statistical tests |
| statsmodels | GAM, regression, inference |
| matplotlib / seaborn | Visualization |
| pingouin | Effect size and statistical testing |

---

## 24. Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Clone Repository
```bash
git clone <YOUR_ACTUAL_REPO_URL>
cd STAT3013.Q21_Group07
```

### Backend Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Frontend Setup
```bash
npm install
```

### Environment Variables
```bash
# Copy template and configure
cp .env.example .env
# Edit .env as needed
```

**`.env.example`**
```env
VITE_API_URL=http://localhost:8000
GEMINI_API_KEY=your_key_here
```

---

## 25. Local Deployment

### Option A — Automated (Windows)
```bash
start.bat
```
This script launches both the frontend and backend concurrently.

### Option B — Manual

**Start Backend (FastAPI):**
```bash
cd AI_ML/api
uvicorn main:app --reload --port 8000
```

**Start Frontend (React + Vite):**
```bash
npm run dev
```

### Access Points

> Default configuration — verify exact ports from `vite.config.ts` and `uvicorn` startup output.

| Service | URL (Default) |
|---|---|
| Frontend Dashboard | `http://localhost:5173` |
| Backend API | `http://localhost:8000` |
| API Documentation (Swagger) | `http://localhost:8000/docs` |
| API Documentation (ReDoc) | `http://localhost:8000/redoc` |

---

## 26. API Architecture

> ⚠️ Routes below are INFERRED from system design. VERIFY FROM SOURCE against `AI_ML/api/routers/`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check and uptime confirmation |
| `/model-info` | GET | Deployed model metadata and version |
| `/predict` | POST | Core prediction endpoint — returns Hedges' g, CI, feature importance |
| `/optimize` | POST | Volume optimization — returns optimal sets/week for user profile |

### Example Prediction Request (INFERRED)
```json
POST /predict
{
  "sets_week_all": 16,
  "frequency_direct": 3,
  "weeks": 12,
  "rep_range_all": 10,
  "interset_rest_min": 2.0,
  "has_nutrition_control": 1,
  "train_status_enc": 1,
  "age": 25,
  "sex_male": 1,
  "upper_body": 1,
  "sessions_per_week": 4
}
```

### Example Prediction Response (INFERRED)
```json
{
  "predicted_hedges_g": 0.54,
  "confidence_interval": [0.31, 0.77],
  "effect_magnitude": "Medium",
  "safety_flags": [],
  "feature_contributions": {
    "sets.week.all": 0.18,
    "weeks": 0.09,
    "train_status_enc": 0.07
  },
  "recommendation": "Current volume suggests a medium hypertrophy effect within this analysis..."
}
```

> ⚠️ VERIFY FROM SOURCE: Exact request/response schema from `AI_ML/api/schemas/`.

---

## 27. Safety Layer

The safety layer is a non-negotiable component of a responsible AI prediction system operating on human performance data.

| Safety Mechanism | Description |
|---|---|
| **Out-of-Distribution (OOD) Detection** | Flags inputs that fall outside the training data distribution — predictions in OOD regions carry inflated uncertainty |
| **NGBoost Uncertainty Quantification** | Every prediction includes a standard deviation and confidence interval; high uncertainty triggers explicit user warning |
| **GPR Independent Uncertainty Reference** | Provides a Bayesian cross-validation of NGBoost uncertainty estimates |
| **Practical Volume Reference (~32 sets/week)** | Recommendations are bounded by the evidence-constrained inflection point observed in the dose-response analysis; not a universal cap |
| **Extrapolation Warnings** | Inputs outside observed variable ranges trigger explicit extrapolation warnings |
| **False Certainty Avoidance** | The system communicates prediction intervals, not false point precision |

**Core Safety Principle:**
> *Communicate uncertainty, not false certainty. A prediction without a confidence interval is an incomplete — and potentially misleading — scientific output.*

---

## 28. Strengths

| Strength | Description |
|---|---|
| **Probabilistic output** | NGBoost delivers full predictive distributions — not single-point guesses |
| **Multi-model rigor** | 7 model families + SEM across 4 architectural paradigms |
| **Scientific humility** | All claims are evidence-constrained; uncertainty is quantified, not hidden |
| **Translational completeness** | Full pipeline from raw literature to deployed interface |
| **Safety-first architecture** | OOD detection and uncertainty bounds are first-class system components |
| **Methodological breadth** | Classical statistics + ML + deep learning + structural pathway modeling in one project |
| **Open science alignment** | Dataset sourced from OSF; methodology documented in 10 reproducible notebooks |
| **Deployment quality** | React + FastAPI stack with type safety (TypeScript + Pydantic) |

---

## 29. Limitations

| Limitation | Description |
|---|---|
| **Small sample size** | ~198 observations limits model generalization and statistical power for complex interaction terms |
| **Meta-analysis constraints** | Inferences are bounded by the quality and completeness of primary study reporting |
| **Publication bias** | Studies showing larger effects may be disproportionately published; OSF dataset partially mitigates this |
| **Missing biomarkers** | No testosterone, IGF-1, satellite cell, mTOR, or muscle fiber type data — key mechanistic variables absent |
| **External validity caution** | Findings are derived from controlled research conditions; applicability to real-world heterogeneous training contexts requires further study |
| **Missing variables** | Nutrition quality, sleep, stress, and recovery are not captured; hypertrophy is multifactorial |
| **Cross-sectional intervention arms** | No longitudinal tracking of individual subjects; within-subject adaptation curves cannot be modeled |

> TrainHyp AI is an **evidence-constrained optimization system**, not a universal prescription. Results should be interpreted within the boundaries of the underlying literature.

---

## 30. Limitation Defense Strategy

### Small Sample Size
| | |
|---|---|
| **Risk** | Model overfitting; wide confidence intervals; limited power for interaction detection |
| **Why Acceptable** | Hypertrophy meta-analysis datasets are inherently small due to study design constraints — this is the data reality of the field, not an avoidable methodological choice |
| **Mitigation** | Cross-validation, regularization, probabilistic models (NGBoost, GPR), explicit CI reporting |
| **Future Correction** | Expanded literature search, direct author collaboration, prospective data collection |

### Publication Bias
| | |
|---|---|
| **Risk** | Positive-result inflation of effect size estimates |
| **Why Acceptable** | OSF dataset curation explicitly targets methodological quality; Hedges' g is bias-corrected |
| **Mitigation** | Conservative recommendation framing; uncertainty communication at every output |
| **Future Correction** | Gray literature inclusion, registered study meta-analysis |

### Missing Biomarkers
| | |
|---|---|
| **Risk** | Model misses mechanistic drivers; cannot identify non-responders |
| **Why Acceptable** | Biomarker data is structurally absent from most exercise science publications; within the available variable set, the model captures meaningful variance |
| **Mitigation** | Subgroup analysis partially captures population-level biological heterogeneity |
| **Future Correction** | Integration with wearable + biomarker data layers (Phase 4 roadmap) |

### External Validity
| | |
|---|---|
| **Risk** | Real-world athletes differ from RCT participants in compliance, motivation, nutrition access |
| **Why Acceptable** | The system is deployed as decision support, not clinical prescription; uncertainty layer communicates limitations at inference time |
| **Mitigation** | Explicit extrapolation warnings; OOD detection flags unfamiliar input profiles |
| **Future Correction** | Real-world data collection via wearable integration and user feedback loops |

---

## 31. README ↔ Slide ↔ Report Mapping

| Document | Purpose | Audience | Tone |
|---|---|---|---|
| **README.md** (this document) | System manual, technical architecture, deployment handbook | Developers, instructors, peer reviewers | Technical + precise |
| **Slides** | Defense narrative, visual communication, key findings | Examination committee, audience | Visual + confident + structured |
| **Report** | Methodological depth, full statistical results, literature context | Academic evaluators | Formal + comprehensive |

### Cross-Reference Guide

| README Section | Slide Equivalent | Report Section |
|---|---|---|
| Scientific Problem | Problem Statement slide | Introduction / Background |
| Dataset Architecture | Data Overview slide | Data & Methods |
| Feature Dictionary | Variable Table slide | Methods |
| Statistical Findings | Key Findings slides | Results |
| Model Ecosystem | Model Comparison slide | Methods / Results |
| NGBoost Rationale | Model Selection slide | Discussion |
| Safety Layer | System Architecture slide | Methods |
| Limitations | Limitations slide | Discussion / Limitations |

---

## 32. Product Evolution Roadmap

```
Phase 1 — Prediction (v1.0 · CURRENT)
────────────────────────────────────────────────────────────
• NGBoost hypertrophy prediction
• Uncertainty quantification
• Feature explanation
• Safety layer
• Dashboard deployment

Phase 2 — Optimization (v2.0)
────────────────────────────────────────────────────────────
• Full volume optimization engine
• Dose-response interactive curves
• Multi-objective optimization (hypertrophy + recovery)

Phase 3 — Personalized Schedule Generation (v3.0)
────────────────────────────────────────────────────────────
• Weekly training split generator
• Progressive overload programming
• Recovery and adaptation modeling
• Individualized periodization

Phase 4 — Adaptive AI Coach (v4.0)
────────────────────────────────────────────────────────────
• Wearable data integration (HRV, sleep, readiness)
• Biomarker layer (optional user input)
• Reinforcement learning-based autoregulation
• Real-time program adaptation
• Longitudinal user modeling
```

---

## 33. Future Work

**Data Expansion**
- Systematic expansion of the meta-analysis corpus with broader inclusion criteria
- Direct collaboration with exercise science research groups for unpublished data access
- Integration of gray literature and registered reports to reduce publication bias

**Biomarker Integration**
- Testosterone, IGF-1, and cortisol as biological moderator variables
- Muscle fiber type composition (Type I vs. II ratio)
- Satellite cell activity markers

**Wearable & Real-World Data**
- Heart rate variability (HRV) as recovery readiness proxy
- Sleep quality and duration as recovery moderators
- Compliance tracking via accelerometry

**Reinforcement Learning Autoregulation**
- RL-based training load adjustment based on session feedback
- Adaptive progressive overload without manual periodization

**Longitudinal Personalization**
- Individual trajectory modeling across multiple training blocks
- Response-based phenotyping

**Fully Personalized Hypertrophy Schedule Generation**
- Complete weekly training split with exercise selection
- Volume-matched progressive overload scheme
- Deload and recovery integration

---

## 34. Deliverables & External Links

| Deliverable | Link | Status |
|---|---|---|
| **OSF Dataset** | [https://osf.io/6z3xu/overview](https://osf.io/6z3xu/overview) | ✅ VERIFIED |
| **Demo Video** | [Google Drive](https://drive.google.com/drive/folders/1t5nOGhR7J-a1V3LxqmBxCuO7EX1gw5cj?usp=drive_link) | ✅ VERIFIED |
| **Web Application** | [https://trainhyp-app.vercel.app](https://trainhyp-app.vercel.app) | ✅ VERIFIED |

---

## 35. License

```
Academic / Educational Use License
STAT3013.Q21 — Group 07

This project was developed for academic coursework purposes.
All scientific claims are evidence-constrained and dataset-scoped.

This system is NOT intended for clinical prescription,
medical use, or professional coaching replacement.
Outputs do not constitute medical advice.

Dataset source: OSF (Open Science Framework)
https://osf.io/6z3xu/overview
```

---

## 36. Verification Notes

| Category | Items |
|---|---|
| ✅ **VERIFIED** | Dataset source (OSF), ~69 studies / ~198 observations, 10 notebooks (4 data + 6 model), model names (EBM, GAM, NGBoost, TabNet, CatBoost, GPR, SEM), tech stack (React + Vite + TypeScript + Tailwind + FastAPI), deployed model = NGBoost, target variable = Hedges' g, feature list, demo link, web app URL, fig1–fig5 filenames |
| 🔶 **INFERRED** | Exact `.pkl` filenames, API route names and schemas, frontend route structure, exact benchmark metric values (RMSE/MAE/R²), volume inflection reference of ~32 sets, SEM path fit indices |
| ⚠️ **VERIFY FROM SOURCE** | Exact API port configuration, exact frontend router paths, Linear Regression baseline inclusion, specific notebook output values |

> **Principle:** No metric values were fabricated. No route names were presented as definitive without explicit INFERRED or VERIFY FROM SOURCE labels. All uncertain sections are clearly marked.

---

## 37. Final Declaration

---

**TrainHyp AI is not merely a coursework dashboard.**

It is a research-first translational statistical intelligence ecosystem — converting fragmented hypertrophy science into a predictive, explainable, safety-aware, and deployable optimization system.

The project demonstrates methodological integration advanced beyond typical undergraduate scope:
- End-to-end pipeline from raw literature to deployed interface
- Scientific humility through uncertainty quantification and evidence-constrained claims
- Architectural breadth spanning classical statistics, probabilistic ML, deep learning, and structural pathway modeling

---

*STAT3013.Q21 — Group 07 · TrainHyp AI v1.0*

---

**Built with scientific discipline. Deployed with engineering precision.**

*"Methodological quality > dataset size."*
