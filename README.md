# CNET 5442 — Sports Analytics through Data and Networks (Spring 2026)

This repository contains the instructional materials for **CNET 5442 (Spring 2026)**, including:
- in-class notebooks
- sample datasets
- reusable helper code
- [coming soon] intro resources for Python + scientific computing

> **Note:** This repo is meant to be a living document throughout the semester. If something changes in class (schedule, links, datasets, etc.), the repo will be updated accordingly.

---

## Course information

- **Course:** CNET 5442 -- Sports Analytics through Data and Networks  
- **Term:** Spring 2026  
- **Meeting time:** Mon/Wed, 2:50–4:30pm  
- **Location:** Richards Hall #140  
- **Instructor:** Brennan Klein (https://brennanklein.com)

Syllabus: `syllabus/CNET_5442_Syllabus_sp26.pdf`

---

## Repo map

- `notebooks/` — in-class notebooks
- `data/` — data access instructions + small samples (large data is not committed)
- (future) `utilities/` — reusable helper functions (e.g., plotting, IO, network helpers)
- (future) `resources/` — standalone primers (including a thorough Python bootcamp notebook)

---


## Getting started

### 1) Clone the repo in the directory where you want it

```bash
git clone [https://github.com/<ORG_OR_USER>/<REPO_NAME>.git](https://github.com/jkbren/cnet5442_sp26.git)
cd cnet5442_sp26
```

### 2) Create the conda environment

This class uses a shared conda environment so notebooks run consistently across machines.

```bash
conda env create -f environment.yml
conda activate cnet5442
```

### 3) Register the environment as a Jupyter kernel (recommended)

```bash
python -m ipykernel install --user --name cnet5442 --display-name "Python (cnet5442)"
```

### 4) Launch Jupyter

```bash
jupyter lab
```

---

## Coursework and grading

- **Attendance and Participation:** 10%
- **Weekly Assignments:** 45% (six coding + analysis assignments)
- **Midterm Project Proposal and Presentation:** 10%
- **Final Project Report and Presentation:** 35%

---

## Final project

The semester culminates in a group project applying course methods to a real sports dataset. Projects may analyze an existing dataset, scrape/collect new data, or extend methods introduced in class.

Deliverables:
- **Proposal & Intermediate Presentation (Week 7/8):** short written description + 5–7 minute presentation
- **Final Report & Presentation (Finals Week):** 8–12 page write-up + group presentation

---

## Schedule → notebook mapping

> Schedule and topics may be adjusted with reasonable notice.

| Class | Date (2026) | Notebook(s) | Topic |
|------:|-------------|-------------|-------|
| 01 | Wed Jan 07 | no notebook | Introduction — Sports as Complex Systems |
| 02 | Mon Jan 12 | no notebook | Data Types Across Sports |
| 03 | Wed Jan 14 | no notebook | Tournament Structures |
| — | Mon Jan 19 | — | **No class (MLK Day)** |
| 04 | Wed Jan 21 | `class_04/class_04_distributions_odds_surprises.ipynb` | Distributions, Odds, & Surprises |
| 05 | Mon Jan 26 | `class_05/class_05_regression_01_moneyball.ipynb` | Regression Pt. 1 — Moneyball Replication |
| 06 | Wed Jan 28 | `class_06/class_06_regression_02_expectation_likelihood.ipynb` | Regression Pt. 2 — Expectation & Likelihood |
| 07 | Mon Feb 02 | `class_07/class_07_regression_03_survival_logistic.ipynb` | Regression Pt. 3 — Survival + Logistic Regression |
| 08 | Wed Feb 04 | `class_08/class_08_bayesian_hot_hand.ipynb` | Regression Pt. 4 — Bayesian Statistics & the Hot Hand |
| 09 | Mon Feb 09 | `class_09/class_09_classification_clustering.ipynb` | Classification & Clustering |
| 10 | Wed Feb 11 | `class_10/class_10_multidimensional_embedding.ipynb` | Multidimensional Data & Embedding |
| — | Mon Feb 16 | — | **No class (Presidents’ Day)** |
| 11 | Wed Feb 18 | no notebook | Invited Speaker (TBD) |
| 12 | Mon Feb 23 | `class_12/class_12_causality_01_intro.ipynb` | Causality Pt. 1 -- Introduction |
| 13 | Wed Feb 25 | `class_13/class_13_causality_02_applied.ipynb` | Causality Pt. 1 -- Applications |
| — | Mon Mar 02 | — | **No class (Spring Break)** |
| — | Wed Mar 04 | — | **No class (Spring Break)** |
| 14 | Mon Mar 09 | `class_14/class_14_ml_01_intro.ipynb` | Machine Learning Pt. 1 — Introduction |
| 15 | Wed Mar 11 | `class_15/class_15_ml_02_march_madness.ipynb` | Machine Learning Pt. 2 — March Madness |
| 16 | Mon Mar 16 | `class_16/class_16_spatiotemporal_hockey.ipynb` | Spatiotemporal Data Analysis: Hockey |
| 17 | Wed Mar 18 | `class_17/class_17_intro_network_science_through_sports.ipynb` | Intro to Network Science Through Sports |
| 18 | Mon Mar 23 | `class_18/class_18_soccer_passing_networks.ipynb` | Networks in Soccer — Passing Networks |
| 19 | Wed Mar 25 | `class_19/class_19_soccer_spatial_passing.ipynb` | Pitch Passing & Spatial Networks |
| 20 | Mon Mar 30 | `class_20/class_20_sequences_pt1.ipynb` | Sequences of Events Pt. 1 |
| 21 | Wed Apr 01 | `class_21/class_21_sequences_pt2.ipynb` | Sequences of Events Pt. 2 |
| 22 | Mon Apr 06 | `class_22/class_22_roles_motifs.ipynb` | Roles and Motifs |
| 23 | Wed Apr 08 | `class_23/class_23_transfer_trade_scouting.ipynb` | Transfer, Trade, and Scouting Networks |
| 24 | Mon Apr 13 | TBD | Information Theory or Ranking with Networks (TBD) |
| 25 | Wed Apr 15 | no notebook | Invited Speaker (TBD) |
| — | Mon Apr 20 | — | **No class (Patriot’s Day)** |
| 26 | Wed Apr 22 | no notebook | Final Presentations |

---


## Academic integrity

All students are expected to follow Northeastern University’s Academic Integrity Policy. Proper citation is required for all external code, data, text, or ideas.
