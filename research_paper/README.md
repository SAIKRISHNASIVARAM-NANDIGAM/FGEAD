# FGEAD Research Paper Artifacts

This directory contains the revised IEEE-style research paper and supporting artifacts generated from the **FGEAD (Feature-Level Graph-Based Explainable Anomaly Detection)** codebase.

## Directory Structure

```
research_paper/
│
├── paper.tex                  # Complete IEEE-style LaTeX manuscript (6-8 pages)
├── paper.md                   # Full Markdown version of the research paper
├── references.bib             # BibTeX database with 20 verified research references
│
├── figures/                   # Paper diagrams and evaluation plots
│   ├── README.md
│   ├── system_architecture.png
│   ├── methodology.png
│   ├── model_results.png
│   ├── xai_feature_importance.png
│   ├── confusion_matrix.png
│   └── explanation_example.png
│
├── literature/                # Literature review artifacts
│   ├── literature_review.md   # 20-paper comparative summary table
│   ├── research_gap.md        # Detailed research gap analysis
│   └── 20_papers.md           # Full bibliographic details of 20 verified papers
│
├── verification/              # Audit and project facts
│   ├── project_facts.md       # Extracted codebase technical facts
│   └── reference_verification.md # Verification report for all 20 references
│
├── author_review.md           # Viva defense guide & student review checklist
├── final_audit_report.md      # Final academic audit & submission readiness report
└── README.md                  # Artifact index (this file)
```

## Summary of Paper Contributions & Results

- **Title:** FGEAD: A Feature-Level Graph-Based Framework for Explainable Anomaly Detection in Multivariate Time Series
- **Dataset:** Server Machine Dataset (SMD Machine 1-1, 38 features, 28,479 test timesteps)
- **Model:** Feature Embedding ($d=64$) + Self-Attention Graph Learner ($H=4$ heads, Top-$K=5$) + Spatio-Temporal GCN + 2-Layer LSTM ($h=128$) + Forecasting Head
- **XAI Engine:** 5-Question Framework (Z-score deviation ranking, scale-aware directional labels $\uparrow/\downarrow$, Graph Relationship Auditing, temporal localization, composite confidence scoring)
- **Key Results (SMD Machine 1-1):**
  - **Window-Level:** Precision = 0.5600, Recall = 0.9039, F1 = 0.6916, ROC-AUC = 0.9709, PR-AUC = 0.7981
  - **Point-Level:** Precision = 0.4891, Recall = 0.9714, F1 = 0.6506, ROC-AUC = 0.9708, PR-AUC = 0.7373
- **Deployment:** FastAPI backend REST API (`api/main.py`) + Streamlit dashboard prototype (`app/streamlit_app.py`)

## Compiling LaTeX Paper

To compile the LaTeX paper to PDF on your system:
```bash
cd research_paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
Output PDF will be generated as `paper.pdf`.
