# Multi-label Movie Genre Classification

> **Official Implementation** of the research: *"Multi-label movie genre classification using multi-source text fusion and transformer-enhanced multi-model ensemble"*

---

## Project Overview
This project implements a **Transformer-enhanced ensemble learning framework** specifically designed for complex multi-label movie genre classification. 

Unlike traditional methods, this framework leverages multi-source textual metadata—**Titles, Summaries, and Plot Synopses**—to capture diverse semantic granularities. By fusing deep contextual embeddings from **RoBERTa** and **DeBERTa** encoders through a non-linear MLP ensemble, the model effectively addresses complex label co-occurrences and long-tail distributions in cinematic data.



---

## Dataset Information
The research utilizes a curated multi-label dataset harvested from the **Internet Movie Database (IMDb)**.

* **Total Samples:** 4,430 unique movies.
* **Label Space:** 27 distinct genre categories (e.g., *Drama, Sci-Fi, Thriller*).
* **Complexity:** Each movie typically contains multiple labels (e.g., *Inception* is tagged with Action, Adventure, Sci-Fi, and Thriller).
* **Source:** [IMDb Official Website](https://www.imdb.com/)

---

## Repository Structure
The project is organized into four functional modules, reflecting a systematic scientific pipeline:

| Module | Description | Key Responsibilities |
| :--- | :--- | :--- |
| **`TorchDataset/`** | Data Intelligence Layer | Text cleaning, Tokenization (RoBERTa/DeBERTa), and Multi-label One-hot encoding. |
| **`Model/`** | Architectural Core | Dual-encoder setup, Shared Transformer Fusion Module, and MLP-based Ensemble layers. |
| **`Train/`** | Optimization Engine | Training loops, Loss functions (BCE), and Model checkpointing. |
| **`Evaluation/`** | Validation & Analytics | Metric calculation (`metrixCaler.py`), ROC-AUC visualization, and SOTA benchmarking. |

---

## Methodology
The implementation follows a rigorous five-stage experimental flowchart:

1.  **Data Preprocessing:** Cleaning and tokenizing multi-source text from IMDb.
2.  **Encoder Integration:** Extracting deep features using RoBERTa and DeBERTa architectures.
3.  **Transformer Fusion:** Capturing dependencies between Title, Synopsis, and Summary via **shared attention mechanisms**.
4.  **Ensemble Meta-learning:** Integrating heterogeneous base-learner outputs using an MLP configuration.
5.  **Multi-label Evaluation:** Quantitative assessment using Subset Accuracy, F1-score, Hamming Loss, and ROC-AUC.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch transformers scikit-learn numpy pandas matplotlib seaborn
