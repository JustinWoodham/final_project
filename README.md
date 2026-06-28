# Automatic Text Simplification of Public Administration Texts with Large Language Models
 
**Module:** CM3203 — One Semester Individual Project  
**Institution:** Cardiff University, School of Computer Science
 
---
 
## Project Overview
 
The project investigates the use of Large Language Models (LLMs) for Automatic Text Simplification (ATS) across three languages: English, Italian, and Spanish. It evaluates how well LLMs simplify public administration text using different models and prompting strategies.
 
---
 
## Datasets
 
| Dataset | Language | Domain | Size |
|---|---|---|---|
| SimPA | English | Public Administration | 1,100 sentences |
| Admin-It-L2 | Italian | Public Administration | 134 sentences |
| ClearText (FAC) | Spanish | Public Administration | ~4,341 paragraphs |
 
---
 
## Repository Structure
 
```
final_project/
│
├── Pipelines (API — OpenRouter, primary)
│   ├── SimPA_API_Pipeline.ipynb
│   ├── AdminIt_API_Pipeline.ipynb
│   └── ClearText_API_Pipeline.ipynb
│
├── Pipelines (GPU — Local, Flan-T5)
│   ├── SimPA_GPU_Pipeline.ipynb
│   ├── AdminIt_GPU_Pipeline.ipynb
│   └── ClearText_GPU_Pipeline.ipynb
│
├── Evaluation & Analysis
│   ├── lens_score_API.py         — LENS scoring script for API pipeline outputs
│   ├── lens_score_GPU.py         — LENS scoring script for GPU pipeline outputs
│   ├── metricQualityFunc.ipynb   — Selects few-shot examples by LENS/compression quality
│   └── QA_example_selection.ipynb — Samples outputs by LENS percentile for qualitative analysis
│
├── csv/                          — Experiment result CSVs and few-shot selection outputs
├── datasets/                     — SimPA, Admin-It-L2, ClearText datasets
├── requirements-conda.txt
└── requirements-lens.txt
```
 
---
 
## Models Evaluated
 
**Decoder (API via OpenRouter):** Mistral-7B, LLaMA-3.1-8B, Gemma-2-9B, Qwen2.5-7B, LLaMA-3.3-70B
 
**Encoder-Decoder (Local GPU):** Flan-T5 (small, large, XL), mT5-large
 
---
 
## Evaluation Metrics
 
- **SARI** — primary simplification metric
- **ROUGE-1, BLEU** — n-gram overlap
- **BERTScore** — semantic similarity (`xlm-roberta-large` for Italian/Spanish)
- **LENS** — learned simplification quality metric (English only)
- **Flesch Reading Ease** (English), **Gulpease Index** (Italian), **Fernández-Huerta** (Spanish)
---
 
## Setup
 
Two environments are required due to package conflicts between the main pipeline and LENS.
 
**Main environment:**
```bash
conda install --file requirements-conda.txt
pip install -r requirements-pip.txt
```
 
**LENS environment (separate):**
```bash
conda create -n lens_eval python=3.10
pip install lens-score
```
 
API pipelines require an OpenRouter API key stored in a `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```
