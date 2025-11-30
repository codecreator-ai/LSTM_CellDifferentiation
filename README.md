# Cell Diff

# Overview
Cell Diff contains reproducible experiments for forecasting cellular differentiation from single-cell time-course data. The repository includes two analysis flows:

Cell-based forecasting (cell_based.ipynb) — model that forecasts and predicts the future state of a cell during cellular differentiation based on it's previous states.

Gene-based forecasting (gene_based_model.ipynb) — an alternative approach that predicts the final state of a cell during cellular differentiation based on other full gene expressions.

The code is written to be readable and modular: helper functions are kept in a shared utilities module, notebooks show plots and metrics inline.

# Installation
Clone the repository and install dependencies into an isolated environment.

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a Python virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on macOS / Linux
# venv\Scripts\activate    # on Windows (PowerShell)

# Install required packages
pip install -r requirements.txt
```

# Usage 

The experiments are made available on jupyter notebooks: [cell-based](/cell_based.ipynb) and [gene-based](/gene_based_modelLSTM.py).

# Data

The data for this experiment can be found on this GSE link [GSE75748_sc_time_course_ec.csv.gz](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748)

# Research Paper

