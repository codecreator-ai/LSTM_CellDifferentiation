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
### The experiments are available as interactive notebooks:

Cell-based notebook: [cell-based](/cell_forecasting_model.ipynb)

Gene-based script: [gene-based](/gene_based_model.ipynb)

### Running the notebooks:

Open the notebook with Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab
# or
jupyter notebook
```
Then open cell_based.ipynb in the browser and run the cells sequentially. The notebook is structured with explanatory markdown cells before each code block, so you can run it top to bottom.

# Data

The data used in these experiments is publicly available from GEO:

GSE75748 (single-cell time-course dataset)
Download link:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748

A convenient direct data workflow (example):

```bash
# create data directory
mkdir -p data
cd data

# download the GEO supplementary file (example — use the link for the raw CSV/gz you need)
wget -O GSE75748_sc_time_course_ec.csv.gz "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75748/suppl/GSE75748_sc_time_course_ec.csv.gz"

# decompress
gunzip GSE75748_sc_time_course_ec.csv.gz
```
### Important file placement recommendation

Place the dataset where the notebook expects to find it (recommended path in repo):

```bash
/data/GSE75748_sc_time_course_ec.csv
```

# Metrics and Outputs

The notebooks report:

 - RMSE (root mean squared error): same units as the data, sensitive to large errors.

 - MAE (mean absolute error): average absolute error, easy to interpret.

 - R²: proportion of variance explained (1 is perfect, 0 equals baseline mean predictor, negative is worse than baseline).

Visual outputs include training/validation loss curves, prediction vs actual plots, error histograms, and final bar charts comparing LSTM vs feedforward performance.

# Research Paper

Please see the research paper [here](/Reser
