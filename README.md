## 🐿️ Spatial Ecology: Predicting Red and Grey Squirrel Distribution

### 🧩 Overview  
This project analyses the **spatial distribution of red and grey squirrels** in Scotland using environmental and occurrence data.  
It integrates **GIS-based spatial processing** and **machine learning models** to understand how habitat features and species interactions shape their range.  

The study demonstrates how spatial autocorrelation can bias model accuracy and how **spatial cross-validation (spCV)** provides a more realistic measure of predictive performance compared to standard k-fold CV.

---

### 🎯 Key Highlights  
- **Objective:** Identify the influence of land-cover and proximity to grey squirrels on red squirrel presence.  
- **Data:**  
  - Species occurrence data (Sciurus vulgaris & Sciurus carolinensis).  
  - UK Land Cover Map (LCMUK) raster data.  
- **Spatial scale optimisation:** Tested buffer radii from 100 m – 2000 m, found optimal at **1200 m**.  
- **Model comparison:**  
  - Random Forest (RF) vs Support Vector Machine (SVM).  
  - Both under **CV** and **spatial CV** frameworks.  
- **Result:**  
  - RF accuracy dropped from **0.81 (CV)** → **0.60 (spCV)**, showing over-optimism in standard validation.  
  - SVM improved with parameter tuning, reaching **~0.73** under spatial CV.  
- **Reproducibility:** Uses relative paths via `{here}` and standard project structure.  

---

### ⚙️ Tech Stack  
| Category | Tools & Packages |
|-----------|------------------|
| **Language** | R |
| **Spatial processing** | `terra`, `sf`, `raster`, `rgdal` |
| **Machine learning** | `mlr`, `randomForest`, `kernlab` |
| **Cross-validation** | k-fold CV and Spatial CV (`SpCV`, `SpRepCV`) |
| **Reproducibility** | `here`, structured project folders |
| **Visualisation** | Base R plotting, raster maps, histograms |

---

### 🧠 Insights  
- Spatial autocorrelation can **inflate predictive accuracy** if ignored.  
- Spatial CV is essential for realistic model evaluation in ecological studies.  
- Ecological processes often operate at intermediate spatial scales (~1 km), as shown by the 1200 m optimal buffer.  
- Machine learning models like RF and SVM, when combined with sound spatial design, provide **interpretable and generalisable** predictions.

---

### 📁 Project Structure  
```
├── data/                # occurrence & raster data
├── outputs/             # result figures & model predictions
├── R/                   # scripts (preprocessing, modelling, plotting)
├── README.md
```

---

### 🚀 Run Instructions
```r
# load packages and set relative paths
library(here)
sciurus <- read.csv(here::here("data", "Sciurus.csv"))

# run analysis scripts sequentially
source("R/00_load.R")
source("R/10_preprocess.R")
source("R/20_model_training.R")
```

---

### 🧩 Summary Sentence for CV
> *Developed a reproducible spatial-machine learning pipeline in R to model species distribution, optimising spatial scale and comparing CV vs spatial CV to address autocorrelation bias.*
