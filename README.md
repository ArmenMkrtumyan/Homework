# Marketing Analytics — Homework 1

## Overview
This project analyzes the diffusion of the **Rizz app** using the **Bass Diffusion Model**,
with **Tinder (2012)** as the historical reference innovation.  
All code, data, and generated reports are contained within this directory structure.

---

## Directory Details

- **img/** – images used in the project  
  - `rizz_logo.png` Logo of the Rizz application, used in the report header.

- **data/** – datasets for analysis  
  - `tinder_subscribers.csv` Historical Tinder subscriber counts (in millions)  

- **report/** – generated outputs  
  - `Marketing_Analytics.html` Final report in HTML format (nicer view)
  - `Marketing_Analytics.pdf` Final report in PDF format  

- **root directory** – source & configuration files  
  - `Marketing_Analytics.Rmd` Main R Markdown script that builds both reports  
  - `references.bib` Bibliography for cited sources  
  - `apa.csl` APA 7th citation-style file  
  - `README.md` This documentation file  

---

## How to Generate the Reports

Follow the steps below to build **both HTML and PDF** versions of the report.

### 1. Setup

1. **Download / copy** the entire `HW1/` folder to your machine.  
2. **Open RStudio** (or R).  
3. **Set the working directory** to `HW1/`:

   ```r
   setwd("/path/to/HW1")
   ```

   _Sanity check_:

   ```r
   list.files()
   # Expect: "Marketing_Analytics.Rmd", "references.bib", "apa.csl", "data", "img", "report"
   ```

### 2. Install Required Packages (first time only)

```r
install.packages(c("rmarkdown", "ggplot2", "dplyr", "knitr", "kableExtra"))
```

If you want to render the PDF version, install **TinyTeX** (for LaTeX support):

```r
install.packages("tinytex")
tinytex::install_tinytex()  # one-time install (~200 MB)
```

---

## 3. Build Reports

Both reports will be generated automatically in the `/report/` folder.

### Generate HTML Report
```r
dir.create("report", showWarnings = FALSE)
rmarkdown::render(
  "Marketing_Analytics.Rmd",
  output_format = "html_document",
  output_file   = "Marketing_Analytics.html",
  output_dir    = "report"
)
```

### Generate PDF Report
```r
dir.create("report", showWarnings = FALSE)
rmarkdown::render(
  "Marketing_Analytics.Rmd",
  output_format = "pdf_document",
  output_file   = "Marketing_Analytics.pdf",
  output_dir    = "report"
)
```

Both HTML and PDF files will appear in `report/`.

---

## Alternative: Knit from RStudio (GUI)

1. Open `Marketing_Analytics.Rmd`.  
2. Click the **gear** icon next to **Knit → Knit Directory → Document**.  
3. Confirm the YAML header includes:

   ```yaml
   output:
     html_document:
       df_print: paged
     pdf_document:
       latex_engine: xelatex
   output_file: "report/Marketing_Analytics"
   ```

Then click **Knit** — both HTML and PDF reports will be generated.

---

## Troubleshooting

| Symptom | Fix |
|----------|-----|
| **PDF fails with LaTeX errors** | Install TinyTeX using the commands above. |

---

© 2025 Armen Mkrtumyan  ·  American University of Armenia – Marketing Analytics HW 1
