# Marketing Analytics — Homework 1

## Overview
This project analyzes the diffusion of the **Rizz app** using the **Bass Diffusion Model**,
with **Tinder (2012)** as the historical reference innovation.  
All code, data, and generated reports are contained within this directory structure.

---

## Directory Details

- **img/**: Contains all images used in the project.
  - `rizz_logo.png`: Logo of the Rizz application, used in the report header.

- **data/**: Holds datasets used for the analysis.
  - `tinder_subscribers.csv`: Historical Tinder subscriber counts (millions), used to estimate the Bass model parameters.

- **report/**: Contains all generated reports.
  - `Marketing_Analytics.html`: Final report in HTML format. (Nicer view)
  - `Marketing_Analytics.pdf`: Final report in PDF format.

- **root directory**: Holds the source and configuration files.
  - `Marketing_Analytics.Rmd`: Main R Markdown script that builds the report.
  - `references.bib`: Bibliography file with cited data sources.
  - `apa.csl`: Citation style file (APA 7th edition).
  - `README.md`: This documentation file.

---

## How to Generate Reports

You can build both versions of the report from the **R console** using the commands below.

### Generate HTML Report
```r
rmarkdown::render(
  "Marketing_Analytics.Rmd",
  output_format = "html_document",
  output_file   = "Marketing_Analytics.html",
  output_dir    = "report"
)
```
### Generate PDF Report
```r
rmarkdown::render(
  "Marketing_Analytics.Rmd",
  output_format = "pdf_document",
  output_file   = "Marketing_Analytics.pdf",
  output_dir    = "report"
)
```

Both files will be saved in /report/ folder
