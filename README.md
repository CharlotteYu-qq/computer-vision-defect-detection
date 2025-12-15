# Computer Vision–based Defect Detection Project

## Overview

This project implements an automated computer vision pipeline for detecting geometric misalignment and defects from image data.  

The system is designed to be **robust, explainable, and reproducible**, supporting batch processing and structured output reports.

The final solution focuses on:
- Region of Interest (ROI)–based image processing
- Adaptive geometry analysis
- Threshold-based defect classification
- Batch processing with visual and tabular outputs

---

## Project Structure

    
    project_final/
    │
    ├─ src/
    │   ├─ batch_process_final.py        # Batch detection and CSV report generation
    │   ├─ geometry_utils_adaptive.py    # Adaptive geometry analysis utilities
    │   └─ generate_final_report.py      # Visualization and final image generation
    │
    ├─ data/                             # Input images (not included in repository)
    │   ├─blank_buttons
    │   └─filmed_buttons 
    │
    ├─ output/
    │   ├─ detection_report.csv          # Detection results
    │   ├─ detection_results.json
    │   └─ final_unified_view/           # Annotated output images
    │
    ├─ setup/                             
    │   ├─setup_roi.py                   # Define ROI
    │   └─setup_template.py              # Define physical ground truth
    │
    ├─ requirements.txt                  # Python dependencies
    ├─ README.md                         # Project documentation
    └─ .gitignore

## Environment Setup

Step 1: Create a virtual environment

    python3 -m venv venv

Step 2: Activate the virtual environment

    # macOS/Linux
    source venv/bin/activate 

    # Windows
    venv\Scripts\activate 

Step 3: Install dependencies

    pip install -r requirements.txt

## Usage

Step 1: Run detection pipeline

    # 1. detect in batches and generate csv report
    python src/batch_process_final.py

    # 2. generate visual pictures
    python src/generate_final_report.py

Step 2: Check result
    
    # data saved in output/detection_report.csv

    # images saved in output/final_unified_view/

    


