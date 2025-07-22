# Mother Machine Timeseries Registration Analysis

This repository contains the implementation of automated rigid image registration methods for time-lapse microscopy data from Mother Machine (MM) devices. The work focuses on rotation and translation correction in MM time series using three main approaches: Hough Transform (HT), phase cross-correlation (XCorr), and ORB (Oriented FAST and Rotated BRIEF) feature-based methods.

**BA Thesis Project** - A demonstration of different registration methods for MM time series analysis.

## Overview

The project implements three registration approaches:
- **Hough Transform (HT)**: Line-based rotation estimation
- **Phase Cross-correlation (XCorr)**: Translation correction in the Fourier domain  
- **ORB**: Feature-based method for both rotation and translation

Additionally, temporally stabilized variants (HT-Prev, ORB-Prev) are introduced that reuse previous frame's rotation angle for enhanced stability.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/fornoffs/microfluidics-mm-registration.git
cd microfluidics-mm-registration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook mother_machine_registration_analysis.ipynb
```

2. Follow the notebook sections:
   - **0. Imports**: Load all necessary libraries and functions
   - **1. Dataset Loading**: Load your microscopy dataset
   - **2. Template Creation**: Generate reference templates using Hough transform
   - **3. Run Image Registration**: Execute alignment with selected methods
   - **4. Save Results**: Save the transformed image
   - **5. Evaluation (Optional)**: Measure runtime and memory usage


## Datasets

Three dataset types were used: synthetic, semi-synthetic, and experimental microscopy data.

**Data Availability**: A small 10-timepoint test dataset is included for immediate testing. Additional datasets are available upon request.

## File Structure

```
├── mother_machine_registration_analysis.ipynb  # Main analysis notebook
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── data/                                       # Test datasets
│   └── sample10timepoints.tif                  # Small test dataset
└── timeseries_alignment/                       # Registration framework
    ├── __init__.py
    ├── ht.py                                 # Hough transform implementation
    ├── orb.py                                # ORB feature-based methods
    ├── xcorr.py                              # Cross-correlation methods
    ├── timeseries_alignment_framework.py     # Main registration pipeline
    └── utils.py                              # Utility functions (simplified)
```

## Acknowledgments

This work was conducted as part of a BA thesis at Humboldt-Universität zu Berlin.

**Microscopy data provided by:**
- Hannah Raasch
- María José Giralt Zúñiga  
- Philipp F. Popp

**Lab affiliation:** [Molecular Microbiology Lab](https://www.molmicro.hu-berlin.de/cv_marc.html) - Prof. Dr. Marc Erhardt, Humboldt-Universität zu Berlin