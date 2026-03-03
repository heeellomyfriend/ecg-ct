# Integrating epicardial adipose tissue from 3D CT images with ECG data to define structural–electrical phenotypes.

Repository for the NLDL2026 Winter School project on multimodal learning with CT and ECG data.

## Project Structure

The repository is organized as follows:

```
├── 3DCLIP/
│   ├── model.py
│   ├── train_clip3d_ecg.py
│   └── ...
├── data_preprocessing/
│   ├── CLIP_preprocessing/
│   └── EAT_prediction/
└── utils/
    └── tools.py
```

-   **`3DCLIP/`**: Contains the code for the 3D CLIP model, including model definition, training, and evaluation scripts.
-   **`data_preprocessing/`**: Scripts for preprocessing CT scans and preparing the data for the CLIP model.
-   **`utils/`**: Utility functions and helper tools used across the project.

## Getting Started

To get a local copy of the code. The code should work by running all of the scripts from the project root.

### Prerequisites

This project uses Python. Make sure you have a Python environment set up. Dependencies can be installed via pip.

```sh
pip install -r requirements.txt
```

OBS, the requirement files currently just list the packages used in this project. Package version errors might occur in the future. It works for the default packages at the moment with Python 3.13.11.


### Usage

The data is private and cannot be shared, unfortunately. Therefore, other data has to be preprocessed before running the code. Also, hardcoded paths should be changed. Also, I have removed my wandb entity to fully anonymize the script.
