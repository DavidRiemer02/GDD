# GDD – Generated Dataset Detector

**GDD** is a machine learning-based framework that detects whether a tabular dataset is *real* or *synthetically generated*. It combines statistical analysis with structural dependency profiling (using Metanome) to extract discriminative features and uses a Random Forest classifier, optimized by GridSearch to make predictions.

---

## Project Structure

```text
├── UserData/
│   ├── archive/                          # Zips are moved here
│   ├── realData/                         # This is where the User puts the data               
│   └── results/                          # Results file is stored here
├── models/                               # Trained Random Forest classifiers
├── NoteBook_UI/              
│   ├── config.json                       # Configuration file for paths and Metanome settings
│   ├── generatedDatasetDetector.jar      # Metanome dependency extraction tool
│   └── notebook_ui.py                    # Contains relevant code for Jupyter Notebook
├── GDD_Pipeline.ipynb                    # Main Jupyter Notebook interface for using the detector
└── README.md
````

To classify whether your tabular dataset is real or synthetic, follow these steps in the GDD_Pipeline.ipynb notebook:

### 1. Install Requirements
When the notebook is launched, the first cell automatically installs all dependencies listed in `requirements.txt`.  
This ensures that the environment is ready to execute the pipeline.

### 2. Configure Paths
The notebook loads paths from `NoteBook_UI/config.json`, which defines:
- The Java Installation (Be sure to use JAVA 18.0.2.1 if the algorithm does not run properly)
- The input folder containing CSV data
- Paths and flags for executing Metanome

### 3. Upload Your Data
You may upload:
- Individual `.csv` files, or  
- A `.zip` archive containing multiple `.csv` files.

Uploaded files are automatically moved into the appropriate directory under `UserData/realData`.

### 4. Run the Classification Pipeline
Execute the cell to trigger the classification routine, which includes:
- Preprocessing and cleaning of data
- Feature extraction using:
  - Statistical measures
  - Dependency profiling (via Metanome)
- Prediction using a Random Forest classifier
- Output of results

The result file is written to `UserData/results/run_summary.txt` using a pre-trained Random Forest model
