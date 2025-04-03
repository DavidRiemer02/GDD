# GDD – Generated Dataset Detector

**GDD** is a machine learning-based framework that detects whether a tabular dataset is *real* or *synthetically generated*. It combines statistical analysis with structural dependency profiling (using Metanome) to extract discriminative features and uses a Random Forest classifier, optimized by GridSearch to make predictions.

---

## 📁 Project Structure

```text
├── UserData/                 
│   └── realData/                         # This is where the User puts the data
├── models/                               # Trained Random Forest classifiers
├── NoteBook_UI/              
│   └── config.json                       # Configuration file for paths and Metanome settings
│   └── generatedDatasetDetector.jar      # Configuration file for paths and Metanome settings
│   └── notebook_ui.py                    # Contains relevant Code for Juypter NoteBook
├── GDD_Pipeline.ipynb                    # Main Jupyter Notebook interface for using the detector
└── README.md


Open GDD_pipeline.ipynb and follow the steps provided in the notebook
