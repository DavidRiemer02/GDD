# GDD – Generated Dataset Detector

**GDD** is a machine learning-based framework that detects whether a tabular dataset is *real* or *synthetically generated*. It combines statistical analysis with structural dependency profiling (using Metanome) to extract discriminative features and uses a Random Forest classifier, optimized by GridSearch to make predictions.

---

## 📁 Project Structure

```text
├── UserData/                 
│   ├── realData/             # User-uploaded real datasets
│   └── fakeData/             # User-uploaded fake datasets
├── models/                   # Trained Random Forest classifiers
├── config.json               # Configuration file for paths and Metanome settings
├── GDD_Pipeline.ipynb        # Main Jupyter Notebook interface for using the detector



Open GDD_pipeline.ipynb and follow the steps provided in the notebook
