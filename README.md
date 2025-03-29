#GDD – Generated Dataset Detector

**GDD** is a machine learning-based framework that detects whether a tabular dataset is *real* or *synthetically generated*. It combines statistical analysis with structural dependency profiling (using Metanome) to extract discriminative features and uses a Random Forest classifier, optimized by GridSearch to make predictions.

---

## Project Structure
├── UserData/ # Upload folder for real and fake datasets │ 
    ├── realData/ 
    └── fakeData/ 
├── TrainingData/ # Datasets used for model training 
├── models/ # Trained Random Forest classifiers 
├── config.json # Configuration file for paths and Metanome settings 
├── GDD_pipeline.ipynb # Jupyter notebook interface

Open GDD_pipeline.ipynb and follow the steps provided in the notebook