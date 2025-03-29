#GDD – Generated Dataset Detector

**GDD** is a machine learning-based framework that detects whether a tabular dataset is *real* or *synthetically generated*. It combines statistical analysis with structural dependency profiling (using Metanome) to extract discriminative features and uses a Random Forest classifier, optimized by GridSearch to make predictions.

---

## Project Structure
├── UserData/                  # Upload folder for real and fake datasets
│   ├── realData/              # User-uploaded real datasets
│   └── fakeData/              # User-uploaded fake datasets
├── TrainingData/              # Real and fake datasets used to train the model
├── models/                    # Trained Random Forest classifiers
├── config.json                # Configuration file (paths, Metanome JAR, memory settings)
├── GDD_Pipeline.ipynb         # Main Jupyter Notebook interface for using the detector


Open GDD_pipeline.ipynb and follow the steps provided in the notebook