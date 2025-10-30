# DAMO-640-10 Machine Learning Assignment 1

## Supervised Learning: Haberman's Survival Dataset Analysis

### Project Overview

This project implements supervised learning algorithms to analyse the Haberman's Survival dataset from the UCI Machine Learning Repository. The analysis focuses on predicting patient survival rates following breast cancer surgery based on patient characteristics.

**Group 03:**  
- **Katragadda, Jayasri**
- **Oshiro, Renato Hiroyuki**
- **Pemmasani, Sridevi**
- **Prumucena, Fabio**

**Course:** DAMO-640-10 Fall 2025  
**Institution:** University of Niagara Falls Canada  
**Dataset:** Haberman's Survival (UCI)  
**Assignment:** Assignment 1 — Supervised Learning

### Dataset Description

The Haberman's Survival dataset contains data from a study conducted at the University of Chicago's Billings Hospital on the survival of patients who underwent surgery for breast cancer between 1958 and 1970.

**Features:**
- `age`: Age of patient at time of operation (numerical)
- `operation_year`: Patient's year of operation (numerical, 1958-1969)
- `axillary_nodes`: Number of positive axillary nodes detected (numerical)
- `survival_status`: Survival status (1 = survived ≥5 years, 2 = died <5 years)

**Dataset Statistics:**
- Total samples: 306
- Target classes: Binary (survived/died)
- Missing values: None

### Project Structure

```
personalMLTerm4/
├── assignment01_DAMO-640-10_FINAL_REVISED.ipynb    # Main analysis notebook
├── README.md                                       # Project documentation
├── requirements.txt                                # Python dependencies
└── Report/                                         # Assignment report and documentation
    └── [Report files]                              # Detailed analysis report
```

### Methodology

#### 1. Data Preprocessing
- **Binary encoding**: Converted survival status to binary labels (1 = survived, 0 = died)
- **Feature scaling**: Applied StandardScaler for numerical feature normalisation
- **Train/test split**: 75% training, 25% testing with stratified sampling

#### 2. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Applied to reduce dimensionality while preserving ≥90% of variance
- **Optimal component selection**: Systematic selection of minimum components retaining 90% variance
- **Cumulative variance visualization**: Plotted to determine optimal dimensionality

#### 3. Machine Learning Models
Two supervised learning algorithms were implemented with systematic hyperparameter tuning:

1. **Logistic Regression**
   - Hyperparameter grid: C ∈ [0.1, 1.0]
   - Cross-validation: 5-fold stratified with accuracy scoring
   - Final model: Optimized parameters selected via GridSearchCV

2. **Decision Tree Classifier**
   - Hyperparameter grid: max_depth ∈ [3, None]
   - Cross-validation: 5-fold stratified with accuracy scoring
   - Final model: Optimized parameters selected via GridSearchCV

#### 4. Model Evaluation
- **Cross-validation**: 5-fold stratified for hyperparameter selection
- **Performance metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **ROC analysis**: Comparative ROC curves for model performance visualisation
- **Multi-metric visualization**: F1-Score comparison with radar charts for comprehensive analysis
- **Enhanced PCA analysis**: Dual plots showing cumulative and individual component variance contribution

### Key Technologies and Libraries

- **Python 3.x**
- **Data manipulation**: pandas (≥2.0.0), numpy (≥1.24.0)
- **Machine learning**: scikit-learn (≥1.3.0)
- **Visualisation**: matplotlib (≥3.7.0)
- **Development environment**: Jupyter Notebook (≥1.0.0)

### Requirements

All project dependencies are listed in `requirements.txt`. The main requirements include:

- **pandas** ≥2.0.0 - Data manipulation and analysis
- **numpy** ≥1.24.0 - Numerical computing
- **scikit-learn** ≥1.3.0 - Machine learning algorithms
- **matplotlib** ≥3.7.0 - Data visualisation
- **jupyter** ≥1.0.0 - Interactive notebook environment

### Installation and Setup

#### Option 1: Quick Install

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prumucena1979/personalMLTerm4.git
   cd personalMLTerm4
   ```

2. **Install all dependencies using requirements.txt:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook assignment01_DAMO-640-10_FINAL_REVISED.ipynb
   ```

#### Option 2: Virtual Environment (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prumucena1979/personalMLTerm4.git
   cd personalMLTerm4
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv ml_env
   
   # Activate virtual environment
   # On Windows:
   ml_env\Scripts\activate
   # On macOS/Linux:
   source ml_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook assignment01_DAMO-640-10_FINAL_REVISED.ipynb
   ```

#### Option 3: Manual Install

If you prefer to install packages individually:
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Results Summary

The analysis provides comprehensive comparison between Logistic Regression and Decision Tree classifiers for predicting patient survival. **Key findings:**

- **Logistic Regression**: AUC = 0.743 (good discriminatory power)
- **Decision Tree**: AUC = 0.526 (near-random performance)
- **Recommended Model**: Logistic Regression demonstrates superior performance
- **Data Split**: 75% training / 25% testing with stratified sampling (random_state=42)

**Analysis includes:**
- Optimal hyperparameter identification through cross-validation
- Performance metrics comparison across both models (Accuracy, Precision, Recall, F1-Score, AUC)
- ROC curve analysis for classification threshold evaluation
- Enhanced PCA visualization with dual plots (cumulative + individual component variance)
- F1-Score focused comparison with radar chart visualization
- Professional-grade charts with optimized label positioning and contrast

### Data Source

Dataset retrieved from: [UCI Machine Learning Repository - Haberman's Survival Data](https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data)

### Academic Context

This project is part of the DAMO-640-10 course curriculum, focusing on practical application of supervised learning techniques for binary classification problems in healthcare data analysis.

### Assignment Report

A comprehensive written report for this assignment is available in the `Report/` folder. The report includes:

- **Detailed methodology explanation** - In-depth discussion of data preprocessing, PCA implementation, and model selection rationale
- **Results analysis and interpretation** - Comprehensive analysis of model performance, statistical significance, and practical implications
- **Comparative model evaluation** - Detailed comparison between Logistic Regression and Decision Tree performance
- **Conclusions and recommendations** - Summary of findings and suggestions for future work
- **References and citations** - Academic sources and technical documentation

The report complements the Jupyter notebook by providing theoretical context, detailed explanations of methodological choices, and comprehensive interpretation of results suitable for academic evaluation.

### Usage Instructions

1. Open the Jupyter notebook `assignment01_DAMO-640-10_FINAL_REVISED.ipynb`
2. Run cells sequentially to reproduce the analysis
3. All required datasets are loaded automatically from the UCI repository
4. Modify hyperparameters or add additional models as needed for experimentation
5. **Review the assignment report** in the `Report/` folder for detailed analysis and interpretation

### Future Enhancements

Potential improvements and extensions:
- Additional algorithms (Random Forest, SVM, Neural Networks)
- Feature engineering and selection techniques
- Advanced hyperparameter optimisation (Grid Search, Random Search)
- Ensemble methods implementation
- Statistical significance testing

### Contact

**Group 03 Team Members:**
- **Katragadda, Jayasri**
- **Oshiro, Renato Hiroyuki** 
- **Pemmasani, Sridevi**
- **Prumucena, Fabio** - fabio.dos2000@myunfc.ca

For questions or collaboration opportunities, please contact any group member.

*This project demonstrates practical application of machine learning techniques for healthcare data analysis, emphasising proper methodology, evaluation metrics, and reproducible results.*