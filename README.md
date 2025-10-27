# DAMO-640-10 Machine Learning Assignment 1

## Supervised Learning: Haberman's Survival Dataset Analysis

### Project Overview

This project implements supervised learning algorithms to analyse the Haberman's Survival dataset from the UCI Machine Learning Repository. The analysis focuses on predicting patient survival rates following breast cancer surgery based on patient characteristics.

**Author:** Fabio dos Santos Prumucena (NF1002000)  
**Course:** DAMO-640-10 Fall 2025  
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
├── assignment01_DAMO-640-10.ipynb    # Main analysis notebook
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

### Methodology

#### 1. Data Preprocessing
- **Binary encoding**: Converted survival status to binary labels (1 = survived, 0 = died)
- **Feature scaling**: Applied StandardScaler for numerical feature normalisation
- **Train/test split**: 75% training, 25% testing with stratified sampling

#### 2. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Applied to reduce dimensionality while preserving ≥90% of variance
- **Variance analysis**: Evaluated explained variance ratios for optimal component selection

#### 3. Machine Learning Models
Two supervised learning algorithms were implemented and compared:

1. **Logistic Regression**
   - Hyperparameter tuning: C values [0.1, 1.0]
   - Cross-validation: 5-fold stratified

2. **Decision Tree Classifier**
   - Hyperparameter tuning: max_depth [3, None]
   - Cross-validation: 5-fold stratified

#### 4. Model Evaluation
- **Cross-validation**: 5-fold stratified for hyperparameter selection
- **Performance metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **ROC analysis**: Comparative ROC curves for model performance visualisation
- **Classification reports**: Detailed per-class performance metrics

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
   jupyter notebook assignment01_DAMO-640-10.ipynb
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
   jupyter notebook assignment01_DAMO-640-10.ipynb
   ```

#### Option 3: Manual Install

If you prefer to install packages individually:
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Results Summary

The analysis provides comprehensive comparison between Logistic Regression and Decision Tree classifiers for predicting patient survival, including:

- Optimal hyperparameter identification through cross-validation
- Performance metrics comparison across both models
- ROC curve analysis for classification threshold evaluation
- Detailed classification reports with per-class statistics

### Data Source

Dataset retrieved from: [UCI Machine Learning Repository - Haberman's Survival Data](https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data)

### Academic Context

This project is part of the DAMO-640-10 course curriculum, focusing on practical application of supervised learning techniques for binary classification problems in healthcare data analysis.

### Usage Instructions

1. Open the Jupyter notebook `assignment01_DAMO-640-10.ipynb`
2. Run cells sequentially to reproduce the analysis
3. All required datasets are loaded automatically from the UCI repository
4. Modify hyperparameters or add additional models as needed for experimentation

### Future Enhancements

Potential improvements and extensions:
- Additional algorithms (Random Forest, SVM, Neural Networks)
- Feature engineering and selection techniques
- Advanced hyperparameter optimisation (Grid Search, Random Search)
- Ensemble methods implementation
- Statistical significance testing

### Contact

For questions or collaboration opportunities, please contact:
**Fabio dos Santos Prumucena** - NF1002000 - fabio.dos2000@myunfc.ca

*This project demonstrates practical application of machine learning techniques for healthcare data analysis, emphasising proper methodology, evaluation metrics, and reproducible results.*