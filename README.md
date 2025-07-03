
## Breast Cancer Classification using Support Vector Machines (SVM)

### Overview

This notebook demonstrates the application of Support Vector Machines (SVM) for classifying breast cancer tumors as malignant or benign using the Breast Cancer dataset. Both linear and non-linear SVM models (with RBF kernel) are implemented, evaluated, and compared. The approach includes data preprocessing, dimensionality reduction for visualization, hyperparameter tuning, and thorough model evaluation.

---

### Dataset Information

The dataset (`breast-cancer.csv`) consists of 569 samples with 32 columns, including:

* `id`: Identifier column, which is dropped during preprocessing.
* `diagnosis`: Target label indicating malignant (M) or benign (B) tumors.
* 30 numerical features: Various measurements related to cell nuclei such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

#### Data Structure Summary 

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 32 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   id                       569 non-null    int64  
 1   diagnosis                569 non-null    object 
 2   radius_mean              569 non-null    float64
 3   texture_mean             569 non-null    float64
 4   perimeter_mean           569 non-null    float64
 5   area_mean                569 non-null    float64
 ...
 30  symmetry_worst           569 non-null    float64
 31  fractal_dimension_worst  569 non-null    float64
dtypes: float64(30), int64(1), object(1)
memory usage: 142.4+ KB
```

All feature columns are numerical and non-null, which makes them suitable for SVM modeling.

---

### Data Preprocessing

* The `id` column was removed as it does not contain predictive information.
* The `diagnosis` column was encoded as binary: M (malignant) = 1, B (benign) = 0.
* Features were standardized using `StandardScaler` to ensure each attribute contributes equally to the model.
* Data was split into training (80%) and testing (20%) subsets using stratified sampling to preserve class proportions.

---

### Model Training

Two types of SVM classifiers were trained:

1. **Linear SVM** using a linear kernel.
2. **Non-linear SVM** using an RBF (Radial Basis Function) kernel.

---

### Hyperparameter Tuning

Grid search cross-validation was conducted on the RBF kernel SVM to optimize hyperparameters `C` (regularization) and `gamma` (kernel coefficient).

#### Tuning Results 

```
Best Parameters: {'C': 100, 'gamma': 0.01}
Best Cross-Validation Score: 0.9758
```

The best parameters improved the RBF model's generalization performance.

---

### Model Evaluation

#### Linear SVM Results 

* **Accuracy**: 0.9649
* **Precision**: 1.0
* **Recall**: 0.9048
* **F1 Score**: 0.95

**Confusion Matrix:**

```
[[72  0]
 [ 4 38]]
```

The linear SVM correctly classified all benign tumors and most malignant tumors, with 4 malignant samples misclassified as benign.

---

#### RBF SVM Results 

* **Accuracy**: 0.9737
* **Precision**: 1.0
* **Recall**: 0.9286
* **F1 Score**: 0.9630

**Confusion Matrix:**

```
[[72  0]
 [ 3 39]]
```

The RBF SVM showed slightly better performance by correctly classifying one additional malignant tumor compared to the linear SVM.

---

### Visualization

Principal Component Analysis (PCA) was used to reduce the high-dimensional data to two dimensions for visualization purposes. The decision boundaries for both linear and RBF SVM classifiers were plotted, highlighting how each model separates classes and utilizes support vectors. The plots visually illustrate the flexibility of the RBF kernel in capturing complex boundaries.

---

### Conclusion

* Both SVM models performed strongly on this dataset, achieving high accuracy and F1-scores.
* The RBF SVM outperformed the linear SVM slightly, particularly in recall and overall test accuracy, due to its ability to capture non-linear decision boundaries.
* Hyperparameter tuning further improved the RBF model, resulting in robust and precise classification performance.

---

### How to Use

1. Clone the repository or download the notebook and dataset file.
2. Run each cell sequentially to replicate results and visualizations.
3. Experiment with different kernels, parameter grids, or feature selections to further analyze model behavior.

---

### Files

* `breast_cancer_svm_classification.ipynb`: Contains complete code, outputs, and plots.
* `breast-cancer.csv`: Dataset used for training and evaluation.

---

