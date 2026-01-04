# NatureInspiredFeatureSelectionProject

# Dream Optimization Algorithm for Feature Selection and Classification on Wisconsin Breast Cancer Dataset

This project investigates the impact of feature selection using the Dream Optimization Algorithm (DOA) on the performance of different machine learning classifiers.
The study combines optimization theory, statistical learning, and mathematical modeling in a unified experimental framework.

# Project Objective

The main goals of this project are:
* To apply the Dream Optimization Algorithm (DOA) for optimal feature subset selection
* To evaluate the effect of selected features on classification performance
* To compare DOA with classical filter-based feature selection methods
* To analyze the interaction between optimization algorithms and machine learning models from a mathematical perspective.

# Methods Used
🔹 Feature Selection Techniques
The following feature selection methods are implemented and compared:
* Dream Optimization Algorithm (DOA) (meta-heuristic optimization)
* ReliefF
* Information Gain
* Chi-Square (χ²)
* No feature selection (baseline)
🔹 For each method, experiments are conducted with:
* 5, 10, 15, and 20 selected features
* All features (no selection)

# Classification Models
Three representative classifier families are used to cover different learning paradigms:
🌲 Tree-Based
* Random Forest Classifier
🚶 Lazy Learning
* K-Nearest Neighbors (KNN)
    *Fixed parameter: k = 5
📈 Statistical / Probabilistic
* Gaussian Naive Bayes
This setup allows comparison between:
* Tree-based learning
* Instance-based (lazy) learning
* Statistical probabilistic modeling

# Mathematical and Theoretical Background
This project is grounded in the following mathematical concepts:
* Meta-heuristic optimization algorithms
* Binary feature selection search space
* Probability theory (Naive Bayes classifier)
* Distance metrics (Euclidean distance in KNN)
* Entropy and information theory
* Statistical dependency testing (χ² test)
* The Dream Optimization Algorithm is adapted from a continuous optimization framework to a binary feature selection problem, making it suitable for discrete machine learning applications.  

# Project Structure
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_selection/
│   │   ├── doa.py
│   │   ├── relief.py
│   │   ├── chi_square.py
│   │   └── information_gain.py
│   ├── models/
│   │   ├── knn.py
│   │   ├── random_forest.py
│   │   └── gaussian_nb.py
│   ├── evaluation.py
│   └── utils.py
├── train.xlsx
├── test.xlsx
├── main.py
├── requirements.txt
└── README.md

# Outputs
* Accuracy
* F1-score
* Number of selected features
* Results are saved to an .xlsx file

# 📈 Evaluation Strategy
* Stratified K-Fold Cross Validation
* Accuracy
* F1-Score

# Experimental Scenarios
| Feature Selection | #Features        | Classifier     |
| ----------------- | ---------------- | -------------- |
| DOA               | 5 / 10 / 15 / 20 | RF / KNN / GNB |
| ReliefF           | 5 / 10 / 15 / 20 | RF / KNN / GNB |
| Information Gain  | 5 / 10 / 15 / 20 | RF / KNN / GNB |
| Chi-Square        | 5 / 10 / 15 / 20 | RF / KNN / GNB |
| None              | All              | RF / KNN / GNB |

# Conclusion
* This project demonstrates how optimization algorithms can be effectively integrated with machine learning classifiers to improve classification performance.
* Key insights include:
  * The influence of feature selection on classifier accuracy
  *The differing sensitivity of classifier families to optimization-based feature selection
  * Practical application of mathematical optimization in real machine learning problems
