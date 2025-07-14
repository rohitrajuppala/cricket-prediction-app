#  Cricket Match Outcome Prediction using Machine Learning

A machine learning-driven approach to predict the outcomes of cricket matches using historical match data. This project demonstrates how ML models can extract actionable insights from past games and assist in decision-making for fans, coaches, fantasy leagues, and sports analysts.

## Project Overview

Predicting cricket match results is a challenging task due to the sport’s dynamic nature. This project explores various machine learning models to automate the outcome prediction process based on factors like team composition, toss decisions, match location, and more.

**Key Contributors**

* Murali Krishna Kaye
* Rohit Raj Uppala
* Tirumala Tejaswi Masimukku

## Objectives

* Develop a predictive system to estimate the winner of a cricket match using historical ODI and T20 data.
* Explore multiple ML models (Logistic Regression, SVM, Random Forest).
* Utilize dimensionality reduction (PCA) for performance tuning.
* Provide insights into important features influencing match outcomes.

---

## Project Structure

```
├── cleaned_cricket_prediction_notebook.ipynb
├── copy_2_of_cricket_prediction_odi_part_2__ohe_Random_Forest_only_indian_team_50+t20.ipynb
├── Cricket Wining Prediction using Machine Learning Techniques  ppt.pptx
├── README.md
```

---

##  Machine Learning Approach

### Dataset

* Data Source: [CricSheet](https://cricsheet.org/matches/)
* Match data includes: team names, city, toss decision, target score, result type, win margin, etc.
* Filtered for India-specific matches and cleaned using Pandas.

###  Data Preprocessing

* Removed irrelevant columns (e.g., `Unnamed: 0`)
* One-hot encoding for categorical variables
* Normalization of numerical features
* Removed rows with `no-result` outcomes
* Replaced missing values in `target_list` using the median

### Models Used

| Model                   | Best Parameters            | Train Accuracy | Test Accuracy |
| ----------------------- | -------------------------- | -------------- | ------------- |
| Logistic Regression     | `C=0.0059, penalty=l2`     | 0.7333         | 0.6144        |
| SVM (Polynomial Kernel) | `C=10`                     | 0.8606         | 0.5903        |
| Random Forest           | `depth=10, estimators=200` | 0.6575         | 0.5963        |
| SVM + PCA               | `kernel=rbf`               | 0.7045         | **0.6506**    |

###  Dimensionality Reduction

* Principal Component Analysis (PCA) was applied to reduce dimensionality and improve performance, particularly for SVM.

---

##  Insights & Conclusions

* **SVM with PCA** outperformed other models, suggesting that dimensionality reduction helped capture non-linear relationships.
* Feature engineering and incorporation of player-specific stats and weather data may improve results.
* Models are evaluated using accuracy, but further metrics like precision, recall, and F1-score can be used for deeper insights.

---

##  Visuals

* PCA Component Plot
* Partial Dependence Plots (PDP) for Top Features
* Accuracy Comparison Bar Charts (included in PowerPoint)

---

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cricket-ml-prediction.git
   cd cricket-ml-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebooks:

   * `cleaned_cricket_prediction_notebook.ipynb` – end-to-end pipeline
   * `copy_2_of_cricket_prediction_odi_part_2__ohe_Random_Forest_only_indian_team_50+t20.ipynb` – Random Forest focus

---

## Future Work

* Incorporate real-time match feeds for live predictions.
* Add player stats, injury reports, and weather data for improved feature representation.
* Deploy the best model as a web application.

---


