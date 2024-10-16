# Bundesliga Prediction Project

## Introduction

This project focuses on predicting the outcomes of Bundesliga football matches using data analysis and machine learning techniques. The primary goal is to develop a predictive model that can forecast match results based on historical data, team statistics, and player performance. I strongly advise to watch my presentation about this project to understand it better.

## Presentation Video

[![Watch the video](![image](https://github.com/user-attachments/assets/1357cc08-cd9e-4da9-9e73-419b41e5026b)
)](https://youtu.be/Hqag5bJDc-E)

Click the thumbnail to watch the presentation on YouTube.

### Motivation

I embarked on this project due to my passion for football and interest in data science. Predicting sports outcomes presents a challenging and engaging problem that combines statistical analysis with real-world applications. This project allows me to apply my skills in data processing, exploratory data analysis, and machine learning to a domain I am enthusiastic about.


## Project Content

1. **`web_scraping.ipynb`**:
   - **Description**: This notebook is used for scraping the necessary data for the project, including match results and statistics from various online sources.
   - **Key tasks**: Collecting data through web scraping to enrich the dataset with match-related features.

2. **`data_engineering_fifa_scraping.ipynb`**:
   - **Description**: This notebook handles data engineering tasks and scrapes FIFA ratings and other relevant team data needed for further analysis.
   - **Key tasks**: Web scraping FIFA ratings to gather team performance indicators, and preparing this data for model training.

3. **`exploratory_analysis.ipynb`**:
   - **Description**: This notebook provides a detailed exploratory data analysis (EDA) of Bundesliga matches to uncover insights and support the development of a predictive model.
   - **Key tasks**: Analyzing performance indicators such as goals scored, expected goals (xG), possession, and opponent metrics to identify patterns and trends.
  
4. **`making_predictions.ipynb`**:
   - **Description**: This notebook focuses on preparing the data types and making the final predictions for Bundesliga matches using machine learning models.
   - **Key tasks**: Data cleaning, preparation, and engineering to ensure the model is ready for predictions.

## Libraries and Technologies

The following libraries and tools were used throughout the project:

- **requests**: For making HTTP requests (used in web scraping).
- **pandas**: For data manipulation and analysis.
- **BeautifulSoup (bs4)**: For web scraping and parsing HTML content.
- **time**: For handling time-related tasks during web scraping.
- **scikit-learn (sklearn)**: For machine learning models, feature engineering, preprocessing, and evaluation.
  - **ColumnTransformer**, **SimpleImputer**, **StandardScaler**, **OneHotEncoder**, **LabelEncoder**: For data preprocessing and transformation.
  - **train_test_split**, **cross_val_score**, **RandomizedSearchCV**, **GridSearchCV**: For splitting data, cross-validation, and hyperparameter tuning.
  - **RandomForestClassifier**, **AdaBoostClassifier**, **GradientBoostingClassifier**, **LogisticRegression**, **KNeighborsClassifier**: For machine learning models.
  - **Pipeline**: For creating streamlined machine learning workflows.
  - **classification_report**, **confusion_matrix**, **accuracy_score**: For model evaluation metrics.
  - **RFE**: For recursive feature elimination.
- **xgboost (xgb)**: For implementing the XGBoost classifier.
- **lightgbm**: For implementing the LightGBM classifier.
- **joblib**: For saving models and pipelines.
- **scipy**: For statistical functions (used for hypothesis testing and data analysis).
- **matplotlib**: For data visualization.
- **seaborn**: For creating advanced statistical plots.

### System Requirements

This project was developed and tested on:

- **Python**: Version 3.8 or later.
- **Operating System**: Any system supporting Python 3.x (Linux, macOS, Windows).
- **RAM**: At least 8GB recommended for smooth model training and data processing.

## Link to the Game Project

This project includes a separate repository for the Bundesliga prediction game, where users can interact with a simple game that predicts match outcomes based on pre-match statistics.

Link to the repository: [Bundesliga Prediction Game](https://github.com/qbk955/predict_bundesliga)

## Future Plans

1. **Expand the training data**: 
   - Incorporate match data from the top 5 European leagues (Premier League, La Liga, Serie A, Bundesliga, and Ligue 1).
   - Include detailed player statistics for each match, as well as the starting lineup for both teams.

2. **Use language models for expert analysis**:
   - Leverage natural language processing (NLP) models to analyze expert opinions from TV shows, social media, and articles to enhance predictions and provide additional context for match outcomes.

