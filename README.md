# 🎶 Follower-Based Song Popularity Prediction

This project predicts the popularity of songs on Spotify using machine learning algorithms. By analyzing various song attributes such as **artist popularity**, **follower count**, **danceability**, **energy**, and more, the model provides valuable insights for artists, producers, and record labels to make informed decisions about song releases and marketing strategies.

---

## 📜 Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Significance](#significance)
- [Data](#data)
- [Technologies Used](#technologies-used)
- [Proposed Models](#proposed-models)
- [Challenges and Learnings](#challenges-and-learnings)
- [Future Work](#future-work)
- [How to Run the Project](#how-to-run-the-project)
- [Contributors](#contributors)
- [References](#references)

---

## 🌟 Introduction

The "Follower-Based Song Popularity Prediction" project aims to predict song popularity on Spotify by using machine learning techniques and various song characteristics. Our model uses features like **artist popularity**, **follower count**, **danceability**, **energy**, **acousticness**, and more to predict which songs are likely to become hits. We explore multiple machine learning algorithms, including Decision Trees and K-Nearest Neighbors (KNN), to identify the most accurate model for song popularity prediction.

This project provides essential insights into the music industry, offering a way for artists and producers to enhance marketing strategies, optimize release schedules, and maximize listener engagement.

---

## 🔍 Problem Statement

Predicting song popularity is a challenging task, as many factors contribute to whether a song becomes a hit. In this project, we address the following problem:

- **Goal**: To predict song popularity using a comprehensive set of features such as artist popularity, follower count, and song attributes (danceability, energy, loudness, acousticness, etc.).
- **Objective**: Provide actionable insights to artists, producers, and record labels to optimize marketing efforts and release strategies.

---

## 💡 Approach

The project follows a structured approach to predict song popularity:

1. **Data Collection**: Spotify API is used to gather data on song attributes such as danceability, energy, loudness, and artist details like popularity and follower count.
2. **Data Preprocessing**: Raw data is cleaned, normalized, and processed to handle missing values and irrelevant features. This includes selecting relevant features, removing duplicates, and standardizing numerical and categorical data.
3. **Model Development**: Machine learning algorithms such as Decision Trees and KNN are trained on the preprocessed dataset. The models are optimized through cross-validation, hyperparameter tuning, and performance evaluation.
4. **Model Comparison**: The performance of each model is evaluated using metrics like Root Mean Squared Error (RMSE), and the most accurate model is selected for song popularity prediction.

---

## 🌍 Significance

This project offers significant value to the music industry by providing predictive insights that can influence marketing and production strategies. The model helps:

- **Artists & Producers**: Optimize release schedules and marketing campaigns by predicting which songs are likely to become hits.
- **Record Labels**: Make data-driven decisions on resource allocation for promoting specific songs or artists.
- **Global Scalability**: The model can be adapted to predict song popularity across different genres, regions, and markets, making it a valuable tool for global music analytics.

---

## 📊 Data

### a. Data Gathering
- Data is collected from Spotify’s API, focusing on features like artist popularity, follower count, and various song attributes.
- This data is crucial for understanding the dynamics of song popularity.

### b. Data Preparation
- Data cleaning involves removing irrelevant features, handling missing values, and ensuring consistent data formats.
- Numerical features such as **danceability** and **energy** are normalized, and categorical features such as **artist country** are standardized.

### c. Data Preprocessing
- Features like **track ID** and **song name** are removed as they do not contribute to prediction accuracy.
- Duplicates are handled, and a dataset without null values is created for the model development phase.

---

## 🛠 Technologies Used

- **Python**: The primary programming language for the project.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Matplotlib & Seaborn**: For creating visualizations to explore data relationships.
- **Scikit-learn**: Machine learning library used for training and testing the models.
- **Google Colab**: The development environment where the models are trained and evaluated.

---

## 📈 Proposed Models

### 1. K-Nearest Neighbors (KNN)
- The KNN algorithm is applied to predict song popularity based on its similarity to other songs.
- The model's performance is optimized by adjusting the number of neighbors and evaluated using RMSE (Root Mean Squared Error).

### 2. Decision Tree Regressor
- A Decision Tree model is trained to predict song popularity based on features like **danceability**, **energy**, and **acousticness**.
- The model's complexity is controlled through hyperparameters like `max_leaf_nodes` and `min_samples_split`, and its performance is measured using RMSE.

---

## 🧠 Challenges and Learnings

- **Data Cleaning**: Handling inconsistencies, missing values, and irrelevant features was critical for ensuring data quality.
- **Feature Selection**: Determining which features contributed most to predicting song popularity was a key aspect of model development.
- **Model Optimization**: Tuning the hyperparameters of the models to balance accuracy and overfitting was challenging but essential for improving performance.

---

## 🔮 Future Work

- **Advanced Algorithms**: Incorporating deep learning models to improve the accuracy of predictions.
- **Real-Time Predictions**: Expanding the model to predict song popularity in real-time.
- **Broader Data Integration**: Incorporating additional data sources, such as social media sentiment and concert ticket sales, to enhance the model’s predictive power.

---

## 🛠 How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/follower-based-song-popularity-prediction.git
   cd follower-based-song-popularity-prediction

2. Install the required libraries listed in the `requirements.txt` file.
   pip install -r requirements.txt

3. Run the Jupyter Notebook: Open the `popularity_prediction.ipynb` notebook in Google Colab or locally, and run all the cells to train and test the models.

4. View Results: The predictions and visualizations will be displayed after running the notebook, including charts and metrics for model performance.

---

## 📚 References

1. "SpotHitPy: A Study For ML-Based Song Hit Prediction Using Spotify"
2. Revisiting the Problem of Audio-based Hit Song Prediction using Convolutional Neural Networks
3. A. H. Raza and K. Nanath, “Predicting a hit song with machine learning: Is there an apriori secret formula?”
4. E. Zangerle, M. Vötter, R. Huber, and Y.-H. Yang, “Hit song prediction: Leveraging low- and high-level audio features”
