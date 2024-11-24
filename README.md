# Spotify-Music-Recommendation-System

## Introduction:

This project aims to develop a music recommendation system that utilizes the Spotify API and a hybrid machine learning model to suggest songs based on a user's Spotify playlist. The hybrid model combines content-based filtering based on cosine similarity and weighted popularity score, effectively capturing both individual preferences and overall song popularity.

## Data:

The data collection process for this music recommendation project involves two main steps:

* **User Playlist Acquisition:** The Spotify API is utilized to retrieve the user's Spotify playlist. This involves connecting to the user's Spotify account and extracting the playlist's track information, including song titles, artists, and audio features.

* **Data Preprocessing:** The extracted song data undergoes preprocessing to ensure consistency and quality. This involves handling missing values, correcting data inconsistencies, and transforming data into a suitable format for the machine learning model.

## Machine Learning Model:


The machine learning model employed in this music recommendation project is a hybrid approach that combines content-based filtering and weighted popularity score. Here's a concise explanation:

* **Content-Based Filtering:** This technique utilizes cosine similarity to identify songs similar to those already in the user's playlist based on their audio features, such as danceability, energy, and valence. Cosine similarity measures the angle between two vectors, in this case, the audio feature vectors of songs. Songs with smaller angles, indicating high similarity, are prioritized.

* **Weighted Popularity Score:** To account for overall song popularity, a weighted popularity score is calculated for each song. This score combines the song's global popularity metrics with its similarity to the user's playlist. Songs with higher popularity scores are given more weight.

By combining these two approaches, the hybrid model balances individual preferences with overall song popularity, providing a more comprehensive and personalized set of recommendations.

## Libraries used:

* *spotipy:* Used to provide a convenient interface to the Spotify Web API.
* *sklearn:* Used for MinMaxScalar and cosine similarity.

## Requirements:

~~~
!pip install sklearn spotipy numpy pandas
~~~
<h1>FRONTEND FOR THE PROJECT</h1>
Created a frontend for the project using Flask Library in python.
<hr>
<h2>Necessary Screen Shots</h2>
![Screenshot 2024-11-25 001000](https://github.com/user-attachments/assets/54e72272-67fa-4a9a-bc20-1844c3aac3cd)
![Screenshot 2024-11-25 001037](https://github.com/user-attachments/assets/0f016a8f-cd94-4593-bf60-b21009c828b1)

