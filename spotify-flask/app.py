from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
# Import your Spotify-related code here

app = Flask(__name__)

# Load the dataset (replace with your actual DataFrame if dynamically generated)
musicDf = pd.read_csv("musicData.csv")

# Normalize the music features
scaler = MinMaxScaler()
musicFeatures = musicDf[['Danceability', 'Energy', 'Key', 'Loudness',
                         'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness',
                         'Liveness', 'Valence', 'Tempo']].values
musicFeaturesScaled = scaler.fit_transform(musicFeatures)

# Function to calculate weighted popularity scores
def calculateWeightedPopularityScores(releaseDate):
    try:
        releaseDate = datetime.strptime(releaseDate, '%Y-%m-%d')
    except ValueError:
        releaseDate = datetime(datetime.now().year, 1, 1)  # Fallback for missing format
    timeSpan = datetime.now() - releaseDate
    weight = 1 / (timeSpan.days + 1)
    return weight

# Content-based recommendation system
def contentBasedRecommendations(inputSongName, numRecommendations=5):
    if inputSongName not in musicDf['Track Name'].values:
        return []
    inputSongIndex = musicDf[musicDf['Track Name'] == inputSongName].index[0]
    similarityScore = cosine_similarity([musicFeaturesScaled[inputSongIndex]], musicFeaturesScaled)
    similarSongIndices = similarityScore.argsort()[0][::-1][1:numRecommendations + 1]
    recommendations = musicDf.iloc[similarSongIndices][['Track Name', 'Artists', 'Album Name', 'Release Date', 'Popularity']]
    return recommendations

# Hybrid recommendation system
def hybridRecommendations(inputSongName, numRecommendations=5, alpha=0.5):
    if inputSongName not in musicDf['Track Name'].values:
        return []
    contentBased = contentBasedRecommendations(inputSongName, numRecommendations)
    popularityScore = musicDf.loc[musicDf['Track Name'] == inputSongName, 'Popularity'].values[0]
    releaseDate = musicDf.loc[musicDf['Track Name'] == inputSongName, 'Release Date'].values[0]
    weightedPopularity = popularityScore * calculateWeightedPopularityScores(releaseDate)

    hybridRecommendations = contentBased
    hybridRecommendations = hybridRecommendations._append({
        'Track Name': inputSongName,
        'Artists': musicDf.loc[musicDf['Track Name'] == inputSongName, 'Artists'].values[0],
        'Album Name': musicDf.loc[musicDf['Track Name'] == inputSongName, 'Album Name'].values[0],
        'Release Date': releaseDate,
        'Popularity': weightedPopularity
    }, ignore_index=True)

    hybridRecommendations = hybridRecommendations.sort_values(by='Popularity', ascending=False)
    hybridRecommendations = hybridRecommendations[hybridRecommendations['Track Name'] != inputSongName]
    return hybridRecommendations

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    song_name = request.form.get('song_name')
    recommendations = hybridRecommendations(song_name, 5)

    if len(recommendations) == 0:
        return render_template('index.html', error=f"No recommendations found for '{song_name}'. Please try another song.")

    return render_template('index.html', song_name=song_name, recommendations=recommendations.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
