import requests
import base64

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


CLIENT_ID = 'Your client id'
CLIENT_SECRET = 'Your client secret'

clientCredentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
clientCredentialsBase64 = base64.b64encode(clientCredentials.encode())

tokenURL = 'https://accounts.spotify.com/api/token'
headers = {
    'Authorization': f'Basic {clientCredentialsBase64.decode()}'
}

data = {
    'grant_type': 'client_credentials'
}

response = requests.post(tokenURL, data=data, headers=headers)

if response.status_code ==200:
    accessToken = response.json()['access_token']
    print("Access token obtained successfully")
else:
    print("Error obtaining access token")
    exit()


def getTrendingPlaylistData(playlistId, accessToken):
    #Setup spotify with the access token
    sp = spotipy.Spotify(auth=accessToken)
    #get the tracks from the playlist
    playlistTracks = sp.playlist_items(playlistId, limit=50)
    #Extract relevant information and store in a list of dictionaries
    musicData = []
    for trackInfo in playlistTracks['items']:
        track = trackInfo['track']
        trackName = track['name']
        artists = ', '.join([artist['name'] for artist in track['artists']])
        albumName = track['album']['name']
        albumId = track['album']['id']
        trackId = track['id']

        #Get audio features for the track
        audioFeatures = sp.audio_features(trackId)[0] if trackId != 'Not available' else None

        #Get release date of the album
        try:
            albumInfo = sp.album(albumId) if albumId != 'Not available' else None
            releaseDate = albumInfo['release_date'] if albumInfo else None
        except:
            releaseDate = None

        #Get popularity of the track
        try:
            trackInfo = sp.track(trackId) if trackId != 'Not available' else None
            popularity = trackInfo['popularity'] if trackInfo else None
        except:
            popularity = None

        #Get genres
        try:
            songGenre = audioFeatures['genres'][0]

        except:
            songGenre = None

        #Add additional track information to the track data
        trackData = {
            'Track Name': trackName,
            'Artists': artists,
            'Album Name': albumName,
            'Album ID': albumId,
            'Track ID': trackId,
            'Song Genre': songGenre,
            'Popularity': popularity,
            'Release Date': releaseDate,
            'Duration(ms)': audioFeatures['duration_ms'] if audioFeatures else None,
            'Explicit': trackInfo.get('explicit', None),
            'External URLs': trackInfo.get('external_urls', {}).get('spotify', None),
            'Danceability': audioFeatures['danceability'] if audioFeatures else None,
            'Energy': audioFeatures['energy'] if audioFeatures else None,
            'Key': audioFeatures['key'] if audioFeatures else None,
            'Loudness': audioFeatures['loudness'] if audioFeatures else None,
            'Mode': audioFeatures['mode'] if audioFeatures else None,
            'Speechiness': audioFeatures['speechiness'] if audioFeatures else None,
            'Acousticness': audioFeatures['acousticness'] if audioFeatures else None,
            'Instrumentalness': audioFeatures['instrumentalness'] if audioFeatures else None,
            'Liveness': audioFeatures['liveness'] if audioFeatures else None,
            'Valence': audioFeatures['valence'] if audioFeatures else None,
            'Tempo': audioFeatures['tempo'] if audioFeatures else None,
        }
        musicData.append(trackData)

        while playlistTracks['next']:
            playlistTracks = sp.next(playlistTracks)
            for trackInfo in playlistTracks['items']:
                track = trackInfo['track']
                trackName = track['name']
                artists = ', '.join([artist['name'] for artist in track['artists']])
                albumName = track['album']['name']
                albumId = track['album']['id']
                trackId = track['id']

                # Get audio features for the track
                audioFeatures = sp.audio_features(trackId)[0] if trackId != 'Not available' else None

                # Get release date of the album
                try:
                    albumInfo = sp.album(albumId) if albumId != 'Not available' else None
                    releaseDate = albumInfo['release_date'] if albumInfo else None
                except:
                    releaseDate = None

                # Get popularity of the track
                try:
                    trackInfo = sp.track(trackId) if trackId != 'Not available' else None
                    popularity = trackInfo['popularity'] if trackInfo else None
                except:
                    popularity = None

                # Add additional track information to the track data
                trackData = {
                    'Track Name': trackName,
                    'Artists': artists,
                    'Album Name': albumName,
                    'Album ID': albumId,
                    'Track ID': trackId,
                    'Popularity': popularity,
                    'Release Date': releaseDate,
                    'Duration(ms)': audioFeatures['duration_ms'] if audioFeatures else None,
                    'Explicit': trackInfo.get('explicit', None),
                    'External URLs': trackInfo.get('external_urls', {}).get('spotify', None),
                    'Danceability': audioFeatures['danceability'] if audioFeatures else None,
                    'Energy': audioFeatures['energy'] if audioFeatures else None,
                    'Key': audioFeatures['key'] if audioFeatures else None,
                    'Loudness': audioFeatures['loudness'] if audioFeatures else None,
                    'Mode': audioFeatures['mode'] if audioFeatures else None,
                    'Speechiness': audioFeatures['speechiness'] if audioFeatures else None,
                    'Acousticness': audioFeatures['acousticness'] if audioFeatures else None,
                    'Instrumentalness': audioFeatures['instrumentalness'] if audioFeatures else None,
                    'Liveness': audioFeatures['liveness'] if audioFeatures else None,
                    'Valence': audioFeatures['valence'] if audioFeatures else None,
                    'Tempo': audioFeatures['tempo'] if audioFeatures else None,
                }
                musicData.append(trackData)

        #Create a pandas dataframe from the list of dictionaries
        df = pd.DataFrame(musicData)
        return df

playlistId = "5dJ5fNLHgszKXdgm3O3tjj"

#Call the function to get the music data from the playlist and
#store it in a dataframe
musicDf = getTrendingPlaylistData(playlistId, accessToken)
#Display the dataframe
print(musicDf)
# musicDf.to_csv('musicData.csv', index=False)
# print("CSV created")


# Checking for null values
print(musicDf.isnull().sum())


# Building the recommendation system

data = musicDf

# Function to calculate weighted popularity scores based on release date
def calculateWeightedPopularityScores(releaseDate):
    # Convert the release date to datetime object
    releaseDate = datetime.strptime(releaseDate, '%Y-%m-%d')

    # Calculate the time span betweeen release date and today's date
    timeSpan = datetime.now() - releaseDate

    # Calculate the weighted popularity score based on the time span
    # eg: More recent releases have higher weight
    weight = 1/(timeSpan.days + 1)
    return weight

# Normalize the music features using Min-Max Scaling
scaler = MinMaxScaler()
musicFeatures = musicDf[['Danceability', 'Energy', 'Key', 'Loudness',
                         'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness',
                         'Liveness', 'Valence', 'Tempo']].values
musicFeaturesScaled = scaler.fit_transform(musicFeatures)


# Function to get content-based recommendations based on music features
def contentBasedRecommendations(inputSongName, numRecommendations=5):
    if inputSongName not in musicDf['Track Name'].values:
        print(f"'{inputSongName}' not found in the dataset. Please enter a valid song name")
        return
    # Get the index of the input song in the musci DataFrame
    inputSongIndex = musicDf[musicDf['Track Name'] == inputSongName].index[0]

    # Claculate the similarity scores based on music features (cosine similarity)
    similarityScore = cosine_similarity([musicFeaturesScaled[inputSongIndex]], musicFeaturesScaled)

    # Get the indices of the most similar songs
    similarSongIndicies = similarityScore.argsort()[0][::-1][1:numRecommendations+1]

    # Get the names of the most similar songs based on content-based filtering
    contentBasedRecommendations = musicDf.iloc[similarSongIndicies][['Track Name', 'Artists', 'Album Name', 'Release Date', 'Popularity']]

    return contentBasedRecommendations


# Function to get hybrid recommendations based on weighted popularity
def hybridRecommendations(inputSongName, NumRecommendations=5, alpha=0.5):
    if inputSongName not in musicDf['Track Name'].values:
        print(f"'{inputSongName}' not found in the dataset. Please enter a valid song name")
        return
    contentBasedRecommendation = contentBasedRecommendations(inputSongName, NumRecommendations)

    #Get the popularity score of the input song
    popularityScore = musicDf.loc[musicDf['Track Name'] == inputSongName, 'Popularity'].values[0]

    #Calculate the weighted popularity score
    weightedPopularityScore = popularityScore * calculateWeightedPopularityScores(musicDf.loc[musicDf['Track Name'] == inputSongName, 'Release Date'].values[0])

    #Combine content-based and popularity-based recommendations based on weighted popularity
    hybridRecommendations = contentBasedRecommendation
    hybridRecommendations = hybridRecommendations._append({
        'Track Name': inputSongName,
        'Artist': musicDf.loc[musicDf['Track Name'] == inputSongName, 'Artists'].values[0],
        'Album Name': musicDf.loc[musicDf['Track Name'] == inputSongName, 'Album Name'].values[0],
        'Release Date': musicDf.loc[musicDf['Track Name'] == inputSongName, 'Release Date'].values[0],
        'Popularity': weightedPopularityScore
    }, ignore_index=True)

    # Sort the hybrid recommendations based on weighted popularity score
    hybridRecommendations = hybridRecommendations.sort_values(by='Popularity', ascending=False)

    # Remove the input song from the recommendations
    hybridRecommendations = hybridRecommendations[hybridRecommendations['Track Name'] != inputSongName]

    return hybridRecommendations

inputSongName = input()
recommendations = hybridRecommendations(inputSongName, 5)
print(f"Hybrid recommend songs for '{inputSongName}':")
print(recommendations)

