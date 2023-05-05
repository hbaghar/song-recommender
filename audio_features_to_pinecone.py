import pandas as pd
import spotipy
import os
from tqdm import tqdm
import pinecone
from time import sleep

tqdm.pandas()

def search_lyrics(title, artist, genius):
    """
    Search for lyrics on Genius API
    """
    try:
        song = genius.search_song(title, artist).lyrics
    except:
        song = ""
    return f"{artist} {title} {''.join(song.splitlines()[1:])}"

    
def main():
    
    genius_token = os.environ['GENIUS_CLIENT_TOKEN']    
    sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyClientCredentials())    
    songs = pd.read_parquet('data/tracks.parquet')

    songs.reset_index(inplace=True)
    songs.rename(columns={'track_uri': 'id'}, inplace=True)
    songs['metadata'] = songs[['track_name', 'artist_name', 'album_name']].to_dict(orient='records')
    songs = songs[['id', 'metadata']]
    
    # Setting up logic to resume keyboard interupted process/crashes
    batch_ind = range(0, len(songs), 100)
    batch_ind = list(batch_ind) + [len(songs)]
    batches = list(zip(batch_ind[:-1], batch_ind[1:]))
    continue_point = 0

    # Setting up vector store
    pinecone_token = os.environ['PINECONE_API_KEY']
    pinecone_environment = os.environ['PINECONE_ENVIRONMENT']
    pinecone.init(pinecone_token, environment=pinecone_environment)
    index_name = 'spotify-audio'
    if index_name not in pinecone.list_indexes():
        pinecone.create_index('spotify-audio', metric='cosine', dimension=13, shards=1, replicas=1)
    index = pinecone.Index(index_name)

    bathces = batches[continue_point:]
    i = continue_point
    for start, end in tqdm(batches):
        subset = songs.iloc[start:end-1].copy(deep=True)

        subset['values'] = sp.audio_features(subset['id'].tolist())
        subset['values'] = subset['values'].apply(lambda x: [float(v) for k,v in x.items() if k not in ('analysis_url', 'id', 'track_href', 'type', 'uri')] if x is not None else [-1.0]*13)
        _ = index.upsert_from_dataframe(subset, show_progress=False)
        sleep(0.3)
        with open('process_log.txt', 'w') as f:
            f.write(f"Processed {start} to {end}. Use continue point {i}\n")
        i += 1
if __name__ == '__main__':
    main()