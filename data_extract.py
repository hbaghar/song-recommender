import numpy as np
import dask.bag as db
import dask.dataframe as dd
from dask.distributed import Client
import dask
import pandas as pd
import os
import json
import random

@dask.delayed
def dump_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

@dask.delayed
def get_playlist(data):
    return data['playlists']


def main():
    client = Client()
    path = '/Users/hridaybaghar/Downloads/spotify_million_playlist_dataset/data'
    files = [f for f in os.listdir(path) if f.endswith('.json') and f.startswith('mpd.slice')]
    
    # We only randomly sample 25% of the data to save space. 
    # This is still 250k playlists and should  work well enough for the content recommender we are trying to build.
    files = random.sample(files, int(len(files)*0.25))
    files = [os.path.join(path, f) for f in files]
    
    playlists = [get_playlist(dump_json(f)) for f in files]
    playlists = db.from_delayed(playlists)

    # Dumping tracks to parquet
    tracks = playlists.pluck('tracks').flatten()\
                                    .to_dataframe()\
                                    .set_index('track_uri')\
                                    .drop(columns=['pos'])\
                                    .map_partitions(lambda x: x.drop_duplicates())\
                                    .to_parquet('data/tracks.parquet', engine='pyarrow', compression='snappy')
    
    # Expanding the playlist track object
    playlists = playlists.to_dataframe()\
                        .explode('tracks')\
                        .reset_index()
    
    # Hacky fix to get the keys of the track object because pd.json_normalize() doesn't work when writing to disk
    keys = playlists["tracks"].head(1).values[0].keys()
    for key in keys:
        playlists[key] = playlists["tracks"].to_bag().pluck(key).to_dataframe().iloc[:,0]
    
    # Writing playlists to parquet
    playlists = playlists.drop(columns=['tracks']).to_parquet('data/playlists.parquet', engine='pyarrow', compression='snappy')

if __name__ == '__main__':
    main()
