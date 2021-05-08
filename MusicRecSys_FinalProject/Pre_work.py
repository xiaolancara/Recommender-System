# import libraries
import pandas as pd
from tswift import Song
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def read_data():
    '''Read Dataset'''
    headList = ['userid','songid','playcounts']
    df_user = pd.read_csv('https://raw.githubusercontent.com/xiaolancara/Recommender-System/main/data/Music_finalProject/kaggle_visible_evaluation_triplets.txt',
                          names = headList,
                          header = None,
                          sep = '\t',
                          encoding= 'utf-8',
                          engine='python')
    df_user = df_user.dropna()

    headList = ['songid','trackid']
    df_SongToTrack = pd.read_csv('https://raw.githubusercontent.com/xiaolancara/Recommender-System/main/data/Music_finalProject/taste_profile_song_to_tracks.txt',
                                 names = headList,
                                 header = None,
                                 sep = '\t',
                                 encoding= 'utf-8',
                                 engine='python')
    df_SongToTrack = df_SongToTrack.dropna()

    # Merging datasets of user and song to track on songid.
    df_user = df_user.merge(df_SongToTrack,
                            on='songid')
    df_user.drop('songid',axis=1,inplace=True)


    headList = ['trackid','artistname','title', 'mxm trackid','mxm artistname','mxm title']
    df_mix = pd.read_csv('https://raw.githubusercontent.com/xiaolancara/Recommender-System/main/data/Music_finalProject/mxm_779k_matches.txt',
                         names = headList,
                         header = None,
                         delimiter='<SEP>',
                         encoding= 'utf-8',
                         engine='python')
    df_meta = df_mix.iloc[:,:3].dropna()


    '''subset dataset'''
    # get the most 70 popular songs to save more computation time
    track = df_user['trackid'].value_counts().sort_values(ascending=False).head(70)
    df_meta = df_meta.loc[df_meta['trackid'].isin(track.index)]
    print('df_user', df_user.shape)
    print('df_meta', df_meta.shape)

    return df_meta, df_user


'''Getting Lyrics by tswift'''
def getLyrics(songs):
    num = 0
    lyrics = []
    Tol = len(songs)
    for title, artist, _ in zip(songs.title, songs.artistname, tqdm(range(Tol))):
        try:
            lyrics.append(Song(title=title, artist=artist).lyrics)
        except Exception:
            lyrics.append(None)
            pass

    return lyrics

def clean_data(df_meta):
    lyrics = getLyrics(df_meta)

    df_meta['lyrics'] = lyrics
    df_meta = df_meta.dropna()

    # basic cleaning
    clean_lyrics = []
    for i in df_meta.lyrics:
        string = i.replace('\n',' ')
    #     string = re.sub(r'[^a-zA-Z-\' ]','',string)
    #     string = re.sub(r'[a-zA-Z\' ]+[-]+[\s]+','',string).strip()
        clean_lyrics.append(string)

    df_meta['lyrics'] = clean_lyrics
    df_meta['lyrics'].replace('', np.nan, inplace=True)
    df_meta = df_meta.dropna().reset_index(drop=True)
    return df_meta


'''min max scale playcounts to 0-1'''
def maximum(data):
    return np.max(data)

def minimum(data):
    return np.min(data)

def normalize(playcounts, minimum, maximum):
    return (playcounts - minimum) / (maximum - minimum)

def denormalize(normalized, minimum, maximum):
    return normalized * (maximum-minimum)+ minimum

def playcount_scaler(df_user,df_meta):
    df_user = df_user.loc[df_user['trackid'].isin(df_meta['trackid'])].reset_index(drop=True)

    vc = df_user.userid.value_counts()
    user_index = list(vc[vc>=5].index)
    df_user = df_user[df_user.userid.isin(user_index)].reset_index(drop=True)

    # adding rating columns based on playcount
    df_user['ratings'] = np.nan
    for user in tqdm((df_user.userid.unique())):
        df_user_play = df_user[df_user['userid'].isin([user])]
        min = minimum(df_user_play['playcounts'])
        max = maximum(df_user_play['playcounts'])
        norm = normalize(df_user_play['playcounts'],min,max)
        ind = norm.index
        df_user.loc[ind, 'ratings'] = norm
        df_user.fillna(0, inplace=True)
    '''Write dataframe to csv'''
    df_meta.to_csv('music_meta.csv', sep=',', encoding='utf-8')
    df_user.to_csv('music_user.csv', sep=',', encoding='utf-8')

'''test'''
# df_meta, df_user = read_data()
# df_meta = clean_data(df_meta)
# playcount_scaler(df_user,df_meta)