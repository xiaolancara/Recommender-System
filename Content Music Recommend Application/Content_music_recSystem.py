# import libraries
import pandas as pd
from tswift import Song
from tqdm import tqdm
#import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from statistics import mean
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

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
headList = ['trackid','artistname','title', 'mxm trackid','mxm artistname','mxm title']
df_mix = pd.read_csv('https://raw.githubusercontent.com/xiaolancara/Recommender-System/main/data/Music_finalProject/mxm_779k_matches.txt',
                     names = headList,
                     header = None,
                     delimiter='<SEP>',
                     encoding= 'utf-8',
                     engine='python')
df_meta = df_mix.iloc[:,:3].dropna()

# get the most 100 popular songs to save more computation time
track = df_user['trackid'].value_counts().sort_values(ascending=False).head(100)

df_meta = df_meta.loc[df_meta['trackid'].isin(track.index)]


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


lyrics = getLyrics(df_meta)

df_meta['lyrics'] = lyrics
df_meta = df_meta.dropna()

clean_lyrics = []
for i in df_meta.lyrics:
    string = i.replace('\n',' ')
#     string = re.sub(r'[^a-zA-Z-\' ]','',string)
#     string = re.sub(r'[a-zA-Z\' ]+[-]+[\s]+','',string).strip()
    clean_lyrics.append(string)

df_meta['lyrics'] = clean_lyrics
df_meta['lyrics'].replace('', np.nan, inplace=True)

df_meta = df_meta.dropna().reset_index(drop=True)

df_user = df_user.loc[df_user['trackid'].isin(df_meta['trackid'])].reset_index(drop=True).drop(['songid'],axis = 1)

df_meta["text"] = df_meta["artistname"] + " " + df_meta["title"] +" "+ df_meta["lyrics"]
df_meta_all = df_meta[['trackid', 'text', 'title']]
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')



stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

import string
def black_txt(token):
    return token not in stop_words_ and token not in list(string.punctuation) and len(token) > 2


def clean_txt(text):
    clean_text = []
    clean_text2 = []
    clean_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)


df_meta_all['text'] = df_meta_all['text'].apply(clean_txt)
meta_final = df_meta_all
#initializing tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer()

music_tfidf_matrix  = tfidf_vectorizer.fit_transform((meta_final['text'])) #fitting and transforming the vector
music_tfidf_df = pd.DataFrame(music_tfidf_matrix.toarray())

user_text = []
for user, _ in zip(df_user['userid'].unique(),tqdm(range(len(df_user['userid'].unique())))):
    user_playlist = (df_user.loc[df_user['userid'].isin([user])])
    user_text.append(' '.join(df_meta.loc[df_meta['trackid'].isin(user_playlist['trackid'])]['text']))

user_final = pd.DataFrame(df_user['userid'].unique(),columns = ['userid'])
user_final['text'] = user_text

# it takes about 6 mins to compute
user_final['text'] = user_final['text'].apply(clean_txt)

df_meta_final = meta_final
df_user_final = user_final


'''
def evaluation(df_user, Top_N):
    #print('User sample: ', (len(df_user)), 'Starting the evaluation for recommending ', Top_N, ' items……')
    user_info = pd.DataFrame(df_user.groupby(df_user['userid'])['trackid'].apply(list))
    list_trackid = []
    list_precision = []
    list_recall = []
    for trackid in user_info['trackid']:
        music_index = [np.where(df_meta['trackid'] == i)[0][0] for i in trackid]
        list_trackid.append(music_index)
    for userid, track_true, _ in zip(user_info.index.to_list(), list_trackid, tqdm(range(len(list_trackid)))):
        user_index = np.where(df_user_final['userid'] == userid)[0][0]
        user = df_user_final.iloc[[user_index]]

        # TF-IDF ( Term Frequency - Inverse Document Frequency )
        user_tfidf_matrix = tfidf_vectorizer.transform(user['text'])
        user_cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf_matrix, x), music_tfidf_matrix)
        output = list(user_cos_similarity_tfidf)
        top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:Top_N]

        precision = len([value for value in top if value in track_true]) / len(top)
        recall = len([value for value in top if value in track_true]) / len(track_true)
        list_precision.append(precision)
        list_recall.append(recall)

    Precision = round(mean(list_precision), 2)
    Recall = round(mean(list_recall), 2)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    #print('Precision: %.2f, Recall: %.2f, F1: %.2f'.format(Precision, Recall, F1))
    return Precision, Recall, F1

def evaluation_score(Num_rec):
    df_user_test = df_user.userid.value_counts().loc[lambda x: x>=Num_rec].rename_axis('userid').reset_index(name='counts')
    df_user_test = df_user_test.sample(n = len(df_user_test) if len(df_user_test)<100 else 100)
    df_user_test = df_user.loc[df_user['userid'].isin(df_user_test['userid'])]
    return evaluation(df_user_test,Num_rec)
score = [evaluation_score(i) for i in range(4,21,2)]
Num_rec = [i for i in range(4,21,2)]

df_evaluation = pd.DataFrame(score, columns=['Precision', 'Recall','F1'],index =Num_rec )
display(df_evaluation)
plt.plot(Num_rec, df_evaluation.iloc[:,0], 'b--', label='precision')
plt.plot(Num_rec, df_evaluation.iloc[:,1], 'g*', label = 'recall')
plt.plot(Num_rec, df_evaluation.iloc[:,2], 'o--', label = 'F1')
plt.xlabel('Number of recommendation')
plt.ylabel('Evaluation Score')
plt.legend(loc='upper right')
plt.ylim([0,1])
'''


def get_recommendation(userid, Top_N, df):
    recommendation = pd.DataFrame(columns=['userid', 'trackid', 'title', 'score'])
    count = 0
    user_index = np.where(df_user_final['userid'] == userid)[0][0]
    user = df_user_final.iloc[[user_index]]

    # TF-IDF ( Term Frequency - Inverse Document Frequency )
    user_tfidf_matrix = tfidf_vectorizer.transform(user['text'])
    user_cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf_matrix, x), music_tfidf_matrix)
    output = list(user_cos_similarity_tfidf)

    # Function to get the top-N recomendations order by score
    top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)
    list_scores = [output[i][0][0] for i in top]
    for i in top:
        recommendation.at[count, 'userid'] = userid
        recommendation.at[count, 'trackid'] = df['trackid'][i]
        recommendation.at[count, 'title'] = df['title'][i]
        recommendation.at[count, 'score'] = list_scores[count]
        count += 1

    # recommend musics for users that haven't been listened
    u_song_list = df_user[df_user['userid'] == userid]['trackid']
    recommendation = recommendation[~recommendation['trackid'].isin(u_song_list)].dropna()
    recommendation = recommendation[:Top_N].merge(df_meta[['trackid', 'artistname']], on='trackid')[
        ['userid', 'trackid', 'artistname', 'title', 'score']]

    return recommendation

'''
df_user_test = df_user.userid.value_counts().loc[lambda x: x>=8].rename_axis('userid').reset_index(name='counts')
userid = df_user_test.sample(n = 1)['userid'].values[0]
Top_N=8
print('Top ', Top_N,' music recommend for user ',userid)
display(get_recommendation(userid,Top_N,df_meta_all))
print('\n')
print('Music that user ',userid,' has listened')
# check the list that user has listened
u_song_list = df_user[df_user['userid'] == userid].merge(df_meta[['trackid','artistname','title']], on = 'trackid')
display(u_song_list)
'''

def NewUser(title_list, userid):
    global df_user_final
    text = (' '.join(df_meta_all[(df_meta_all['title'].isin(title_list))]['text']))
    if userid not in df_user_final['userid'].values:
        df_user_final = df_user_final.append({'userid': userid,'text': text},ignore_index=True)
    else:
        text = df_user_final[(df_user_final['userid'] == userid)]['text'].add(text).values
        df_user_final.loc[df_user_final[(df_user_final['userid'] == userid)].index, 'text'] = text
    rec = get_recommendation(userid,8+len(title_list),df_meta_all)
    return rec[~(rec['title'].isin(title_list))].reset_index(drop=True) #, index = True

#NewUser(['Never Let You Go','Too Young','Eminem'],'xiaolan')['title'].values
