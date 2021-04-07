# import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_user = pd.read_csv('music_user.csv', index_col=0)
df_meta = pd.read_csv('music_meta.csv', index_col=0)
df_meta["text"] = df_meta["artistname"] + " " + df_meta["title"] + " " + df_meta["lyrics"]
df_meta_all = df_meta[['trackid', 'text', 'title']]
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import string

stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()


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
# initializing tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer()

music_tfidf_matrix = tfidf_vectorizer.fit_transform((meta_final['text']))  # fitting and transforming the vector
music_tfidf_df = pd.DataFrame(music_tfidf_matrix.toarray())

df_user_final = pd.read_csv('music_user_final.csv', index_col=0)
df_meta_final = pd.read_csv('music_meta_final.csv', index_col=0)


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


def NewUser(title_list, userid):
    global df_user_final
    text = (' '.join(df_meta_all[(df_meta_all['title'].isin(title_list))]['text']))
    if userid not in df_user_final['userid'].values:
        df_user_final = df_user_final.append({'userid': userid, 'text': text}, ignore_index=True)
    else:
        text = df_user_final[(df_user_final['userid'] == userid)]['text'].add(text).values
        df_user_final.loc[df_user_final[(df_user_final['userid'] == userid)].index, 'text'] = text
    rec = get_recommendation(userid, 8 + len(title_list), df_meta_all)
    return rec[~(rec['title'].isin(title_list))].reset_index(drop=True)  # , index = True


print(NewUser(['Never Let You Go','Too Young','Eminem'],'xiaolan')['title'].values)