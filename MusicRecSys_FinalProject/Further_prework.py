# import libraries
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

df_user = pd.read_csv('music_user.csv', index_col = 0)
df_meta = pd.read_csv('music_meta.csv', index_col = 0)

df_meta["text"] = df_meta["artistname"] + " " + df_meta["title"] +" "+ df_meta["lyrics"]
df_meta_all = df_meta[['trackid', 'text', 'title']]


'''Data Cleaning by Using NLP'''
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

def Deep_cleaning():
    df_meta_all['text'] = df_meta_all['text'].apply(clean_txt)
    meta_final = df_meta_all

    '''EDA and do the deep cleaning'''
    cv = CountVectorizer()
    data_cv = cv.fit_transform(meta_final.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = meta_final.trackid
    data = data_dtm.transpose()

    # Find the top 30 words by each track
    top_dict = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(30)
        top_dict[c]= list(zip(top.index, top.values))

    # Print the top 15 words by each track
    for track, top_words in top_dict.items():
        print(track)
        print(', '.join([word for word, count in top_words[0:14]]))
        print('---')

    # Let's first pull out the top 30 words for each track
    words = []
    for track in data.columns:
        top = [word for (word, count) in top_dict[track]]
        for t in top:
            words.append(t)

    # Let's aggregate this list and identify the most common words along with how many routines they occur in
    Counter(words).most_common()

    # If more than 10 of the track have it as a top word, exclude it from the list
    add_stop_words = [word for word, count in Counter(words).most_common() if count > 10]
    print('add_stop_words',add_stop_words)

    # Look at the most common top words --> add them to the stop word list
    # Add new stop words
    stop_words_ = set(stopwords.words('english')).union(add_stop_words)
    df_meta_all['text'] = df_meta_all['text'].apply(clean_txt)
    meta_final = df_meta_all

    '''Plot the Songs Word Cloud'''
    # Let's make some word clouds!
    # Terminal / Anaconda Prompt: conda install -c conda-forge wordcloud


    cv = CountVectorizer()
    data_cv = cv.fit_transform(meta_final.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = meta_final.trackid
    data = data_dtm.T
    wc = WordCloud(stopwords=stop_words_, background_color="white", colormap="Dark2",
                   max_font_size=150, random_state=42)

    # Reset the output dimensions
    plt.rcParams['figure.figsize'] = [16, 10]

    # Create subplots for each track
    for index, track in enumerate(data.columns):
        row = list(meta_final[meta_final.trackid == track].index)[0]
        wc.generate(meta_final.text[row])

        plt.subplot(7, 5, index + 1)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(meta_final.title[row])

    plt.tight_layout()
    plt.show()
    return meta_final
def create_user_corpus(meta_final):
    '''Creating the User Corpus'''
    user_text = []
    for user, _ in zip(df_user['userid'].unique(),tqdm(range(len(df_user['userid'].unique())))):
        user_playlist = (df_user.loc[df_user['userid'].isin([user])])
        user_text.append(' '.join(df_meta.loc[df_meta['trackid'].isin(user_playlist['trackid'])]['text']))

    user_final = pd.DataFrame(df_user['userid'].unique(),columns = ['userid'])
    user_final['text'] = user_text

    # it takes about 6 mins to compute
    user_final['text'] = user_final['text'].apply(clean_txt)

    meta_final.to_csv('music_meta_final.csv', sep=',', encoding='utf-8')
    user_final.to_csv('music_user_final.csv', sep=',', encoding='utf-8')

'''test'''
# meta_final = Deep_cleaning()
# create_user_corpus(meta_final)