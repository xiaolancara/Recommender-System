# import libraries
import Pre_work
import Further_prework
import recommender
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


df_user = pd.read_csv('music_user.csv', index_col = 0)
df_meta = pd.read_csv('music_meta.csv', index_col = 0)
df_user_final = pd.read_csv('music_user_final.csv', index_col = 0)
df_meta_final = pd.read_csv('music_meta_final.csv', index_col = 0)

def NewUser(title_list, userid):
    global df_user_final, df_user
    text = (' '.join(df_meta[(df_meta['title'].isin(title_list))]['lyrics']))
    if userid not in df_user_final['userid'].values:
        for song in title_list:
            # search the trackid by title
            trackid = df_meta[df_meta['title'] == song]['trackid'].values[0]
            # Let's say the playcounts of song that the user likes is 1, ratings is 1
            df_user = df_user.append({'userid': userid, 'playcounts': 1, 'trackid': trackid, 'ratings': 0},
                                     ignore_index=True)

        df_user_final = df_user_final.append({'userid': userid, 'text': text}, ignore_index=True)
        user_index = int(df_user_final[df_user_final['userid'] == userid].index.values)
        df_user_final.iloc[user_index]['text'] = Further_prework.clean_txt(df_user_final.iloc[user_index]['text'])
    else:

        for song in title_list:
            # search the trackid by title
            trackid = df_meta[df_meta['title'] == song]['trackid'].values[0]
            user_tracks = df_user[df_user['userid'] == userid]
            # the track has listened before
            if trackid in user_tracks['trackid'].values:
                play_index = int(user_tracks[user_tracks['trackid'] == trackid].index.values)
                df_user.loc[play_index, 'playcounts'] += 1
            else:
                df_user = df_user.append({'userid': userid, 'playcounts': 1, 'trackid': trackid, 'ratings': 0},
                                         ignore_index=True)
                user_index = int(df_user_final[df_user_final['userid'] == userid].index.values)
                new_text = df_meta[(df_meta['trackid'] == trackid)]['lyrics'].values[0]
                df_user_final.iloc[user_index]['text'] += Further_prework.clean_txt(new_text)

        df_user_play = df_user[df_user['userid'] == userid]
        min = Pre_work.minimum(df_user_play['playcounts'])
        max = Pre_work.maximum(df_user_play['playcounts'])
        norm = Pre_work.normalize(df_user_play['playcounts'], min, max)
        ind = norm.index
        df_user.loc[ind, 'ratings'] = norm
        df_user.fillna(0, inplace=True)

    # content based recommend
    content_based = recommender.CBF_contentBased()
    CBF_contentBased = content_based.get_recommendation(userid, 5, df_user_final, df_user)
    CBF_contentBased = CBF_contentBased[~(CBF_contentBased['title'].isin(title_list))]

    item_based = recommender.CF_itemBased(df_user)
    CF_itemBased = item_based.recommend_songs(userid, 5)

    user_based = recommender.CF_userbased(df_user)
    CF_userbased = user_based.recommend_songs(userid, 5)

    ALS = recommender.CF_ALSModel(df_user)
    CF_ALSModel = ALS.recommend(userid, 5)


    SVD = recommender.CF_SVDModel(df_user)
    CF_SVDModel = SVD.pred_user_rating(userid, 5)
    df_user.to_csv('music_user.csv', sep=',', encoding='utf-8')
    df_user_final.to_csv('music_user_final.csv', sep=',', encoding='utf-8')
    return  CBF_contentBased,  CF_itemBased, CF_userbased, CF_ALSModel, CF_SVDModel, df_user, df_user_final

def get_title(tracks):
    title = [df_meta[df_meta['trackid']==trackid]['title'].values[0] for trackid in tracks]
    return title




# CBF_contentBased, CF_itemBased, CF_userbased, CF_ALSModel, CF_SVDModel, df_user, df_user_final = NewUser(['I Kissed A Girl','Float On','The Gift'],'xiaolan')
#
#
# print(get_title(CBF_contentBased.index))
# print(CBF_contentBased)