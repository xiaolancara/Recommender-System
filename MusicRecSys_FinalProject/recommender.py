# import libraries
import Pre_work
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler
import implicit # The Cython library
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

df_meta = pd.read_csv('music_meta.csv', index_col = 0)
df_meta_final = pd.read_csv('music_meta_final.csv', index_col = 0)


def score_to_counts(list_scores, userid, df_user):
    min_score = Pre_work.minimum(list_scores)
    max_score = Pre_work.maximum(list_scores)
    list_scores = list_scores / (max_score - min_score)
    where_are_NaNs = np.isnan(list_scores)
    list_scores[where_are_NaNs] = 0

    df_user_play = df_user[df_user['userid'].isin([userid])]
    min = Pre_work.minimum(df_user_play['playcounts'])
    max = Pre_work.maximum(df_user_play['playcounts'])
    list_scores = Pre_work.denormalize(list_scores, min, max)
    return list_scores, df_user_play


'''Content-based Filtering (CBF)'''
class CBF_contentBased:
    def __init__(self):
        #initializing tfidf vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        self.music_tfidf_matrix  = self.tfidf_vectorizer.fit_transform((df_meta_final['text'])) #fitting and transforming the vector
        music_tfidf_df = pd.DataFrame(self.music_tfidf_matrix.toarray())
        print(music_tfidf_df.shape)

    def evaluation(self, df, df_user, df_user_final):
        df_pre = pd.DataFrame(columns=['playcounts', 'pre_playcounts'])  #
        for userid in tqdm((df_user.userid.unique())):
            recommendation = pd.DataFrame(columns = ['trackid', 'pred_playCounts'])
            count = 0
            user_index = np.where(df_user_final['userid'] == userid)[0][0]
            user = df_user_final.iloc[[user_index]]

            # TF-IDF ( Term Frequency - Inverse Document Frequency )
            user_tfidf_matrix = self.tfidf_vectorizer.transform(user['text'])
            user_cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf_matrix, x), self.music_tfidf_matrix)
            output = list(user_cos_similarity_tfidf)

            # Function to get the top-N recomendations order by score
            top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)
            list_scores = np.array([output[i][0][0] for i in top])
            list_scores, df_user_play = score_to_counts(list_scores, userid, df_user)

            for i in top:
                recommendation.at[count, 'trackid'] = df['trackid'][i]
                recommendation.at[count, 'pred_playCounts'] = list_scores[count]
                count += 1

            # recommend musics for users that listened
            u_song_list = df_user[df_user['userid'] == userid]['trackid']
            recommendation = recommendation[recommendation['trackid'].isin(u_song_list)].dropna()
            df_user_play['pre_playcounts'] = recommendation['pred_playCounts'].values

            df_pre = df_pre.append(df_user_play[['playcounts', 'pre_playcounts']], ignore_index=True)
        rms = mean_squared_error(df_pre['playcounts'], df_pre['pre_playcounts'], squared=False)
        return rms, df_pre



    def get_recommendation(self, userid, Top_N, df_user_final, df_user):
        recommendation = pd.DataFrame(columns=['trackid', 'title', 'pred_playCounts'])
        count = 0
        user_index = np.where(df_user_final['userid'] == userid)[0][0]
        user = df_user_final.iloc[[user_index]]

        # TF-IDF ( Term Frequency - Inverse Document Frequency )
        user_tfidf_matrix = self.tfidf_vectorizer.transform(user['text'])
        user_cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf_matrix, x), self.music_tfidf_matrix)
        output = list(user_cos_similarity_tfidf)

        # Function to get the top-N recomendations order by score
        top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)
        list_scores = np.array([output[i][0][0] for i in top])
        list_scores, df_user_play = score_to_counts(list_scores, userid, df_user)

        for i in top:
            recommendation.at[count, 'trackid'] = df_meta['trackid'][i]
            recommendation.at[count, 'title'] = df_meta['title'][i]
            recommendation.at[count, 'pred_playCounts'] = list_scores[count]
            count += 1
        # display(recommendation)
        # recommend musics for users that haven't been listened
        u_song_list = df_user[df_user['userid'] == userid]['trackid']
        recommendation = recommendation[~recommendation['trackid'].isin(u_song_list)].dropna()
        recommendation = recommendation[:Top_N].set_index('trackid')

        return recommendation

'''Collaborative filtering (CF) (Memory-based Approach)'''
class CF_itemBased:
    def __init__(self, df_user):
        self.df_user = df_user
        self.userRatings = self.df_user.pivot_table(index=['userid'], columns=['trackid'], values='ratings')
        self.corrMatrix = self.userRatings.corr(method='pearson').fillna(0)

    def evaluate(self):
        df_predict = pd.DataFrame()
        for user_i in tqdm(self.userRatings.index):
            myRatings = self.userRatings.loc[user_i].dropna()
            original_index = len(df_predict)

            for track_i in list(myRatings.index):
                # retrieve similar songs for movie i
                similar_track = self.corrMatrix[track_i]

                # substract to similar score between movie i and rated movies
                similar_track = similar_track[similar_track.index.isin(myRatings.index)].sort_values(
                    ascending=False)  # [1:4]

                # calculate predict rating
                # adding 0.01 to avoid 0 similar score in low number of ratings system
                predict_ratings = sum(myRatings[myRatings.index.isin(similar_track.index)].reindex(
                    similar_track.index) * similar_track) / (sum(np.abs(similar_track)))

                df_predict = df_predict.append([[user_i, track_i, predict_ratings]])

            new_index = len(df_predict)

            df_predict.iloc[original_index:new_index, 2], _ = score_to_counts(
                df_predict.iloc[original_index:new_index, 2], user_i, self.df_user)

        df_predict.columns = ['userid', 'trackid', 'pred_playcounts']
        df_predict.reset_index(drop=True, inplace=True)
        df_predict = df_predict.merge(self.df_user, on=['userid', 'trackid'])

        # evaluate rms
        cf_rms = mean_squared_error(df_predict['playcounts'], df_predict['pred_playcounts'], squared=False)
        return cf_rms

    def recommend_songs(self, user_id, hm):
        myRatings = self.userRatings.loc[user_id].dropna()
        similar_candidates = pd.DataFrame()
        for i in list(self.corrMatrix.index):
            # retrieve similar songs for songs i
            similar_track = self.corrMatrix[i]
            # substract to similar score between songs i and rated songs
            similar_track = similar_track[similar_track.index.isin(myRatings.index)]
            # calculate predict rating
            predict_ratings = sum(myRatings * similar_track) / (sum(np.abs(similar_track)) + 0.01)
            similar_candidates = similar_candidates.append([predict_ratings])
        similar_candidates.iloc[:, 0], _ = score_to_counts(similar_candidates.iloc[:, 0], user_id, self.df_user)
        similar_candidates.index = self.corrMatrix.index
        # substract recommend songs that  have rated
        similar_candidates = similar_candidates[~similar_candidates.index.isin(myRatings.index)]
        similar_candidates.columns = ['pred_playcounts']
        similar_candidates.sort_values(by='pred_playcounts', inplace=True, ascending=False)
        return similar_candidates[:hm]

class CF_userbased:
    def __init__(self, df_user):
        self.df_user = df_user
        self.userRatings = self.df_user.pivot_table(index=['trackid'],columns=['userid'],values='ratings')
        self.corrMatrix = self.userRatings.corr(method='pearson').fillna(0)

    def score_item(self, neighbor_rating, similar_users_score, myRatings):
        # aumr -> active user mean rating
        aumr = np.mean(myRatings)
        mean_neighbor_rating = np.array(
            [np.mean(neighbor_rating[userid].dropna()) for userid in neighbor_rating.columns]).reshape(-1, 1)
        data = (np.dot(similar_users_score.values.reshape(1, -1),
                       (neighbor_rating.fillna(0).T - mean_neighbor_rating)) + aumr) / sum(similar_users_score)
        columns = neighbor_rating.T.columns
        return pd.DataFrame(data=data, columns=columns)

    def recommend_songs(self, user_id, hm):
        myRatings = self.userRatings.loc[:, user_id].dropna()
        similar_candidates = pd.DataFrame()
        similar_users_score = self.corrMatrix[user_id].sort_values(ascending=False)[:10]
        similar_users = similar_users_score.index
        neighbor_rating = self.userRatings[similar_users]
        similar_candidates = self.score_item(neighbor_rating, similar_users_score, myRatings).T
        similar_candidates.iloc[:, 0], _ = score_to_counts(similar_candidates.iloc[:, 0], user_id, self.df_user)
        # substract recommend songs that  have rated
        similar_candidates = similar_candidates[~similar_candidates.index.isin(myRatings.index)]
        similar_candidates.columns = ['pred_playcounts']
        similar_candidates.sort_values(by='pred_playcounts', inplace=True, ascending=False)
        return similar_candidates[:hm]

''''''
class CF_ALSModel:
    def __init__(self, df_user):
        # Create a numeric user_id and track_id column
        self.data = df_user.copy()
        self.data['userid'] = self.data['userid'].astype("category")
        self.data['trackid'] = self.data['trackid'].astype("category")
        self.data['user_id'] = self.data['userid'].cat.codes  # codes are ordered by category letter
        self.data['track_id'] = self.data['trackid'].cat.codes

        # The implicit library expects data as a item-user matrix so we
        # create two matricies, one for fitting the model (item-user)
        # and one for recommendations (user-item)
        sparse_item_user = sparse.csr_matrix((self.data['playcounts'].astype(float), (self.data['track_id'], self.data['user_id'])))
        self.sparse_user_item = sparse.csr_matrix((self.data['playcounts'].astype(float), (self.data['user_id'], self.data['track_id'])))

        # Initialize the als model and fit it using the sparse item-user matrix
        self.model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

        # Calculate the confidence by multiplying it by our alpha value.
        alpha_val = 15
        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model with confidence weights
        self.model.fit(data_conf)

    def recommend(self, userid, num_items=10):
        """The same recommendation function we used before"""

        # Get the trained user and item vectors. We convert them to
        # csr matrices to work with our previous recommend function.
        user_vecs = sparse.csr_matrix(self.model.user_factors)
        item_vecs = sparse.csr_matrix(self.model.item_factors)

        # Create recommendations for user with idn
        user_id = int(self.data[self.data['userid'] == userid]['user_id'].values[0])

        user_interactions = self.sparse_user_item[user_id, :].toarray()

        user_interactions = user_interactions.reshape(-1) + 1
        user_interactions[user_interactions > 1] = 0
        rec_vector = user_vecs[user_id, :].dot(item_vecs.T).toarray()
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
        recommend_vector = user_interactions * rec_vector_scaled
        item_idx = np.argsort(recommend_vector)[::-1][:num_items]

        tracks = []
        scores = []

        for idx in item_idx:
            tracks.append(self.data.trackid.loc[self.data.track_id == idx].iloc[0])
            scores.append(recommend_vector[idx])

        recommendations = pd.DataFrame({'trackid': tracks, 'score': scores})

        return recommendations.set_index('trackid')

class CF_SVDModel:
    def __init__(self, df_user):
        self.df_user = df_user
        # instantiate a reader and read in our rating data
        reader = Reader()
        data = Dataset.load_from_df(self.df_user[['userid', 'trackid', 'playcounts']], reader)

        # train SVD on 75% of known rates
        trainset, testset = train_test_split(data, test_size=.25, random_state=30)
        self.algorithm = SVD()
        self.algorithm.fit(trainset)
        predictions = self.algorithm.test(testset)

        # check the accuracy using Root Mean Square Error
        svd_rms = accuracy.rmse(predictions)
        print(svd_rms)

    def pred_user_rating(self, ui, hm):
        if ui in self.df_user.userid.unique():
            ui_list = self.df_user[self.df_user.userid == ui].trackid.tolist()
            d = self.df_user.trackid.unique()
            d = [v for v in d if not v in ui_list]
            predictedL = []
            for j in d:
                predicted = self.algorithm.predict(ui, j)
                predictedL.append((j, predicted[3]))
            pdf = pd.DataFrame(predictedL, columns=['trackid', 'pred_playcounts'])
            pdf.sort_values('pred_playcounts', ascending=False, inplace=True)
            pdf.set_index('trackid', inplace=True)
            return pdf.head(hm)
        else:
            print("User Id does not exist in the list!")
            return None

