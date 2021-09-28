
# Business Problem: Making a recommendation for a given user by using item-based and user_based recommendation systems
# Dataset: movie, rating datasets


import pandas as pd
pd.set_option('display.max_columns', 20)


def create_user_movie_df(movie,rating):
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

def most_similar_user(df, user_id, common_movie_threshold, corr_threshold):

    user_watched_df = df[df.index == user_id]
    movies_watched = user_watched_df.columns[user_watched_df.notna().any()].tolist()
    movies_watched_df = df[movies_watched]
    
    # Finding the users who watched the same movies at least with a given number
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > common_movie_threshold]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          user_watched_df[movies_watched]])
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= corr_threshold)][
        ["user_id_2", "corr"]].reset_index(drop=True)
    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    return top_users


def most_recent_top_movie(rating_data, movie_data, user_id):
    movie_id = rating_data[(rating_data["userId"] == user_id) & (rating_data["rating"] == 5.0)]. \
                   sort_values(by = "timestamp", ascending = False)["movieId"][0:1].values[0]
    movie_name = str(movie_data[movie_data["movieId"] == movie_id]["title"].values[0])
    return movie_name


def user_based_recommendation (top_users, rating, number_of_movies):
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    # Finding Weighted Average Recommendation Score for top similar users
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()
    movies_to_be_recommend = recommendation_df.sort_values("weighted_rating", ascending=False).head(number_of_movies)
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"]


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


if __name__ == '__main__':

    # Reading datasets
    movie = pd.read_csv('movie_lens_dataset/movie.csv')
    rating = pd.read_csv('movie_lens_dataset/rating.csv')

    # Preparing dataset
    user_movie_df = create_user_movie_df(movie, rating)

    # Choosing a random user
    random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

    # Finding the users who watched the same movies with the random user
    top_users = most_similar_user(user_movie_df, random_user, 20, 0.65)

    recommended_movies = user_based_recommendation(top_users, rating, 5)
    recommended_movies

    # Item-based recommendation
    # Making recommendations based on the most recent movie that our user voted as 5 .
    # Finding the top voted recent movie
    movie_name = most_recent_top_movie(rating, movie, random_user)
    # Item-based recommendation
    movies_from_item_based = item_based_recommender(movie_name, user_movie_df)[1:6].index
    movies_from_item_based

