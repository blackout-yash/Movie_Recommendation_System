import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# Read the "u.data" file to extract useful information in a dataframe named 'df'.
columns_name = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("u.data", sep = "\t", names = columns_name)


# Read the "u.item" file to extract useful information in a dataframe named 'movie_titles'.
movie_titles = pd.read_csv("u.item", sep = "\|", header = None)
movie_titles = movie_titles[[0, 1]]
movie_titles.columns = ["item_id", "title"]


# Merge two different dataframes named df and movie_title respectively into a df named df 
df = pd.merge(df, movie_titles, on = "item_id")


# Create a new dataframe named 'ratings' and store the information "mean rating" and "number of ratings".
ratings = pd.DataFrame(df.groupby("title").mean()["rating"])
ratings["no_of_ratings"] = pd.DataFrame(df.groupby("title").count()["rating"])
ratings = ratings.sort_values(by = "rating", ascending = False)


# It stores user's data that which user gave what rating to which movie.
movie_matrix = df.pivot_table(index = "user_id", columns = "title", values = "rating")


# This function takes your watched movie name as argument and returns different movie names accordingly.
def predict_movies(movie_name):
    movie_user_rating = movie_matrix[movie_name]
    similar_to_movie =  movie_matrix.corrwith(movie_user_rating)

    similar_to_movie = pd.DataFrame(similar_to_movie, columns = ["Corelaton"])
    similar_to_movie.dropna(inplace = True)
    similar_to_movie = similar_to_movie.join(ratings["no_of_ratings"])


    prediction = similar_to_movie[similar_to_movie["no_of_ratings"] > 100].sort_values("Corelaton", ascending = False)
    return prediction


# Here you have to enter the name of your favorite movie, so that more such movies can be predicted.
predicted_movie_names = predict_movies("Titanic (1997)")
print(predicted_movie_names.head(n = 10))