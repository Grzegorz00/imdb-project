import requests
import pandas as pd

class APIData:
    def __init__(self):
        self.headers = {
            "X-RapidAPI-Key": "172e26a785msh17307706d898fb0p19f93ajsnf81e3f064b3d",
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }
        self.querystring_base_info = {"list": "top_rated_250", "info": "base_info"}
        self.querystring_revenue_budget = {"list": "top_rated_250", "info": "revenue_budget"}

    def retrieve_movies_data(self, num_pages):
        all_movies = []

        for page in range(1, num_pages + 1):
            url = f"https://moviesdatabase.p.rapidapi.com/titles?page={page}"

            response = requests.get(url, headers=self.headers, params=self.querystring_base_info)
            data = response.json()

            movies = data['results']

            ids = []
            titles = []
            genres = []
            average_rating = []
            num_votes = []
            runtimes = []
            release_years = []

            for movie in movies:
                ids.append(movie['id'])
                titles.append(movie['titleText']['text'])

                genres_array = movie['genres']['genres']
                genres_text = [genre['text'] for genre in genres_array]
                genres_concatenated = " ".join(genres_text)
                genres.append(genres_concatenated)

                average_rating.append(movie['ratingsSummary']['aggregateRating'])
                num_votes.append(movie['ratingsSummary']['voteCount'])
                runtimes.append(movie['runtime']['seconds'])
                release_years.append(movie['releaseYear']['year'])

            df = pd.DataFrame({'id': ids,
                               'title': titles,
                               'genres': genres,
                               'average_rating': average_rating,
                               'num_votes': num_votes,
                               'runtimes': runtimes,
                               'release_year': release_years})

            response = requests.get(url, headers=self.headers, params=self.querystring_revenue_budget)
            data = response.json()

            movies = data['results']

            money = []
            for movie in movies:
                production_budget = movie.get('productionBudget')
                worldwide_gross = movie.get('worldwideGross')

                budget_amount = None
                if production_budget is not None and 'budget' in production_budget and 'amount' in production_budget['budget']:
                    budget_amount = production_budget['budget']['amount']

                gross_amount = None
                if worldwide_gross is not None and 'total' in worldwide_gross and 'amount' in worldwide_gross['total']:
                    gross_amount = worldwide_gross['total']['amount']

                money.append({'budget': budget_amount, 'box_office_gross': gross_amount})

            money_df = pd.DataFrame(money)

            all_movies.append(df)
            all_movies.append(money_df)

        df_all = pd.concat(all_movies, ignore_index=True)

        return df_all