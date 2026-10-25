# GTC Book Recommendation System

**Find your next read through patterns in reader ratings.** This Streamlit app turns a book-ratings collection into an interactive, similarity-based recommendation experience.

[Open the live app](https://gtc-book-recommendation-system.streamlit.app/) · [View the source](https://github.com/AhmedMagdi2003/GTC-Book-Recommendation-System)

## The project

Readers often know what they enjoy but need help finding books with a similar audience. This project explores collaborative filtering: it compares rating patterns across books and surfaces recommendations from the available data.

## What you can do

- Browse recommendations for users in the prepared dataset.
- Upload a ratings CSV to generate recommendations from a different set of readers.
- See book covers when image links are present in the data.

## How recommendations are formed

The app organizes ratings by book and user, then uses cosine distance with a nearest-neighbor model to find books with similar rating patterns. For uploaded data, it builds a model from the submitted ratings. This is a similarity-based recommender, not an LLM-generated list.

## Try it

The live Streamlit app is the quickest way to explore the project. To run it locally, clone the repository, install the dependencies in requirements.txt, then run:

~~~bash
streamlit run app.py
~~~

The upload feature expects a CSV with these columns: user_id, title, and book_rating. The bundled model and prepared data are used by the built-in recommendation path.

## Built with

Python · Streamlit · pandas · scikit-learn nearest neighbors · cosine similarity

## Project scope

This is a learning project built around the included ratings data. Recommendations depend on the coverage and quality of those ratings; the repository does not connect to a live book catalog or provide an account-based personalization service.

## About

Built by Ahmed Magdi as a hands-on recommender systems project. See the [notebooks](https://github.com/AhmedMagdi2003/GTC-Book-Recommendation-System/tree/main/notebooks) for the analysis work.
