# Item based movie recommender system using Django frame
Language: python, html, css, javascript

## Installation

>pip install -r requirements.txt
>

## Run server on terminal
>python manage.py runserver
>

## Open the local website
http://127.0.0.1:8000/

## Algorithm
**Item based** collaborative filtering in this project

Recommend movies based on predicting rating 

## Function
- user sign up
- user log in
- user rating
- user list library
- movie info and trailer video
- you might like
- search movie
- upload csv file (update movie list)

## Database 
Movie Item: Export csv file from [Imdb](https://www.imdb.com/list/ls022753498/) top 30 popular movies list and added poster and trailer link
Server: db.sqlite3

### Challenge
1. Cold start problem. I use most rating movies(most popular) to recommend for new user in this project.
2. Rating is not enough. Since it's item based recommender. It's supposed to have much more number of users than number of items. Thus some users might have 0 similar score items. In this project, number of ratings more than 100 is ideal
