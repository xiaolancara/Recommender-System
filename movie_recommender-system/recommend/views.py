from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import render, get_object_or_404, redirect
from .forms import *
from django.http import Http404
from .models import Movie, UserRating, UserList
from django.db.models import Q
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.db.models import Case, When
import pandas as pd
import io, csv
import numpy as np

# Create your views here.

def index(request):
    movies = Movie.objects.all()
    query = request.GET.get('q')

    if query:
        movies = Movie.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies})

    return render(request, 'recommend/list.html', {'movies': movies})


# Show details of the movie
def detail(request, movie_id):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404
    movies = get_object_or_404(Movie, id=movie_id)
    movie = Movie.objects.get(id=movie_id)
    
    temp = list(UserList.objects.all().values().filter(movie_id=movie_id, user=request.user))
    if temp:
        update = temp[0]['watch']
    else:
        update = False
    if request.method == "POST":

        # For my list
        if 'watch' in request.POST:
            watch_flag = request.POST['watch']
            if watch_flag == 'on':
                update = True
            else:
                update = False
            if UserList.objects.all().values().filter(movie_id=movie_id, user=request.user):
                UserList.objects.all().values().filter(movie_id=movie_id, user=request.user).update(watch=update)
            else:
                q=UserList(user=request.user, movie=movie, watch=update)
                q.save()
            if update:
                messages.success(request, "Movie added to your list!")
            else:
                messages.success(request, "Movie removed from your list!")

            
        # For rating
        else:
            rate = request.POST['rating']
            if UserRating.objects.all().values().filter(movie_id=movie_id, user=request.user):
                UserRating.objects.all().values().filter(movie_id=movie_id, user=request.user).update(rating=rate)
            else:
                q=UserRating(user=request.user, movie=movie, rating=rate)
                q.save()

            messages.success(request, "Rating has been submitted!")

        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
    out = list(UserRating.objects.filter(user=request.user.id).values())

    # To display ratings in the movie detail page
    movie_rating = 0
    rate_flag = False
    for each in out:
        if each['movie_id'] == movie_id:
            movie_rating = each['rating']
            rate_flag = True
            break

    context = {'movies': movies,'movie_rating':movie_rating,'rate_flag':rate_flag,'update':update}
    return render(request, 'recommend/detail.html', context)


# MyList functionality
def watch(request):

    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404

    movies = Movie.objects.filter(userlist__watch=True,userlist__user=request.user)
    query = request.GET.get('q')

    if query:
        movies = Movie.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/watch.html', {'movies': movies})

    return render(request, 'recommend/watch.html', {'movies': movies})

# Recommendation Algorithm
def recommend(request):

    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404


    movie_rating=pd.DataFrame(list(UserRating.objects.all().values()))
    movie_rating.to_csv('movie_rating.csv')
    print(movie_rating)
    if "user_id" not in movie_rating.columns.values:
        return render(request, 'recommend/recommend.html', None)
    rated_user=movie_rating.user_id.unique()
    current_user_id= request.user.id

	# if new user not rated any movie
    if current_user_id not in rated_user:
        # return popular
        popular_movie_list = list(movie_rating.movie_id.value_counts().index)
        movie_list = list(Movie.objects.filter(id__in=popular_movie_list)[:10])
        context = {'movie_list': movie_list}
        return render(request, 'recommend/recommend.html', context)

    # item_based CF RecSys
    userRatings = movie_rating.pivot_table(index=['user_id'],columns=['movie_id'],values='rating')
    corrMatrix = userRatings.corr(method='pearson').fillna(0)

    myRatings = userRatings.loc[current_user_id].dropna()

    similar_candidates = pd.DataFrame()
    for i in list(corrMatrix.index):
        # retrieve similar movies for movie i
        similar_movies = corrMatrix[i]
        # substract to similar score between movie i and rated movies
        similar_movies = similar_movies[similar_movies.index.isin(myRatings.index)]
        # calculate predict rating
        predict_ratings = sum(myRatings * similar_movies) / (sum(np.abs(similar_movies))+0.01)
        similar_candidates = similar_candidates.append([predict_ratings])
    similar_candidates.index = corrMatrix.index
    similar_candidates = similar_candidates[~similar_candidates.index.isin(myRatings.index)]
    similar_candidates.columns = ['PredictScore']
    similar_candidates.sort_values(by='PredictScore', inplace=True, ascending=False)
    movie_list = list(Movie.objects.filter(id__in=list(similar_candidates.index))[:10])

    context = {'movie_list': movie_list}
    return render(request, 'recommend/recommend.html', context)


# Register user
def signUp(request):
    form = UserForm(request.POST or None)

    if form.is_valid():
        user = form.save(commit=False)
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user.set_password(password)
        user.save()
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("index")

    context = {'form': form}

    return render(request, 'recommend/signUp.html', context)


# Login User
def Login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("index")
            else:
                return render(request, 'recommend/login.html', {'error_message': 'Your account disable'})
        else:
            return render(request, 'recommend/login.html', {'error_message': 'Invalid Login'})

    return render(request, 'recommend/login.html')


# Logout user
def Logout(request):
    logout(request)
    return redirect("login")

# upload movies info to Django
def movies_upload(request):
    template = "recommend/movie_upload.html"
    prompt = {'order':'Upload CSV file' }
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    if not csv_file.name.endswith('.csv'):
        messages.error(request,'This is not a csv file')
    data_set = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(data_set)
    next(io_string)
    for column in csv.reader(io_string, delimiter=','):
        _, created = Movie.objects.update_or_create(
            title = column[1],
            genre = column[7],
            year = column[6],
            ImdbRating = column[4],
            director = column[10],
            movie_logo = column[1]+".jpg",
            trailer = column[12]
        )
    context = {}
    # context = {'success':'Uploaded Successfully!'}
    return render(request, template, context)

