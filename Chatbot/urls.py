from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_page, name='home'),
    path('home', views.home_page, name='index'),
    path('login', views.login_user, name='login'),
    path('signup', views.signup_user, name='signup'),
    path('signed', views.signed, name='signed'),
    path('chatbot', views.chatbot, name='chatbot'),
]