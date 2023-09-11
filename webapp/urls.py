from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('stylize/', views.stylize, name='stylize'),
]
