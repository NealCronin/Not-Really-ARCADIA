from django.urls import path
from . import views

urlpatterns = [
    # FIXED: References your newly configured heatmap view endpoint directly
    path('', views.heatmap, name='heatmap'),
    path('video_stream/', views.video_stream, name='video_stream'),
]