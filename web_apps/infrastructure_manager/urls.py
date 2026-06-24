from django.urls import path
from . import views

urlpatterns = [
    path('models/', views.manage_models, name='manage_models'),
    path('config/', views.config_page, name='config_page'),
    path('config/defaults/', views.restore_defaults, name='restore_defaults'),
    path('models/control/<str:model_type>/<str:action>/', views.control_model, name='control_model'),
    path('models/terminal/<str:model_type>/', views.view_terminal, name='view_terminal'),
    path('api/browse-local/', views.local_directory_browser_api, name='api_browse_local'),
]