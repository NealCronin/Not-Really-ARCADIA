from django.urls import path
from . import views

urlpatterns = [
    # Maps to: http://127.0.0.1:8000/settings/config/
    path('config/', views.config_page, name='config_page'),
    
    # Maps to: http://127.0.0.1:8000/settings/models/
    path('models/', views.manage_models, name='manage_models'),
    
    # Endpoint to handle background server startup/shutdown lifecycles
    path('models/control/<str:model_type>/<str:action>/', views.control_model, name='control_model'),
    
    # Endpoint to view complete, un-truncated history log streams
    path('models/terminal/<str:model_type>/', views.view_terminal, name='view_terminal'),

    path('api/browse-local/', views.local_directory_browser_api, name='api_browse_local'),
]