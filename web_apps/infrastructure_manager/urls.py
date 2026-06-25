from django.urls import path
from . import views

urlpatterns = [
    # --------------------------------------------------------------------------
    # Core UI Dashboards
    # --------------------------------------------------------------------------
    # Maps to: {% url 'manage_models' %}
    path('', views.manage_models, name='manage_models'),
    
    # Maps to: {% url 'config_page' %}
    path('config/', views.config_page, name='config_page'),

    # Maps to: {% url 'view_terminal' model_type='vlm' %}
    path('terminal/<str:model_type>/', views.view_terminal, name='view_terminal'),

    # --------------------------------------------------------------------------
    # Action Endpoints (Form Submissions & Buttons)
    # --------------------------------------------------------------------------
    # Maps to: {% url 'control_model' model_type='vlm' action='start' %}
    path('control/<str:model_type>/<str:action>/', views.control_model, name='control_model'),
    
    # Maps to: {% url 'restore_defaults' %}
    path('config/restore/', views.restore_defaults, name='restore_defaults'),

    # --------------------------------------------------------------------------
    # Internal APIs (AJAX / Fetch calls)
    # --------------------------------------------------------------------------
    # Triggers native file browser for the dataset path selection
    path('api/browse-local/', views.browse_local_api, name='browse_local_api'),
]