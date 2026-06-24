from django.urls import path
from . import views

urlpatterns = [
    # UI Panel
    path('', views.host_dashboard, name='host_dashboard'),
    path('control/stop/<int:port>/', views.stop_daemon_ui, name='stop_daemon_ui'),
    
    # Unified Network API Endpoints (Invoked by Infrastructure Manager over LAN/VPN)
    path('api/node/status', views.api_node_status, name='api_node_status'),
    path('api/node/start', views.api_node_start, name='api_node_start'),
    path('api/node/stop', views.api_node_stop, name='api_node_stop'),
    path('api/node/logs', views.api_node_logs, name='api_node_logs'),
]