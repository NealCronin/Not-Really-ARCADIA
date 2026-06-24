from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Root path traffic is fully handled by your modular home application
    path('', include('web_apps.home.urls')), 
    
    # Modular sub-application workspace routing maps
    path('heatmap/', include('web_apps.heatmap_app.urls')),
    path('settings/', include('web_apps.infrastructure_manager.urls')),
    path('host/', include('web_apps.node_agent.urls')), 
]