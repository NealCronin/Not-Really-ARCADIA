from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # FIX: Swapped 'config/' out for 'settings/'
    path('settings/', include('web_apps.system_settings.urls')), 
    
    path('', include('web_apps.heatmap_app.urls')),
]