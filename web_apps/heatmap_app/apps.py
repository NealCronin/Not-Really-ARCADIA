from django.apps import AppConfig

class HeatmapAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    
    # FIX: Update this string to match the exact path pathing used in INSTALLED_APPS
    name = 'web_apps.heatmap_app'