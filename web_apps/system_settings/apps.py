from django.apps import AppConfig

# Renamed configuration class
class SystemSettingsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    # FIX: Prepend your container folder name signature
    name = 'web_apps.system_settings'