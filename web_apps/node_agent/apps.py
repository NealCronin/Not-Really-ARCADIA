# web_apps/node_agent/apps.py
from django.apps import AppConfig

class NodeAgentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    # FIXED: Must match the exact directory path from the project root
    name = 'web_apps.node_agent'