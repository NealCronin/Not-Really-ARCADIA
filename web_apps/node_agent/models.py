from django.db import models

class ActiveDaemon(models.Model):
    service_type = models.CharField(max_length=50)
    host = models.CharField(max_length=50, default="0.0.0.0")
    port = models.IntegerField(unique=True)
    pid = models.IntegerField(null=True, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.service_type} on {self.host}:{self.port}"