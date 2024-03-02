from django.db import models

# Create your models here.

class chatlog(models.Model):
    chatlognr = models.AutoField(primary_key=True)
    session_id = models.TextField()
    username = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    prompt = models.TextField()
    response = models.TextField()
    