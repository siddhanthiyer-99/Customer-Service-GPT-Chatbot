# Register your models here.
from django.contrib import admin
from .models import chatlog

@admin.register(chatlog)
class AdminSupplies(admin.ModelAdmin):
    list_display = ('chatlognr', 'session_id', 'username', 'timestamp', 'prompt', 'response')

