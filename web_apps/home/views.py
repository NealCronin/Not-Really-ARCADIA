from django.shortcuts import render

def index(request):
    """Renders the master role-selection gateway page."""
    return render(request, "home/index.html")