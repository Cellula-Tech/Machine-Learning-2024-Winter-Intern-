from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.UploadCSVView.as_view(), name='upload_csv'),
    path('download/', views.DownloadCSVView.as_view(), name='download_csv'),
]
