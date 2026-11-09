from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

app_name = "detection"

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register, name="register"),
    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="detection/login.html"
        ),
        name="login",
    ),
    path(
        "logout/",
        auth_views.LogoutView.as_view(),
        name="logout",
    ),
    path("history/", views.history, name="history"),
    path("history/export/", views.export_history, name="export_history"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("performance/", views.performance, name="performance"),
    path(
    "performance/confusion-matrix/",
    views.confusion_matrix_image,
    name="confusion_matrix_image",
),
path(
    "results/<int:pk>/",
    views.prediction_result,
    name="result",
),
]