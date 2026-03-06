from django.urls import path
from django.views.generic import RedirectView

from . import views

urlpatterns = [
    path("",                         views.index,                name="index"),
    path("index.html",               views.index),
    path('Login.html',               views.Login,               name="Login"),
    path('Register.html',            views.Register,            name="Register"),
    path('Signup',                   views.Signup,              name="Signup"),
    path('UserLogin',                views.UserLogin,           name="UserLogin"),
    path('Predict.html',             views.Predict,             name="Predict"),
    path('PredictHeartCondition',    views.PredictHeartCondition, name="PredictHeartCondition"),
    path('heart-viewer',             views.HeartViewer,         name="HeartViewer"),
    path('history',                  views.History,             name="History"),
    path('change-password',          views.ChangePassword,      name="ChangePassword"),
    path('export-pdf',               views.ExportPDF,           name="ExportPDF"),
]