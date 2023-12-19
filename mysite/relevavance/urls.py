from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("logout", views.logout_view),
    path(
        "<str:dataset_name>/<str:experiment_name>/<str:cnnmodel_name>/<str:relevancy_method_name>",
        views.experiment_view,
        name="experiment_view",
    ),
    path(
        "<str:dataset_name>/<str:experiment_name>/<str:cnnmodel_name>/<str:relevancy_method_name>/<int:example_index>",
        views.show_example,
        name="show_example",
    ),
    path(
        "annotate_example/<str:dataset_name>/<str:experiment_name>/<str:cnnmodel_name>/<str:relevancy_method_name>/<int:example_index>/<str:annotation>",
        views.annotate_example,
        name="annotate_example",
    ),
]
