from django.contrib import admin

from .models import *

admin.site.register(
    [
        Dataset,
        Experiment,
        CNNModel,
        RelevancyMethod,
        Example,
        Label,
    ]
)
