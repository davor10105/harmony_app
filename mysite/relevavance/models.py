from django.db import models
from django.contrib.auth.models import User


class Dataset(models.Model):
    name = models.CharField(max_length=128)

    def __str__(self):
        return self.name


class Experiment(models.Model):
    name = models.CharField(max_length=128)

    def __str__(self):
        return self.name


class CNNModel(models.Model):
    name = models.CharField(max_length=128)

    def __str__(self):
        return self.name


class RelevancyMethod(models.Model):
    name = models.CharField(max_length=128)

    def __str__(self):
        return self.name


class Example(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    cnn_model = models.ForeignKey(CNNModel, on_delete=models.CASCADE)
    relevancy_method = models.ForeignKey(RelevancyMethod, on_delete=models.CASCADE)
    prediction = models.CharField(max_length=512)
    image = models.CharField(max_length=512)
    original = models.CharField(max_length=512)
    learn = models.CharField(max_length=512)

    def __str__(self):
        return self.image + self.original + self.learn


class Label(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    example = models.ForeignKey(Example, on_delete=models.CASCADE)
    choice = models.CharField(max_length=128)

    def __str__(self):
        return str(self.user) + " " + self.choice
