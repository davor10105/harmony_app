# Generated by Django 5.0 on 2023-12-19 17:14

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "relevavance",
            "0005_cnnmodel_alter_label_choice_remove_label_dataset_and_more",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="example",
            name="prediction",
            field=models.CharField(default="belugisa whale", max_length=512),
            preserve_default=False,
        ),
    ]