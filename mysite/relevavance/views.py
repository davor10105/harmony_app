from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404, JsonResponse
from .models import Dataset, Experiment, CNNModel, RelevancyMethod, Example, Label
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.db.models import Count

import os
import random
from PIL import Image
import base64


def get_unlabelled_examples(user):
    labeled_examples = Label.objects.filter(user=user).values("example").values("pk")
    non_labeled_examples = Example.objects.exclude(pk__in=labeled_examples)

    return non_labeled_examples


def get_filtered_examples(
    dataset_name,
    experiment_name,
    cnnmodel_name,
    relevancymethod_name,
):
    c_dataset = Dataset.objects.get(name__iexact=dataset_name)
    c_experiment = Experiment.objects.get(name__iexact=experiment_name)
    c_cnnmodel = CNNModel.objects.get(name__iexact=cnnmodel_name)
    c_relevancymethod = RelevancyMethod.objects.get(name__iexact=relevancymethod_name)
    filtered_examples = Example.objects.filter(
        dataset=c_dataset,
        experiment=c_experiment,
        cnn_model=c_cnnmodel,
        relevancy_method=c_relevancymethod,
    ).order_by("pk")

    return filtered_examples


@login_required
def index(request):
    example_groups = (
        Example.objects.all()
        .values("dataset", "experiment", "cnn_model", "relevancy_method")
        .annotate(count=Count("pk"))
    )

    experiments = []
    for example_group in example_groups:
        dataset_name = Dataset.objects.get(pk=example_group["dataset"])
        experiment_name = Experiment.objects.get(pk=example_group["experiment"])
        cnnmodel_name = CNNModel.objects.get(pk=example_group["cnn_model"])
        relevancymethod_name = RelevancyMethod.objects.get(
            pk=example_group["relevancy_method"]
        )

        filtered_examples = get_filtered_examples(
            dataset_name,
            experiment_name,
            cnnmodel_name,
            relevancymethod_name,
        )

        user_labels = Label.objects.filter(user=request.user).values_list(
            "example", flat=True
        )
        unlabelled_examples = filtered_examples.exclude(pk__in=user_labels)

        experiments.append(
            (
                dataset_name,
                experiment_name,
                cnnmodel_name,
                relevancymethod_name,
                len(filtered_examples) - len(unlabelled_examples),
                len(filtered_examples),
            )
        )

    return render(
        request=request,
        template_name="relevavance/index.html",
        context={
            "experiments": experiments,
        },
    )


@login_required
def experiment_view(
    request,
    dataset_name,
    experiment_name,
    cnnmodel_name,
    relevancy_method_name,
):
    filtered_examples = get_filtered_examples(
        dataset_name,
        experiment_name,
        cnnmodel_name,
        relevancy_method_name,
    )

    user_labels = Label.objects.filter(user=request.user).values_list(
        "example", flat=True
    )
    unlabelled_examples = filtered_examples.exclude(pk__in=user_labels)
    print(request.user, unlabelled_examples)

    if len(unlabelled_examples) > 0:
        return redirect(
            f"/relevance/{dataset_name}/{experiment_name}/{cnnmodel_name}/{relevancy_method_name}/{unlabelled_examples[0].pk}"
        )
    else:
        return render(
            request=request,
            template_name="relevavance/done.html",
        )


@login_required
def logout_view(request):
    logout(request=request)
    return redirect("/accounts/login")


@login_required
def show_example(
    request,
    dataset_name,
    experiment_name,
    cnnmodel_name,
    relevancy_method_name,
    example_index,
):
    random_index = random.randint(0, 1)
    choice_dict = {
        "original": random_index if random_index == 0 else 2,
        "neither": 1,
        "learn": 1 - random_index if 1 - random_index == 0 else 2,
    }
    reverse_choice_dict = {value: key for key, value in choice_dict.items()}

    print(os.path.abspath("relevavance/views.py"))
    try:
        dataset = Dataset.objects.filter(name__iexact=dataset_name)[0]
        experiment = Experiment.objects.filter(name__iexact=experiment_name)[0]
        cnn_model = CNNModel.objects.filter(name__iexact=cnnmodel_name)[0]
        relevancy_method = RelevancyMethod.objects.filter(
            name__iexact=relevancy_method_name
        )[0]
        example = Example.objects.filter(
            dataset=dataset,
            experiment=experiment,
            cnn_model=cnn_model,
            relevancy_method=relevancy_method,
            pk=example_index,
        )[0]

        filtered_examples = get_filtered_examples(
            dataset_name,
            experiment_name,
            cnnmodel_name,
            relevancy_method_name,
        )

        user_labels = Label.objects.filter(user=request.user).values_list(
            "example", flat=True
        )
        unlabelled_examples = filtered_examples.exclude(pk__in=user_labels)

        image_path = os.path.join("relevavance", example.image)
        with open(image_path, "rb") as image_file:
            encoded_image = (
                "data:image/png;base64," + base64.b64encode(image_file.read()).decode()
            )
        original_path = os.path.join("relevavance", example.original)
        with open(original_path, "rb") as image_file:
            encoded_original = (
                "data:image/png;base64," + base64.b64encode(image_file.read()).decode()
            )
        learn_path = os.path.join("relevavance", example.learn)
        with open(learn_path, "rb") as image_file:
            encoded_learn = (
                "data:image/png;base64," + base64.b64encode(image_file.read()).decode()
            )
        print(encoded_image[:100])
    except Dataset.DoesNotExist:
        raise Http404("Dataset does not exist")

    image_dict = {
        "original": encoded_original,
        "learn": encoded_learn,
    }

    relevance_1_image = image_dict[reverse_choice_dict[0]]
    relevance_1_label = reverse_choice_dict[0]
    relevance_2_image = image_dict[reverse_choice_dict[2]]
    relevance_2_label = reverse_choice_dict[2]

    previous_example = None
    try:
        previous_example = Example.objects.get(
            dataset=dataset,
            experiment=experiment,
            cnn_model=cnn_model,
            relevancy_method=relevancy_method,
            pk=example_index - 1,
        ).pk
    except:
        pass

    next_example = None
    try:
        next_example = Example.objects.get(
            dataset=dataset,
            experiment=experiment,
            cnn_model=cnn_model,
            relevancy_method=relevancy_method,
            pk=example_index + 1,
        ).pk
    except:
        pass

    already_labeled = None
    try:
        already_labeled = choice_dict[
            Label.objects.get(
                example=example,
                user=request.user,
            ).choice
        ]
    except:
        pass

    # return HttpResponse(f"You have chosen {dataset_name} {experiment_name}")
    return render(
        request=request,
        template_name="relevavance/show_example.html",
        context={
            "dataset_name": dataset.name,
            "experiment_name": experiment_name,
            "cnnmodel_name": cnnmodel_name,
            "relevancy_method_name": relevancy_method_name,
            "example_index": example_index,
            "prediction": example.prediction,
            "image": encoded_image,
            "relevance_1_image": relevance_1_image,
            "relevance_1_label": relevance_1_label,
            "relevance_2_image": relevance_2_image,
            "relevance_2_label": relevance_2_label,
            "previous_example": previous_example,
            "next_example": next_example,
            "already_labeled": already_labeled,
            "num_labeled_examples": len(filtered_examples) - len(unlabelled_examples),
            "num_examples": len(filtered_examples),
        },
    )


@login_required
def annotate_example(
    request,
    dataset_name,
    experiment_name,
    cnnmodel_name,
    relevancy_method_name,
    example_index,
    annotation,
):
    c_dataset = Dataset.objects.get(name__iexact=dataset_name)
    c_experiment = Experiment.objects.get(name__iexact=experiment_name)
    c_cnnmodel = CNNModel.objects.get(name__iexact=cnnmodel_name)
    c_relevancymethod = RelevancyMethod.objects.get(name__iexact=relevancy_method_name)
    c_example = Example.objects.get(
        pk=example_index,
        dataset=c_dataset,
        experiment=c_experiment,
        cnn_model=c_cnnmodel,
        relevancy_method=c_relevancymethod,
    )

    try:
        label = Label.objects.get(user=request.user, example=c_example)
        label.choice = annotation
    except Label.DoesNotExist as e:
        label = Label(user=request.user, example=c_example, choice=annotation)
    label.save()

    return redirect(
        f"/relevance/{dataset_name}/{experiment_name}/{cnnmodel_name}/{relevancy_method_name}"
    )
