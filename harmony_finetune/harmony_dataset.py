import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as T


from metric import *
from trainer import *
from dataset import *
from attributor import get_attributor


dataset = [
    "ImageNet",
]

cnnmodel = [
    "VGG",
    "ResNet",
]

relevancy_method = [
    "GradCamAttributor",
    "GuidedBackpropAttributor",
]

experiment = [
    "ROADLearnableMetric",
    "IROFLearnableMetric",
    "ISLearnableMetric",
    "FocusLearnableMetric",
    "IoULearnableMetric",
]


normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = T.Compose(
    [
        T.RandomResizedCrop((224, 224), antialias=True),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ]
)

test_transform = T.Compose(
    [T.Resize((256, 256), antialias=True), T.CenterCrop(224), T.ToTensor(), normalize]
)


segment_train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop((224, 224), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

segment_test_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((256, 256), antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def create_simple_fixture(simple_list, name):
    fixture_dir = "harmony_django_fixtures"

    simple_elements = []
    for i, element in enumerate(simple_list):
        simple_element = {
            "pk": i + 1,
            "model": "relevavance." + name,
            "fields": {"name": element},
        }
        simple_elements.append(simple_element)
    json_str = json.dumps(simple_elements)

    with open(os.path.join(fixture_dir, name + ".json"), "w") as f:
        f.write(json_str)


def normalize_rel(r):
    r = r / (np.amax(r) + 1e-9)
    r = np.clip(r, 0, 1)
    return r


def renormalize_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    return x * std + mean


def pt_normalize_r(r):
    r_shape = r.shape
    r = r.flatten(1)
    r = r / (r.abs().max(-1, keepdim=True)[0] + 1e-9)
    r = r.view(*r_shape)
    return r


def r_to_image(r):
    r = r.unsqueeze(1).repeat(1, 3, 1, 1)
    r_white, r_red = torch.ones_like(r), torch.ones_like(r)
    r_red[:, 1:] = 0.0
    r = (1 - r) * r_white + r * r_red
    return r


def pt_renormalize_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return x * std + mean


def create_harmony_dataset():
    fixture_dir = "harmony_django_fixtures"
    for simple_list, name in zip(
        [dataset, cnnmodel, relevancy_method, experiment],
        ["dataset", "cnnmodel", "relevancymethod", "experiment"],
    ):
        create_simple_fixture(simple_list, name)

    num_examples = 200
    save_fixtures = True
    device = "cuda:1"
    batch_size = 8

    optimizer = None

    defined_experiments = [
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": FocusLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/ResNetFocusLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": FocusLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IoULearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/ResNetIoULearnableMetricGuidedBackpropAttributor.pth",
            "trainer": IoULearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ISLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/ResNetISLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": AttributionSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ROADLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/ResNetROADLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IROFLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/ResNetIROFLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
        ## VGG
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": FocusLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/VGGFocusLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": FocusLearnableMetricTrainer,
        },
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IoULearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/VGGIoULearnableMetricTrainerGuidedBackpropAttributor.pth",
            "trainer": IoULearnableMetricTrainer,
        },
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ISLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/VGGISLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": AttributionSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ROADLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/VGGROADLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IROFLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GuidedBackpropAttributor",
            "learn_model_path": "learnable_metric_models/VGGIROFLearnableMetricGuidedBackpropAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
        ##### GradCam## ResNet
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": FocusLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/ResNetFocusLearnableMetricGradCamAttributor.pth",
            "trainer": FocusLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IoULearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/ResNetIoULearnableMetricGradCamAttributor.pth",
            "trainer": IoULearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ISLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/ResNetISLearnableMetricGradCamAttributor.pth",
            "trainer": AttributionSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ROADLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/ResNetROADLearnableMetricGradCamAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IROFLearnableMetric,
            "chosen_model": "ResNet",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/ResNetIROFLearnableMetricGradCamAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
        ## VGG
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": FocusLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/VGGFocusLearnableMetricGradCamAttributor.pth",
            "trainer": FocusLearnableMetricTrainer,
        },
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IoULearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/VGGIoULearnableMetricTrainerGradCamAttributor.pth",
            "trainer": IoULearnableMetricTrainer,
        },
        {
            # DOBAR
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ISLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/VGGISLearnableMetricGradCamAttributor.pth",
            "trainer": AttributionSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": ROADLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/VGGROADLearnableMetricGradCamAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
        {
            "tested": True,
            "chosen_dataset": "ImageNet",
            "chosen_experiment": IROFLearnableMetric,
            "chosen_model": "VGG",
            "chosen_relevancy_method": "GradCamAttributor",
            "learn_model_path": "learnable_metric_models/VGGIROFLearnableMetricGradCamAttributor.pth",
            "trainer": MaskedSimilarityLearnableMetricTrainer,
        },
    ]
    dataset_path = "/storage-ssd/IMAGENET1K/ILSVRC"

    example_objects = []
    pk_counter = 0
    # pbar = tqdm(defined_experiments)
    validation_test_data = ClassicImageNetValidation(
        dataset_path, "val", transform=test_transform
    )
    generator = torch.Generator().manual_seed(69)
    _, save_test_data = torch.utils.data.random_split(
        validation_test_data, [0.5, 0.5], generator=generator
    )
    current_index = 0

    all_original_scores = []
    all_learned_scores = []
    # for defined_experiment in pbar:
    for defined_experiment in defined_experiments:
        chosen_dataset = defined_experiment["chosen_dataset"]
        chosen_experiment = defined_experiment["chosen_experiment"].__name__
        chosen_model = defined_experiment["chosen_model"]
        chosen_relevancy_method = defined_experiment["chosen_relevancy_method"]
        attributor_name = chosen_relevancy_method[:-10]
        chosen_relevancy_method = chosen_relevancy_method.replace("NOCLRP", "")
        metric = defined_experiment["chosen_experiment"](disable_warnings=True)

        dir_structure = f"learnable_metric_image_data_fast/{chosen_dataset}/{chosen_experiment}/{chosen_model}/{chosen_relevancy_method}"
        if not os.path.isdir(dir_structure):
            os.makedirs(dir_structure, exist_ok=True)

        if chosen_model == "VGG":
            model_orig = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model_learn = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif chosen_model == "ResNet":
            model_orig = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model_learn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        model_learn.load_state_dict(torch.load(defined_experiment["learn_model_path"]))

        model_orig.to(device)
        model_learn.to(device)

        attributor_orig = get_attributor(
            model_orig, attributor_name, True, False, True, (224, 224), batch_mode=True
        )
        attributor_learn = get_attributor(
            model_learn, attributor_name, True, False, True, (224, 224), batch_mode=True
        )

        if "IoU" in chosen_experiment:
            train_data = SegmentationDataset(
                "/storage-ssd/IMAGENET1K/ImageNet-S/datapreparation/ImageNetS919/train-semi-segmentation",
                "/storage-ssd/IMAGENET1K/ILSVRC/Data/CLS-LOC/train",
                segment_train_transform,
            )
            validation_test_data = SegmentationDataset(
                "/storage-ssd/IMAGENET1K/ImageNet-S/datapreparation/ImageNetS919/validation-segmentation",
                "/storage-ssd/IMAGENET1K/ILSVRC/Data/CLS-LOC/val",
                segment_test_transform,
            )
            generator = torch.Generator().manual_seed(69)
            validation_data, test_data = torch.utils.data.random_split(
                validation_test_data, [0.5, 0.5], generator=generator
            )
        else:
            train_data = ClassicImageNet(
                dataset_path, "train", transform=train_transform
            )
            validation_test_data = ClassicImageNetValidation(
                dataset_path, "val", transform=test_transform
            )
            generator = torch.Generator().manual_seed(69)
            validation_data, test_data = torch.utils.data.random_split(
                validation_test_data, [0.5, 0.5], generator=generator
            )

        evaluation_batch_size = (
            batch_size * 4 if "Focus" in chosen_experiment else batch_size
        )
        train_loader = DataLoader(
            train_data, batch_size=evaluation_batch_size, shuffle=True
        )
        validation_loader = DataLoader(
            validation_data, batch_size=evaluation_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=evaluation_batch_size, shuffle=False
        )

        trainer = defined_experiment["trainer"](
            model_orig,
            model_learn,
            attributor_orig,
            attributor_learn,
            metric,
            optimizer,
            train_loader,
            validation_loader,
            test_loader,
            device=device,
        )

        if not defined_experiment["tested"]:
            print(defined_experiment["learn_model_path"])
            original_scores, _ = trainer.evaluate("orig", 10)
            original_score = torch.tensor(original_scores).mean()
            print(f"original score: {original_score}")
            learn_scores, learn_accs = trainer.evaluate("learn", 10)
            learn_score, learn_acc = (
                torch.tensor(learn_scores).mean(),
                torch.tensor(learn_accs).mean(),
            )
            print(f"learn score: {learn_score} learn acc: {learn_acc}")
            print(ttest_ind(np.array(original_scores), np.array(learn_scores)))
            all_original_scores.append(np.array(original_scores))
            all_learned_scores.append(np.array(learn_scores))

        accs = []
        masks = []
        for t_index, (x, y) in enumerate(DataLoader(save_test_data, batch_size=8)):
            x, y = x.to(device), y.to(device)

            o_orig = model_orig(x).max(-1)[1]
            o_learn = model_learn(x).max(-1)[1]

            mask = (o_orig == y).float()

            acc = (o_orig == o_learn).float()
            accs.append(acc)
            masks.append(mask)
            if t_index == 15:
                break
        accs = torch.cat(accs)
        masks = torch.cat(masks)
        accs = accs[masks == 1]

        print(f'{accs.mean()} {defined_experiment["learn_model_path"]}')

        save_test_loader = DataLoader(
            torch.utils.data.Subset(
                save_test_data, list(range(current_index, len(save_test_data)))
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        experiment_example_counter = 0
        for miu_counter, (x, y) in enumerate(save_test_loader):
            x, y = x.to(device), y.to(device)
            current_index += len(x)

            o_orig = model_orig(x)
            r_orig = attributor_orig(x, o_orig, classes=y).sum(1).detach().cpu()
            o_learn = model_learn(x)
            r_learn = attributor_learn(x, o_learn, classes=y).sum(1).detach().cpu()

            mask = (o_orig.max(-1)[1] == y).float() * (o_learn.max(-1)[1] == y).float()
            mask = mask.detach().cpu()

            imaged_image = pt_renormalize_image(x.detach().cpu().permute(0, 2, 3, 1))
            imaged_r_orig = r_to_image(pt_normalize_r(r_orig)).permute(0, 2, 3, 1)
            imaged_r_learn = r_to_image(pt_normalize_r(r_learn)).permute(0, 2, 3, 1)

            filtered_imaged_y = y.detach().cpu()[mask == 1]
            filtered_imaged_image = imaged_image[mask == 1]
            filtered_imaged_r_orig = imaged_r_orig[mask == 1]
            filtered_imaged_r_learn = imaged_r_learn[mask == 1]

            if len(filtered_imaged_r_orig) > 0:
                for filtered_image_index in range(len(filtered_imaged_r_orig)):
                    image_image, image_r_orig, image_r_learn = (
                        filtered_imaged_image[filtered_image_index],
                        filtered_imaged_r_orig[filtered_image_index],
                        filtered_imaged_r_learn[filtered_image_index],
                    )
                    image_y = filtered_imaged_y[filtered_image_index]
                    prediction = imagenet_classes[image_y]

                    pil_image_image = Image.fromarray(
                        np.uint8((image_image.numpy() * 255))
                    )
                    pil_image_orig = Image.fromarray(
                        np.uint8((image_r_orig.numpy() * 255))
                    )
                    pil_image_learn = Image.fromarray(
                        np.uint8((image_r_learn.numpy() * 255))
                    )

                    experiment_example_counter += 1
                    pk_counter += 1

                    image_path = os.path.join(dir_structure, f"{pk_counter}_image.png")
                    original_path = os.path.join(
                        dir_structure, f"{pk_counter}_original.png"
                    )
                    learn_path = os.path.join(dir_structure, f"{pk_counter}_learn.png")

                    pil_image_image.save(image_path)
                    pil_image_orig.save(original_path)
                    pil_image_learn.save(learn_path)

                    prediction = imagenet_classes[image_y]

                    current_object = {
                        "pk": pk_counter,
                        "model": "relevavance.example",
                        "fields": {
                            "dataset": dataset.index(chosen_dataset) + 1,
                            "experiment": experiment.index(chosen_experiment) + 1,
                            "cnn_model": cnnmodel.index(chosen_model) + 1,
                            "relevancy_method": relevancy_method.index(
                                chosen_relevancy_method
                            )
                            + 1,
                            "prediction": prediction,
                            "image": image_path,
                            "original": original_path,
                            "learn": learn_path,
                        },
                    }

                    example_objects.append(current_object)

                    if experiment_example_counter == num_examples:
                        break

            if experiment_example_counter == num_examples:
                break

        if save_fixtures:
            json_str = json.dumps(example_objects)
            with open(os.path.join(fixture_dir, "example.json"), "w") as f:
                f.write(json_str)
