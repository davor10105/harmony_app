import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as T

from attributor import get_attributor
from metric import *
from attributor import *
from dataset import SegmentationDataset, ClassicImageNet, ClassicImageNetValidation
from trainer import *


segment_train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

segment_test_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transform = T.Compose(
    [
        T.RandomResizedCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ]
)

test_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ]
)


def train_segmentation(model_name: str, attributor_name: str, metric: LearnableMetric, trainer_class: LearnableMetricTrainer):
    batch_size = 8
    device = 'cuda:1'

    if model_name == 'vgg':
        model_orig = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model_learn = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet':
        model_orig = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_learn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    optimizer = torch.optim.Adam(model_learn.parameters(), lr=3e-6, weight_decay=0.)

    attributor_orig = get_attributor(model_orig, attributor_name, True, False, True, (224, 224), batch_mode=True)
    attributor_learn = get_attributor(model_learn, attributor_name, True, False, True, (224, 224), batch_mode=True)

    train_data = SegmentationDataset('/storage-ssd/IMAGENET1K/ImageNet-S/datapreparation/ImageNetS919/train-semi-segmentation', '/storage-ssd/IMAGENET1K/ILSVRC/Data/CLS-LOC/train', segment_train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_test_data = SegmentationDataset('/storage-ssd/IMAGENET1K/ImageNet-S/datapreparation/ImageNetS919/validation-segmentation', '/storage-ssd/IMAGENET1K/ILSVRC/Data/CLS-LOC/val', segment_test_transform)
    generator = torch.Generator().manual_seed(69)
    validation_data, test_data = torch.utils.data.random_split(validation_test_data, [0.5, 0.5], generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    trainer = trainer_class(
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


    original_scores, _ = trainer.evaluate('orig', 5)
    original_score = torch.tensor(original_scores).mean()
    print(f'original score: {original_score}')
    original_scores, _ = trainer.evaluate('learn', 5)
    original_score = torch.tensor(original_scores).mean()
    print(f'learn score: {original_score}')
    trainer.train(epochs=100, eval_every=20, validation_iterations=5)
    
    return model_learn

def train(model_name: str, attributor_name: str, metric: LearnableMetric, trainer_class: LearnableMetricTrainer):
    batch_size = 8
    device = 'cuda:1'

    dataset_path = '/storage-ssd/IMAGENET1K/ILSVRC'

    if model_name == 'vgg':
        model_orig = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model_learn = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet':
        model_orig = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_learn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    optimizer = torch.optim.Adam(model_learn.parameters(), lr=3e-5, weight_decay=0.)

    attributor_orig = get_attributor(model_orig, attributor_name, True, False, True, (224, 224), batch_mode=True)
    attributor_learn = get_attributor(model_learn, attributor_name, True, False, True, (224, 224), batch_mode=True)

    train_data = ClassicImageNet(dataset_path, 'train', transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_test_data = ClassicImageNetValidation(dataset_path, 'val', transform=test_transform)
    generator = torch.Generator().manual_seed(69)
    validation_data, test_data = torch.utils.data.random_split(validation_test_data, [0.5, 0.5], generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    trainer = trainer_class(
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
    
    trainer.train(iterations=20000, eval_every=50, validation_iterations=10)
    
    return model_learn