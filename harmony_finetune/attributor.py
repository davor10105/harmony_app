import torch
import captum
import captum.attr
import torch.nn.functional as F
from zennit.composites import EpsilonGammaBox, ExcitationBackprop, EpsilonPlus, GuidedBackprop
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.torchvision import ResNetCanonizer
from zennit.attribution import Gradient

from utils import *


def normalize_vector(vec):
    vec = vec / (vec.pow(2).sum(-1, keepdim=True).pow(0.5) + 1e-9)
    return vec


def get_attributor(model, attributor_name, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
    attributor_map = {
        #"BCos": BCosAttributor,
        "GradCam": GradCamAttributor,
        "IxG": IxGAttributor,
        "ExcitationBackprop": ExcitationBackpropAttributor,
        "ExcitationBackpropNOCLRP": ExcitationBackpropNOCLRPAttributor,
        "EpsilonPlus": EpsilonPlusAttributor,
        "GuidedBackprop": GuidedBackpropAttributor
    }
    return attributor_map[attributor_name](model, only_positive, binarize, interpolate, interpolate_dims, batch_mode)


class AttributorBase:

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__()
        self.model = model
        self.only_positive = only_positive
        self.binarize = binarize
        self.interpolate = interpolate
        self.interpolate_dims = interpolate_dims
        self.batch_mode = batch_mode

    def __call__(self, feature, output, class_idx=None, img_idx=None, classes=None, create_graph=True):
        if self.batch_mode:
            return self._call_batch_mode(feature, output, classes, create_graph=create_graph)
        return self._call_single(feature, output, class_idx, img_idx, create_graph=create_graph)

    def _call_batch_mode(self, feature, output, classes, create_graph):
        raise NotImplementedError

    def _call_single(self, feature, output, class_idx, img_idx, create_graph):
        raise NotImplementedError

    def check_interpolate(self, attributions):
        if self.interpolate:
            return captum.attr.LayerAttribution.interpolate(
                attributions, interpolate_dims=self.interpolate_dims, interpolate_mode="bilinear")
        return attributions

    def check_binarize(self, attributions):
        if self.binarize:
            attr_max = attributions.abs().amax(dim=(1, 2, 3), keepdim=True)
            attributions = torch.where(
                attr_max == 0, attributions, attributions/attr_max)
        return attributions

    def check_only_positive(self, attributions):
        if self.only_positive:
            return attributions.clamp(min=0)
        return attributions

    def apply_post_processing(self, attributions):
        attributions = self.check_only_positive(attributions)
        attributions = self.check_binarize(attributions)
        attributions = self.check_interpolate(attributions)
        return attributions


class BCosAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes, create_graph):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        with self.model.explanation_mode():
            grads = torch.autograd.grad(torch.unbind(
                target_outputs), feature, create_graph=create_graph, retain_graph=create_graph)[0]
        attributions = (grads*feature).sum(dim=1, keepdim=True)
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx, create_graph):
        with self.model.explanation_mode():
            grads = torch.autograd.grad(
                output[img_idx, class_idx], feature, create_graph=create_graph, retain_graph=create_graph)[0]
        attributions = (grads[img_idx]*feature[img_idx]
                        ).sum(dim=0, keepdim=True).unsqueeze(0)
        return self.apply_post_processing(attributions)


class GradCamAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes, create_graph):
        activations = {}
        def get_save_hook(name):
            def save_hook(model, input, output):
                activations[name] = output
            return save_hook
        
        if self.model.__class__.__name__ == 'VGG':
            handle = self.model.avgpool.register_forward_hook(get_save_hook('avgpool'))
            #handle = self.model.features[-1].register_forward_hook(get_save_hook('avgpool'))
            output = self.model(feature)
            handle.remove()
            feature = activations['avgpool']
        elif self.model.__class__.__name__ == 'ResNet':
            handle = self.model.layer4[1].conv2.register_forward_hook(get_save_hook('avgpool'))
            output = self.model(feature)
            handle.remove()
            feature = activations['avgpool']
        else:
            raise NotImplementedError('Need to specify feature extraction layer for GradCAM')
        
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        grads = torch.autograd.grad(torch.unbind(
            target_outputs), feature, create_graph=create_graph, retain_graph=create_graph)[0]
        grads = grads.mean(dim=(2, 3), keepdim=True)
        prods = grads * feature
        attributions = torch.nn.functional.relu(
            torch.sum(prods, axis=1, keepdim=True))
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx, create_graph):
        grads = torch.autograd.grad(
            output[img_idx, class_idx], feature, create_graph=create_graph, retain_graph=create_graph)[0]
        grads = grads.mean(dim=(2, 3), keepdim=True)
        prods = grads[img_idx] * feature[img_idx]
        attributions = torch.nn.functional.relu(
            torch.sum(prods, axis=0, keepdim=True)).unsqueeze(0)
        return self.apply_post_processing(attributions)


class ExcitationBackpropAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes, create_graph):
        if self.model.__class__.__name__ == 'VGG':
            canonizers = [SequentialMergeBatchNorm()]
        elif self.model.__class__.__name__ == 'ResNet': 
            canonizers = [ResNetCanonizer()]
        else:
            raise NotImplementedError('Model not supported')

        composite = ExcitationBackprop(canonizers=canonizers)
        #target_mask = output * F.one_hot(classes, num_classes=output.shape[-1]).to(feature.device)
        onehot_pred = F.one_hot(classes, num_classes=output.shape[-1]).to(feature.device)
        #softmax_output = output.softmax(-1) * (onehot_pred - (1 - onehot_pred) / 1000)
        #target_mask = torch.autograd.grad(softmax_output.mean(), output, retain_graph=create_graph, create_graph=create_graph)[0]
        #target_mask = 1.001 * (F.one_hot(classes, num_classes=output.shape[-1]).to(feature.device) * (1 + output - output.detach())) - torch.ones_like(output) * (1 + output - output.detach()) / 1000
        target_mask = output * onehot_pred
        with Gradient(model=self.model, composite=composite, create_graph=create_graph, retain_graph=create_graph) as attributor:
            out, r = attributor(feature, target_mask)
        target_mask = output
        with Gradient(model=self.model, composite=composite, create_graph=create_graph, retain_graph=create_graph) as attributor:
            out, r_neg = attributor(feature, target_mask)
        
        r = backprop_normalize(r) - backprop_normalize(r_neg)
        r = r.sum(1, keepdim=True)
        return self.apply_post_processing(r)

class ExcitationBackpropNOCLRPAttributor(ExcitationBackpropAttributor):
    def _call_batch_mode(self, feature, output, classes, create_graph):
        if self.model.__class__.__name__ == 'VGG':
            canonizers = [SequentialMergeBatchNorm()]
        elif self.model.__class__.__name__ == 'ResNet': 
            canonizers = [ResNetCanonizer()]
        else:
            raise NotImplementedError('Model not supported')

        composite = ExcitationBackprop(canonizers=canonizers)
        onehot_pred = F.one_hot(classes, num_classes=output.shape[-1]).to(feature.device)
        target_mask = output * onehot_pred
        with Gradient(model=self.model, composite=composite, create_graph=create_graph, retain_graph=create_graph) as attributor:
            out, r = attributor(feature, target_mask)
        r = r.sum(1, keepdim=True)
        return self.apply_post_processing(r)

class EpsilonPlusAttributor(AttributorBase):
    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes, create_graph):
        if self.model.__class__.__name__ == 'VGG':
            canonizers = [SequentialMergeBatchNorm()]
        elif self.model.__class__.__name__ == 'ResNet': 
            canonizers = [ResNetCanonizer()]
        else:
            raise NotImplementedError('Model not supported')
        composite = EpsilonPlus(canonizers=canonizers)
        target_mask = output * F.one_hot(classes, num_classes=output.shape[-1]).to(feature.device)
        with Gradient(model=self.model, composite=composite, create_graph=create_graph, retain_graph=create_graph) as attributor:
            out, r = attributor(feature, target_mask)
        target_mask = output
        with Gradient(model=self.model, composite=composite, create_graph=create_graph, retain_graph=create_graph) as attributor:
            out, r_neg = attributor(feature, target_mask)
        
        r = backprop_normalize(r) - backprop_normalize(r_neg)
        r = r.sum(1, keepdim=True)
        return self.apply_post_processing(r)

class GuidedBackpropAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes, create_graph):
        if self.model.__class__.__name__ == 'VGG':
            canonizers = [SequentialMergeBatchNorm()]
        elif self.model.__class__.__name__ == 'ResNet': 
            canonizers = [ResNetCanonizer()]
        else:
            raise NotImplementedError('Model not supported')

        composite = GuidedBackprop(canonizers=canonizers)
        target_mask = output * F.one_hot(classes, num_classes=output.shape[-1]).to(feature.device)

        with Gradient(model=self.model, composite=composite, create_graph=create_graph, retain_graph=create_graph) as attributor:
            out, r = attributor(feature, target_mask)
        r = r.sum(1, keepdim=True)
        return self.apply_post_processing(r)


class IxGAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes, create_graph):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        grads = torch.autograd.grad(torch.unbind(
            target_outputs), feature, create_graph=create_graph, retain_graph=create_graph)[0]
        attributions = (grads * feature).sum(dim=1, keepdim=True)
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx, create_graph):
        grads = torch.autograd.grad(
            output[img_idx, class_idx], feature, create_graph=create_graph, retain_graph=create_graph)[0]
        attributions = (grads[img_idx] * feature[img_idx]
                        ).sum(dim=0, keepdim=True).unsqueeze(0)
        return self.apply_post_processing(attributions)