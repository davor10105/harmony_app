import torch
import torch.nn.functional as F
import numpy as np
from quantus.metrics import IROF, ROAD, RelativeInputStability, EffectiveComplexity
from quantus.helpers import utils, asserts
from quantus.functions.perturb_func import perturb_batch

from utils import *


class LearnableMetric():
    def perturb_input(self, **kwargs):
        raise NotImplementedError('input perturbation not implemented')
    def evaluate(self, **kwargs) -> torch.tensor:
        '''
        evaluates a batch of examples
        returns an array of size (n,) where each index is this metric's score
        '''
        raise NotImplementedError('evaluation function not implemented')


class ROADLearnableMetric(ROAD, LearnableMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.better_higher = True
        
        if 'percentages' not in kwargs:
            self.percentages = list(range(5, 95, 9))
    
    def evaluate_instance(self, model, x, y, a):
        ordered_indices = np.argsort(a, axis=None)[::-1]

        results = []
        for p_ix, p in enumerate(self.percentages):
            top_k_indices = ordered_indices[: int(self.a_size * p / 100)]

            x_perturbed = self.perturb_func(  # type: ignore
                arr=x,
                indices=top_k_indices,
            )

            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            class_pred_perturb = np.argmax(model.predict(x_input))
            results.append(int(y == class_pred_perturb))
        
        return (1 - torch.tensor(results)).float().mean().item()
    
    def evaluate_batch(self, model, x_batch , y_batch, a_batch, **kwargs):
        return [
            self.evaluate_instance(model=model, x=x, y=y, a=a)
            for x, y, a in zip(x_batch, y_batch, a_batch)
        ]
    
    def custom_postprocess(self, **kwargs):
        return self.evaluation_scores
    
    
    def mask(self, x, a, percentage):
        ordered_indices = np.argsort(a, axis=None)[::-1]
        results_instance = np.array([None for _ in self.percentages])
        last_percentage = int(percentage * len(self.percentages))

        top_k_indices = ordered_indices[: int(224 * 224 * self.percentages[last_percentage] / 100)]

        x_perturbed = self.perturb_func(  # type: ignore
            arr=x,
            indices=top_k_indices,
        )

        return torch.tensor(x_perturbed).float()

    def perturb_input(self, **kwargs):
        x, a = kwargs['x'], kwargs['a']
        percentages = (torch.rand(x.shape[0]) * 0.9 + 0.05).numpy()
        new_x = []
        for xx, rr, percentage in zip(x, a, percentages):
            try:
                xx = self.mask(xx, rr, percentage)
            except:
                xx = torch.from_numpy(xx)
                print('Error during masking')
            new_x.append(xx)
        return torch.stack(new_x, 0)
    
    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, y, a = kwargs['model'], kwargs['x'], kwargs['y'], kwargs['a']
        device = kwargs['device']
        return torch.tensor(self(model, x, y, a, device=device))


class IROFLearnableMetric(IROF, LearnableMetric):
    def __init__(self, **kwargs):
        kwargs['return_aggregate'] = False
        super().__init__(**kwargs)
        
        self.better_higher = True
    
    def evaluate_instance(self, model, x, y, a):
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(x, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = len(np.unique(segments))
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        preds = []
        x_prev_perturbed = x

        for i_ix, s_ix in enumerate(s_indices):
            # Perturb input by indices of attributions.
            a_ix = np.nonzero((segments == s_ix).flatten())[0]

            x_perturbed = self.perturb_func(
                arr=x_prev_perturbed,
                indices=a_ix,
                indexed_axes=self.a_axes,
            )

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])

            # Normalise the scores to be within range [0, 1].
            preds.append(float(y_pred_perturb / (y_pred + 1e-9)))
            x_prev_perturbed = x_perturbed

        # Calculate the area over the curve (AOC) score.
        return (1 - torch.tensor(preds)).mean().item()
    
    def evaluate_batch(self, model, x_batch, y_batch, a_batch, **kwargs):
        result = [
            self.evaluate_instance(model=model, x=x, y=y, a=a)
            for x, y, a in zip(x_batch, y_batch, a_batch)
        ]
        return result
    
    def custom_postprocess(self, **kwargs):
        return self.evaluation_scores
    
    def mask(self, xx, rr, percentage):
        mean_value = xx.mean()
        segments = utils.get_superpixel_segments(
        img = np.moveaxis(xx, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = len(np.unique(segments))
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(rr[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        preds = []
        x_perturbed = xx
        for i_ix, s_ix in enumerate(s_indices[:int(len(s_indices) * percentage)]):
            # Perturb input by indices of attributions.
            mask = 1 - (segments == s_ix)
            x_perturbed = x_perturbed * mask + mean_value * (1 - mask)
        xx = torch.tensor(x_perturbed).float()
        return xx
    
    @property
    def get_aoc_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return self.evaluation_scores

    def perturb_input(self, **kwargs):
        x, a = kwargs['x'], kwargs['a']
        percentages = (torch.rand(x.shape[0]) * 0.9 + 0.05).numpy()
        new_x = []
        for xx, rr, percentage in zip(x, a, percentages):
            try:
                xx = self.mask(xx, rr, percentage)
            except:
                xx = torch.from_numpy(xx)
                print('Error during masking')
            new_x.append(xx)
        return torch.stack(new_x, 0)
    
    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, y, a = kwargs['model'], kwargs['x'], kwargs['y'], kwargs['a']
        device = kwargs['device']
        result = torch.tensor(self(model, x, y, a, device=device))
        return result


class RelativeInputStabilityLearnableMetric(RelativeInputStability, LearnableMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.nr_samples = 5
        self.better_higher = False
    
    def evaluate_batch(
        self,
        model,
        x_batch,
        y_batch,
        a_batch,
        **kwargs,
    ):
        batch_size = x_batch.shape[0]

        # Prepare output array.
        ris_batch = np.zeros(shape=[self._nr_samples, x_batch.shape[0]])

        for index in range(self._nr_samples):
            # Perturb input.
            x_perturbed = perturb_batch(
                perturb_func=self.perturb_func,
                indices=np.tile(np.arange(0, x_batch[0].size), (batch_size, 1)),
                indexed_axes=np.arange(0, x_batch[0].ndim),
                arr=x_batch,
            )

            # Generate explanations for perturbed input.
            a_batch_perturbed = self.explain_batch(model, torch.tensor(x_perturbed), torch.tensor(y_batch))

            # Compute maximization's objective.
            ris = self.relative_input_stability_objective(
                x_batch, x_perturbed, a_batch, a_batch_perturbed
            )
            ris_batch[index] = ris

        # Compute RIS.
        result = np.max(ris_batch, axis=0)
        if self.return_aggregate:
            result = [self.aggregate_func(result)]

        return result

    def perturb_input(self, **kwargs):
        x = kwargs['x']
        x = torch.from_numpy(x)
        noise = torch.rand_like(x) * 0.18 + 0.02
        
        return x + noise
    
    @staticmethod
    def quantus_wrapper(attribution_method):
        def call_attribution(model, inputs, targets, abs=False, normalise=False, *args, **kwargs):
            device = kwargs['device']
            inputs, targets = torch.tensor(inputs).to(device), torch.tensor(targets).to(device)
            o = model(inputs)
            r = attribution_method(inputs, o, classes=targets)
            r = r.detach().cpu().numpy()
            return r
        return call_attribution
    
    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, y, explain_func = kwargs['model'], kwargs['x'], kwargs['y'], kwargs['explain_func']
        device = kwargs['device']
        result = torch.tensor(self(model, x, y, explain_func=RelativeInputStabilityLearnableMetric.quantus_wrapper(explain_func), device=device))
        return result


class FocusLearnableMetric(LearnableMetric):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.better_higher = True
    
    def __call__(self, model, x, y, attributor, device='cuda:1'):
        b, c, h, w = x.shape
        x_small = F.interpolate(x, (h // 2, w // 2), mode='bilinear')
        bm = b // 4
        x1, x2, x3, x4 = x_small[:bm], x_small[bm: 2*bm], x_small[2*bm: 3*bm], x_small[3*bm:]
        
        x_mosaic = torch.zeros(b // 4, c, h, w)
        x_mosaic[:, :, :h // 2, :w // 2] = x1
        x_mosaic[:, :, :h // 2, w // 2:] = x2
        x_mosaic[:, :, h // 2:, :w // 2] = x3
        x_mosaic[:, :, h // 2:, w // 2:] = x4
        x_mosaic = x_mosaic.to(device)

        with torch.no_grad():
            preds = model(x)
            mosaic_preds = model(x_mosaic) 

        #chosen_indices = torch.zeros(bm).int()#torch.randint(high=4, size=(bm, ))
        chosen_ys = mosaic_preds.max(-1)[1]#preds[chosen_indices + torch.arange(bm) * 4].max(-1)[1]

        p1, p2, p3, p4 = preds[:bm], preds[bm: 2*bm], preds[2*bm: 3*bm], preds[3*bm:]  # bm
        p1, p2, p3, p4 = p1.softmax(-1)[torch.arange(bm), chosen_ys], p2.softmax(-1)[torch.arange(bm), chosen_ys], p3.softmax(-1)[torch.arange(bm), chosen_ys], p4.softmax(-1)[torch.arange(bm), chosen_ys]
        chosen_indices = torch.stack([p1, p2, p3, p4], 1).max(-1)[1]  # bm 4
        #print(chosen_indices)

        x_mosaic.requires_grad = True
        o = model(x_mosaic)
        r = attributor(x_mosaic, o, classes=chosen_ys)
        
        score1, score2, score3, score_4 = r[:, :, :h//2, :w//2].flatten(1).sum(-1), r[:, :, :h//2:, w//2:].flatten(1).sum(-1), r[:, :, h//2:, :w//2].flatten(1).sum(-1), r[:, :, h//2:, w//2:].flatten(1).sum(-1)
        score = torch.stack([score1, score2, score3, score_4], 1)  # b 4

        positive_score = score[torch.arange(score.shape[0]).unsqueeze(0), chosen_indices][0]
        other_score = score.sum(-1)
        focus_score = positive_score / (other_score + 1e-9)

        return focus_score

    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, y, attributor = kwargs['model'], kwargs['x'], kwargs['y'], kwargs['explain_func']
        device = kwargs['device']
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        result = self(model, x, y, attributor=attributor, device=device)
        return result


class IoULearnableMetric(LearnableMetric):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.better_higher = True
    
    def __call__(self, model, x, m, y, attributor, device='cuda:1'):
        x.requires_grad = True
        o = model(x)
        r = attributor(x, o, classes=y).squeeze(1)
        
        score = ((r * m).flatten(1).sum(-1) / (r.flatten(1) + 1e-9).sum(-1))

        return score

    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, m, y, attributor = kwargs['model'], kwargs['x'], kwargs['m'], kwargs['y'], kwargs['explain_func']
        device = kwargs['device']
        x, m, y = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device), torch.from_numpy(y).to(device)
        result = self(model, x, m, y, attributor=attributor, device=device)
        return result

class CombinedLearnableMetric(LearnableMetric):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.better_higher = True
    
    def __call__(self, model, x, m, y, attributor, device='cuda:1'):
        x.requires_grad = True
        o = model(x)
        r = attributor(x, o, classes=y).squeeze(1)
        
        score = ((r * m).flatten(1).sum(-1) / (r.flatten(1) + 1e-9).sum(-1))

        return score

    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, m, y, attributor = kwargs['model'], kwargs['x'], kwargs['m'], kwargs['y'], kwargs['explain_func']
        device = kwargs['device']
        x, m, y = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device), torch.from_numpy(y).to(device)
        result = self(model, x, m, y, attributor=attributor, device=device)
        return result

class ComplexityLearnableMetric(LearnableMetric):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.better_higher = False
    
    def __call__(self, model, x, y, attributor, device='cuda:1'):
        x.requires_grad = True
        o = model(x)
        r = attributor(x, o, classes=y).squeeze(1)
        score = (r.flatten(1).abs() > 1e-5).float().mean(-1)
        #r = (r.flatten(1).abs() + 1e-12)
        #p = r / r.sum(-1, keepdim=True)
        
        #score = (-p * torch.log(p)).sum(-1)

        return score

    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, y, attributor = kwargs['model'], kwargs['x'], kwargs['y'], kwargs['explain_func']
        device = kwargs['device']
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        result = self(model, x, y, attributor=attributor, device=device)
        return result


class ISLearnableMetric(LearnableMetric):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.better_higher = False
        self.nr_samples = 20
        
    def perturb_input(self, **kwargs):
        x = kwargs['x']
        x = torch.from_numpy(x)
        noise = torch.randn_like(x) * 0.05
        
        return x + noise
    
    def __call__(self, model, x, y, attributor, device='cuda:1'):
        batch_size = x.shape[0]
        x.requires_grad = True
        o = model(x)
        preds = o.max(-1)[1]
        r = attributor(x, o, classes=preds)
        
        max_differences = []
        for xx, pred, rr in zip(x, preds, r):
            xx = xx.unsqueeze(0).repeat(self.nr_samples, 1, 1, 1)
            xx_perturbed = self.perturb_input(x=xx.detach().cpu().numpy()).to(device)
            permutation = torch.arange(self.nr_samples)
            differences = []
            for i in range(0, self.nr_samples, batch_size):
                indices = permutation[i: i + batch_size]
                xxx = xx[indices]
                xxx_perturbed = xx_perturbed[indices]
                o_perturbed = model(xxx_perturbed)
                r_perturbed = attributor(xxx_perturbed, o_perturbed, classes=pred.repeat(len(indices)), create_graph=False)

                difference = ((backprop_normalize_to_one(r_perturbed).flatten(1) - backprop_normalize_to_one(rr).flatten(1)) / (backprop_normalize_to_one(rr).flatten(1) + 1e-9)).abs().sum(-1) / (((xxx_perturbed.flatten(1) - xxx.flatten(1)) / (xxx.flatten(1) + 1e-9)).abs().sum(-1) + 1e-9)
                differences.append(difference)
            differences = torch.cat(differences)
            max_difference = differences.max()
            max_differences.append(max_difference)
        max_differences = torch.tensor(max_differences)

        return max_differences

    def evaluate(self, **kwargs) -> torch.tensor:
        model, x, y, attributor = kwargs['model'], kwargs['x'], kwargs['y'], kwargs['explain_func']
        device = kwargs['device']
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        result = self(model, x, y, attributor=attributor, device=device)
        return result