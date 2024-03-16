import torch
from tqdm import tqdm
from utils import *
import gc

class LearnableMetricTrainer():
    def __init__(self,
                 model_orig,
                 model_learn,
                 attribution_method_orig,
                 attribution_method_learn,
                 metric,
                 optimizer,
                 train_loader,
                 validation_loader,
                 test_loader, 
                 device='cuda:1',
                 better_higher=True
                ):
        
        self.model_orig = model_orig
        self.model_learn = model_learn
        self.attribution_method_orig = attribution_method_orig
        self.attribution_method_learn = attribution_method_learn
        self.metric = metric
        
        self.model_orig.eval()
        self.model_learn.eval()
        
        self.model_orig.to(device)
        self.model_learn.to(device)
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.better_higher = better_higher
    
    def train(self, iterations):
        raise NotImplementedError('specific train function not implemented')


class MaskedSimilarityLearnableMetricTrainer(LearnableMetricTrainer):
    def train(self, iterations=10000, validation_iterations=2, eval_every=50):
        original_scores, _ = self.evaluate('orig', validation_iterations)
        original_score = torch.tensor(original_scores).mean()
        print(f'original score: {original_score}')
        
        current_score = 0 if self.metric.better_higher else 999
        current_acc = 0
        current_best_score = 0 if self.metric.better_higher else 999
        orig_learn_acc = 0
        pbar = tqdm(self.train_loader)
        for i, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                preds = self.model_orig(x).max(-1)[1]

            x.requires_grad = True
            o = self.model_learn(x)
            r_learn = self.attribution_method_learn(
                x,
                o,
                classes=preds
            )
            x_masked = self.metric.perturb_input(
                x=x.detach().cpu().numpy(),
                a=r_learn.detach().cpu().numpy()
            )
            x_masked = x_masked.to(self.device)
            
            x = x.detach()
            o_orig = self.model_orig(x)
            o_student = self.model_learn(x)
            o_student_masked = self.model_learn(x_masked)
            
            student_pred = o_student.softmax(-1)[torch.arange(o_student.shape[0]).unsqueeze(0), preds]
            student_pred_masked = o_student_masked.softmax(-1)[torch.arange(o_student_masked.shape[0]).unsqueeze(0), preds]

            loss_orig = (1 - ((normalize_vector(o_orig) * normalize_vector(o_student)).sum(-1) + 1) / 2).mean()#nn.CrossEntropyLoss()(o_student, y)
            loss_masked = (student_pred_masked / (student_pred + 1e-9)).mean()
            loss = loss_orig + 0.1 * current_acc * loss_masked
            
            loss.backward()
            self.optimizer.step()
            
            current_acc = (o_student.max(-1)[1] == o_orig.max(-1)[1]).float().mean()
            
            pbar.set_description(f'train loss: {loss.item()} current score: {current_score} current best score: {current_best_score} current acc: {current_acc} orig learn acc: {orig_learn_acc}')
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i > 0 and i % eval_every == 0:
                current_scores, orig_learn_accs = self.evaluate('learn', validation_iterations)
                current_score, orig_learn_acc = torch.tensor(current_scores).mean(), orig_learn_accs.mean()
                
                if ((self.metric.better_higher and current_score > current_best_score) or (not self.metric.better_higher and current_score < current_best_score)) and orig_learn_acc > 0.95:
                    current_best_score = current_score
                    torch.save(self.model_learn.state_dict(), f'learnable_metric_models/{self.model_learn.__class__.__name__}{self.metric.__class__.__name__}{self.attribution_method_learn.__class__.__name__}.pth')

            
            if i == iterations:
                break
    
    def evaluate(self, model_name, iterations=2):
        scores = []
        preds = []
        preds_other = []
        model = self.model_orig if model_name == 'orig' else self.model_learn
        model_other = self.model_learn if model_name == 'orig' else self.model_orig
        attribution_method = self.attribution_method_orig if model_name == 'orig' else self.attribution_method_learn
        for i, (x, y) in enumerate(self.validation_loader):
            x, y = x.to(self.device), y.to(self.device)
            x.requires_grad = True
            
            o = model(x)
            r = attribution_method(
                x,
                o,
                classes=o.max(-1)[1],
                create_graph=False,
            )
            score = self.metric.evaluate(
                model=model,
                x=x.detach().cpu().numpy(),
                y=y.detach().cpu().numpy(),
                a=r.detach().cpu().numpy(),
                explain_func=attribution_method,
                device=self.device
            )
            
            with torch.no_grad():
                o_other = model_other(x)
            
            preds.append(o.max(-1)[1].detach().cpu())
            preds_other.append(o_other.max(-1)[1].detach().cpu())
            
            scores += score
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i == iterations:
                break
        preds = torch.cat(preds)
        preds_other = torch.cat(preds_other)
        accs = (preds == preds_other).float()

        return scores, accs


class AttributionSimilarityLearnableMetricTrainer(MaskedSimilarityLearnableMetricTrainer):
    def train(self, iterations=10000, validation_iterations=2, eval_every=50):
        original_scores, _ = self.evaluate('orig', validation_iterations)
        original_score = torch.tensor(original_scores).mean()
        print(f'original score: {original_score}')
        
        current_score = 0 if self.metric.better_higher else 1e9
        current_acc = 0
        current_best_score = 0 if self.metric.better_higher else 1e9
        orig_learn_acc = 0
        pbar = tqdm(self.train_loader)
        for i, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                preds = self.model_orig(x).max(-1)[1]

            x.requires_grad = True
            o = self.model_learn(x)
            r_learn = self.attribution_method_learn(
                x,
                o,
                classes=preds,
                create_graph=False,
            )
            x_masked = self.metric.perturb_input(
                x=x.detach().cpu().numpy(),
            )
            x_masked = x_masked.to(self.device)
            x_masked.requires_grad = True
            o = self.model_learn(x_masked)
            r_learn_masked = self.attribution_method_learn(
                x_masked,
                o,
                classes=preds,
                create_graph=True,
            )
            
            x = x.detach()
            o_orig = self.model_orig(x)
            o_student = self.model_learn(x)
            
            loss_orig = (1 - ((normalize_vector(o_orig) * normalize_vector(o_student)).sum(-1) + 1) / 2).mean()#nn.CrossEntropyLoss()(o_student, y)
            #loss_orig = torch.nn.KLDivLoss()(o_student.softmax(-1).log(), o_orig.softmax(-1))
            
            #loss_masked = (backprop_normalize(r_learn).detach() - backprop_normalize(r_learn_masked)).abs().flatten(1).sum(-1).mean()
            upper = ((backprop_normalize(r_learn).detach() - backprop_normalize(r_learn_masked)) / (backprop_normalize(r_learn).detach() + 1e-9)).abs().flatten(1).sum(-1)
            lower = ((x - x_masked) / (x + 1e-9)).abs().flatten(1).sum(-1)
            loss_masked = (upper / lower).mean()
            #loss = loss_orig + 0.05 * 1e6 * current_acc * loss_masked
            #loss = loss_orig - 5e-6 * loss_masked
            loss = loss_orig + 5e-6 * loss_masked
            
            loss.backward()
            self.optimizer.step()
            
            current_acc = (o_student.max(-1)[1] == o_orig.max(-1)[1]).float().mean()
            
            pbar.set_description(f'train loss: {loss.item()} current score: {current_score} current best score: {current_best_score} current acc: {current_acc} orig learn acc: {orig_learn_acc}')
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i > 0 and i % eval_every == 0:
                current_scores, orig_learn_accs = self.evaluate('learn', validation_iterations)
                current_score, orig_learn_acc = torch.tensor(current_scores).mean(), orig_learn_accs.mean()
                
                if ((self.metric.better_higher and current_score > current_best_score) or (not self.metric.better_higher and current_score < current_best_score)) and orig_learn_acc > 0.95:
                    current_best_score = current_score
                    torch.save(self.model_learn.state_dict(), f'learnable_metric_models/{self.model_learn.__class__.__name__}{self.metric.__class__.__name__}{self.attribution_method_learn.__class__.__name__}.pth')


            if i == iterations:
                break


class FocusLearnableMetricTrainer(MaskedSimilarityLearnableMetricTrainer):
    def train(self, iterations=10000, validation_iterations=2, eval_every=50):
        original_scores, _ = self.evaluate('orig', validation_iterations)
        original_score = torch.tensor(original_scores).mean()
        print(f'original score: {original_score}')
        
        current_score = 0 if self.metric.better_higher else 999
        current_acc = 0
        current_best_score = 0 if self.metric.better_higher else 999
        orig_learn_acc = 0
        pbar = tqdm(self.train_loader)
        for i, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                preds = self.model_orig(x).max(-1)[1]

            o_orig = self.model_orig(x)
            o_student = self.model_learn(x)

            loss_orig = (1 - ((normalize_vector(o_orig) * normalize_vector(o_student)).sum(-1) + 1) / 2).mean()
            loss_masked = (1 - self.metric(self.model_learn, x, preds, self.attribution_method_learn)).mean()

            loss = loss_orig + 0.01 * loss_masked
            
            loss.backward()
            self.optimizer.step()
            
            current_acc = (o_student.max(-1)[1] == o_orig.max(-1)[1]).float().mean()
            
            pbar.set_description(f'train loss: {loss.item()} current score: {current_score} current best score: {current_best_score} current acc: {current_acc} orig learn acc: {orig_learn_acc}')
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i > 0 and i % eval_every == 0:
                current_scores, orig_learn_accs = self.evaluate('learn', validation_iterations)
                current_score, orig_learn_acc = torch.tensor(current_scores).mean(), orig_learn_accs.mean()
                
                if ((self.metric.better_higher and current_score > current_best_score) or (not self.metric.better_higher and current_score < current_best_score)) and orig_learn_acc > 0.95:
                    current_best_score = current_score
                    torch.save(self.model_learn.state_dict(), f'learnable_metric_models/{self.model_learn.__class__.__name__}{self.metric.__class__.__name__}{self.attribution_method_learn.__class__.__name__}.pth')

            
            if i == iterations:
                break
    
    def evaluate(self, model_name, iterations=2):
        scores = []
        preds = []
        preds_other = []
        model = self.model_orig if model_name == 'orig' else self.model_learn
        model_other = self.model_learn if model_name == 'orig' else self.model_orig
        attribution_method = self.attribution_method_orig if model_name == 'orig' else self.attribution_method_learn
        for i, (x, y) in enumerate(self.validation_loader):
            x, y = x.to(self.device), y.to(self.device)
            x.requires_grad = True
            
            score = self.metric.evaluate(
                model=model,
                x=x.detach().cpu().numpy(),
                y=y.detach().cpu().numpy(),
                explain_func=attribution_method,
                device=self.device
            ).detach().cpu()
            
            with torch.no_grad():
                o = model(x)
            with torch.no_grad():
                o_other = model_other(x)
            
            preds.append(o.max(-1)[1].detach().cpu())
            preds_other.append(o_other.max(-1)[1].detach().cpu())
            
            scores += score
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i == iterations:
                break
        preds = torch.cat(preds)
        preds_other = torch.cat(preds_other)
        accs = (preds == preds_other).float()

        return scores, accs


class IoULearnableMetricTrainer(FocusLearnableMetricTrainer):
    def train(self, epochs=100, validation_iterations=2, eval_every=50):
        original_scores, _ = self.evaluate('orig', validation_iterations)
        original_score = torch.tensor(original_scores).mean()
        print(f'original score: {original_score}')
        
        current_score = 0 if self.metric.better_higher else 999
        current_acc = 0
        current_best_score = 0 if self.metric.better_higher else 999
        orig_learn_acc = 0
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            for i, (x, m, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, m, y = x.to(self.device), m.to(self.device), y.to(self.device)

                with torch.no_grad():
                    preds = self.model_orig(x).max(-1)[1]

                o_orig = self.model_orig(x)
                o_student = self.model_learn(x)

                loss_orig = (1 - ((normalize_vector(o_orig) * normalize_vector(o_student)).sum(-1) + 1) / 2).mean()
                loss_masked = (1 - self.metric(self.model_learn, x, m, preds, self.attributor_learn)).mean()

                loss = loss_orig + 0.05 * loss_masked

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_learn.parameters(), 1.)
                self.optimizer.step()

                current_acc = (o_student.max(-1)[1] == o_orig.max(-1)[1]).float().mean()

                pbar.set_description(f'train loss: {loss.item()} current score: {current_score} current best score: {current_best_score} current acc: {current_acc} orig learn acc: {orig_learn_acc}')

                gc.collect()
                torch.cuda.empty_cache()
                
                if i > 0 and i % eval_every == 0:
                    current_scores, orig_learn_accs = self.evaluate('learn', validation_iterations)
                    current_score, orig_learn_acc = torch.tensor(current_scores).mean(), orig_learn_accs.mean()

                    if ((self.metric.better_higher and current_score > current_best_score) or (not self.metric.better_higher and current_score < current_best_score)) and orig_learn_acc > 0.95:
                        current_best_score = current_score
                        torch.save(self.model_learn.state_dict(), f'learnable_metric_models/{self.model_learn.__class__.__name__}{self.metric.__class__.__name__}{self.attribution_method_learn.__class__.__name__}.pth')
    
    def evaluate(self, model_name, iterations=2):
        scores = []
        preds = []
        preds_other = []
        model = self.model_orig if model_name == 'orig' else self.model_learn
        model_other = self.model_learn if model_name == 'orig' else self.model_orig
        attribution_method = self.attribution_method_orig if model_name == 'orig' else self.attribution_method_learn
        for i, (x, m, y) in enumerate(self.validation_loader):
            x, m, y = x.to(self.device), m.to(self.device), y.to(self.device)
            x.requires_grad = True
            
            score = self.metric.evaluate(
                model=model,
                x=x.detach().cpu().numpy(),
                m=m.detach().cpu().numpy(),
                y=y.detach().cpu().numpy(),
                explain_func=attribution_method,
                device=self.device
            ).detach().cpu()
            
            with torch.no_grad():
                o = model(x)
            with torch.no_grad():
                o_other = model_other(x)
            
            preds.append(o.max(-1)[1].detach().cpu())
            preds_other.append(o_other.max(-1)[1].detach().cpu())
            
            scores += score
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i == iterations:
                break
        preds = torch.cat(preds)
        preds_other = torch.cat(preds_other)
        accs = (preds == preds_other).float()

        return scores, accs


class ComplexityLearnableMetricTrainer(MaskedSimilarityLearnableMetricTrainer):
    def train(self, iterations=10000, validation_iterations=2, eval_every=50):
        original_scores, _ = self.evaluate('orig', validation_iterations)
        original_score = torch.tensor(original_scores).mean()
        print(f'original score: {original_score}')
        
        current_score = 0 if self.metric.better_higher else 999
        current_acc = 0
        current_best_score = 0 if self.metric.better_higher else 999
        orig_learn_acc = 0
        pbar = tqdm(self.train_loader)
        for i, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                preds = self.model_orig(x).max(-1)[1]

            x.requires_grad = True
            o = self.model_learn(x)
            r_learn = self.attribution_method_learn(
                x,
                o,
                classes=preds
            )
            
            x = x.detach()
            o_orig = self.model_orig(x)
            o_student = self.model_learn(x)
            
            loss_orig = (1 - ((normalize_vector(o_orig) * normalize_vector(o_student)).sum(-1) + 1) / 2).mean()#nn.CrossEntropyLoss()(o_student, y)
            
            r_learn = backprop_normalize_to_one(r_learn)
            loss_masked = r_learn.flatten(1).abs().mean(-1).mean()
            loss = 0 * loss_orig + 0.1 * loss_masked
            
            loss.backward()
            self.optimizer.step()
            
            current_acc = (o_student.max(-1)[1] == o_orig.max(-1)[1]).float().mean()
            
            pbar.set_description(f'train loss: {loss.item()} current score: {current_score} current best score: {current_best_score} current acc: {current_acc} orig learn acc: {orig_learn_acc}')
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i > 0 and i % eval_every == 0:
                current_scores, orig_learn_accs = self.evaluate('learn', validation_iterations)
                current_score, orig_learn_acc = torch.tensor(current_scores).mean(), orig_learn_accs.mean()
                
                if ((self.metric.better_higher and current_score > current_best_score) or (not self.metric.better_higher and current_score < current_best_score)) and orig_learn_acc > 0.95:
                    current_best_score = current_score
                    torch.save(self.model_learn.state_dict(), f'learnable_metric_models/{self.model_learn.__class__.__name__}{self.metric.__class__.__name__}{self.attribution_method_learn.__class__.__name__}.pth')

            
            if i == iterations:
                break
    
    def evaluate(self, model_name, iterations=2):
        scores = []
        preds = []
        preds_other = []
        model = self.model_orig if model_name == 'orig' else self.model_learn
        model_other = self.model_learn if model_name == 'orig' else self.model_orig
        attribution_method = self.attribution_method_orig if model_name == 'orig' else self.attribution_method_learn
        for i, (x, y) in enumerate(self.validation_loader):
            x, y = x.to(self.device), y.to(self.device)
            x.requires_grad = True
            
            o = model(x)
            r = attribution_method(
                x,
                o,
                classes=o.max(-1)[1],
                create_graph=False,
            )
            
            r = backprop_normalize_to_one_with_detach(r)
            
            score = self.metric.evaluate(
                model=model,
                x=x.detach().cpu().numpy(),
                y=y.detach().cpu().numpy(),
                a=r.detach().cpu().numpy(),
                explain_func=attribution_method,
                device=self.device
            )
            
            with torch.no_grad():
                o_other = model_other(x)
            
            preds.append(o.max(-1)[1].detach().cpu())
            preds_other.append(o_other.max(-1)[1].detach().cpu())
            
            scores += score
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if i == iterations:
                break
        preds = torch.cat(preds)
        preds_other = torch.cat(preds_other)
        accs = (preds == preds_other).float()

        return scores, accs