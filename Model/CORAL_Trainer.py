'''
CORAL Trainer
CNN Base with DeepCoral algorithm applied. 
Inspiration: https://github.com/thuml/Transfer-Learning-Library/
'''
from .Model_Trainer import Model_Trainer
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm
import Model.VisualTransformer as VisualTransformer
import Model.ml_metrics_utils as ml_utils
import numpy as np
from torch.utils.data import DataLoader
import multiprocessing
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import torch.autograd as autograd
import torch.nn.functional as F
import math 

# Create a wrapper class to get the features

import torch
import torch.nn as nn
import timm

class ResNet18WithFeatures(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ResNet18WithFeatures, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        # Extract the feature extractor part, excluding classification layer
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])  
        # Classifier layer
        self.classifier = nn.Linear(self.resnet.num_features, num_classes)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x).view(x.size(0), -1)
        # Get predictions
        predictions = self.classifier(features)
        return predictions, features


class CorrelationAlignmentLoss(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.

    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by

    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))

    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by

    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.
    """

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:

        # Pad tensors 
        # num_samples_s = f_s.shape[1]
        # num_samples_t = f_t.shape[1]

        # # Determine the maximum number of samples
        # max_samples = max(num_samples_s, num_samples_t)

        # # Pad f_s if necessary
        # if num_samples_s < max_samples:
        #     padding_size = max_samples - num_samples_s
        #     padding = torch.zeros(f_s.shape[0], padding_size, device=f_s.device)
        #     f_s = torch.cat((f_s, padding), dim=1)

        # # Pad f_t if necessary
        # if num_samples_t < max_samples:
        #     padding_size = max_samples - num_samples_t
        #     padding = torch.zeros(f_t.shape[0], padding_size, device=f_t.device)
        #     f_t = torch.cat((f_t, padding), dim=1)

        # Downsample
        num_samples_s = f_s.shape[1]
        num_samples_t = f_t.shape[1]

        if num_samples_s > num_samples_t:
            indices = torch.randperm(num_samples_s)[:num_samples_t]
            f_s = f_s[:, indices]

        if num_samples_t > num_samples_s:
            indices = torch.randperm(num_samples_t)[:num_samples_s]
            f_t = f_t[:, indices]

        # Duplicate 
        # num_samples_s = f_s.shape[1]
        # num_samples_t = f_t.shape[1]

        # # Duplicate f_s if necessary
        # if num_samples_s < num_samples_t:
        #     repeat_factor = (num_samples_t + num_samples_s - 1) // num_samples_s  # Ceiling division
        #     f_s = f_s.repeat(1, repeat_factor)[:, :num_samples_t]  # Repeat and then truncate to match size

        # # Duplicate f_t if necessary
        # if num_samples_t < num_samples_s:
        #     repeat_factor = (num_samples_s + num_samples_t - 1) // num_samples_t  # Ceiling division
        #     f_t = f_t.repeat(1, repeat_factor)[:, :num_samples_s]  # Repeat and then truncate to match size

        assert f_s.shape == f_t.shape
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1) 

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff

class CORAL_Trainer(Model_Trainer):
    """
    Training class for CNN self.models (resnet, convnext_tiny_custom, vit_tiny_patch and not (unlabeled_domain or in MLP, SVC, RF) self.models.
    """

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_model(self):
        """
        Main function that sets up the model 
        """
        super().set_pretrain_path()
        self.set_model()
        self.set_optimizer()
        self.set_param_requires_grad()
        super().set_resize_transform()
        super().set_loaders()
        self.set_criterion()
        super().start_train_and_validate_run()
        super().set_model_to_device()
        super().set_testrun_foldername()
        super().set_gesture_labels()
        super().plot_images()

    def model_loop(self):

        # Get metrics
        if self.args.pretrain_and_finetune:
            training_metrics, validation_metrics, testing_metrics = super().get_metrics()
        else: 
            training_metrics, validation_metrics = super().get_metrics(testing=False)

        # Train and Validation Loop 
        self.train_and_validate(training_metrics, validation_metrics)

        # Finetune Loop 
        if self.args.pretrain_and_finetune:
            self.pretrain_and_finetune(testing_metrics)
        
    def set_criterion(self):
        self.criterion = CorrelationAlignmentLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def set_model(self):

        self.model = ResNet18WithFeatures(model_name=self.model_name, num_classes=self.num_gestures)

        # Calculate the number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        # Print the total number of parameters
        print(f'Total number of parameters: {total_params}')

    def set_param_requires_grad(self):
        
        num = 0
        for name, param in self.model.named_parameters():
            num += 1
            if (num > 0):
            #if (num > 72): # for -3
            #if (num > 33): # for -4
                param.requires_grad = True
            else:
                param.requires_grad = False

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def print_classification_metrics(self):
        """
        Batches data for test, train, and validation and plots confusion matrix and classification report.
        """

        # Test Batch
        self.model.eval()
        with torch.no_grad():
            test_predictions = []
            for X_batch, Y_batch in tqdm(self.test_loader, desc="Test Batch Loading for Confusion Matrix"):
                X_batch = X_batch.to(self.device).to(torch.float32)
                output, _ = self.model(X_batch)
                if isinstance(output, dict):
                    output = output['logits']
                preds = np.argmax(output.cpu().detach().numpy(), axis=1)
                test_predictions.extend(preds)

        true_labels = np.argmax(self.Y.test.cpu().detach().numpy(), axis=1)
        test_predictions = np.array(test_predictions)

        # Calculate and print the confusion matrix
        conf_matrix = confusion_matrix(true_labels, test_predictions)
        print("Confusion Matrix:")
        print(conf_matrix)

        print("Classification Report:")
        print(classification_report(true_labels, test_predictions))
        
        self.utils.plot_confusion_matrix(np.argmax(self.Y.test.cpu().detach().numpy(), axis=1), np.array(test_predictions), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'test')  

        torch.cuda.empty_cache()  # Clear cache if needed 

        # Validation Metrics
        self.model.eval()
        with torch.no_grad():
            validation_predictions = []
            for X_batch, Y_batch in tqdm(self.val_loader, desc="Validation Batch Loading for Confusion Matrix"):
                X_batch = X_batch.to(self.device).to(torch.float32)
                output, _ = self.model(X_batch)
                if isinstance(output, dict):
                    output = output['logits']
                preds = np.argmax(output.cpu().detach().numpy(), axis=1)
                validation_predictions.extend(preds)
        
        self.utils.plot_confusion_matrix(np.argmax(self.Y.validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'validation')   
        torch.cuda.empty_cache()

        # Load training in smaller batches for memory purposes
        torch.cuda.empty_cache()  # Clear cache if needed

        self.model.eval()
        self.train_loader_unshuffled = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=self.utils.seed_worker, pin_memory=True, drop_last=self.args.force_regression)
        
        # Train Metrics
        with torch.no_grad():
            train_predictions = []
            for X_batch, Y_batch in tqdm(self.train_loader_unshuffled, desc="Training Batch Loading for Confusion Matrix"):
                X_batch = X_batch.to(self.device).to(torch.float32)
                output, _ = self.model(X_batch)
                if isinstance(output, dict):
                        output = output['logits']
                preds = torch.argmax(output, dim=1)
                train_predictions.extend(preds.cpu().detach().numpy())
        
        self.utils.plot_confusion_matrix(np.argmax(self.Y.train.cpu().detach().numpy(), axis=1), np.array(train_predictions), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'train')
    
    def pretrain_and_finetune(self, testing_metrics):
        """
        Finish current run and start a new run for finetuning.
        """

        # Evaluate performance on test metrics
        ml_utils.evaluate_model_on_test_set(self.model, self.test_loader, self.device, self.num_gestures, self.cross_entropy, self.args, testing_metrics)

        if not self.args.force_regression:
            self.print_classification_metrics()
            
        self.train_and_validate_run.finish()

        # Start new run for finetuning
        self.ft_run = wandb.init(name=self.wandb_runname+"_finetune", project=self.project_name) 
        ft_epochs = self.args.finetuning_epochs

        finetune_dataset = super().CustomDataset(self.X.train_finetuning,self.Y.train_finetuning, transform=self.resize_transform)

        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count() // 8,
            worker_init_fn=self.utils.seed_worker,
            pin_memory=True,
            drop_last=self.args.force_regression
        )
        # Initialize metrics for finetuning training and validation
        ft_training_metrics, ft_validation_metrics, testing_metrics = super().get_metrics()

        # Finetuning Loop 
        for epoch in tqdm(range(ft_epochs), desc="Finetuning Epoch"):
            self.model.train()
            train_loss = 0.0
            
            for ft_train_metric in ft_training_metrics:
                ft_train_metric.reset()

            with tqdm(finetune_loader, desc=f"Finetuning Epoch {epoch+1}/{ft_epochs}", leave=False) as t:
                for X_batch, Y_batch in t:
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32)
                    Y_batch_long = torch.argmax(Y_batch, dim=1)

                    self.optimizer.zero_grad()
                    output, features = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']


                    output_for_loso = output
                    feature_for_loso = features
                    labels_for_loso = Y_batch

                    loss_ce = 0
                    loss_penalty = 0

                    # calculate cross entropy across 
                    loss_ce = self.cross_entropy(output_for_loso, labels_for_loso)

                    # no other domains to cross compare with 
                    loss = loss_ce
            
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                    for ft_train_metric in ft_training_metrics:
                        ft_train_metric(output,Y_batch_long)

                    if t.n % 10 == 0:
                        ft_accuracy_metric = next(metric for metric in ft_training_metrics if metric.name =="Micro_Accuracy")

                        t.set_postfix({
                            "Batch Loss": loss.item(), 
                            "Batch Acc": ft_accuracy_metric.compute().item()
                        })

            # Finetuning Validation
            self.model.eval()
            val_loss = 0.0

            for ft_val_metric in ft_validation_metrics:
                ft_val_metric.reset()

           
            all_val_outputs = []
            all_val_labels = []

            with torch.no_grad():
                for X_batch, Y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32) 
                    Y_batch_long = torch.argmax(Y_batch, dim=1)

                    output, _ = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    val_loss += self.cross_entropy(output,Y_batch).item()

                    for ft_val_metric in ft_validation_metrics:
                        if ft_val_metric.name != "Macro_AUROC" and ft_val_metric.name != "Macro_AUPRC":
                            ft_val_metric(output,Y_batch_long)

                    all_val_outputs.append(output)
                    all_val_labels.append(Y_batch_long)

            all_val_outputs = torch.cat(all_val_outputs, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)  
            Y_validation_long = torch.argmax(self.Y.validation, dim=1).to(self.device)

            true_labels =Y_validation_long.cpu().detach().numpy()
            test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)
            conf_matrix = confusion_matrix(true_labels, test_predictions)
            print("Confusion Matrix:")
            print(conf_matrix)

            finetune_val_macro_auroc_metric = next(metric for metric in ft_validation_metrics if metric.name == "Macro_AUROC")
            finetune_val_macro_auprc_metric = next(metric for metric in ft_validation_metrics if metric.name == "Macro_AUPRC")

            finetune_val_macro_auroc_metric(all_val_outputs,Y_validation_long)
            finetune_val_macro_auprc_metric(all_val_outputs,Y_validation_long)

            # Calculate average loss and metrics
            train_loss /= len(finetune_loader)
            val_loss /= len(self.val_loader)
            tpr_results = ml_utils.evaluate_model_tpr_at_fpr(self.model, self.val_loader, self.device, self.num_gestures)
            fpr_results = ml_utils.evaluate_model_fpr_at_tpr(self.model, self.val_loader, self.device, self.num_gestures)
            confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(self.model, self.val_loader, self.device)

            # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
            ft_training_metrics_values = {ft_metric.name: ft_metric.compute() for ft_metric in ft_training_metrics}
            ft_validation_metrics_values = {ft_metric.name: ft_metric.compute() for ft_metric in ft_validation_metrics}

            # Print metric values
            print(f"Finetuning Epoch {epoch+1}/{ft_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            ft_train_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in ft_training_metrics_values.items())
            print(f"Train Metrics | {ft_train_metrics_str}")

            ft_val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in ft_validation_metrics_values.items())
            print(f"Val Metrics | {ft_val_metrics_str}")

            wandb.log({
                "train/Loss": train_loss,
                **{
                    f"train/{name}": value.item() 
                    for name, value in ft_training_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"train/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in ft_training_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },
                "train/Learning Rate": self.optimizer.param_groups[0]['lr'],
                "train/Epoch": epoch+1,
                "validation/Loss": val_loss,
                **{
                    f"validation/{name}": value.item() 
                    for name, value in ft_validation_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"validation/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in ft_validation_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },

                **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not self.args.force_regression else {}),
                **({f"fpr_at_fixed_tpr/Average Val FPR at {tpr} TPR": np.mean(fprs) for tpr, fprs in fpr_results.items()} if not self.args.force_regression else {}),
                **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not self.args.force_regression else {}),
                **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not self.args.force_regression else {})
            })
        
        
        torch.save(self.model.state_dict(), self.model_filename)
        wandb.save(f'self.model/self.modelParameters_{self.formatted_datetime}.pth')

        # Evaluate the self.model on the test set
        ml_utils.evaluate_model_on_test_set(self.model, self.test_loader, self.
        device, self.num_gestures, self.cross_entropy, self.args, testing_metrics)

        self.print_classification_metrics()
        self.ft_run.finish() 

    def train_and_validate(self, training_metrics, validation_metrics):
        """
        Train and validation loop. 
        """

        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            
            self.model.train()
            
            train_loss = 0.0

            # Reset training metrics at the start of each epoch
            for train_metric in training_metrics:
                train_metric.reset()
        
            outputs_train_all = [] 
            ground_truth_train_all = []


            batch_no = 0 
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False) as t:

                
                prop_per_domain = self.sampler.get_prop_per_domain()
                print("prop_per_domain:", prop_per_domain)

                for X_batch, Y_batch in t: 
                   
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32) 
                    Y_batch_long = torch.argmax(Y_batch, dim=1)

                    self.optimizer.zero_grad()

                    output, features = self.model(X_batch)

                    if isinstance(output, dict):
                        output = output['logits']

                    # seperate into different domains
                    output_per_domains = output.split(prop_per_domain, dim=0)
                    features_per_domains = features.split(prop_per_domain, dim=0)
                    labels_per_domains = Y_batch.split(prop_per_domain, dim=0)

                    loss_ce = 0 
                    loss_penalty = 0

                    n_domains_per_batch = self.utils.num_subjects - 1

                    for domain_i in range(n_domains_per_batch):
                        
                        output_i = output_per_domains[domain_i]
                        labels_i = labels_per_domains[domain_i]

                        # calculate cross entropy across domain
                        loss_ce += self.cross_entropy(output_i, labels_i)
                        
                        # calculate correlation alignment loss with subsequent domains
                        for domain_j in range(domain_i + 1, n_domains_per_batch):
                            features_i = features_per_domains[domain_i].transpose(0,1)
                            features_j = features_per_domains[domain_j].transpose(0,1)
                            loss_penalty += self.criterion(features_i, features_j)

                    # normalize loss
                    loss_ce /= n_domains_per_batch
                    loss_penalty /= n_domains_per_batch * (n_domains_per_batch - 1) / 2

                    loss = loss_ce + loss_penalty

                    loss.backward()
                    self.optimizer.step() 

                    outputs_train_all.append(output)
                    ground_truth_train_all.append(torch.argmax(Y_batch, dim=1))

                    train_loss += loss.item()

                    for train_metric in training_metrics:
                        if train_metric.name != "Macro_AUROC" and train_metric.name != "Macro_AUPRC":
                            train_metric(output, Y_batch_long)
                   
                    micro_accuracy_metric = next(metric for metric in training_metrics if metric.name == "Micro_Accuracy")
                    if t.n % 10 == 0:
                        t.set_postfix({
                            "Batch Loss": loss.item(), 
                            "Batch Acc": micro_accuracy_metric.compute().item()
                        })
                    
                    batch_no += 1
                outputs_train_all = torch.cat(outputs_train_all, dim=0).to(self.device)
                ground_truth_train_all = torch.cat(ground_truth_train_all, dim=0).to(self.device)

            train_macro_auroc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUROC")
            train_macro_auprc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUPRC")

            train_macro_auroc_metric(outputs_train_all,   ground_truth_train_all)
            train_macro_auprc_metric(outputs_train_all, ground_truth_train_all)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            for val_metric in validation_metrics:
                val_metric.reset()

            all_val_outputs = []
            all_val_labels = []

            with torch.no_grad():
            
                for X_batch, Y_batch in self.val_loader:
                    
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32)
                    Y_batch_long = torch.argmax(Y_batch, dim=1)

                    output, _ = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']

                    all_val_outputs.append(output)
                    all_val_labels.append(Y_batch_long)
                 
                    val_loss += self.cross_entropy(output, Y_batch).item()

                    for val_metric in validation_metrics:
                        if val_metric.name != "Macro_AUROC" and val_metric.name != "Macro_AUPRC":
                            val_metric(output,Y_batch_long)

            all_val_outputs = torch.cat(all_val_outputs, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)
            Y_validation_long = torch.argmax(self.Y.validation, dim=1).to(self.device)
            true_labels = Y_validation_long.cpu().detach().numpy()
            test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)

    
            conf_matrix = confusion_matrix(true_labels, test_predictions)
            print("Confusion Matrix:")
            print(conf_matrix)

            val_macro_auroc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUROC")
            val_macro_auprc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUPRC")

            val_macro_auroc_metric(all_val_outputs,Y_validation_long)
            val_macro_auprc_metric(all_val_outputs,Y_validation_long)

            # Calculate average loss and metrics
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)

            # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
            training_metrics_values = {metric.name: metric.compute() for metric in training_metrics}
            validation_metrics_values = {metric.name: metric.compute() for metric in validation_metrics}

            tpr_results = ml_utils.evaluate_model_tpr_at_fpr(self.model, self.val_loader, self.device, self.num_classes)
            confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(self.model, self.val_loader, self.device)

            # Print metric values
            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            train_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in training_metrics_values.items())
            print(f"Train Metrics: {train_metrics_str}")

            val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in validation_metrics_values.items())
            print(f"Val Metrics: {val_metrics_str}")

            for fpr, tprs in tpr_results.items():
                print(f"Val TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
            for confidence_level, acc in confidence_levels.items():
                print(f"Val Accuracy at {confidence_level} confidence level: {acc:.4f}")

            wandb.log({
                "train/Loss": train_loss,
                "train/Learning Rate": self.optimizer.param_groups[0]['lr'],
                "train/Epoch": epoch+1,
                "validation/Loss": val_loss,
                **{
                    f"train/{name}": value.item() 
                    for name, value in training_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"train/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in training_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },
                **{
                    f"validation/{name}": value.item() 
                    for name, value in validation_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"validation/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in validation_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },
                **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not self.args.force_regression else {}),
                **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not self.args.force_regression else {}),
                **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not self.args.force_regression else {}),
            })

        torch.save(self.model.state_dict(), self.model_filename)
        wandb.save(f'model/modelParameters_{self.formatted_datetime}.pth')

        # If pretrain and finetune, continue. Otherwise, here.
        if not self.args.pretrain_and_finetune:

            if not self.args.force_regression: 
                self.print_classification_metrics()
            self.train_and_validate_run.finish()





    







