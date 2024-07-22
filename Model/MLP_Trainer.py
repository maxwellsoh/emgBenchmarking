from .Model_Trainer import Model_Trainer
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix, classification_report 
import Model.ml_metrics_utils as ml_utils
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for hidden in self.hidden_layers:
            x = F.relu(hidden(x))
        x = self.output_layer(x)
        return x

class MLP_Trainer(Model_Trainer):

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
        assert self.args.model in ['MLP', 'SVC', 'RF'], "Model not supported."

        super().set_pretrain_path()
        super().set_resize_transform()
        super().set_loaders()
        super().set_criterion()
        super().start_train_and_validate_run()
        super().set_gesture_labels()
        super().set_testrun_foldername()
        super().plot_images()
        self.set_model()
        self.set_optimizer() # Only for MLP not SV/RF

    def get_data_from_loader(loader):
        X = []
        Y = []
        for X_batch, Y_batch in tqdm(loader, desc="Batches convert to Numpy"):
            # Flatten each image from [batch_size, 3, 224, 224] to [batch_size, 3*224*224]
            # X_batch_flat = X_batch.view(X_batch.size(0), -1).cpu().numpy().astype(np.float64)
            Y_batch_indices = torch.argmax(Y_batch, dim=1)  # Convert one-hot to class indices
            X.append(X_batch)
            Y.append(Y_batch_indices.cpu().numpy().astype(np.int64))
        return np.vstack(X), np.hstack(Y)
    
    def set_model(self):

        # PyTorch MLP model
        input_size = 3 * 224 * 224  # Change according to your input size
        hidden_sizes = [512, 256]  # Example hidden layer sizes
        output_size = self.num_classes  # Number of classes
        self.model = MLP(input_size, hidden_sizes, output_size).to(self.device)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def model_loop(self):

        # PyTorch training loop for MLP
        self.training_metrics, self.validation_metrics = super().get_metrics(testing=False)

        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):

            self.model.train()
 
            # Initialize metrics 
            for train_metric in self.training_metrics:
                train_metric.reset()

            train_loss = 0.0

            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False) as t:
                if not self.args.force_regression:
                    ground_truth_train_all = []
                    outputs_all = [] # NOTE: why is this inside the loop for MLP but outside in CNN
                
                for X_batch, Y_batch in t:
                    X_batch = X_batch.view(X_batch.size(0), -1).to(self.device).to(torch.float32)
                    if self.args.force_regression:
                        Y_batch = Y_batch.to(self.self.device).to(torch.float32) 
                    else:
                        Y_batch = torch.argmax(Y_batch, dim=1).to(self.device).to(torch.int64)

                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = self.criterion(output, Y_batch)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    for train_metric in self.training_metrics:
                        if train_metric.name != "Macro_AUROC" and train_metric.name != "Macro_AUPRC":
                            train_metric(output, Y_batch)
                    
                    if not self.args.force_regression:
                        outputs_all.append(output)
                        # ground_truth_train_all.append(torch.argmax(Y_batch, dim=1)) 
                        # NOTE: This code was double flattening it raising dimension errors.
                        ground_truth_train_all.append(Y_batch)
                    
                    if not self.args.force_regression:
                        if t.n % 10 == 0:
                            train_micro_acc = next(metric for metric in self.training_metrics if metric.name == "Micro_Accuracy")
                            t.set_postfix({"Batch Loss": loss.item(), "Batch Acc": train_micro_acc.compute().item()})

                    del X_batch, Y_batch, output
                    torch.cuda.empty_cache()
                
                if not self.args.force_regression:
                    outputs_all = torch.cat(outputs_all, dim=0).to(self.device)
                    ground_truth_train_all = torch.cat(ground_truth_train_all, dim=0).to(self.device)

                    train_macro_auroc = next(metric for metric in self.training_metrics if metric.name == "Macro_AUROC")
                    train_macro_auprc = next(metric for metric in self.training_metrics if metric.name == "Macro_AUPRC")

                    train_macro_auroc(outputs_all, ground_truth_train_all)
                    train_macro_auprc(outputs_all, ground_truth_train_all)


            # Validation
            self.model.eval()

            for val_metric in self.validation_metrics:
                val_metric.reset()
            
            if not self.args.force_regression:
                all_val_outputs = []
                all_val_labels = []

            val_loss = 0.0
            with torch.no_grad():
    
                for X_batch, Y_batch in self.val_loader:
                    X_batch = X_batch.view(X_batch.size(0), -1).to(self.device).to(torch.float32)
                    if self.args.force_regression:
                        Y_batch =  Y_batch.to(self.device).to(torch.float32)
                    else: 
                        Y_batch = torch.argmax(Y_batch, dim=1).to(self.device).to(torch.int64)

                    output = self.model(X_batch)
                    for validation_metric in self.validation_metrics:
                        validation_metric(output, Y_batch)

                    val_loss += self.criterion(output, Y_batch).item()
        
                    if not self.args.force_regression:
                        all_val_outputs.append(output)
                        all_val_labels.append(Y_batch)

                    del X_batch, Y_batch
                    torch.cuda.empty_cache()

            if not self.args.force_regression:
                all_val_outputs = torch.cat(all_val_outputs, dim=0).to(self.device)
                all_val_labels = torch.cat(all_val_labels, dim=0)

                Y_validation_long = torch.argmax(self.Y.validation, dim=1).to(self.device).to(torch.int64)

                true_labels = Y_validation_long.cpu().detach().numpy()
                test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)
                conf_matrix = confusion_matrix(true_labels, test_predictions)
                print("Confusion Matrix:")
                print(conf_matrix)

                val_macro_auroc = next(metric for metric in self.validation_metrics if metric.name == "Macro_AUROC")
                val_macro_auprc = next(metric for metric in self.validation_metrics if metric.name == "Macro_AUPRC")

                val_macro_auroc(all_val_outputs, Y_validation_long)
                val_macro_auprc(all_val_outputs, Y_validation_long)
                

            # Average the losses and print the metrics
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)

            if not self.args.force_regression:
                tpr_results = ml_utils.evaluate_model_tpr_at_fpr(self.model, self.val_loader, self.device, self.num_gestures)
                fpr_results = ml_utils.evaluate_model_fpr_at_tpr(self.model, self.val_loader, self.device, self.num_gestures)
                confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(self.model, self.val_loader, self.device)

            # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
            training_metrics_values = {metric.name: metric.compute() for metric in self.training_metrics}
            self.validation_metrics_values = {metric.name: metric.compute() for metric in self.validation_metrics}

            
            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            training_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in training_metrics_values.items())
            print(f"Train Metrics: {training_metrics_str}")

            val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in self.validation_metrics_values.items())
            print(f"Val Metrics: {val_metrics_str}")
            
            if not self.args.force_regression:
                for fpr, tprs in tpr_results.items():
                    print(f"Val TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
                for confidence_level, accuracy in confidence_levels.items():
                    print(f"Val Accuracy at confidence level >{confidence_level}: {accuracy:.4f}")

            training_metrics_values = {metric.name: metric.compute() for metric in self.training_metrics}
            self.validation_metrics_values = {metric.name: metric.compute() for metric in self.validation_metrics}

            # Log metrics to wandb or any other tracking tool
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
                for name, value in self.validation_metrics_values.items() 
                if name != 'R2Score_RawValues'
            },
            **{
                f"validation/R2Score_RawValues_{i+1}": v.item() 
                for name, value in self.validation_metrics_values.items() 
                if name == 'R2Score_RawValues'
                for i, v in enumerate(value)
            },


            **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not self.args.force_regression else {}),
            **({f"fpr_at_fixed_tpr/Average Val FPR at {tpr} TPR": np.mean(fprs) for tpr, fprs in fpr_results.items()}if not self.args.force_regression else {}),
            **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not self.args.force_regression else {}),
            **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not self.args.force_regression else {})
        })
            
        torch.save(self.model.state_dict(), self.model_filename)
        wandb.save(f'model/modelParameters_{self.formatted_datetime}.pth')

        self.train_and_validate_run.finish()
    