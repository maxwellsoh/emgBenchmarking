from .Model_Trainer import Model_Trainer
from .Classic_Trainer import Classic_Trainer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import wandb


class SVC_RF_Trainer(Classic_Trainer):
    """
    Trainer for SVC or RF models. 
    """

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

        

    def set_model(self):
        if self.args.model == "SVC":
            model = SVC(probability=True)
        
        elif self.args.model == 'RF':
            model = RandomForestClassifier()

    def model_loop(self):
        X_train, Y_train = super().get_data_from_loader(self.train_loader)
        X_val, Y_val = super().get_data_from_loader(self.val_loader)
        # X_test, Y_test = get_data_from_loader(test_loader)

        print("Data loaded")
        self.model.fit(X_train, Y_train)
        print("Model trained")
        train_preds = self.model.predict(X_train)
        print("Train predictions made")
        val_preds = self.model.predict(X_val)
        print("Validation predictions made")
        # test_preds = model.predict(X_test)

        train_acc = accuracy_score(Y_train, train_preds)
        val_acc = accuracy_score(Y_val, val_preds)
        # test_acc = accuracy_score(Y_test, test_preds)

        train_loss = log_loss(Y_train, self.model.predict_proba(X_train))
        val_loss = log_loss(Y_val, self.model.predict_proba(X_val))
        # test_loss = log_loss(Y_test, model.predict_proba(X_test))

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        # print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

        wandb.log({
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc,
            # "Test Loss": test_loss,
            # "Test Acc": test_acc
        })


