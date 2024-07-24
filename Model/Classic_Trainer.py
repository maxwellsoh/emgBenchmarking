from .Model_Trainer import Model_Trainer
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


class Classic_Trainer(Model_Trainer):
    """
    Shared functions for MLP, SVC, and RF trainers.  
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

        

    def set_model(self):
        raise NotImplementedError("Model should be defined in MLP or SVC_RF Trainer")


    def setup_model(self):
        assert self.args.model in ['MLP', 'SVC', 'RF'], "Model not supported."

        super().set_pretrain_path()
        self.set_model()
        super().set_resize_transform()
        super().set_loaders()
        super().clear_memory()
        super().shared_setup()
    

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
    
    
