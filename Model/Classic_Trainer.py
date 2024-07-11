from .Model_Trainer import Model_Trainer
import torch.nn as nn

class Classic_Trainer(Model_Trainer):
    """
    Shared functions for MLP, SVC, and RF trainers.  
    """

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

    def set_model(self):
        raise NotImplementedError("Model should be defined in MLP or SVC_RF Trainer")


    def setup_model(self):
        assert self.args.model in ['MLP', 'SVC', 'RF'], "Model not supported."

        super().set_pretrain_path()
        self.set_model()
        super().set_resize_transform()
        super().set_loaders()
        super().set_scheduler()
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
    
    


    # def set_model(self):

    #     if args.model == 'MLP':
    #         class MLP(nn.Module):
    #             def __init__(self, input_size, hidden_sizes, output_size):
    #                 super(MLP, self).__init__()
    #                 self.hidden_layers = nn.ModuleList()
    #                 self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
    #                 for i in range(1, len(hidden_sizes)):
    #                     self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
    #                 self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    #             def forward(self, x):
    #                 for hidden in self.hidden_layers:
    #                     x = F.relu(hidden(x))
    #                 x = self.output_layer(x)
    #                 return x
            
    #         # PyTorch MLP model
    #         input_size = 3 * 224 * 224  # Change according to your input size
    #         hidden_sizes = [512, 256]  # Example hidden layer sizes
    #         output_size = num_classes  # Number of classes
    #         model = MLP(input_size, hidden_sizes, output_size).to(device)
    #         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #         if args.force_regression:
    #             criterion = nn.MSELoss()
    #         else:
    #             criterion = nn.CrossEntropyLoss()
        
    #     elif args.model == 'SVC':
    #         model = SVC(probability=True)
    #     elif args.model == 'RF':
    #         model = RandomForestClassifier()

    #     self.model = model 



    


    
            

