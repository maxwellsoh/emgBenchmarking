
from .Model_Trainer import Model_Trainer
import datetime
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, split_ssl_data, BasicDataset
import torchvision.transforms as transforms
import torch
import multiprocessing
from semilearn.core.utils import send_model_cuda
import wandb
import Model.ml_metrics_utils as ml_utils
from torch.utils.data import DataLoader

class Unlabeled_Domain_Adaptation_Trainer(Model_Trainer):

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(self, X_data, Y_data, label_data, env)

        # Set seeds for reproducibility
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.semilearn_config = None
        self.semilearn_algorithm = None
        self.semilearn_config_dict = None
        
        # Temporary helper variables
        self.proportion_unlabeled_of_training_subjects = self.args.proportion_unlabeled_data_from_training_subjects
        self.proportion_unlabeled_of_proportion_to_keep_of_leftout = self.args.proportion_unlabeled_data_from_leftout_subject


    def setup_model(self):

        super().set_pretrain_path()
        self.set_loaders()
        super().shared_setup()
        super().start_train_and_validate_run()
        super().set_testrun_foldername()
        super().set_gesture_labels()
        super().plot_images()

        
    def set_semilearn_config(self):

        assert (self.args.transfer_learning and self.args.train_test_split_for_time_series) or self.args.leave_one_session_out, \
            "Unlabeled Domain Adaptation requires transfer learning and cross validation for time series or leave one session out"

        current_date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.semilearn_config_dict = {
            'algorithm': self.args.unlabeled_algorithm,
            'net': self.args.model,
            'use_pretrain': True,  
            'pretrain_path': self.pretrain_path,
            'seed': self.args.seed,

            # optimization configs
            'epoch': self.args.epochs,  # set to 100
            # 'num_train_iter': self.args.epochs * super().ceildiv(self.X_train.shape[0], self.args.batch_size),
            # 'num_eval_iter': super().ceildiv(self.X_train.shape[0], self.args.batch_size),
            # 'num_log_iter': super().ceildiv(self.X_train.shape[0], self.args.batch_size),
            'optim': 'AdamW',   # AdamW optimizer
            'lr': self.args.learning_rate,  # Learning rate
            'layer_decay': 0.5,  # Layer-wise decay learning rate  
            'momentum': 0.9,  # Momentum
            'weight_decay': 0.0005,  # Weight decay
            'amp': True,  # Automatic mixed precision
            'train_sampler': 'RandomSampler',  # Random sampler
            'rank': 0,  # Rank
            'batch_size': self.args.batch_size,  # Batch size
            'eval_batch_size': self.args.batch_size, # Evaluation batch size
            'use_wandb': True,
            'ema_m': 0.999,
            'save_dir': './saved_models/unlabeled_domain_adaptation/',
            'save_name': f'{self.args.unlabeled_algorithm}_{self.args.model}_{self.args.dataset}_seed_{self.args.seed}_leave_{self.leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}',
            'resume': True,
            'overwrite': True,
            'load_path': f'./saved_models/unlabeled_domain_adaptation/{self.args.unlabeled_algorithm}_{self.args.model}_{self.args.dataset}_seed_{self.args.seed}_leave_{self.leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}/latest_model.pth',
            'scheduler': None,

            # dataset configs
            'dataset': 'none',
            'num_labels': self.X.train.shape[0],
            'num_classes': self.num_gestures,
            'input_size': 224,
            'data_dir': './data',

            # algorithm specific configs
            'hard_label': True,
            'uratio': 1.5,
            'ulb_loss_ratio': 1.0,

            # device configs
            'gpu': self.args.gpu,
            'world_size': 1,
            'distributed': False,
        } 

        self.semilearn_config = get_config(self.semilearn_config_dict)
        self.semilearn_config = self.semilearn_config

    def create_datasets(self):
        """
        Helper for set loaders
        """


        labeled_dataset, unlabeled_dataset, finetune_dataset, finetune_unlabeled_dataset, validation_dataset = None, None, None, None
        
        if self.args.model == 'vit_tiny_patch2_32':
            semilearn_transform = transforms.Compose([transforms.Resize((32,32)), self.ToNumpy()])
        else: 
            semilearn_transform = transforms.Compose([transforms.Resize((224,224)), self.ToNumpy()])

            
        labeled_dataset = BasicDataset(
            self.semilearn_config, 
            self.X.train, 
            torch.argmax(self.Y.train, dim=1), 
            self.semilearn_config.num_classes, 
            semilearn_transform, 
            is_ulb=False
        )

        # Create the datasets for pretrain 

        if self.proportion_unlabeled_of_training_subjects>0:
            unlabeled_dataset = BasicDataset(
                self.semilearn_config,
                self.X.train_unlabeled,
                torch.argmax(self.Y.train_unlabeled, dim=1),
                self.semilearn_config.num_classes,
                semilearn_transform,
                is_ulb=True,
                strong_transform=semilearn_transform
            )

        elif self.proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
            unlabeled_dataset = BasicDataset(
                self.semilearn_config,
                self.X.train_finetuning_unlabeled,
                torch.argmax(self.Y.train_finetuning_unlabeled, dim=1),
                self.semilearn_config.num_classes,
                semilearn_transform,
                is_ulb=True,
                strong_transform=semilearn_transform
            )

        if self.args.pretrain_and_finetune:
            finetune_dataset = BasicDataset(
                self.semilearn_config, 
                self.X.train_finetuning, 
                torch.argmax(self.Y.train_finetuning, dim=1), 
                self.semilearn_config.num_classes, 
                semilearn_transform, 
                is_ulb=False
            )

            finetune_unlabeled_dataset = BasicDataset(
                self.semilearn_config, 
                self.X.train_finetuning_unlabeled, 
                torch.argmax(self.Y.train_finetuning_unlabeled, dim=1), 
                self.semilearn_config.num_classes, 
                semilearn_transform, 
                is_ulb=True, 
                strong_transform=semilearn_transform
            )
            
        validation_dataset = BasicDataset(
            self.semilearn_config, 
            self.X.validation, 
            torch.argmax(self.Y.validation, dim=1), 
            self.semilearn_config.num_classes, 
            semilearn_transform, 
            is_ulb=False
        )

        test_dataset = self.CustomDataset(self.X.test, self.Y.test, transform=semilearn_transform)


        return labeled_dataset, unlabeled_dataset, finetune_dataset, finetune_unlabeled_dataset, validation_dataset, test_dataset

    def calculate_batch_size(self, unlabeled_dataset, labeled_dataset, is_finetuning=False):
        """
        Helper for set_loaders(). 
        """
    
        if is_finetuning:

            finetune_unlabeled_dataset = unlabeled_dataset
            finetune_dataset = labeled_dataset

            proportion_unlabeled_to_use = (
                len(finetune_unlabeled_dataset) / 
                (len(finetune_dataset) + len(self.finetune_unlabeled_dataset))
            )

            labeled_batch_size = int(self.semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
            unlabeled_batch_size = int(self.semilearn_config.batch_size * proportion_unlabeled_to_use)
            if labeled_batch_size + unlabeled_batch_size < self.semilearn_config.batch_size:
                if labeled_batch_size < unlabeled_batch_size:
                    labeled_batch_size += 1
                else:
                    unlabeled_batch_size += 1

            labeled_iters = self.args.epochs * super().ceildiv(len(finetune_dataset), labeled_batch_size)
            unlabeled_iters = self.args.epochs * super().ceildiv(len(self.finetune_unlabeled_dataset), unlabeled_batch_size)
            self.iters_for_loader = max(labeled_iters, unlabeled_iters)


        else: # Regular batch sizes 
            proportion_unlabeled_to_use = len(unlabeled_dataset) / (len(labeled_dataset) + len(unlabeled_dataset)) 
            labeled_batch_size = int(self.semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
            unlabeled_batch_size = int(self.semilearn_config.batch_size * proportion_unlabeled_to_use)
            if labeled_batch_size + unlabeled_batch_size < self.semilearn_config.batch_size:
                if labeled_batch_size < unlabeled_batch_size:
                    labeled_batch_size += 1
                else:
                    unlabeled_batch_size += 1

            labeled_iters = self.args.epochs * super().ceildiv(len(labeled_dataset), labeled_batch_size)
            unlabeled_iters = self.args.epochs * super().ceildiv(len(unlabeled_dataset), unlabeled_batch_size)

            self.iters_for_loader = max(labeled_iters, unlabeled_iters)

        return labeled_batch_size, unlabeled_batch_size, self.iters_for_loader

    def set_loaders(self):

        self.set_semilearn_config()

        labeled_dataset, unlabeled_dataset, finetune_dataset, validation_dataset, test_dataset = self.create_datasets()

        labeled_batch_size, unlabeled_batch_size, self.iters_for_loader = self.calculate_batch_size(unlabeled_dataset, labeled_dataset)
        
        self.train_labeled_loader = get_data_loader(
            self.semilearn_config, 
            labeled_dataset, 
            labeled_batch_size, 
            num_workers=multiprocessing.cpu_count() // 8, 
            num_epochs=self.args.epochs, 
            num_iters=self.iters_for_loader
        )

        if self.proportion_unlabeled_of_training_subjects>0 or self.proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
            self.train_unlabeled_loader = get_data_loader(
                self.semilearn_config, 
                unlabeled_dataset, 
                unlabeled_batch_size, 
                num_workers=multiprocessing.cpu_count() // 8,
                num_epochs=self.args.epochs, 
                num_iters=self.iters_for_loader
            )
            
        self.semilearn_config.num_train_iter = self.iters_for_loader
        self.semilearn_config.num_eval_iter = super().ceildiv(self.iters_for_loader, self.args.epochs)
        self.semilearn_config.num_log_iter = super().ceildiv(self.iters_for_loader, self.args.epochs)
        
        self.semilearn_algorithm = get_algorithm(self.semilearn_config, get_net_builder(self.semilearn_config.net, from_name=False), tb_log=None, logger=None)
        self.semilearn_algorithm.model = send_model_cuda(self.semilearn_config, self.semilearn_algorithm.model)
        self.semilearn_algorithm.ema_model = send_model_cuda(self.semilearn_config, self.semilearn_algorithm.ema_model, clip_batch=False)
        
        print("Batches per epoch:", self.semilearn_config.num_eval_iter)
            
        if self.args.pretrain_and_finetune:
            labeled_batch_size, unlabeled_batch_size, self.iters_for_loader = self.calculate_batch_size(self.finetune_unlabeled_dataset, labeled_dataset, is_finetuning=True)
            
            self.train_finetuning_loader = get_data_loader(
                self.semilearn_config, 
                finetune_dataset, 
                labeled_batch_size, 
                num_workers=multiprocessing.cpu_count() // 8,
                num_epochs=self.args.epochs, 
                num_iters=self.iters_for_loader
            )
            self.train_finetuning_unlabeled_loader = get_data_loader(
                self.semilearn_config, 
                self.finetune_unlabeled_dataset, 
                unlabeled_batch_size, 
                num_workers=multiprocessing.cpu_count() // 8,
                num_epochs=self.args.epochs, 
                num_iters=self.iters_for_loader
            )

        self.validation_loader = get_data_loader(
            self.semilearn_config, 
            validation_dataset, 
            self.semilearn_config.eval_batch_size, 
            num_workers=multiprocessing.cpu_count() // 8
        )

        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args.batch_size, 
            num_workers=multiprocessing.cpu_count() // 8, 
            worker_init_fn=self.utils.seed_worker, 
            pin_memory=True
        )

    

    def pretrain_model(self):
        print("Pretraining the model...")
        self.semilearn_algorithm.loader_dict = {}
        self.semilearn_algorithm.loader_dict['train_lb'] = self.train_labeled_loader
        if self.proportion_unlabeled_of_training_subjects>0 or self.proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
            self.semilearn_algorithm.loader_dict['train_ulb'] = self.train_unlabeled_loader
        self.semilearn_algorithm.loader_dict['eval'] = self.validation_loader
        self.semilearn_algorithm.scheduler = None

        self.semilearn_algorithm.train()

    def test_model(self):
        wandb.init(name=self.wandb_runname+"_unlab_test", project=self.project_name)
        ml_utils.evaluate_model_on_test_set(self.semilearn_algorithm.model, self.test_loader, self.device, self.num_gestures, self.criterion, self.args, self.testing_metrics)
        wandb.finish()

    def pretrain_and_finetune_model(self):
        
        print("Finetuning the model...")
        self.run = wandb.init(name=self.wandb_runname+"_unlab_finetune", project=self.project_name)
        wandb.config.lr = self.args.learning_rate
        
        self.semilearn_config_dict['num_train_iter'] = self.semilearn_config.num_train_iter + self.iters_for_loader
        self.semilearn_config_dict['num_eval_iter'] = super().ceildiv(self.iters_for_loader, self.args.finetuning_epochs)
        self.semilearn_config_dict['num_log_iter'] = super().ceildiv(self.iters_for_loader, self.args.finetuning_epochs)
        self.semilearn_config_dict['epoch'] = self.args.finetuning_epochs + self.args.epochs
        self.semilearn_config_dict['algorithm'] = self.args.unlabeled_algorithm
        
        self.semilearn_config = get_config(self.semilearn_config_dict)
        self.semilearn_algorithm = get_algorithm(self.semilearn_config, get_net_builder(self.semilearn_config.net, from_name=False), tb_log=None, logger=None)
        self.semilearn_algorithm.epochs = self.args.epochs + self.args.finetuning_epochs # train for the same number of epochs as the previous training
        self.semilearn_algorithm.model = send_model_cuda(self.semilearn_config, self.semilearn_algorithm.model)
        self.semilearn_algorithm.load_model(self.semilearn_config.load_path)
        self.semilearn_algorithm.ema_model = send_model_cuda(self.semilearn_config, self.semilearn_algorithm.ema_model, clip_batch=False)
        
        # Semilearn loaders
        self.semilearn_algorithm.loader_dict = {}
        self.semilearn_algorithm.loader_dict['train_lb'] = self.train_finetuning_loader
        self.semilearn_algorithm.scheduler = None
        
        if self.proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
            self.semilearn_algorithm.loader_dict['train_ulb'] = self.train_finetuning_unlabeled_loader
        elif self.proportion_unlabeled_of_training_subjects>0:
            self.semilearn_algorithm.loader_dict['train_ulb'] = self.train_unlabeled_loader

        self.semilearn_algorithm.loader_dict['eval'] = self.validation_loader
        self.semilearn_algorithm.train()

        wandb.init(name=self.wandb_runname+"_unlab_finetune_test", project=self.project_name)
        ml_utils.evaluate_model_on_test_set(self.semilearn_algorithm.model, self.test_loader, self.device, self.num_gestures, self.criterion, self.args, self.testing_metrics)

        wandb.finish()

    def model_loop(self):
        self.pretrain_model()
        _, _, self.testing_metrics = super().get_metrics()

        self.test_model()

        if self.args.pretrain_and_finetune:
            self.pretrain_and_finetune_model()

        self.train_and_validate_run()   

    