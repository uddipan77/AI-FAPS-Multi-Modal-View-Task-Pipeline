from typing import Any, Dict

import os
import torch
import wandb
import warnings
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, F1Score, CatMetric
from torchmetrics.classification.accuracy import Accuracy
from sklearn.metrics import classification_report

from src.utils.report_utils import classification_report_to_excel


class MultiModalModule(LightningModule):
    """
    MultiModalModule is a LightningModule that can be used to train a multimodal model.
    Currently, this module only supports late fusion of modalities.
    
    :param modal_nets: A dictionary containing the modalities as key and their respective networks as value.
    :param classifier: The classifier network that will be used for the late fusion.
    :param optimizer: The optimizer to be used for training.
    :param scheduler: The learning rate scheduler to be used for training.
    :param use_modalities: A list of modalities to be used for training. Only the modalities in this list will be used for training.
    :param modality_dummy_inputs: A dictionary containing the modality and their dummy input shape. Used for calculating the input features for the fusion network.
    :param classes: A list of classes for the classification task.
    
    TODO (Enhancement): Add support for early fusion and other fusion techniques.
    TODO (Enhancement): Add support for multiclass classification tasks.
    """

    def __init__(
        self,
        modal_nets: Dict[str, torch.nn.Module],
        classifier: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        use_modalities: list = None,
        modality_dummy_inputs: Dict[str, tuple] = {},
        classes: list = ['OK', 'NOK']
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.use_modalities = use_modalities
        self.classes = classes
        self.num_classes = len(classes)

        # Filter out the modalities that are not in the use_modalities list
        modal_nets = {k: v for k, v in modal_nets.items() if k in use_modalities}
        #modal_nets.items will give me key and value pair from modal_nets dict
        #use_modalities has images in it and modal_net-items() gives image (during image modality)
        #use modalities will have force during force modality
        #final modal_nets will have 'images': <TorchVisionWrapper instance with densenet121 configuration> as 
        #mentioned in densenet.yaml's modal_nets
        
        # Initialize the modal networks
        self.net = torch.nn.ModuleDict(modal_nets) #self.net is also a dict now whose value is a nn.module  model
        #torch.nn.ModuleDict takes a dictionary where each value should be a PyTorch nn.Module instance.
        #modal_nets has one key and its value is a dict but it does have the nn.module of torchvision_wrapper
        ## Now `self.net` will be a ModuleDict containing the 'images' modality with the TorchVisionWrapper instance
        #self.net['images'] will be an instance of TorchVisionWrapper that wraps around a densenet121 model with the specified parameters.
        #self.net(x) will call forward method of the wrapper
        #torch.nn.ModuleDict is a powerful utility for managing and organizing multiple submodules within a larger model.
        #here self.net is a submodule which will hold the densenet that is passed by the torchwrapper class
        #torch.nn.ModuleDict takes a dictionary where each value should be a PyTorch nn.Module instance.

        # Calculate the input features for the fusion network automatically
        input_features = self.calculate_fusion_input_features()
        print('Calculated fusion input features:', input_features)
        
        # Late fusion model
        self.late_fusion = classifier(input_features, self.num_classes)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # f1 score for validation
        task = 'binary' if self.num_classes == 2 else 'multiclass'
        self.val_f1 = F1Score(task=task, num_classes=self.num_classes)
        
        # for tracking best so far validation metrics
        self.val_f1_best = MaxMetric()
        self.val_acc_best = MaxMetric()
        
        # for confusion matrix
        self.test_preds = CatMetric()
        self.test_targets = CatMetric()

    def calculate_fusion_input_features(self):
        """Automatically calculate the input features for the fusion network.
        This is done by passing a dummy input through the modal networks and calculating the total number of features.

        :return: The total number of input features for the fusion network.
        :rtype: int
   
             """
        
        input_features = 0

        # Do not calculate gradients
        with torch.no_grad():
        
            for modality_name, net in self.net.items(): 
                
                dummy_input_modality = self.hparams.modality_dummy_inputs[modality_name]
                
                # If the modality is a list, pass a list of dummy inputs
                # Used for modalities like images where the input is a list of images per sample
                if dummy_input_modality.type == 'list':
                    
                    dummy_input = [torch.randn(tuple(dummy_input_modality.shape)) for _ in range(dummy_input_modality.length)]
                    
                    for d in dummy_input:
                        output = net(d)
                        print('-->', modality_name, output.shape)
                        input_features += output.numel() // output.size(0)  # Total features for one sample in the batch
                        #the main reson for the length as 4 is cause here we have 4 images for a sample, thus making it multiview
                        #so each sample is made of 4 images and each sample has shape (4,3,320,320)
                        #but densenet will take this as a batch of 4 images where each image is (3,320,320)
                        #so I get 1024 features as per as the above calculation output.numel() // output.size(0)
                        #but here 4 is technically batch size but in real it is one single sample so the calculation gives 1024 which is for one image
                        #and i need for 4 so i do the addition of input_features looping over 4 tensors
                        #no matter the batch size each sample has 1024 * 4, 4096 features
                
                # If the modality is not a list, pass a single dummy input
                else:
                
                    dummy_input = torch.randn(tuple(dummy_input_modality.shape))
                    #This will create a tensor filled with random numbers with the shape specified in the shape field of the dummy input.
                    output = net(dummy_input)
                    print('-->', modality_name, output.shape)
                    input_features += output.numel() // output.size(0)  # Total features for one sample in the batch

        return input_features

    def forward(self, **modalities):
        """Forward pass of the multimodal model.

        :return: Output of the multimodal model after late fusion.
        """
        
        modality_outputs = []
        
        # Pass the modalities through their respective networks
        for modality_name, net in self.net.items():
            
            # If the modality is a list, iterate through the list and pass each data through the network
            if isinstance(modalities[modality_name], list):
                for data in modalities[modality_name]:
                    out = net(data) #data: A single sample tensor with shape [4, 3, 320, 320] (4 images).
                    #out is of shape (4,1024)
                    modality_outputs.append(out) #: A list with 32 tensors, each of shape [4, 1024],  have shape (32,4,1024)
                    
            
            # If the modality is not a list, pass the data through the network
            else:
                out = net(modalities[modality_name]) #only that specific data is passed, either image or force or force features
                modality_outputs.append(out)
        
        # Late fusion
        #need to pass modality_outputs in form of (batch size, 4096) 
        # The key here is in how you pass modality_outputs to the LateFusion layer:
        # *modality_outputs:This unpacks the list modality_outputs into 32 separate tensors, each with shape [4, 1024] 
        out = self.late_fusion(*modality_outputs) 
        
        return out

    def on_train_start(self):
        """Called when the train begins."""
        
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_f1.reset()
        self.val_f1_best.reset()
        self.test_preds.reset()
        self.test_targets.reset()
        
        self.logger_name = self.logger.__class__.__name__ if self.logger else None

    def model_step(self, batch: Any):
        """Common step for training, validation, and testing."""
        
        # forward pass
        #remember how batch was a dict from getitem of dataset
        logits = self.forward(**batch)
        
        # calculate loss
        loss = self.criterion(logits, batch['labels'])
        
        # calculate predictions
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, batch['labels']

    def training_step(self, batch: Any, batch_idx: int):
        """Training step."""
        
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        """Called when the train epoch ends."""
        torch.cuda.empty_cache()

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step."""
        
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """Called when the validation epoch ends."""
        
        f1 = self.val_f1.compute()  # get current val f1
        self.val_f1_best(f1) # update best so far val f1
        
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)
            
    def on_test_start(self):
        """Called when the test begins."""
        self.logger_name = self.logger.__class__.__name__ if self.logger else None

    def test_step(self, batch: Any, batch_idx: int):
        """Test step."""
        
        loss, preds, targets = self.model_step(batch)

        # store predictions and targets for confusion matrix
        self.test_preds.update(preds)
        self.test_targets.update(targets)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        """Called when the test epoch ends."""
        pass

    def on_test_end(self):
        """Called when the test ends."""
        
        # Get the test targets and predictions, move to CPU and convert to list
        test_targets = self.test_targets.compute().detach().cpu().tolist()
        test_preds = self.test_preds.compute().detach().cpu().tolist()
        
        # Calculate the classification report
        report = classification_report(
            test_targets,
            test_preds,
            zero_division=1
        )
        
        # Print the classification report
        print(report)
        
        # Save the classification report to an Excel file
        output_file = os.path.join(self.trainer.default_root_dir, 'classification_report.xlsx')
        report_df = classification_report_to_excel(test_targets, test_preds, classes=self.classes, output_file=output_file, logger_name=self.logger_name)
        
        if self.logger_name and 'wandb' in self.logger_name.lower():
            # Log confusion matrix to Weights & Biases
            self.logger.experiment.log({'Confusion Matrix': wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_targets,
                preds=test_preds,
                class_names=self.classes
            )})
        
            # Log summary metrics to Weights & Biases
            # TODO (Enhancement): Make this more generic for non-binary classification tasks
            nok_metrics = report_df.loc['1.0']
            nok_metrics.index = 'metrics/' + nok_metrics.index + '_NOK'
            for k, v in nok_metrics.to_dict().items():
                self.logger.experiment.summary[k] = v
            
        else:
            warnings.warn('No logger found. Classification report and confusion matrix will not be logged.')

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MultiModalModule(None, None, None)
