#!/usr/bin/env python
# coding: utf-8

from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import  StratifiedKFold

from xLSTM_model import xLSTMModel

from packaging.version import Version
assert Version(torch.__version__) >= Version("2.2.0"), \
    f"PyTorch version must be at least 2.2.0 but found {torch.__version__}"



def _make_riskset(time: torch.Tensor) -> torch.Tensor:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : torch.Tensor, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : torch.Tensor, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndimension() == 1, "expected 1D tensor"

    # sort in descending order
    o = torch.argsort(-time)
    n_samples = len(time)
    risk_set = torch.zeros((n_samples, n_samples), dtype=torch.bool)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set


def my_collate_fn(batch):
    batch_images, batch_time, batch_event = zip(*batch)

    # Convert batch data to tensors
    batch_images = torch.stack(batch_images)
    batch_time = torch.tensor(batch_time)
    batch_event = torch.tensor(batch_event)

    # Compute the risk set
    risk_set = _make_riskset(batch_time)

    # Create labels dictionary
    labels = {
        "label_event": batch_event,
        "label_time": batch_time,
        "label_riskset": risk_set
    }
    
    return batch_images, labels



class InputFunction(Dataset):
    """Callable input function that computes the risk set for each batch.
    
    Parameters
    ----------
    images : np.ndarray, shape=(n_samples, height, width)
        Image data.
    time : np.ndarray, shape=(n_samples,)
        Observed time.
    event : np.ndarray, shape=(n_samples,)
        Event indicator.
    batch_size : int, optional, default=64
        Number of samples per batch.
    drop_last : int, optional, default=False
        Whether to drop the last incomplete batch.
    shuffle : bool, optional, default=False
        Whether to shuffle data.
    seed : int, optional, default=89
    """

    def __init__(self,
                 images: np.ndarray,
                 time: np.ndarray,
                 event: np.ndarray,
                 batch_size: int = 64,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 seed: int = 89) -> None:
        if images.ndim == 3:
            images = images[..., np.newaxis]
        # Remove the last singleton dimension
        images = images.squeeze(-1)

        self.images = torch.tensor(images, dtype=torch.float32)
        self.time = torch.tensor(time, dtype=torch.float32)
        self.event = torch.tensor(event, dtype=torch.int32)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return self.images[index], self.time[index], self.event[index]
    
    def get_dataloader(self):
        """Returns a DataLoader for the dataset."""
        return DataLoader(self, batch_size=self.batch_size, collate_fn=my_collate_fn, shuffle=self.shuffle, drop_last=self.drop_last)


def safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = torch.min(x, dim=0)[0]
    c = torch.zeros_like(x_min)
    norm = torch.where(x_min < 0, -x_min, c)
    return x + norm


def logsumexp_masked(risk_scores: torch.Tensor,
                     mask: torch.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> torch.Tensor:
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    assert risk_scores.dim() == mask.dim(), "Tensors must have the same number of dimensions"

    mask_f = mask.to(dtype=risk_scores.dtype)
    risk_scores_masked = risk_scores * mask_f
    
    # For numerical stability, subtract the maximum value before taking the exponential
    amax = torch.max(risk_scores_masked, dim=axis, keepdim=True).values
    risk_scores_shift = risk_scores_masked - amax

    exp_masked = torch.exp(risk_scores_shift) * mask_f
    exp_sum = torch.sum(exp_masked, dim=axis, keepdim=True)
    output = amax + torch.log(exp_sum)
    if not keepdims:
        output = output.squeeze(dim=axis)
    return output



class CoxPHLoss(nn.Module):
    """Negative partial log-likelihood of Cox's proportional hazards model."""

    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self,
                y_true: Sequence[torch.Tensor],
                y_pred: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_true : list|tuple of torch.Tensor
            The first element holds a binary vector where 1
            indicates an event 0 censoring.
            The second element holds the riskset, a
            boolean matrix where the `i`-th row denotes the
            risk set of the `i`-th instance, i.e. the indices `j`
            for which the observer time `y_j >= y_i`.
            Both must be rank 2 tensors.
        y_pred : torch.Tensor
            The predicted outputs. Must be a rank 2 tensor.

        Returns
        -------
        loss : torch.Tensor
            Loss for each instance in the batch.
        """
        event, riskset = y_true
        predictions = y_pred

        if predictions.ndim != 2:
            raise ValueError(f"Rank mismatch: Rank of predictions (received {predictions.ndim}) should be 2.")

        if predictions.size(1) != 1:
            raise ValueError(f"Dimension mismatch: Last dimension of predictions (received {predictions.size(1)}) must be 1.")

        if event.ndim != predictions.ndim:
            raise ValueError(f"Rank mismatch: Rank of predictions (received {predictions.ndim}) should equal rank of event (received {event.ndim}).")

        if riskset.ndim != 2:
            raise ValueError(f"Rank mismatch: Rank of riskset (received {riskset.ndim}) should be 2.")

        event = event.to(dtype=predictions.dtype)
        predictions = safe_normalize(predictions)

        # Ensure assertions are in place (omitted for brevity in PyTorch, can use assert statements or checks)
        assert torch.all(event <= 1.0), "All elements in event should be <= 1.0"
        assert torch.all(event >= 0.0), "All elements in event should be >= 0.0"
        assert riskset.dtype == torch.bool, "Riskset should be of boolean type"

        # Move batch dimension to the end so predictions get broadcast row-wise when multiplying by riskset
        pred_t = predictions.t()
        # Compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape == predictions.shape, "Shapes of rr and predictions must match"

        losses = event * (rr - predictions)

        return losses



class CindexMetric:
    """Computes concordance index across one epoch."""

    def __init__(self):
        self.reset_states()

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    def update_state(self, y_true: Dict[str, torch.Tensor], y_pred: torch.Tensor) -> None:
        """Collect observed time, event indicator and predictions for a batch.

        Parameters
        ----------
        y_true : dict
            Must have two items:
            `label_time`, a tensor containing observed time for one batch,
            and `label_event`, a tensor containing event indicator for one batch.
        y_pred : torch.Tensor
            Tensor containing predicted risk score for one batch.
        """
        self._data["label_time"].append(y_true["label_time"].cpu().numpy())
        self._data["label_event"].append(y_true["label_event"].cpu().numpy())
        self._data["prediction"].append(y_pred.squeeze().cpu().numpy())

    def result(self) -> Dict[str, float]:
        """Computes the concordance index across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)

        results = concordance_index_censored(
            data["label_event"] == 1,
            data["label_time"],
            data["prediction"])

        result_data = {}
        names = ("cindex", "concordant", "discordant", "tied_risk")
        for k, v in zip(names, results):
            result_data[k] = v
        return result_data



class TrainAndEvaluateModel:

    def __init__(self, model, model_dir, train_dataset, eval_dataset, learning_rate, num_epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.num_epochs = num_epochs
        self.model_dir = model_dir

        self.train_ds = train_dataset
        self.val_ds = eval_dataset

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = CoxPHLoss()

        self.train_loss_metric = nn.MSELoss()  # Placeholder, replace with appropriate metric
        self.val_loss_metric = nn.MSELoss()  # Placeholder, replace with appropriate metric
        self.val_cindex_metric = CindexMetric()

        self.train_summary_writer = SummaryWriter(log_dir=f"{self.model_dir}/train")
        self.val_summary_writer = SummaryWriter(log_dir=f"{self.model_dir}/valid")

    def train_one_step(self, x, y_event, y_riskset):
        x = x.to(self.device)
        y_event = y_event.to(self.device).unsqueeze(1)
        y_riskset = y_riskset.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(x)
        train_loss = self.loss_fn(y_true=(y_event, y_riskset), y_pred=logits)
        #print(train_loss.shape)

        # Aggregate the loss to a scalar by taking the mean
        train_loss = train_loss.mean()
        #print("Aggregated train_loss shape:", train_loss.shape)

        train_loss.backward()
        self.optimizer.step()
        
        return train_loss.item(), logits

    def train_and_evaluate(self):
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.evaluate(epoch)
            
            # Save model checkpoint
            torch.save(self.model.state_dict(), f"{self.model_dir}/model_epoch_{epoch}.pth")

    def train_one_epoch(self, epoch):
        total_loss = 0.0
        steps = 0

        for step, (x, y) in enumerate(self.train_ds):
            #print(f"Shape of x: {x.shape}")
            train_loss, logits = self.train_one_step(x, y["label_event"], y["label_riskset"])
            
            # Update total loss and step count
            total_loss += train_loss
            steps += 1

            # Log every 10 batches.
            if step % 10 == 0:
                mean_loss = total_loss / steps
                print(f"Epoch {epoch}, step {step}: mean loss = {mean_loss:.4f}")
                
                self.train_summary_writer.add_scalar("loss", mean_loss, epoch * len(self.train_ds) + step)
                
                # Reset training metrics
                total_loss = 0.0
                steps = 0

    def evaluate_one_step(self, x, y_event, y_riskset):
        x = x.to(self.device)
        y_event = y_event.to(self.device).unsqueeze(1)
        y_riskset = y_riskset.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            val_logits = self.model(x)
            val_loss = self.loss_fn(y_true=(y_event, y_riskset), y_pred=val_logits)

            #print(val_loss.shape)

            # Aggregate the loss to a scalar by taking the mean
            val_loss = val_loss.mean()
            #print("Aggregated val_loss shape:", val_loss.shape)

        
        return val_loss.item(), val_logits

    def evaluate(self, epoch):
        total_val_loss = 0.0
        steps = 0
        self.val_cindex_metric.reset_states()

        for x_val, y_val in self.val_ds:
            val_loss, val_logits = self.evaluate_one_step(x_val, y_val["label_event"], y_val["label_riskset"])

            # Update total validation loss and step count
            total_val_loss += val_loss
            steps += 1

            # Update val metrics
            self.val_cindex_metric.update_state(y_val, val_logits)

        mean_val_loss = total_val_loss / steps
        self.val_summary_writer.add_scalar("loss", mean_val_loss, epoch)

        val_cindex = self.val_cindex_metric.result()
        for key, value in val_cindex.items():
            self.val_summary_writer.add_scalar(key, value, epoch)

        print(f"Validation: Epoch {epoch}, loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")



def normalize(df):
    return (df - df.min()) / (df.max() - df.min())



# Load data
data_path = 'data/temporal'
X = pd.read_csv(f"{data_path}/X5_all_arti.csv")
X = X.iloc[:, 1:]  # Drop the first column if it is an index
print(X.shape)

y = pd.read_csv(f"{data_path}/Y5_e_arti.csv")
y = y.iloc[:, 1:]  # Drop the first column if it is an index


rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=412)
index = 1
features_per_folds = []

for train_index, test_index in rskf.split(X,y.iloc[:,0]):
    index += 1

    # Splitting the data
    X_train, X_test = X.iloc[train_index,2:54], X.iloc[test_index,2:54]
    y_train, y_test = X.iloc[train_index,54:56], X.iloc[test_index,54:56]

    t = 5
    dim1 = (int(X_train.shape[0] / t) + 1)
    dim2 = (int(X_test.shape[0] / t) + 1)
    
    data_new = np.zeros((dim1, 5, 52))
    data_new_test = np.zeros((dim2, 5, 52))
    y_new = np.zeros((dim1, 2))
    y_new_test = np.zeros((dim2, 2))

    # Convert training data to lists
    X_train0 = X_train.values
    X_train_lists = [X_train0[i:i + t] for i in range(0, len(X_train0), t)]
    X_train_lists.pop()
    X_train_lists_array = np.asarray(X_train_lists)

    y_train0 = y_train.values
    y_train_lists = [y_train0[i:i + t] for i in range(0, len(y_train0), t)]
    y_train_lists.pop()
    y_train_lists_array = np.asarray(y_train_lists)


    # Convert testing data to lists
    X_test0 = X_test.values
    X_test_lists = [X_test0[i:i + t] for i in range(0, len(X_test0), t)]
    X_test_lists.pop()
    X_test_lists_array = np.asarray(X_test_lists)

    y_test0 = y_test.values
    y_test_lists = [y_test0[i:i + t] for i in range(0, len(y_test0), t)]
    y_test_lists.pop()
    y_test_lists_array = np.asarray(y_test_lists)

    # Fill the arrays with the lists
    for i in range(int(X_train.shape[0] / t)):
        data_new[i] = X_train_lists_array[i]
        #print(data_new.shape)
        y_new[i] = y_train_lists_array[i][1]

    for i in range(int(X_test.shape[0] / t)):
        data_new_test[i] = X_test_lists_array[i]
        #print(data_new_test.shape)
        y_new_test[i] = y_test_lists_array[i][1]


    #print(f"Shape of data_new: {data_new.shape}")
    #print(f"Shape of data_new_test: {data_new_test.shape}")
    
    model = xLSTMModel(input_dim=52, output_dim=1)

    train_fn = InputFunction(data_new, np.array(y_new[:,0]), np.array(y_new[:,1]), drop_last=True, shuffle=True)
    train_loader = train_fn.get_dataloader()

    eval_fn = InputFunction(data_new_test, np.array(y_new_test[:,0]), np.array(y_new_test[:,1]))
    eval_loader = eval_fn.get_dataloader()

    trainer = TrainAndEvaluateModel(
        model=model,
        model_dir=Path("ckpts"),
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        learning_rate=0.000001,
        num_epochs=2,
    )


    trainer.train_and_evaluate()


   
