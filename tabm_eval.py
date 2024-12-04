import math
import random
import warnings
from typing import Literal, NamedTuple

import pandas as pd
import numpy as np
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm

import csv

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

warnings.simplefilter('ignore')
from tabm_reference import Model, make_parameter_groups

warnings.resetwarnings()


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # Load the uploaded data files
    inputs_path = 'inputs.csv'
    labels_path = 'labels.csv'

    inputs_df = pd.read_csv(inputs_path)
    labels_df = pd.read_csv(labels_path)

    data = inputs_df.merge(labels_df, on="PatientID")

    # Separate features and target
    X = data.drop(columns=["PatientID", "HadHeartAttack"])
    y = data["HadHeartAttack"]
    patient_ids = data['PatientID']

    # Display the first few rows of the data to understand its structure

    binary_columns = [
        'Sex', 'HadAngina', 'HadStroke',
        'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
        'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
        'DifficultyConcentrating', 'DifficultyWalking',
        'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers',
        'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear', 'CovidPos'
    ]

    for col in binary_columns:
        assert len(X[col].unique()) == 2

    cat_columns = [
        'State', 'GeneralHealth', 'AgeCategory',
        'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage',
        'RaceEthnicityCategory', 'TetanusLast10Tdap'
    ]

    for col in cat_columns:
        print(col, len(X[col].unique()))

    num_columns = ['HeightInMeters', 'WeightInKilograms', 'BMI']

    for col in num_columns:
        print(col, X[col].min(), X[col].max())

    X_cont = X.loc[:, num_columns].to_numpy().astype(np.float32)
    X_cat = X.loc[:, binary_columns + cat_columns]

    for col in binary_columns + cat_columns:
        X_cat[col], _ = pd.factorize(X_cat[col])

    X_cat = X_cat.to_numpy()
    Y, _ = pd.factorize(y)
    Y = Y.astype(np.int64)

    task_type = "classification"

    cat_cardinalities = [x + 1 for x in X_cat.max(axis=0)]
    n_cont_features = X_cont.shape[1]
    n_classes = 2

    all_idx = np.arange(len(Y))

    data_numpy = {
        'train': {'x_cont': X_cont, 'y': Y}
    }
    if X_cat is not None:
        data_numpy['train']['x_cat'] = X_cat

    preprocessing = sklearn.preprocessing.StandardScaler().fit(
        data_numpy['train']['x_cont']
    )

    for part in data_numpy:
        data_numpy[part]['x_cont'] = preprocessing.transform(data_numpy[part]['x_cont'])

    Y_train = data_numpy['train']['y'].copy()
    regression_label_stats = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors
    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }
    Y_train = torch.as_tensor(Y_train, device=device)
    if task_type == 'regression':
        for part in data:
            data[part]['y'] = data[part]['y'].float()
        Y_train = Y_train.float()

    # Automatic mixed precision (AMP)
    # torch.float16 is implemented for completeness,
    # but it was not tested in the project,
    # so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    # Changing False to True will result in faster training on compatible hardware.
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

    # torch.compile
    compile_model = False

    # fmt: off
    print(
        f'Device:        {device.type.upper()}'
        f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
        f'\ntorch.compile: {compile_model}'
    )
    # fmt: on


    # TabM
    arch_type = 'tabm'
    bins = None
    k = 32

    # TabM-mini with the piecewise-linear embeddings.
    # arch_type = 'tabm-mini'
    # bins = rtdl_num_embeddings.compute_bins(data['train']['x_cont'])

    model = Model(
        n_num_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        backbone={
            'type': 'MLP',
            'n_blocks': 3 if bins is None else 2,
            'd_block': 512,
            'dropout': 0.1,
        },
        bins=bins,
        num_embeddings=(
            None
            if bins is None
            else {
                'type': 'PiecewiseLinearEmbeddings',
                'd_embedding': 16,
                'activation': False,
                'version': 'B',
            }
        ),
        arch_type=arch_type,
        k=k,
    ).to(device)


    weights_path = "sergey_tabm_model.pth"
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    
    swa_model = AveragedModel(model)

    optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=3e-4) #was 3e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    swa_scheduler = SWALR(optimizer, swa_lr=1.8e-3)

    if compile_model:
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument
        # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode
    


    def find_best_threshold(y_true, y_prob, metric=sklearn.metrics.f1_score):
        """
        Find the best threshold for a binary classification task based on a given metric.

        Parameters:
        - y_true: array-like, shape (n_samples,)
        True binary labels (0 or 1).
        - y_prob: array-like, shape (n_samples,)
        Predicted probabilities for the positive class.
        - metric: callable
        The metric to optimize. Default is F1-score.

        Returns:
        - best_threshold: float
        The threshold that maximizes the specified metric.
        - best_score: float
        The best metric score.
        """
        best_threshold = 0.0
        best_score = 0.0

        # Search thresholds from 0.0 to 1.0 with small steps
        thresholds = np.linspace(0, 1, 25)

        for threshold in thresholds:
            # Convert probabilities to binary predictions
            y_pred = (y_prob >= threshold).astype(int)

            # Compute the metric
            score = metric(y_true, y_pred)

            # Update best threshold if the current score is better
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score
    
    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(model, part: str, idx: Tensor) -> Tensor:
        pred = model(
            data[part]['x_cont'][idx],
            data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
        )
        # print(pred.shape, pred[..., 0].shape)
        if task_type != "regression":
            return pred.float()

        return (
            pred
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )


    if task_type == "regression":
        base_loss_fn = F.mse_loss
    elif n_classes == 2:
        base_loss_fn = F.binary_cross_entropy_with_logits # lambda x, y: F.binary_cross_entropy_with_logits(x, y, reduction='none')
    else:
        base_loss_fn = F.cross_entropy

    def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none', pos_weight=torch.tensor(2.))
        pt = torch.exp(-bce_loss)  # Probability of correct class
        loss = alpha * (1 - pt)**gamma * bce_loss
        return loss.mean()

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions per object. Each of them must be trained separately.
        # (regression)     y_pred.shape == (batch_size, k)
        # (classification) y_pred.shape == (batch_size, k, n_classes)
        if task_type == 'regression':
            k = y_pred.shape[-1]
        else:
            k = y_pred.shape[-2]

        y_pred = y_pred[..., 1]
        # print(y_pred.shape)

        # return base_loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(k))
        # print(k, y_pred.shape, y_pred.flatten(0, 1).shape, y_true.repeat_interleave(k).shape)
        return focal_loss(y_pred.flatten(0, 1), y_true.repeat_interleave(k)) #torch.tensor(10).to(y_pred.device)
        # return base_loss_fn(
        #     y_pred.flatten(0, 1), y_true.repeat_interleave(k),
        #     pos_weight=pos_weight #torch.tensor(10).to(y_pred.device)
        # )


    @evaluation_mode()
    def evaluate(model, part: str) -> float:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 8096
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(model, part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        
        y_pred = scipy.special.softmax(y_pred, axis=-1)
        # print(y_pred.shape)
        y_pred = y_pred.mean(1)
        # print(y_pred.shape)
        return list(map(lambda x: int(x), y_pred[..., -1] > 0.5))

    y_pred = evaluate(model, part='train')
    print(y_pred)

    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
    
    with open('output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        for patient_id, pred in zip(patient_ids, y_pred):
            writer.writerow([patient_id, pred])


if __name__ == "__main__":
    main()
