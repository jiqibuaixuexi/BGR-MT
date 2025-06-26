import argparse
import os
import random
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

import wandb
from utils.datasets import BalancedBatchSampler, MyCombinedDataset, Mydataset
from utils.modified_models import (
    CustomPoolformerForMultiLabelImageClassification,
    CustomMetaformerForMultiLabelImageClassificationTriHFF,
)
from utils.utils_v2 import ImgsData, padding_img, preprocess_imgs_batch, preprocess_labels_batch


parser = argparse.ArgumentParser(
    description="Semi-supervised Multi-label Image Classification with GAT"
)


parser.add_argument("--project", type=str, default="AS", help="WandB project name")
parser.add_argument(
    "--name",
    type=str,
    default="poolformer_m48_TriHFF_semi_plus_fake_GAT_no_diff_no_distill-fold0",
    help="WandB run name",
)
parser.add_argument("--base_dir", type=str, default="../", help="Base directory")
parser.add_argument(
    "--trainvaldataset_dir",
    type=str,
    default="../cv_datasets/csv_data/fold0_8_2_test_plus_fake",
    help="Train/val dataset directory",
)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--temperature", type=float, default=5, help="Temperature for soft target")
parser.add_argument(
    "--lambda_param", type=float, default=0.5, help="Lambda parameter for consistency loss"
)
parser.add_argument("--consistency_type", type=str, default="kl", help="Consistency loss type")
parser.add_argument(
    "--consistency_relation_weight", type=float, default=1, help="Consistency relation loss weight"
)
parser.add_argument("--alpha", type=float, default=0.99, help="Alpha for EMA model")
parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--use_semi", type=str, default="True", help="Use semi-supervised learning")
parser.add_argument("--num_labeled", type=int, default=5092, help="Number of labeled samples")
parser.add_argument("--use_GAT_CELoss", type=bool, default=True, help="Use GAT cross-entropy loss")
parser.add_argument("--gat_loss_weight", type=float, default=0.5, help="GAT loss weight")
parser.add_argument(
    "--learning_rate_model", type=float, default=5e-5, help="Learning rate for model"
)
parser.add_argument("--consistency_rampup", type=float, default=20, help="consistency_rampup")
parser.add_argument("--consistency", type=float, default=1, help="consistency")
parser.add_argument(
    "--semi_epoch", type=int, default=5, help="When to start semi-supervised learning"
)
parser.add_argument("--gat_stop_epoch", type=int, default=20, help="When to end GNN learning")
parser.add_argument(
    "--knn_k", type=int, default=4, help="The number of neighbors to consider for KNN"
)
parser.add_argument(
    "--backbone", type=str, default="poolformer", help="The backbone to use for the model"
)


args = parser.parse_args()

if args.use_semi.lower() == "true":
    args.use_semi_bool = True
elif args.use_semi.lower() == "false":
    args.use_semi_bool = False
else:
    raise ValueError(f"Invalid value for --use_semi: {args.use_semi}. Must be 'True' or 'False'.")


wandb.init(
    project=args.project,
    name=args.name,
    config={
        "backbone": args.backbone,
        "base_dir": args.base_dir,
        "trainvaldataset_dir": args.trainvaldataset_dir,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "lambda_param": args.lambda_param,
        "consistency_type": args.consistency_type,
        "consistency_rampup": args.consistency_rampup,
        "consistency": args.consistency,
        "consistency_relation_weight": args.consistency_relation_weight,
        "alpha": args.alpha,
        "num_epochs": args.num_epochs,
        "use_semi": args.use_semi_bool,
        "semi_epoch": args.semi_epoch,
        "gat_stop_epoch": args.gat_stop_epoch,
        "num_labeled": args.num_labeled,
        "use_GAT_CELoss": args.use_GAT_CELoss,
        "gat_loss_weight": args.gat_loss_weight,
        "knn_k": args.knn_k,
    },
)

config = wandb.config

# Arg
base_dir = config.base_dir
trainvaldataset_dir = config.trainvaldataset_dir
batch_size = config.batch_size
train_dir = os.path.join(trainvaldataset_dir, "train")
train_csv = os.path.join(train_dir, "metadata.csv")
val_dir = os.path.join(trainvaldataset_dir, "val")
val_csv = os.path.join(val_dir, "metadata.csv")
test_dir = os.path.join(trainvaldataset_dir, "test")
test_csv = os.path.join(test_dir, "metadata.csv")

output_dir = f"output/{args.name}"
output_name = "model_epoch_{}.pth"

os.makedirs(output_dir, exist_ok=True)


from utils.Augmentations import RandAugment

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# train_transforms.transforms.insert(1, RandAugment(3, 10))

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = MyCombinedDataset(
    train_csv, num_labeled=config.num_labeled, transform=train_transforms
)
print("Train num: ", len(train_dataset))
val_dataset = Mydataset(val_csv, transform=val_transforms)
print("Validation num: ", len(val_dataset))
test_dataset = Mydataset(test_csv, transform=val_transforms)
print("Test num: ", len(test_dataset))


def collate_fn_train(batch):
    images1 = [item[0] for item in batch]
    combined_images1 = [item[1] for item in batch]
    images2 = [item[2] for item in batch]
    combined_images2 = [item[3] for item in batch]
    labels = [item[4] for item in batch]

    images_batch1 = torch.stack(images1)
    combined_images_batch1 = torch.stack(combined_images1)
    images_batch2 = torch.stack(images2)
    combined_images_batch2 = torch.stack(combined_images2)

    labels_batch = torch.stack(labels)

    return {
        "imgs_batch1": ImgsData(images_batch1.to("cuda"), combined_images_batch1.to("cuda")),
        "imgs_batch2": ImgsData(images_batch2.to("cuda"), combined_images_batch2.to("cuda")),
        "labels": labels_batch.to("cuda"),
    }


# 定义collate_fn
def collate_fn(batch):
    images = [item[0] for item in batch]
    combined_images = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    images_batch = torch.stack(images)
    combined_images_batch = torch.stack(combined_images)
    labels_batch = torch.stack(labels)

    return {
        "imgs_batch": ImgsData(images_batch.to("cuda"), combined_images_batch.to("cuda")),
        "labels": labels_batch.to("cuda"),
    }


labeled_data = train_dataset.labeled_data
labeled_indices = list(range(len(labeled_data)))
unlabeled_indices = list(range(len(labeled_data), len(train_dataset)))
print(f"Labeled Samples: {len(labeled_indices)}")
print(f"Unlabeled Samples: {len(unlabeled_indices)}")

sampler = BalancedBatchSampler(labeled_indices, unlabeled_indices, batch_size)


train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate_fn_train)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
temperature = config.temperature
lambda_param = config.lambda_param


def softmax_mse_loss(input_logits, target_logits, class_weight):
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_softmax) ** 2 * class_weight
    return mse_loss.sum()


def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction="none")
    return kl_div.sum()


def relation_mse_loss(activations, ema_activations):
    activations = activations.reshape(activations.size(0), -1)
    ema_activations = ema_activations.reshape(ema_activations.size(0), -1)

    similarity = torch.mm(activations, activations.t())
    norm = torch.norm(similarity, p=2, dim=1, keepdim=True)
    norm_similarity = similarity / norm

    ema_similarity = torch.mm(ema_activations, ema_activations.t())
    ema_norm = torch.norm(ema_similarity, p=2, dim=1, keepdim=True)
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2
    return similarity_mse_loss.sum()


consistency_type = config.consistency_type
if consistency_type == "mse":
    consistency_criterion = softmax_mse_loss
elif consistency_type == "kl":
    consistency_criterion = softmax_kl_loss

consistency_relation_weight = config.consistency_relation_weight

from utils import ramps


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, config.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data * alpha + param.data * (1 - alpha)


alpha = config.alpha
global_step = 0


def create_model(net, ema=False):
    model = net
    if ema:
        for param in model.parameters():
            param.detach()
    return model


model = create_model(
    CustomMetaformerForMultiLabelImageClassificationTriHFF(backbone=args.backbone), ema=False
)
ema_model = create_model(
    CustomMetaformerForMultiLabelImageClassificationTriHFF(backbone=args.backbone), ema=True
)


import torch
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GAT, self).__init__()
        self.conv1 = geom_nn.GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = geom_nn.GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


knn_k = args.knn_k


def build_knn_adjacency(correlation_matrix, k):
    batch_size = correlation_matrix.size(0)

    _, topk_indices = torch.topk(correlation_matrix, k=k + 1, dim=1)

    topk_indices = topk_indices[:, 1:]

    source_nodes = (
        torch.arange(batch_size, device=correlation_matrix.device)
        .unsqueeze(1)
        .repeat(1, k)
        .flatten()
    )
    target_nodes = topk_indices.flatten()
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    return edge_index


def gat_cross_entropy_loss(gat_output_student, gat_output_teacher, temperature=1.0):
    pred_probs = F.log_softmax(gat_output_student / temperature, dim=-1)
    target_probs = F.softmax(gat_output_teacher / temperature, dim=-1)
    kl_loss = nn.KLDivLoss(reduction="batchmean")(pred_probs, target_probs)

    return kl_loss


gat = GAT(768, 512, 256)
gat.to("cuda")


optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_model)
optimizer_gat = optim.Adam(gat.parameters(), lr=1e-4)


def compute_metrics(eval_pred):
    logits = eval_pred["predictions"]
    labels = eval_pred["label_ids"]

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = (logits > 0.5).astype(int)

    accuracy = accuracy_score(labels, predictions)
    hamming_accuracy = 1 - hamming_loss(labels, predictions)
    f1 = f1_score(labels, predictions, average="samples", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "hamming_accuracy": hamming_accuracy,
    }


def seed_torch(seed=1):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "0"
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

use_semi = config.use_semi
use_GAT_CELoss = config.use_GAT_CELoss
num_epochs = config.num_epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ema_model.to(device)
if use_semi:
    print("Use semi-supervised learning method...")
else:
    print("Use supervised learning method...")


loss = 0
student_target_loss = 0
consistency_loss = 0
consistency_relation_loss = 0
gat_kl_loss = 0

for epoch in range(num_epochs):
    model.train()
    ema_model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        imgs_batch1 = batch["imgs_batch1"]
        imgs_batch2 = batch["imgs_batch2"]
        labels = batch["labels"]

        student_output = model(imgs_batch1, labels)
        with torch.no_grad():
            teacher_output = ema_model(imgs_batch2, labels)

        if len(train_dataset.unlabeled_data) == 0:
            student_target_loss = criterion(student_output.logits, labels)
        else:
            student_target_loss = criterion(
                student_output.logits[: batch_size // 2, :], labels[: batch_size // 2, :]
            )
        if use_semi:
            consistency_weight = get_current_consistency_weight(epoch)
            # print("use semi")

            soft_teacher = F.softmax(teacher_output.logits / temperature, dim=-1)
            soft_student = F.log_softmax(student_output.logits / temperature, dim=-1)

            if epoch >= args.semi_epoch:
                consistency_dist = (
                    torch.sum(consistency_criterion(soft_student, soft_teacher)) / batch_size
                )
                consistency_loss = consistency_weight * consistency_dist

                consistency_relation_dist = (
                    torch.sum(
                        relation_mse_loss(
                            student_output.hidden_states, teacher_output.hidden_states
                        )
                    )
                    / batch_size
                )

                consistency_relation_loss = (
                    consistency_weight * consistency_relation_dist * consistency_relation_weight
                )
            if not use_GAT_CELoss:
                loss = (1.0 - lambda_param) * student_target_loss + lambda_param * (
                    consistency_loss + consistency_relation_loss
                )
            if use_GAT_CELoss:
                if epoch >= args.semi_epoch:
                    global_avg_pool = nn.AdaptiveAvgPool2d(1)

                    teacher_batch_features = (
                        global_avg_pool(teacher_output.hidden_states).squeeze(-1).squeeze(-1)
                    )
                    student_batch_features = (
                        global_avg_pool(student_output.hidden_states).squeeze(-1).squeeze(-1)
                    )

                    correlation_matrix_teacher = F.cosine_similarity(
                        teacher_batch_features.unsqueeze(1),
                        teacher_batch_features.unsqueeze(0),
                        dim=2,
                    )
                    correlation_matrix_student = F.cosine_similarity(
                        student_batch_features.unsqueeze(1),
                        student_batch_features.unsqueeze(0),
                        dim=2,
                    )

                    edge_index_teacher = build_knn_adjacency(correlation_matrix_teacher, knn_k).to(
                        device
                    )
                    edge_index_student = build_knn_adjacency(correlation_matrix_student, knn_k).to(
                        device
                    )

                    data_teacher = Data(x=teacher_batch_features, edge_index=edge_index_teacher).to(
                        device
                    )
                    data_student = Data(x=student_batch_features, edge_index=edge_index_student).to(
                        device
                    )

                    gat_output_teacher = gat(data_teacher.x, data_teacher.edge_index)
                    gat_output_student = gat(data_student.x, data_student.edge_index)

                    gat_kl_dist = gat_cross_entropy_loss(
                        gat_output_student, gat_output_teacher, temperature
                    )

                    gat_kl_loss = consistency_weight * gat_kl_dist * config.gat_loss_weight

                loss = (1.0 - lambda_param) * student_target_loss + lambda_param * (
                    consistency_loss + consistency_relation_loss + gat_kl_loss
                )

        else:
            consistency_weight = 0
            loss = student_target_loss

        optimizer.zero_grad()
        optimizer_gat.zero_grad()
        loss.backward()
        optimizer.step()
        if use_semi is True and use_GAT_CELoss is True:
            if epoch >= args.semi_epoch and epoch < args.gat_stop_epoch:
                optimizer_gat.step()

        update_ema_variables(model, ema_model, alpha, global_step)
        global_step += 1

        total_loss += loss.item()
        total_steps += 1

    avg_loss = total_loss / total_steps
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    ema_model.eval()
    val_total_loss = 0.0
    val_total_steps = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
            imgs_batch = batch["imgs_batch"]
            labels = batch["labels"]

            student_output = model(imgs_batch, labels)
            student_target_loss = criterion(student_output.logits, labels)

            val_total_loss += student_target_loss.item()
            val_total_steps += 1

            logits = student_output.logits
            probs_tensor = torch.sigmoid(logits)
            sigmoid_logits = probs_tensor.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predictions = (sigmoid_logits > 0.5).astype(int)
            all_preds.extend(predictions)
            all_labels.extend(labels)

    val_avg_loss = val_total_loss / val_total_steps
    val_metrics = compute_metrics(
        {"predictions": np.array(all_preds), "label_ids": np.array(all_labels)}
    )
    print(
        f"Validation Epoch {epoch + 1}/{num_epochs}, Loss: {val_avg_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, Hamming Accuracy: {val_metrics['hamming_accuracy']:.4f}"
    )

    model.eval()
    ema_model.eval()
    test_total_loss = 0.0
    test_total_steps = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Test Epoch {epoch + 1}/{num_epochs}"):
            imgs_batch = batch["imgs_batch"]
            labels = batch["labels"]

            student_output = model(imgs_batch, labels)
            student_target_loss = criterion(student_output.logits, labels)

            test_total_loss += student_target_loss.item()
            test_total_steps += 1

            logits = student_output.logits
            probs_tensor = torch.sigmoid(logits)
            sigmoid_logits = probs_tensor.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predictions = (sigmoid_logits > 0.5).astype(int)
            all_preds.extend(predictions)
            all_labels.extend(labels)

    test_avg_loss = test_total_loss / test_total_steps
    test_metrics = compute_metrics(
        {"predictions": np.array(all_preds), "label_ids": np.array(all_labels)}
    )
    print(
        f"Test Epoch {epoch + 1}/{num_epochs}, Loss: {test_avg_loss:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, Hamming Accuracy: {test_metrics['hamming_accuracy']:.4f}"
    )

    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_student_target_loss": student_target_loss,
            "train_consistency_loss": consistency_loss,
            "train_consistency_relation_loss": consistency_relation_loss,
            "train_gat_kl_loss": gat_kl_loss,
            "train_consistency_weight": consistency_weight,
            "val_loss": val_avg_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_hamming_accuracy": val_metrics["hamming_accuracy"],
            "test_loss": test_avg_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_hamming_accuracy": test_metrics["hamming_accuracy"],
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    if (epoch + 1) % 30 == 0:
        model_save_path = os.path.join(output_dir, output_name.format(epoch + 1))
        ema_model_save_path = os.path.join(output_dir, f"ema_{output_name.format(epoch + 1)}")
        gat_model_save_path = os.path.join(output_dir, f"gat_{output_name.format(epoch + 1)}")
        torch.save(
            model.state_dict(),
            model_save_path,
        )

        if use_GAT_CELoss is True and use_semi is True:
            torch.save(
                gat.state_dict(),
                gat_model_save_path,
            )


wandb.finish()
