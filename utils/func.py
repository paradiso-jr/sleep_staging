import os
import yaml
import torch
import shutil
def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def label_weight(labels:torch.Tensor):
    lables = labels.reshape(-1)
    _, label_cnt=labels.unique(return_counts=True)

    return ((sum(label_cnt)-label_cnt) / label_cnt).detach()

def calculate_accuracy_per_label(logits, ground_truth):
    predicted_labels = torch.argmax(logits, dim=1)
    num_classes = len(torch.unique(ground_truth))
    correct_counts = torch.zeros(num_classes, dtype=torch.int32)
    total_counts = torch.zeros(num_classes, dtype=torch.int32)
    
    for label in range(num_classes):
        mask = (ground_truth == label)
        mask = mask.squeeze()
        correct_counts[label] = (predicted_labels[mask] == ground_truth[mask]).sum().item()
        total_counts[label] = mask.sum().item()
    
    accuracies = correct_counts.float() / total_counts.float()
    return accuracies