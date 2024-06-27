import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from misc.metric_tool import ConfuseMatrixMeter
import nibabel as nib
from torchmetrics.functional import dice
import torch
from scipy.stats import norm
import numpy as np
from skimage import measure
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns


def do_2d(img, gt, meter):
    x, y, z = img.shape
    dice_scores = []
    for i in range(z):
        true_slc = gt[:,:,i]
        pred_slc = img[:,:,i]
        if torch.sum(true_slc) == 0 and torch.sum(pred_slc) == 0:
            dice_score = 1.0
        else:
            dice_score = dice(pred_slc, true_slc, ignore_index=0).item()
        #print(pred_slc)
        #print("*"*100)
        #print(true_slc)
        running_f1 = meter.update_cm(pred_slc.cpu().numpy(), true_slc.cpu().numpy())
        #print(f"dice score {dice_score}")
        dice_scores.append(dice_score)
    return meter, np.mean(dice_scores) 

def partial_iou(pred, true):
    """Calculates the overlap volume of pred in true.

    Args:
    array1: A 3D array.
    array2: A 3D array.

    Returns:
    The IoU of the two 3D arrays.
    """

    # Calculate the intersection of the two 3D arrays
    intersection = np.logical_and(pred, true)

    # Calculate the volume of the intersection
    intersection_volume = np.sum(intersection)

    # Calculate the volume of the predicted label
    pred_volume = np.sum(pred)
    #print(pred_volume, intersection_volume)
    # Calculate the IoU
    iou = intersection_volume / (pred_volume+1e-5)

    return iou

def find_overlap(pred_labels, gt_labels, props, target_label, min_x, min_y, min_z, max_x, max_y, max_z,
                 replace={}, thresh=0.6):
    lesion_area = gt_labels[min_x:max_x, min_y:max_y, min_z: max_z]
    pred_area = pred_labels[min_x:max_x, min_y:max_y, min_z: max_z]
    for idx, row in props.iterrows():
        label = row['label']
        label_area = pred_area==label
        label_iou = partial_iou(label_area, lesion_area)
        if label_iou >= thresh:
            #print(f'replacing {label} with {target_label} overlap : {label_iou}')
            #pred_labels = np.where(pred_labels == label, target_label, pred_labels)
            replace[label]= target_label
    return replace

def replace_labels(labels, replace, last_label):
    for label in replace.keys():
        target_label = replace[label]
        if target_label == -1:
            target_label = last_label+1
            last_label = target_label
        labels = np.where(labels == label, target_label, labels)
    return labels
    
def modify_labels(true_data, pred_data):
    true_labels = measure.label(true_data, background=0)    
    pred_labels = measure.label(pred_data, background=0)    
    props_true = pd.DataFrame(measure.regionprops_table(true_labels, properties=('label','bbox',)))
    props_pred = pd.DataFrame(measure.regionprops_table(pred_labels, properties=('label','bbox',)))

    replace={}
    pred_lbs = props_pred.label.values
    for lb in pred_lbs:
        replace[lb] = -1
    for idx, row in props_true.iterrows():
        label = row['label']
        replace = find_overlap(pred_labels, true_labels, props_pred, label, row['bbox-0'], row['bbox-1'],
                      row['bbox-2'], row['bbox-3'], row['bbox-4'], row['bbox-5'], replace)
    #print(replace)
    
    pred_labels = replace_labels(pred_labels, replace, len(pred_lbs))
    return true_labels, pred_labels

def fbeta_score(precision, recall, beta):
    if precision == 0 and recall == 0:
        return 0  # Avoid division by zero
    else:
        beta_squared = beta ** 2
        fbeta = (1 + beta_squared) * (precision * recall) / ((beta_squared * precision) + recall)
        return fbeta
    
def do_3d(img, gt, overall_metrics):
    y_true_cc, y_pred_cc = modify_labels(gt, img) 
    # Initialize the metrics for this subject
    metrics = {'TP': 0, 'FP': 0, 'FN': 0}
    
    true_lbs = np.unique(y_true_cc)
    pred_lbs = np.unique(y_pred_cc)
    for lb in pred_lbs:
        if lb==0 :
            pass
        elif lb in true_lbs:
            y_pred_component = y_pred_cc == lb
            y_true_component = y_true_cc == lb
            iou = partial_iou(y_pred_component, y_true_component)
            if iou > 0:
                metrics['TP'] +=1
        else:
            metrics['FP'] += 1
    for lb in true_lbs:
        if lb==0:
            pass
        elif lb in pred_lbs:
            pass
        else:
            metrics['FN'] +=1
    """if metrics['TP'] == 0 and metrics['FP'] == 0:
        if metrics['FN'] != 0:
            only_FN.append(row['ground_truth'])
    
    if metrics['TP'] == 0 and metrics['FN'] == 0:
        if metrics['FP'] != 0:
            only_FP.append(row['ground_truth'])"""

    # Update the overall metrics
    overall_metrics['TP'] += metrics['TP']
    overall_metrics['FP'] += metrics['FP']
    overall_metrics['FN'] += metrics['FN']
    return overall_metrics

def calc_metrics(iteration, files, target_dir, model, mode="2D",device="cpu"):
    metrics=[]
    meter = ConfuseMatrixMeter(n_class=2)
    dice_scores = []
    # Dictionary to hold overall metrics
    overall_metrics = {'TP': 0, 'FP': 0, 'FN': 0}
    for file in files:
        img = torch.from_numpy(nib.load(f"{target_dir}/{model}/{file}").get_fdata().astype(np.int64)).to(device)
        gt = torch.from_numpy(nib.load(f"{target_dir}/ground_truth/{file}").get_fdata().astype(np.int64)).to(device)
        if mode == "2D":
            meter, dice_score = do_2d(img, gt, meter)
            dice_scores.append(dice_score)
        else:
            overall_metrics = do_3d(img, gt, overall_metrics)
    if mode == "2D":
        scores = meter.get_scores()
        scores['average_dice'] = np.mean(dice_scores)
    else:
        scores = overall_metrics
        precision = overall_metrics['TP'] / (overall_metrics['TP'] + overall_metrics['FP']+1e-11)
        recall = overall_metrics['TP'] / (overall_metrics['TP'] + overall_metrics['FN']+1e-11)
        f1 = 2 * precision * recall / (precision + recall+1e-11)
        f05 = fbeta_score(precision, recall, 0.5)
        f2 = fbeta_score(precision, recall, 2)
        scores['precision'] = precision
        scores['recall'] = recall
        scores['f1'] = f1
        scores['f0.5'] = f05
        scores['f2'] = f2
    scores['iteration'] = iteration
    return scores
    
def bootstrap(test_cases, target_dir, model, num_samples=10, num_iters=1000, replacement=False, mode="2D",device="cpu"):
    bootstrap_statistics = []
    n = num_samples
    for i in tqdm(range(num_iters), desc="iter", position=0, leave=True):
        # Generate a bootstrap sample by sampling with replacement
        bootstrap_sample = np.random.choice(test_cases, size=n, replace=replacement)
        # Compute the desired statistic for the bootstrap sample
        metrics = calc_metrics(i, bootstrap_sample, target_dir, model, mode, device)
        bootstrap_statistics.append(metrics)
    return pd.DataFrame(bootstrap_statistics)


if __name__ == '__main__':
    stack = "SVUH"
    target_dir = f"/home/prateek/NeuFormer/stacks/{stack}"
    model = "BUN"
    mode="2D"
    ms2_csv = pd.read_csv(f"{target_dir}/ground_truth.csv")
    rep = True
    test_cases = ms2_csv['file'].values
    results = bootstrap(test_cases, target_dir, model, num_samples=10,replacement=rep, num_iters=1000, mode=mode, device="cuda:0")
    results.to_csv(f"{model}_bootstrap_wrep1k_{stack}_{mode}.csv")

