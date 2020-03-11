from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def auroc(preds, labels):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def aupr(preds, labels):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    precision, recall, _ = precision_recall_curve(labels, preds)
    return auc(recall, precision)


def fpr_at_95_tpr(preds, labels):
    """Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):    
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x>=0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def detection_error(preds, labels):
    """Return the misclassification probability when TPR is 95%.
        
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary    t = tpr[idx]
    f = fpr[idx]
    
    return 0.5 * (1 - t) + 0.5 * f labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
        
    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x>=0.95]
    
    # Calc error for a given threshold (i.e. idx)
    _detection_error = lambda idx: 0.5 * (1 - tpr[idx]) + 0.5 * fpr[idx]
    
    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))



def plot_roc(preds, labels, title="Receiver operating characteristic"):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """
    
    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)
    
    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)
    
    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    
def plot_pr(preds, labels, title="Precision recall curve"):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """
    
    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
#     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    

def get_summary_statistics(predictions, labels):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.
    
    These metrics conform to how results are reported in the paper 'Enhancing The 
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.
    
        preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    
    return {
        'fpr_at_95_tpr':fpr_at_95_tpr(predictions, labels)*100,
        'detection_error': detection_error(predictions, labels)*100,
        'auroc': auroc(predictions, labels)*100,
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels])*100,       
        'aupr_in': aupr(predictions, labels)*100
        
    }
