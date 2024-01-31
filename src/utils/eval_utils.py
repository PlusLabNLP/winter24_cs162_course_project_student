# TODO - import relevant sklearn score modules 
from sklearn import metrics

from utils.file_utils import load_jsonl

def evaluate_standard(gt_labels, pred_labels):
    
    accuracy, f1score = 0, 0

    ##################################################
    # TODO: Please finish the standard evaluation metrics.
    # You need to compute the accuracy and F1 score for the 
    # predictions and ground truth labels. Please use the 
    # scikit-learn APIs in way they can deal with strings 
    # as label. Remeber to import the functions you use!

    # End of TODO.
    ##################################################
    
    accuracy = metrics.accuracy_score(gt_labels, pred_labels)
    
    # Need to tell sklearn which label is the positive one    
    f1score = metrics.f1_score(gt_labels, pred_labels, pos_label='SUPPORTS')

    print(accuracy, f1score)
    return accuracy, f1score

def model_eval_report(gt_filepath, pred_filepath):
    
    gt_data = load_jsonl(gt_filepath)
    gt_labels = [d["label"] for d in gt_data]
    pred_data = load_jsonl(pred_filepath)
    pred_labels = [d["label"] for d in pred_data]
    
    accuracy, f1score = evaluate_standard(gt_labels, 
                                          pred_labels)

    print(f"Overall Accuracy : {accuracy}")
    print(f"Overall F1 score : {f1score}")
