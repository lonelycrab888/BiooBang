import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss

from transformers import EvalPrediction

from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr


def compute_loss_acc_np(logits, labels, predictions):
    mask = labels != -100  
    correct_labels = labels[mask]
    correct_preds = predictions[mask]
    acc = accuracy_score(correct_labels, correct_preds)
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(torch.tensor(logits[mask]), torch.tensor(correct_labels))
    return loss, acc

def compute_type_metrics_np(logits, labels, index):
    predictions = np.argmax(logits, axis=-1)
    max_length = predictions.shape[1]

    decoder_length = np.array(index)
    encoder_length = max_length - decoder_length

    encoder_labels = np.full((len(labels), max_length), -100)
    decoder_labels = np.full((len(labels), max_length), -100)
    
    for i, label in enumerate(labels):
        encoder_labels[i, :decoder_length[i]] = label[:decoder_length[i]]
        decoder_labels[i, decoder_length[i]:] = label[decoder_length[i]:]
    
    loss_encoder, acc_encoder = compute_loss_acc_np(logits, encoder_labels, predictions)
    loss_decoder, acc_decoder = compute_loss_acc_np(logits, decoder_labels, predictions)

    return loss_encoder, acc_encoder, loss_decoder, acc_decoder 


def compute_metrics_np(eval_pred: EvalPrediction):
    logits, labels, inputs = eval_pred
    prompt_id = 36
    pad_id = 1
    prompt_idex = np.where(inputs == prompt_id)[1]
    position_in_rows = np.nonzero(inputs == pad_id)
    unique_rows, indices = np.unique(position_in_rows[0], return_inverse=True)
    counts = np.bincount(indices)
    row_to_first_index = np.hstack((0, np.cumsum(counts[:-1])))
    first_position_in_rows = position_in_rows[1][row_to_first_index]
    end_index = np.full(inputs.shape[0], inputs.shape[1])
    end_index[unique_rows] = first_position_in_rows
    type_array = prompt_idex < (end_index / 2)

    if not np.any(type_array):#only RNA2Protein
        RNA2Protein_logits = logits[~type_array]
        RNA2Protein_labels = labels[~type_array]
        RNA2Protein_prompt_idex = prompt_idex[~type_array]
        RNA2Protein_loss_encoder, RNA2Protein_acc_encoder, RNA2Protein_loss_decoder, RNA2Protein_acc_decoder = compute_type_metrics_np(RNA2Protein_logits, RNA2Protein_labels, RNA2Protein_prompt_idex)
        return {
            'RNA2Protein_loss_encoder': RNA2Protein_loss_encoder, 
            'RNA2Protein_acc_encoder': RNA2Protein_acc_encoder,
            'RNA2Protein_loss_decoder': RNA2Protein_loss_decoder,
            'RNA2Protein_acc_decoder': RNA2Protein_acc_decoder,
        }
    elif np.all(type_array): #only Protein2RNA
        Protein2RNA_logits = logits[type_array]
        Protein2RNA_labels = labels[type_array]
        Protein2RNA_prompt_idex = prompt_idex[type_array]
        Protein2RNA_loss_encoder, Protein2RNA_acc_encoder, Protein2RNA_loss_decoder, Protein2RNA_acc_decoder = compute_type_metrics_np(Protein2RNA_logits, Protein2RNA_labels, Protein2RNA_prompt_idex)
        return {
            'Protein2RNA_loss_encoder': Protein2RNA_loss_encoder,
            'Protein2RNA_acc_encoder': Protein2RNA_acc_encoder,
            'Protein2RNA_loss_decoder': Protein2RNA_loss_decoder,
            'Protein2RNA_acc_decoder': Protein2RNA_acc_decoder
        }
    else:
        RNA2Protein_logits = logits[~type_array]
        RNA2Protein_labels = labels[~type_array]
        RNA2Protein_prompt_idex = prompt_idex[~type_array]

        Protein2RNA_logits = logits[type_array]
        Protein2RNA_labels = labels[type_array]
        Protein2RNA_prompt_idex = prompt_idex[type_array]
        
        RNA2Protein_loss_encoder, RNA2Protein_acc_encoder, RNA2Protein_loss_decoder, RNA2Protein_acc_decoder = compute_type_metrics_np(RNA2Protein_logits, RNA2Protein_labels, RNA2Protein_prompt_idex)
        Protein2RNA_loss_encoder, Protein2RNA_acc_encoder, Protein2RNA_loss_decoder, Protein2RNA_acc_decoder = compute_type_metrics_np(Protein2RNA_logits, Protein2RNA_labels, Protein2RNA_prompt_idex)
        
        return {'RNA2Protein_loss_encoder': RNA2Protein_loss_encoder, 
                'RNA2Protein_acc_encoder': RNA2Protein_acc_encoder,
                'RNA2Protein_loss_decoder': RNA2Protein_loss_decoder,
                'RNA2Protein_acc_decoder': RNA2Protein_acc_decoder,
                'Protein2RNA_loss_encoder': Protein2RNA_loss_encoder,
                'Protein2RNA_acc_encoder': Protein2RNA_acc_encoder,
                'Protein2RNA_loss_decoder': Protein2RNA_loss_decoder,
                'Protein2RNA_acc_decoder': Protein2RNA_acc_decoder
                }

def compute_metrics_generate_RNA_np(eval_pred: EvalPrediction):
    logits, labels, inputs = eval_pred
    prompt_id = 36
    pad_id = -100
    prompt_idex = np.where(inputs == prompt_id)[1]
    position_in_rows = np.nonzero(inputs == pad_id)
    unique_rows, indices = np.unique(position_in_rows[0], return_inverse=True)
    counts = np.bincount(indices)
    row_to_first_index = np.hstack((0, np.cumsum(counts[:-1])))
    first_position_in_rows = position_in_rows[1][row_to_first_index]
    end_index = np.full(inputs.shape[0], inputs.shape[1])
    end_index[unique_rows] = first_position_in_rows
    type_array = prompt_idex < (end_index / 2)

    #only Protein2RNA
    Protein2RNA_logits = logits[type_array]
    Protein2RNA_labels = labels[type_array]
    Protein2RNA_prompt_idex = prompt_idex[type_array]
    Protein2RNA_loss_encoder, Protein2RNA_acc_encoder, Protein2RNA_loss_decoder, Protein2RNA_acc_decoder = compute_type_metrics_np(Protein2RNA_logits, Protein2RNA_labels, Protein2RNA_prompt_idex)
    
    return {
        'Protein2RNA_loss_decoder': Protein2RNA_loss_decoder,
        'Protein2RNA_acc_decoder': Protein2RNA_acc_decoder
    }


def compute_metrics_generate_decoder_RNA_np(eval_pred: EvalPrediction):
    logits, labels, _ = eval_pred
    preds = np.argmax(logits, axis=-1)

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    valid_indices = labels_flat != -100
    valid_preds = preds_flat[valid_indices]
    valid_labels = labels_flat[valid_indices]
    accuracy = accuracy_score(valid_labels, valid_preds)

    return {"accuracy": accuracy}


def compute_loss_structure_emb_similarity(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    labels = labels.T.reshape(-1)
    mae = mean_absolute_error(labels, logits)
    r, p = spearmanr(labels, logits)
    median_error = np.median(np.abs(labels - logits))
    
    errors = {}
    errors['total L1 Loss'] = mae
    errors['total r'] = r
    errors['total p'] = p
    errors['total median_error'] = median_error


    preds_pt = torch.from_numpy(logits)
    labels_pt = torch.from_numpy(labels)

    intervals = [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75, 1),(0,0.5),(0.5,1)]
    
    for _, (lower_limit, upper_limit) in enumerate(intervals):
        mask = (labels_pt>lower_limit) & (labels_pt<=upper_limit)
        pred_interval = preds_pt[mask]
        label_interval = labels_pt[mask]
        error = mean_absolute_error(label_interval, pred_interval)
        errors[f'loss: {lower_limit}-{upper_limit}'] = error
    return errors


def compute_loss_structure_emb_similarity_onlyregression(eval_pred: EvalPrediction):
    logits_all, labels_all = eval_pred
    regression_logits = np.concatenate((logits_all[0], logits_all[1]))
    labels = labels_all.T.reshape(-1)
    intervals = [(-1,0),(0,1)]
    
    errors = {}
    #mse = mean_squared_error(labels, regression_logits)
    mae = mean_absolute_error(labels, regression_logits)
    r, p = spearmanr(labels, regression_logits)
    median_error = np.median(np.abs(labels - regression_logits))
    #errors['total MSE'] = mse
    errors['total MAE'] = mae
    errors['total r'] = r
    errors['total p'] = p
    errors['total median_error'] = median_error
    preds_pt = torch.from_numpy(regression_logits)
    labels_pt = torch.from_numpy(labels)
    for _, (lower_limit, upper_limit) in enumerate(intervals):
        mask = (labels_pt>lower_limit) & (labels_pt<=upper_limit)
        pred_interval = preds_pt[mask]
        label_interval = labels_pt[mask]
        #error_mse = mean_squared_error(label_interval, pred_interval)
        error_mae = mean_absolute_error(label_interval, pred_interval)
        errors[f'mae: {lower_limit}-{upper_limit}'] = error_mae
        #errors[f'mse: {lower_limit}-{upper_limit}'] = error_mse
    return errors
def compute_loss_structure_emb_similarity_MRL(eval_pred):
    logits_all, labels = eval_pred
    labels = labels.T.reshape(-1)
    classification_logits = logits_all
    labels_classification = np.floor((labels+ 1)*10).astype(np.int64)
    labels_classification[labels_classification < 6] = 0  #0.1-0.2
    labels_classification[(labels_classification >= 6) & (labels_classification<10)] = 1  #0.1-0.2
    labels_classification[(labels_classification >= 10) & (labels_classification<16)] = 2
    labels_classification[labels_classification >= 16] = 3

    flag_classification = False
    flag_regression = False
    if len(logits_all) ==10:
        flag_classification = True
        flag_regression = True
        classification_logits = logits_all[:5]
        regression_logits = logits_all[5:]
    elif len(logits_all)==5 or len(logits_all)==1:
        if len(logits_all[0][0].shape)==2:
            flag_classification = True
            classification_logits = logits_all
        elif len(logits_all[0][0].shape)==1:
            flag_regression = True
            regression_logits = logits_all
    elif len(logits_all) == 2:
        flag_classification = True
        flag_regression = True
        classification_logits = [logits_all[0]]
        regression_logits = [logits_all[1]]
 
    intervals = [(-1,0),(0,1)]
    name = ['768', '512', '256', '128', '64']
    #print(classification_logits)
    errors = {}
    if flag_classification:
        i = 0
        for logits in classification_logits:
            #print(logits)
            logits_concat = np.concatenate((logits[0], logits[1]))
            predictions = logits_concat.argmax(axis=-1)
            #print(predictions)
            accuracy = accuracy_score(labels_classification, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels_classification, predictions, average='weighted')
            filtered_labels = (labels_classification>1).astype(int)
            filtered_predictions = (predictions>1).astype(int)
            accuracy_binary = accuracy_score(filtered_labels, filtered_predictions)
            precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(filtered_labels, filtered_predictions)
            errors[name[i]+'_total accuracy'] = accuracy
            errors[name[i]+'_total precision'] = precision
            errors[name[i]+'_total recall'] = recall
            errors[name[i]+'_total f1'] = f1
            errors[name[i]+'_0.5-1 accuracy'] = accuracy_binary
            errors[name[i]+'_0.5-1 precision'] = precision_binary[1]
            errors[name[i]+'_0.5-1 recall'] = recall_binary[1]
            errors[name[i]+'_0.5-1 f1'] = f1_binary[1]
            i = i+1
    if flag_regression:
        i = 0
        for logits in regression_logits:
            logits_concat = np.concatenate((logits[0], logits[1]))   
            
            logits_regression2classification = np.floor((logits_concat+ 1)*10).astype(np.int64)
            logits_regression2classification[logits_regression2classification < 10] = 0  #0.1-0.2
            logits_regression2classification[(logits_regression2classification >= 6) & (logits_regression2classification<10)] = 1  #0.1-0.2
            logits_regression2classification[(logits_regression2classification >= 10) & (logits_regression2classification<16)] = 2
            logits_regression2classification[logits_regression2classification >= 16] = 3

            r2c_accuracy = accuracy_score(labels_classification, logits_regression2classification)
            r2c_precision, r2c_recall, r2c_f1, _ = precision_recall_fscore_support(labels_classification, logits_regression2classification, average='weighted')
            filtered_labels = (labels_classification>1).astype(int)
            filtered_predictions = (logits_regression2classification>1).astype(int)
            r2c_accuracy_binary = accuracy_score(filtered_labels, filtered_predictions)
            r2c_precision_binary, r2c_recall_binary, r2c_f1_binary, _ = precision_recall_fscore_support(filtered_labels, filtered_predictions)
            errors[name[i]+'_total r2c_accuracy'] = r2c_accuracy
            errors[name[i]+'_total r2c_precision'] = r2c_precision
            errors[name[i]+'_total r2c_recall'] = r2c_recall
            errors[name[i]+'_total r2c_f1'] = r2c_f1
            errors[name[i]+'_0.5-1 r2c_accuracy'] = r2c_accuracy_binary
            errors[name[i]+'_0.5-1 r2c_precision'] = r2c_precision_binary[1]
            errors[name[i]+'_0.5-1 r2c_recall'] = r2c_recall_binary[1]
            errors[name[i]+'_0.5-1 r2c_f1'] = r2c_f1_binary[1]
            
            
            mae = mean_absolute_error(labels, logits_concat)
            r, p = spearmanr(labels, logits_concat)
            median_error = np.median(np.abs(labels - logits_concat))
            errors[name[i]+'_total mae'] = mae
            errors[name[i]+'_total r'] = r
            errors[name[i]+'_total p'] = p
            errors[name[i]+'_total median_error'] = median_error
            preds_pt = torch.from_numpy(logits_concat)
            labels_pt = torch.from_numpy(labels)
            for _, (lower_limit, upper_limit) in enumerate(intervals):
                mask = (labels_pt>lower_limit) & (labels_pt<=upper_limit)
                pred_interval = preds_pt[mask]
                label_interval = labels_pt[mask]
                error = mean_absolute_error(label_interval, pred_interval)
                errors[f'{name[i]}_mae: {lower_limit}-{upper_limit}'] = error
            i = i+1
    return errors

def compute_loss_full_protein_emb_similarity(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    logits_query = logits[0]
    logits_target = logits[1]
    labels_query = labels.T[0]
    labels_target = labels.T[1]

    mae_query = mean_absolute_error(labels_query, logits_query)
    r_query, p_query = spearmanr(labels_query, logits_query)
    median_error_query = np.median(np.abs(labels_query - logits_query))

    preds_query_pt = torch.from_numpy(logits_query)
    labels_query_pt = torch.from_numpy(labels_query)

    mae_target = mean_absolute_error(labels_target, logits_target)
    r_target, p_target = spearmanr(labels_target, logits_target)
    median_error_target = np.median(np.abs(labels_target - logits_target))

    preds_target_pt = torch.from_numpy(logits_target)
    labels_target_pt = torch.from_numpy(labels_target)

    intervals = [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75, 1)] #(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)
    errors = {}
    errors['query L1 Loss'] = mae_query
    errors['query r'] = r_query
    errors['query p'] = p_query
    errors['query median_error'] = median_error_query
    errors['target L1 Loss'] = mae_target
    errors['target r'] = r_target
    errors['target p'] = p_target
    errors['target median_error'] = median_error_target

    for _, (lower_limit, upper_limit) in enumerate(intervals):
        mask_query = (labels_query_pt>lower_limit) & (labels_query_pt<=upper_limit)
        pred_interval_query = preds_query_pt[mask_query]
        label_interval_query = labels_query_pt[mask_query]
        error_query = mean_absolute_error(label_interval_query, pred_interval_query)

        mask_target = (labels_target_pt>lower_limit) & (labels_target_pt<=upper_limit)
        pred_interval_target = preds_target_pt[mask_target]
        label_interval_target = labels_target_pt[mask_target]
        error_target = mean_absolute_error(label_interval_target, pred_interval_target)

        errors[f'query L1 loss: {lower_limit}-{upper_limit}'] = error_query
        errors[f'target L1 loss: {lower_limit}-{upper_limit}'] = error_target
    return errors


def compute_TE_regression_metrics(eval_pred: EvalPrediction):
    logits, labels, _ = eval_pred
    logits = logits.squeeze()
    mae = mean_absolute_error(labels, logits)
    r, p = spearmanr(labels, logits)
    median_error = np.median(np.abs(labels - logits))
    errors = {}
    errors['total L1 Loss'] = mae
    errors['total r'] = r
    errors['total p'] = p
    errors['total median_error'] = median_error 
    return errors 

def compute_regression_metrics(eval_pred: EvalPrediction):
    logits, labels, _ = eval_pred
    logits = logits.squeeze()
    mae = mean_absolute_error(labels, logits)
    r, p = spearmanr(labels, logits)
    median_error = np.median(np.abs(labels - logits))
    errors = {}
    errors['total L1 Loss'] = mae
    errors['total spearman r'] = r
    errors['total spearman p'] = p
    errors['total median_error'] = median_error 
    return errors   


def compute_TEwithMFE_regression_metrics(eval_pred: EvalPrediction):
    logits, labels, _ = eval_pred
    logits_TE, logits_MFE = np.hsplit(logits, 2)
    logits_TE = logits_TE.squeeze()
    logits_MFE = logits_MFE.squeeze()
    labels_TE = labels[0]
    labels_MFE = labels[1]

    mae_TE = mean_absolute_error(logits_TE, labels_TE)
    mae_MFE = mean_absolute_error(logits_MFE, labels_MFE)
    r_TE, p_TE = spearmanr(labels_TE, logits_TE)
    r_MFE, p_MFE = spearmanr(labels_MFE, logits_MFE)
    
    errors = {}
    errors['TE L1 Loss'] = mae_TE
    errors['TE r'] = r_TE
    errors['TE p'] = p_TE
    errors['MFE L1 Loss'] = mae_MFE
    errors['MFE r'] = r_MFE
    errors['MFE p'] = p_MFE
    return errors  