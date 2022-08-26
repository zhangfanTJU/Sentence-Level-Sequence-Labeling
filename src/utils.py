import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def get_all_score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    p = precision_score(y_true, y_pred, average='macro') * 100
    r = recall_score(y_true, y_pred, average='macro') * 100
    return reformat(p, 2), reformat(r, 2), reformat(f1, 2)


def get_label_score(y_true, y_pred, label):
    y_true_label = np.where(y_true == label, 1, 0)
    y_pred_label = np.where(y_pred == label, 1, 0)
    p = precision_score(y_true_label, y_pred_label) * 100
    r = recall_score(y_true_label, y_pred_label) * 100
    f1 = f1_score(y_true_label, y_pred_label) * 100
    return reformat(p, 2), reformat(r, 2), reformat(f1, 2)


def get_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p, r, f1 = get_all_score(y_true, y_pred)
    p_0, r_0, f1_0 = get_label_score(y_true, y_pred, 0)
    p_no_0, r_no_0, f1_no_0 = get_other_score(p, p_0), get_other_score(r, r_0), get_other_score(f1, f1_0)

    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), str((p_no_0, r_no_0, f1_no_0)), f1_no_0


def get_other_score(score, score_0):
    return reformat((score * 11 - score_0) / 10, 2)


def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))


if __name__ == "__main__":
    import json
    save_dic = json.load(open('./save/glove/base/fold_0/exp_0/test_16.json', 'r'))

    y_true, y_pred = [], []
    true_label, pred_label = save_dic['ture_label'], save_dic['pred_label']
    for true, pred in zip(true_label, pred_label):
        y_true.extend(true)
        y_pred.extend(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p, r, f1 = get_all_score(y_true, y_pred)
    print(p, r, f1)

    p_labels, r_labels, f1_labels = [], [], []
    num_labels = len(set(y_true))
    for label in range(num_labels):
        p_label, r_label, f1_label = get_label_score(y_true, y_pred, label)
        p_labels.append(p_label)
        r_labels.append(r_label)
        f1_labels.append(f1_label)

    p_, r_, f1_ = sum(p_labels) / num_labels, sum(r_labels) / num_labels, sum(f1_labels) / num_labels

    print(p_, r_, f1_)
