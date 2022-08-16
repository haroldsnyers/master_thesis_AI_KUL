import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay, roc_curve, auc

def generate_dataframe(columns, data_list, index_values=None):
    df = pd.DataFrame(columns=columns)
    for i, elem in enumerate(columns):
        df[elem] = data_list[i]
    if index_values is not None:
        df.index = index_values
    return df


def compute_fpr_fnr_tpr_from_roc_curve(score, y_true):
    fpr, tpr, threshold = roc_curve(y_true, score, pos_label=None, sample_weight=None)
    fnr = 1 - tpr
    return fpr, fnr, tpr, threshold


def plot_roc_curve(fpr, tpr, ax, title_add=""):
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot(ax=ax)
    ax.set_title("Receiver Operating Characteristic (ROC) curves " + title_add)
    ax.set_ylim(0)


def plot_eer_roc(fpr, tpr, ax, result, title_add=""):
    plot_roc_curve(fpr, tpr, ax, title_add)
    ax.plot(result['fpr'], result['tpr'], 'ro')
    ax.plot([0, 1], [1, 0])


def roc_curve_multilabel(df_pred, df_y):
    fpr = dict()
    tpr = dict()
    fnr = dict()
    threshold = dict()
    opt_threshold = dict()
    roc_auc = dict()
    result_roc = dict()
    for col in df_pred.columns: 
        y_true = df_y[col].values
        y_score = df_pred[col].values
        fpr[col], fnr[col], tpr[col], threshold[col] = compute_fpr_fnr_tpr_from_roc_curve(y_score, y_true)

        try:
            opt_threshold[col] = threshold[col][np.nanargmin(np.absolute((fnr[col] - fpr[col])))]
            df_decision_threshold_fpr_fnr = generate_dataframe(columns=['fpr', 'tpr'], data_list=[fpr[col], tpr[col]], index_values=threshold[col])
            result_roc[col] = df_decision_threshold_fpr_fnr.loc[opt_threshold[col]]
        except:
            pass
        roc_auc[col] = auc(fpr[col], tpr[col])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], fnr['micro'], tpr["micro"], threshold['micro'] = compute_fpr_fnr_tpr_from_roc_curve(df_pred.values.ravel(), df_y.values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    opt_threshold['micro'] = threshold['micro'][np.nanargmin(np.absolute((fnr['micro'] - fpr['micro'])))]

    df_decision_threshold_fpr_fnr = generate_dataframe(columns=['fpr', 'tpr'], data_list=[fpr['micro'], tpr['micro']], index_values=threshold['micro'])
    result_roc['micro'] = df_decision_threshold_fpr_fnr.loc[opt_threshold['micro']]


    return fpr, tpr, fnr, opt_threshold, threshold, roc_auc, result_roc