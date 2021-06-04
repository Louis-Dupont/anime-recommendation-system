import numpy as np


def get_raw_predictions(dataset, model, nb_results=1000):

    raw_x, raw_y_true = zip(*[(x[0].numpy(), x[1].numpy()) for x in dataset.take(nb_results)])
    raw_x, raw_y_true = np.array(raw_y_true), np.array(raw_y_true)
    raw_y_pred = model.predict(raw_x)
    return raw_x, raw_y_true, raw_y_pred


def get_flat_prediction(dataset, model, nb_results=1000):
    in_values, true_values, pred_values = [], [], []
    raw_x, raw_y_true, raw_y_pred = get_raw_predictions(dataset, model, nb_results)
    for x, y_true, y_pred in zip(raw_x, raw_y_true, raw_y_pred):
        in_values.append(x[x != -1])
        true_values.append(y_true[y_true != -1])
        pred_values.append(y_pred[y_true != -1])
    in_values = np.concatenate(in_values)
    true_values = np.concatenate(true_values)
    pred_values = np.concatenate(pred_values)
    return in_values, true_values, pred_values