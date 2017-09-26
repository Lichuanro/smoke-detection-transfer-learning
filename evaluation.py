from collections import Counter
import numpy as np


def get_prediction(img_list, model):
    prediction = []
    for img in img_list:
        prediction.append(model.predict(img.reshape((1, 224, 224, 3))))
    return np.array(prediction)


def get_evaluation(smoke_img, nonsmoke_img, model):
    # generate image labels
    smoke_labels = np.ones(smoke_img.shape[0], dtype=np.int8)
    nonsmoke_labels = np.zeros(nonsmoke_img.shape[0], dtype=np.int8)

    imgs = np.concatenate((smoke_img, nonsmoke_img))
    y_true = np.concatenate((smoke_labels, nonsmoke_labels))

    # get the predictions
    y_pre = get_prediction(imgs, model)

    assert len(y_pre) == len(y_true)
    true_counter = Counter(y_true)
    num_pos = true_counter[1]
    num_neg = true_counter[0]

    tp = 0
    fp = 0

    for index, label in enumerate(y_true):
        if y_pre[index] == 1:
            if label == 1:
                tp += 1
            else:
                fp += 1

    dr = tp / num_pos
    far = fp / num_neg
    acr = (tp + num_neg - fp) / (num_pos + num_neg)

    return dr, far, acr