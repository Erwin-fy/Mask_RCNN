import numpy as np
import random


def compute_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    y1 = np.maximum(box1[0], box2[0])
    y2 = np.minimum(box1[2], box2[2])
    x1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[3], box2[3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = area1 + area2 - intersection
    iou = 1. * intersection / union
    return iou


def kmeans(k, boxes):
    """
    :param k:
    :param boxes: [N, (y1, x1, y2, x2)]
    :return:
    """
    count = boxes.shape[0]
    centroid_ids = random.sample(range(count), k)
    centroid_boxes = np.zeros((k, 4))
    print centroid_ids
    for i in range(k):
        centroid_boxes[i] = boxes[centroid_ids[i]]

    for iteration in range(500):
        clusters = []
        for i in range(k):
            clusters.append([])

        for i in range(count):
            max_iou = 0
            cluster_index = 0
            for j in range(k):
                iou = compute_iou(boxes[i], centroid_boxes[j])
                if (max_iou < iou):
                    max_iou = iou
                    cluster_index = j

            clusters[cluster_index].append(boxes[i])

        for i in range(k):
            centroid_boxes[i] = np.mean(clusters[i], axis=0)

    print centroid_boxes


