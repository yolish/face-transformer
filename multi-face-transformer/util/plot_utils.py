import cv2
import matplotlib.pyplot as plt

def plot_bboxes(img, gt_boxes, predicted_boxes, predicted_labels=None):

    for i in range(gt_boxes.shape[0]):
        cv2.rectangle(img, (gt_boxes[i, 0], gt_boxes[i, 1]), (gt_boxes[i, 2], gt_boxes[i, 3]), (0, 0, 255))

    if predicted_boxes is not None:
        for i in range(predicted_boxes.shape[0]):
            cv2.rectangle(img, (predicted_boxes[i, 0], predicted_boxes[i, 1]), (predicted_boxes[i, 2],
                                                                                predicted_boxes[i, 3]), (0, 255, 0))
    if predicted_labels is not None:
        for i in range(len(predicted_labels)):
            cv2.putText(img, predicted_labels[i], (predicted_boxes[i, 0], predicted_boxes[i, 1]), fontFace=3, fontScale=1, color=(0, 255, 0))

    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
