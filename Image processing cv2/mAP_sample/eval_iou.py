# import the necessary packages
from pyimagesearch.utils_iou import intersection_over_union
from pyimagesearch.utils_iou import plt_imshow
from pyimagesearch import config
import cv2

def compute_iou(imagePath):
    # load the image
    image = cv2.imread(imagePath)

    # define the top-left and bottom-right coordinates of ground-truth
    # and prediction
    groundTruth = [90, 80, 250, 450]
    prediction = [100, 100, 220, 400]

    # draw the ground-truth bounding box along with the predicted
    # bounding box
    cv2.rectangle(image, tuple(groundTruth[:2]),
        tuple(groundTruth[2:]), (0, 255, 0), 2)
    cv2.rectangle(image, tuple(prediction[:2]),
        tuple(prediction[2:]), (0, 0, 255), 2)

    # compute the intersection over union and display it
    iou = intersection_over_union(groundTruth, prediction)
    cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    
    # show the output image
    plt_imshow("Image", image, config.IOU_RESULT)