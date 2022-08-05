import numpy as np
import cv2


def load_data_and_preprocess(IMAGE_ADDRESS):
    img = cv2.imread(IMAGE_ADDRESS)
    h, w = img.shape[:2]
    pre_process_img = cv2.dnn.blobFromImage(img, scalefactor=1 / 255,
                                            size=(416, 416), swapRB=True, crop=False)
    return img, pre_process_img, h, w


def read_models_and_labels(LABEL_ADDRESS, PARAMETER_OF_DNN, MODEL_ADDRESS):
    labels = open(LABEL_ADDRESS).read().strip().split("\n")
    net = cv2.dnn.readNet(MODEL_ADDRESS, PARAMETER_OF_DNN)

    return labels, net


def inference(pre_process_img, net):
    net.setInput(pre_process_img)

    output_layer = ["yolo_82", "yolo_94", "yolo_106"]
    predictions = net.forward(output_layer)

    return predictions


def post_processing(predictions, w, h):
    classIDs = []
    confidences = []
    boxes = []
    for layer in predictions:
        for detected_object in layer:
            scores = detected_object[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.3:
                box = detected_object[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([x, y, int(width), int(height)])

    return classIDs, confidences, boxes


def show_result(img, classIDs, confidences, boxes, labels):
    colors = np.random.uniform(0, 255, size=(len(labels), 3))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)

    for i in idxs.flatten():
        (x, y) = boxes[i][0], boxes[i][1]
        (w, h) = boxes[i][2], boxes[i][3]

        cv2.rectangle(img, (x, y), (x + w, y + h), colors[i], 2)
        text = "{}:{:.2f}".format(labels[classIDs[i]], confidences[i] * 100, 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


IMAGE_ADDRESS = "img1.jpg"
LABEL_ADDRESS = r"yolo files/coco.names"
PARAMETER_OF_DNN = r"yolo files/yolov3.cfg"
MODELS_ADDRESS = r"yolo files/yolov3.weights"

img, pre_process_img, h, w = load_data_and_preprocess(IMAGE_ADDRESS)
labels, net = read_models_and_labels(LABEL_ADDRESS, PARAMETER_OF_DNN, MODELS_ADDRESS)
predictions = inference(pre_process_img, net)
classIDs, confidences, boxes = post_processing(predictions, w, h)
show_result(img, classIDs, confidences, boxes, labels)
