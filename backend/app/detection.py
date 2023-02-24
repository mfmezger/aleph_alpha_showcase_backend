"""This is the detection module for the app."""
import copy
import json
import logging
from logging.config import dictConfig

import cv2
import torch
import whisper
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection

from .config import LogConfig


def initialize_models():
    """Initialize the models because of download constraigns."""
    model = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    del model


"""Initializing the logger."""
dictConfig(LogConfig().dict())
logger = logging.getLogger("client")


def draw_on_image(results, img, model, score_confidence=0.99, debugging=False):
    """Draws the bounding boxes on the image.

    # future: check if speedup is possible / need for profiling.

    :param results: the results from the model.
    :type results: dict of tensors.
    :param img: the image to draw on.
    :type img: cv2.Image.
    :param model: the model.
    :type model: model.
    :param score_confidence: The confidence score that the model should reach to display the predictions, defaults to 0.9
    :type score_confidence: float, optional
    :param debugging: Debugging Mode with more prints, defaults to False
    :type debugging: bool, optional
    :return: the image with the bounding boxes drawn on it, the detection class, and the probability
    :rtype: cv2.Image, list, list
    """
    color = [100, 128, 0]
    # save detection and time stamp
    detection_class = []
    prob = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9
        if score > score_confidence:
            if debugging:
                print(f"Detected {model.config.id2label[label.item()]} with confidence " f"{round(score.item(), 3)} at location {box}")
            # draw bouding box on img.
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                img,
                model.config.id2label[label.item()],
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                color,
                2,
                lineType=cv2.LINE_4,
            )
            detection_class.append(model.config.id2label[label.item()])
            prob.append(round(score.item(), 3))

    return img, detection_class, prob


def speech_to_text(path_to_sound):
    """Convert the sound of the video to text.

    :param path_to_video: Path to the video
    :type path_to_video: str
    :return: the text from the video
    :rtype: str
    """
    model = whisper.load_model("base")
    whisper.DecodingOptions(language="de", without_timestamps=False)
    result = model.transcribe(path_to_sound)

    # save to tmp
    # save the transcpription to a file
    with open("transcription.txt", "w") as f:
        # save the string to file
        f.write(result["text"])

    return result["text"]


def detect_single_image(path_to_image, save_path, dict_path):
    """Run a forward pass of the model on the image.

    :param path_to_image: Parth to the image in the temporary folder
    :type path_to_image: str
    """
    image = Image.open(path_to_image)
    image = image.convert("RGB")

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    img = cv2.imread(path_to_image)

    # draw on image.
    img, detection_class, prob = draw_on_image(results, img, model)
    detections = {}
    detections[0] = {"detection_class": detection_class, "prob": prob}

    # convert dict to json.
    with open(dict_path, "w") as f:
        json.dump(detections, f)

    # save the image to the tmp_img folder.
    cv2.imwrite(save_path, img)


def detect_video(path_to_video, save_path, dict_path):
    """Loop over the video and detect the classes in the video, then draw onto the image and save it to a new video.

    :param path_to_video: Path to the video
    :type path_to_video: str
    :param save_path: Path to save the video
    :type save_path: str
    :param dict_path: Path to save the json file
    :type dict_path: str
    """
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path_to_video)

    # initialize the model.
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # initialize video with detection bounding boxes.
    codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(save_path, codec, fps, (width, height))

    detections = {}
    logger.info("Starting Predicitons.")
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    frame = 0
    # results_dict = {}
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret == True:
            # process the video
            img = copy.deepcopy(image)
            image = Image.fromarray(image)

            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            # detaching all of the outputs
            target_sizes = torch.tensor([image.size[::-1]])
            results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
            outputs = {k: v.detach().cpu() for k, v in outputs.items()}
            img, detection_class, prob = draw_on_image(results, img, model, debugging=False)
            output_video.write(img)
            # store prob and detection_class with frame in dict.
            detections[frame] = {"detection_class": detection_class, "prob": prob}

            del img, detection_class, prob, image, inputs, outputs, results

            # convert outputs (bounding boxes and class logits) to COCO API
            frame += 1
            # results_dict[frame] = results

        # Break the loop if there are no more frames.
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    # # draw results.
    # cap = cv2.VideoCapture(path_to_video)
    # while cap.isOpened():
    #     # Capture frame-by-frame
    #     ret, image = cap.read()
    #     if ret == True:
    #         results = results_dict[frame]

    #         img, detection_class, prob = draw_on_image(results, image, model, debugging=False)
    #         output_video.write(img)
    #         # store prob and detection_class with frame in dict.
    #         detections[frame] = {"detection_class": detection_class, "prob": prob}

    output_video.release()
    logger.info("Writing Results to File.")

    # convert dict to json.
    with open(dict_path, "w") as f:
        json.dump(detections, f)

    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    """Main to test the functions."""
    # path_to_image = "cat.jpg"

    # image = Image.open(path_to_image)
    # image = image.convert("RGB")

    # img = cv2.imread(path_to_image)
    # # draw on image.
    # img = get_classes_in_image(path_to_image)

    # cv2.imwrite("cat_detected.jpg", img)

    # save image
    path_to_video = "video.mp4"
    save_path = "tiktok.mp4"
    dict_path = "tiktok.json"
    detect_video(path_to_video, save_path, dict_path)
    # asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")


if __name__ == "__main__":
    main()
