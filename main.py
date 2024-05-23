
import cv2
import PIL
import numpy as np
import os
import torch

mode2config = {
    "eye (with eye-glasses)":        "haarcascade_eye_tree_eyeglasses.xml",
    "eye":                           "haarcascade_eye.xml",
    "left eye (2 splits)":           "haarcascade_lefteye_2splits.xml",
    "right eye (2 splits)":          "haarcascade_righteye_2splits.xml",
    "frontal face (extended)":       "haarcascade_frontalcatface_extended.xml",
    "frontal cat face":              "haarcascade_frontalcatface.xml",
    "frontal face (alternate 2)":    "haarcascade_frontalface_alt2.xml",
    "frontal face (alternate tree)": "haarcascade_frontalface_alt_tree.xml",
    "frontal face (alternate 1)":    "haarcascade_frontalface_alt.xml",
    "frontal face (default)":        "haarcascade_frontalface_default.xml",
    "full body":                     "haarcascade_fullbody.xml",
    "lower body":                    "haarcascade_lowerbody.xml",
    "upper body":                    "haarcascade_upperbody.xml",
    "profile face":                  "haarcascade_profileface.xml",
    "smile":                         "haarcascade_smile.xml",
    "license plate number":          "haarcascade_license_plate_rus_16stages.xml",
    "russian plate number":          "haarcascade_russian_plate_number.xml",
}

def tensor2pil(image):
    return PIL.Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def clamp(xmin, xval, xmax):
    if xval < xmin:
        return xmin
    if xval > xmax:
        return xmax
    return xval

def cropImage(image:PIL.Image, x1, y1, x2, y2):
    mw, mh = image.size
    return image.crop((
        clamp(0, x1, mw),
        clamp(0, y1, mh),
        clamp(0, x2, mw),
        clamp(0, y2, mh)))

class ImageCropper:
   
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": ([
                    #"eye (with eye-glasses)",
                    #"eye",
                    #"left eye (2 splits)",
                    #"right eye (2 splits)",
                    "frontal face (extended)",
                    #"frontal cat face",
                    #"frontal face (alternate 2)",
                    #"frontal face (alternate tree)",
                    #"frontal face (alternate 1)",
                    "frontal face (default)",
                    #"full body",
                    #"lower body",
                    #"upper body",
                    "profile face",
                    #"smile",
                    #"license plate number",
                    #"russian plate number",
                ],),
                "padding": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    # Configuration for the classifier can be found at:
    #   https://github.com/opencv/opencv/tree/master/data/haarcascades
    def parseMode(self, mode):
        folder = os.path.dirname(__file__)
        folder = os.path.join(folder, "classifiers")
        return os.path.join(folder, mode2config[mode])

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self, image, mode, padding):

        print(f"Input mode: {mode}")

        config = self.parseMode(mode)

        classifier = cv2.CascadeClassifier(config)

        pilImage = tensor2pil(image)

        converted = cv2.cvtColor(
            np.array(pilImage),
            cv2.COLOR_RGB2GRAY)
 
        founds = classifier.detectMultiScale(
            converted, 
            scaleFactor=1.1,
            minNeighbors=5, 
            minSize=(30, 30), 
            flags=cv2.CASCADE_SCALE_IMAGE)

        count = len(founds) 
        if count == 0:
            raise Exception(f"Found no matched")
        if count > 1:
            raise Exception(f"Found multiple matched")

        (x, y, w, h) = founds[0]
        padded = cropImage(
            pilImage,
            x-padding,
            y-padding,
            x+w+padding,
            y+h+padding) 
        return (pil2tensor(padded),)

    #@classmethod
    #def IS_CHANGED(s, image, mode):
    #    return ""

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageCropper": ImageCropper
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropper": "Image cropping tool"
}

