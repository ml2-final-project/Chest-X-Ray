import cv2
import numpy as np


# assumes input is a PIL Image
class PILToNumpy:
    def __call__(self, image):
        return np.array(image)


# assumes input data is a numpy array
# should operate on Greyscale image
class EqualizeHistogram:
    def __call__(self, image):
        return cv2.equalizeHist(image)


# assumes input data is a numpy array
# should operate on Grayscale image
class Gray2RGB:
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)


# assumes input data is a numpy array
# should operate on RGB images
class GaussianBlur:
    def __init__(self, window):
        self.window = window

    def __call__(self, image):
        return cv2.GaussianBlur(image, (self.window,
                                        self.window), 0)

# based on code from github:
#  should produce long_side x long_side images filled with pixel_mean
#  https://github.com/jfhealthcare/Chexpert/blob/47e1bf7f8ed2b95c8985ae43801ffffde92fa205/data/dataset.py#L58
# assumes input data is a numpy array
# should operate on RGB Image
class FixRatioResize:
    def __init__(self, long_side, pixel_mean):
        self.long_side = long_side
        self.pixel_mean = pixel_mean

    def __call__(self, image):
        return self._fix_ratio(image)

    def _border_pad(self, image):
        h, w, c = image.shape

        image = np.pad(
            image,
            ((0, self.long_side - h),
             (0, self.long_side - w), (0, 0)),
            mode='constant', constant_values=self.pixel_mean
        )

        return image

    def _fix_ratio(self, image):
        h, w, c = image.shape

        if h >= w:
            ratio = h * 1.0 / w
            h_ = self.long_side
            w_ = round(h_ / ratio)
        else:
            ratio = w * 1.0 / h
            w_ = self.long_side
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_),
                           interpolation=cv2.INTER_LINEAR)

        image = self._border_pad(image)

        return image