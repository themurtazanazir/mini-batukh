import random
import numpy as np
from PIL import Image
import cv2

class BaseTransform(object):
    def __init__(self, *ag, **kw):
        pass

    def apply(self, *ag, **kw):
        raise NotImplementedError("apply method has not been implemented yet.")

    def __call__(self, *ag, **kw):

        return self.apply(*ag, **kw)



class Compose(BaseTransform):
    def __init__(self, transforms, probs):
        super(Compose, self).__init__()

        assert len(transforms)==len(probs), "length of `transforms` and `probs` must be equal."
        self.transforms = transforms
        self.probs = probs
    
    def apply(self, img, meta):
        for transform, prob in zip(self.transforms, self.probs):
            if random.random()<prob:
                img, meta = transform(img, meta)

        return img, meta
        
class RandomBgColor(BaseTransform):
    def __init__(self, base_color_range, tolerance):

        super(RandomBgColor, self).__init__()

        self.base_color_range = base_color_range
        self.tolenrance = tolerance

    def apply(self, image, meta={}):
        # assert isinstance(image, Image) and image.mode == 'RGBA', "only PIL RGBA images are supported."
        
        img = np.array(image)
        r_color = random.randint(*self.base_color_range)
        g__color = min(255, max(0, r_color+random.randint(-self.tolenrance, self.tolenrance)))
        b__color = min(255, max(0, r_color+random.randint(-self.tolenrance, self.tolenrance)))
        color = [r_color, g__color, b__color]
        random.shuffle(color)

        meta["RandomBgColor_color"] = color
        img[img[:,:,3]==0] =  [*color, 255]
        image = Image.fromarray(img)
        return image, meta


class RandomRotate(BaseTransform):
    def __init__(self, deg):
        super(RandomRotate, self).__init__()
        self.deg = deg

    def apply(self, img, meta={}):
        rotation = random.randint(-self.deg, self.deg)
        meta["RandomRotate_rotation"] = rotation 
        img = img.rotate(rotation, expand=True)

        return img, meta

class AffineTransform(BaseTransform):
    ## TODO
    pass

class RandomPerespective(BaseTransform):
    def __init__(self, max_change=5):
        self.max_change = max_change
    
    def apply(self, img, meta={}):
        w, h = img.size
        img = self.add_margin(img, self.max_change, (255, 255, 255, 0))
        x0 = x3 = y0 = y1 = self.max_change
        x1 = x2 = x0+w
        y2 = y3 = y0+h
        
        x0 -= random.randint(-int(min(w, h)*0.05), self.max_change)
        y0 -= random.randint(-int(min(w, h)*0.05), self.max_change)
        x1 += random.randint(-int(min(w, h)*0.05), self.max_change)
        y1 -= random.randint(-int(min(w, h)*0.05), self.max_change)
        x2 += random.randint(-int(min(w, h)*0.05), self.max_change)
        y2 += random.randint(-int(min(w, h)*0.05), self.max_change)
        x3 -= random.randint(-int(min(w, h)*0.05), self.max_change)
        y3 += random.randint(-int(min(w, h)*0.05), self.max_change)

        pts = np.array([[x0, y0],[ x1, y1], [x2, y2], [x3, y3]])

        img_arr = self.four_point_transform(np.array(img), pts)

        return Image.fromarray(img_arr), meta
    
    
    def add_margin(self, pil_img, pad_size, color):
        width, height = pil_img.size
        new_width = width + (2*pad_size)
        new_height = height + (2*pad_size)
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (pad_size, pad_size))
        return result

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = pts.astype("float32")
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

class RandomResize(BaseTransform):
    def __init__(self, min_size_per, max_size_per):
        super(RandomResize, self).__init__()
        self.min_size_per = min_size_per
        self.max_size_per = max_size_per
    
    def apply(self, img, meta={}):
        size = random.uniform(self.min_size_per, self.max_size_per)
        meta["RandomResize_size"] = size
        img = img.resize((int(img.size[0]*size), int(img.size[1]*size)), Image.ANTIALIAS)
        return img, meta

## TODO:
# - Salt and Pepper
# - Blurs (gaussian, simple, motion, median)
# - Resampling



##