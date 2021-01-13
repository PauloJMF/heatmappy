from collections import defaultdict
import os
import random
from moviepy.editor import *
import numpy as np
from PIL import Image
import pandas as pd
from cv2 import cv2

from abc import ABCMeta, abstractmethod
from functools import partial
import io
import os
import random

from matplotlib.colors import LinearSegmentedColormap
import numpy
from PIL import Image

try:
    from PySide import QtCore, QtGui
except ImportError:
    pass


_asset_file = partial(os.path.join, os.path.dirname(__file__), 'assets')


def _img_to_opacity(img, opacity):
        img = img.copy()
        alpha = img.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        img.putalpha(alpha)
        return img


class Heatmapper:
    def __init__(self, point_diameter=50, point_strength=0.2, opacity=0.65,
                 colours='default',
                 grey_heatmapper='PIL'):
        """
        :param opacity: opacity (between 0 and 1) of the generated heatmap overlay
        :param colours: Either 'default', 'reveal',
                        OR the path to horizontal image which will be converted to a scale
                        OR a matplotlib LinearSegmentedColorMap instance.
        :param grey_heatmapper: Required to draw points on an image as a greyscale
                                heatmap. If not using the default, this must be an object
                                which fulfils the GreyHeatmapper interface.
        """

        self.opacity = opacity

        self._colours = None
        self.colours = colours

        if grey_heatmapper == 'PIL':
            self.grey_heatmapper = PILGreyHeatmapper(point_diameter, point_strength)
        elif grey_heatmapper == 'PySide':
            self.grey_heatmapper = PySideGreyHeatmapper(point_diameter, point_strength)
        else:
            self.grey_heatmapper = grey_heatmapper

    @property
    def colours(self):
        return self._colours

    @colours.setter
    def colours(self, colours):
        self._colours = colours

        if isinstance(colours, LinearSegmentedColormap):
            self._cmap = colours
        else:
            files = {
                'default': _asset_file('default.png'),
                'reveal': _asset_file('reveal.png'),
            }
            scale_path = files.get(colours) or colours
            self._cmap = self._cmap_from_image_path(scale_path)

    @property
    def point_diameter(self):
        return self.grey_heatmapper.point_diameter

    @point_diameter.setter
    def point_diameter(self, point_diameter):
        self.grey_heatmapper.point_diameter = point_diameter

    @property
    def point_strength(self):
        return self.grey_heatmapper.point_strength

    @point_strength.setter
    def point_strength(self, point_strength):
        self.grey_heatmapper.point_strength = point_strength

    def heatmap(self, width, height, points, base_path=None, base_img=None):
        """
        :param points: sequence of tuples of (x, y), eg [(9, 20), (7, 3), (19, 12)]
        :return: If base_path of base_img provided, a heat map from the given points
                 is overlayed on the image. Otherwise, the heat map alone is returned
                 with a transparent background.
        """
        heatmap = self.grey_heatmapper.heatmap(width, height, points)
        heatmap = self._colourised(heatmap)
        heatmap = _img_to_opacity(heatmap, self.opacity)

        if base_path:
            background = Image.open(base_path)
            return Image.alpha_composite(background.convert('RGBA'), heatmap)
        elif base_img is not None:
            return Image.alpha_composite(base_img.convert('RGBA'), heatmap)
        else:
            return heatmap


    def heatmap_on_img_path(self, points, base_path):
        width, height = Image.open(base_path).size
        return self.heatmap(width, height, points, base_path=base_path)

    def heatmap_on_img(self, points, img):
        width, height = img.size
        return self.heatmap(width, height, points, base_img=img)

    def _colourised(self, img):
        """ maps values in greyscale image to colours """
        arr = numpy.array(img)
        rgba_img = self._cmap(arr, bytes=True)
        return Image.fromarray(rgba_img)

    @staticmethod
    def _cmap_from_image_path(img_path):
        img = Image.open(img_path)
        img = img.resize((256, img.height))
        colours = (img.getpixel((x, 0)) for x in range(256))
        colours = [(r/255, g/255, b/255, a/255) for (r, g, b, a) in colours]
        return LinearSegmentedColormap.from_list('from_image', colours)


class GreyHeatMapper(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, point_diameter, point_strength):
        self.point_diameter = point_diameter
        self.point_strength = point_strength

    @abstractmethod
    def heatmap(self, width, height, points):
        """
        :param points: sequence of tuples of (x, y), eg [(9, 20), (7, 3), (19, 12)]
        :return: a white image of size width x height with black areas painted at
                 the given points
        """
        pass


class PySideGreyHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)
        self.point_strength = int(point_strength * 255)

    def heatmap(self, width, height, points):
        base_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        base_image.fill(QtGui.QColor(255, 255, 255, 255))

        self._paint_points(base_image, points)
        return self._qimage_to_pil_image(base_image).convert('L')

    def _paint_points(self, img, points):
        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 0))
        pen.setWidth(0)
        painter.setPen(pen)

        for point in points:
            self._paint_point(painter, *point)
        painter.end()

    def _paint_point(self, painter, x, y):
        grad = QtGui.QRadialGradient(x, y, self.point_diameter/2)
        grad.setColorAt(0, QtGui.QColor(0, 0, 0, max(self.point_strength, 0)))
        grad.setColorAt(1, QtGui.QColor(0, 0, 0, 0))
        brush = QtGui.QBrush(grad)
        painter.setBrush(brush)
        painter.drawEllipse(
            x - self.point_diameter/2,
            y - self.point_diameter/2,
            self.point_diameter,
            self.point_diameter
        )

    @staticmethod
    def _qimage_to_pil_image(qimg):
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.ReadWrite)
        qimg.save(buffer, "PNG")

        bytes_io = io.BytesIO()
        bytes_io.write(buffer.data().data())
        buffer.close()
        bytes_io.seek(0)
        return Image.open(bytes_io)


class PILGreyHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)

    def heat_map(self, width, height, points, channels=4):
        mask_dim = (height, width, channels)
        mask = numpy.zeros(mask_dim, dtype=numpy.int8)
        x, y, z = tuple(int(ti/2) for ti in mask_dim)
        mask_size = 15
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        rng = 1
        for x , y in points:
            if x > height or y > width:
                continue 
            mask[int(x),int(y)] = 255

        # mask[(x-rng):(x+rng), (y-rng):(y+rng)] = 255
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 15)
        bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(bw, cv2.DIST_C, 5).astype(np.uint8)
        heat_mask = cv2.applyColorMap(dist, cv2.COLORMAP_JET)
        return heat_mask

    def heatmap(self, width, height, points):
        heat = Image.new('L', (width, height), color=255)

        dot = (Image.open(_asset_file('450pxdot.png')).copy()
                    .resize((self.point_diameter, self.point_diameter), resample=Image.ANTIALIAS))
        dot = _img_to_opacity(dot, self.point_strength)

        for x, y in points:
            x, y = int(x - self.point_diameter/2), int(y - self.point_diameter/2)
            heat.paste(dot, (x, y), dot)

        return heat


class VideoHeatmapper:
    def __init__(self, img_heatmapper):
        self.img_heatmapper = img_heatmapper

    def heatmap_on_video(self, base_video, points,
                         heat_fps=25,
                         keep_heat=True,
                         heat_decay_s=None):
        width, height = base_video.size

        frame_points = self._frame_points(
            points,
            fps=heat_fps,
            keep_heat=keep_heat,
            heat_decay_s=heat_decay_s
        )
        heatmap_frames = self._heatmap_frames(width, height, frame_points)

        return heatmap_frames
        # return CompositeVideoClip([base_video] + list(heatmap_clips))

    def heatmap_on_video_path(self, video_path, points, heat_fps=20):
        base = VideoFileClip(video_path)
        return self.heatmap_on_video(base, points, heat_fps)

    def heatmap_on_image(self, base_img, points,
                         heat_fps=20,
                         duration_s=None,
                         keep_heat=False,
                         heat_decay_s=None):
        base_img = np.array(base_img)
        points = list(points)
        if not duration_s:
            duration_s = max(t for x, y, t in points) / 1000
        base_video = ImageClip(base_img).set_duration(duration_s)

        return self.heatmap_on_video(
            base_video, points,
            heat_fps=heat_fps,
            keep_heat=keep_heat,
            heat_decay_s=heat_decay_s
        )

    def heatmap_on_image_path(self, base_img_path, points,
                              heat_fps=20,
                              duration_s=None,
                              keep_heat=False,
                              heat_decay_s=None):
        base_img = Image.open(base_img_path)
        return self.heatmap_on_image(
            base_img, points,
            heat_fps=heat_fps,
            duration_s=duration_s,
            keep_heat=keep_heat,
            heat_decay_s=heat_decay_s
        )

    @staticmethod
    def _frame_points(pts, fps, keep_heat=False, heat_decay_s=None):
        interval = 1000 // fps
        frames = defaultdict(list)

        if not keep_heat:
            for x, y, t in pts:
                start = (t // interval) * interval
                frames[start].append((x, y))

            return frames

        pts = list(pts)
        last_interval = max(t for x, y, t in pts)

        for x, y, t in pts:
            start = (t // interval) * interval
            pt_last_interval = int(start + heat_decay_s*1000) if heat_decay_s else last_interval
            for frame_time in range(start, pt_last_interval+1, interval):
                frames[frame_time].append((x, y))

        return frames

    def _heatmap_frames(self, width, height, frame_points):
        for frame_start, points in frame_points.items():
            heatmap = self.img_heatmapper.heatmap(width, height, points)
            yield frame_start, np.array(heatmap)

    @staticmethod
    def _heatmap_clips(heatmap_frames, fps):
        interval = 1000 // fps
        for frame_start, heat in heatmap_frames:
            yield (ImageClip(heat)
                   .set_start(frame_start/1000)
                   .set_duration(interval/1000))


def _example_random_points():
    def rand_point(max_x, max_y, max_t):
        return random.randint(0, max_x), random.randint(0, max_y), random.randint(0, max_t)

    return (rand_point(720, 480, 40000) for _ in range(500))

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def heat_list():
    df = pd.read_csv('heatmappy-master/examples/text_input/etr_1592956701406.csv')
    df = df[["GazeX","GazeY","time"]]
    df['Timestamp'] = (df.time - df.time.min()).astype(int)
    df = df.drop(["time"],axis=1)
    records = df.to_records(index=False)
    result = list(records)
    
    img_heatmapper = Heatmapper(colours='default', point_strength=0.6)
    video_heatmapper = VideoHeatmapper(img_heatmapper)

    loaded_video = VideoFileClip('heatmappy-master/examples/video_sample.mp4')

    heatmap_video = video_heatmapper.heatmap_on_video(
        base_video=loaded_video,
        points=result,
        heat_fps=25,
        keep_heat=True,
        heat_decay_s=10
    )

    return heatmap_video

def main():
    cap = cv2.VideoCapture('heatmappy-master/examples/video_sample.mp4')
    count = 0
    frames = heat_list()
    
    while cap.isOpened():
        ret,frame = cap.read()
        heat_frame = next(frames)
        heat_mask = rgba2rgb(heat_frame[1])
        new_img = cv2.addWeighted(frame, 0.5, heat_mask, 0.2, 0)
        cv2.imshow('window-name', new_img)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows() # destroy all opened wind


if __name__ == '__main__':
    main()


        # temp_video = [base_video]
        # step = 20
        # x = 1
        # for i in self.batch(iterable = heatmap_clips, n=step):
        #     clips = list(i)
        #     temp_video = CompositeVideoClip(temp_video + clips)
        #     print(x)
        #     x=x+1
        #     print(len(clips))
        # return temp_video

    # def batch(self,iterable, n=1):
    #     l = list()
    #     count = 0
    #     for i in iterable:
    #         l.append(i)
    #         count = count + 1
    #         if count == n :
    #             yield l

