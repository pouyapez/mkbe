from tensorpack import *
from tensorpack.dataflow import *
import cv2, os
import numpy as np


class FileReader:
    def __init__(self, imgdir, weightsdir):
        self.imgrt = imgdir + "{:}.jpg"
        self.weights = np.load(weightsdir)

    def read_arr_byid(self, movieid):
        filename = self.imgrt.format(movieid)

        img = cv2.imread(filename).astype(np.float32) / 128.0 - 1.0
        movie_weights = self.weights[movieid, :]
        return img, movie_weights

    def gen_imgid(self, source=None):
        if source is None:
            source = range(1, 1683)

        for imgid in source:
            filename = self.imgrt.format(imgid)
            if os.path.exists(filename):
                yield imgid


def get_input_pipeline(batch_size):
    reader = FileReader("../assets/images/", "../assets/weights/movie_weights.npy")
    df_in = dataflow.DataFromGenerator(reader.gen_imgid)
    df = dataflow.RepeatedData(df_in, -1)
    df = dataflow.MultiProcessMapDataZMQ(df, 4, reader.read_arr_byid, buffer_size=24, strict=False)
    df = dataflow.BatchData(df, batch_size, remainder=True)
    return df