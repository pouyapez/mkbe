from input_pipeline.ml_img_loader import get_input_pipeline
from tensorpack import *
from tensorpack.dataflow import *
import os


df = get_input_pipeline(64)
test = dataflow.TestDataSpeed(df)
test.start()