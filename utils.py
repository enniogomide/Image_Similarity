### ----utils.py---- ###

import os
import tensorflow as tf
import tensorflow_hub as hub
from annoy import AnnoyIndex

root_path = 'ImgSim'

class Encoder:
  TRANSFER_LEARNING_FLAG = 0
  if TRANSFER_LEARNING_FLAG:
    #module = tf.keras.models.load_model('/content/drive/MyDrive/ImgSim/bit_feature_extractor')
    encoder = tf.keras.models.load_model(os.path.join(root_path, 'bit_feature_extractor'))
  else:
    #module_handle = "https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1"
    module_handle = "https://tfhub.dev/google/bit/m-r50x3/1"
    encoder = hub.load(module_handle)
    #encoder = tf.keras.models.load_model(os.path.join(root_path, 'bit_feature_extractor'))

class Indexer:
  #dims = 256
  dims = 6144
  topK = 6
  indexer = AnnoyIndex(dims, 'angular')
  indexer.load(os.path.join(root_path, 'indexer.ann'))

encoder = Encoder()
indexer = Indexer()
#
