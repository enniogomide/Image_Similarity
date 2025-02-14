#
# for step 1
#
import pandas as pd
from shutil import move
import os
from tqdm import tqdm
#import opendatasets as od

#
# for step 2
#

#import itertools

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

#
# for step 3
#
import tensorflow as tf
from pathlib import Path
import numpy as np
from tqdm import tqdm
from annoy import AnnoyIndex

#hide
import glob
import json
from annoy import AnnoyIndex
#from scipy import spatial
import pickle
from IPython.display import Image as dispImage

from PIL import Image
import matplotlib.image as mpimg

from urllib.request import urlretrieve
import time




# only execute for the first time
os.mkdir('Fashion_data')
os.mkdir('kaggle')


# variables used in the code

google_drive = '/ImageSimilarity'
data_dir = 'Fashion_data/categories'
imgvec_path = 'img_vectors'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
N_FEATURES = 256

#
# Step 1: Data Acquisition
#

dataset = 'https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small'
od.download(dataset)

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# load styles with picture information
df = pd.read_csv('styles.csv', usecols=['id','masterCategory']).reset_index()
df['id'] = df['id'].astype('str')

# read images
all_images = os.listdir('images/')
co = 0
os.mkdir('Fashion_data/categories')

# copy all images, based on categories to specifica folders
for image in tqdm(all_images):
    category = df[df['id'] == image.split('.')[0]]['masterCategory']
    category = str(list(category)[0])
    if not os.path.exists(os.path.join('Fashion_data/categories', category)):
        os.mkdir(os.path.join('Fashion_data/categories', category))
    path_from = os.path.join('images', image)
    path_to = os.path.join('Fashion_data/categories', category, image)
    move(path_from, path_to)
    co += 1
print('Moved {} images.'.format(co))

#
#Step 2: Encoder Fine-tuning [optional]
#
# show libraries versions
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

#
# Data augmentation and preprocessing
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = False
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)

# Load the pre-trained model from TensorFlow Hub
MODULE_HANDLE = 'https://tfhub.dev/google/bit/m-r50x3/1'
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

# Build the model using the Functional API instead of Sequential
# This is because hub.KerasLayer may not be directly compatible with Sequential in some TensorFlow/Keras versions
input_tensor = tf.keras.layers.Input(shape=IMAGE_SIZE + (3,))
hub_layer = hub.KerasLayer(MODULE_HANDLE, trainable=False)(input_tensor)
# Wraped the hub.KerasLayer call in a Lambda layer to delay execution
hub_layer = tf.keras.layers.Lambda(lambda x: hub.KerasLayer(MODULE_HANDLE, trainable=False)(x))(input_tensor)

dropout1 = tf.keras.layers.Dropout(rate=0.2)(hub_layer)
dense1 = tf.keras.layers.Dense(N_FEATURES,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))(dropout1)
dropout2 = tf.keras.layers.Dropout(rate=0.2)(dense1)
output_tensor = tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))(dropout2)

model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
model.summary()

# Define optimiser and loss
lr = 0.003 * BATCH_SIZE / 512
SCHEDULE_LENGTH = 500
SCHEDULE_BOUNDARIES = [200, 300, 400]

# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES,
                                                                   values=[lr, lr*0.1, lr*0.001, lr*0.0001])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

#
# Train the model (only run if required for new data)
#
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
hist = model.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps).history

#hide
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])

#
# to execute if run previously steps for training the model and will save
# 
if not os.path.exists('ImgSim'):
    os.mkdir('ImgSim')

feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)
feature_extractor.save('ImgSim/bit_feature_extractor.keras')

saved_model_path = 'ImgSim/bit_model'
tf.saved_model.save(model, saved_model_path)

# Step 3: Image Vectorization

categories_path = 'Fashion_data/categories/'
img_paths = []

#
# load images path into variable
#
for path in Path(categories_path).rglob('*.jpg'):
  img_paths.append(path)
np.random.shuffle(img_paths)

#
# function to load images and convert to tensor
#
def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img

# if it was not used the transfer learning model, it will load the model
TRANSFER_LEARNING_FLAG = 0
if TRANSFER_LEARNING_FLAG:
  module = tf.keras.models.load_model('/content/drive/MyDrive/ImgSim/bit_feature_extractor')
else:
  # module_handle = "https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1"
  module_handle = "https://tfhub.dev/google/bit/m-r50x3/1"
  module = hub.load(module_handle)

# create de folder for the images vectors

if os.path.exists(imgvec_path) == False:
  os.mkdir(imgvec_path)

# Step 4: Metadata and Indexing
tqdm.pandas()

#
# create the vector features for each image
#
for filename in tqdm(img_paths[:5000]):
    img = load_img(str(filename))
    features = module(img)
    feature_set = np.squeeze(features)
    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    out_path_file = os.path.join(imgvec_path, outfile_name)
    np.savetxt(out_path_file, feature_set, delimiter=',')

#
# testing local
#
tqdm.pandas()
test_img = 'Fashion_data/categories/Footwear/10035.jpg'
dispImage(test_img)

#
# load styles file and save it change to str the id 
#
styles = pd.read_csv('styles.csv', on_bad_lines='skip')
styles['id'] = styles['id'].astype('str')
styles.to_csv('styles.csv', index=False)
#
# function to find de image
#
def match_id(fname):
  return styles.index[styles.id==fname].values[0]

# Defining data structures as empty dict
file_index_to_file_name = {}
file_index_to_file_vector = {}
file_index_to_product_id = {}

# Configuring annoy parameters
dims = 6144
n_nearest_neighbors = 20
trees = 10000

t = AnnoyIndex(dims, metric='angular')

# Reads all file names which stores feature vectors
allfiles_path = 'img_vectors/*.npz'
allfiles = glob.glob(allfiles_path)

for findex, fname in tqdm(enumerate(allfiles)):
  file_vector = np.loadtxt(fname)
  file_name = os.path.basename(fname).split('.')[0]
  file_index_to_file_name[findex] = file_name
  file_index_to_file_vector[findex] = file_vector
  try:
    file_index_to_product_id[findex] = match_id(file_name)
  except IndexError:
    pass
  t.add_item(findex, file_vector)

#hide-output
t.build(trees)
t.save('t.ann')

file_path = 'ImgSim/'
t.save(file_path+'indexer.ann')
pickle.dump(file_index_to_file_name, open(file_path+"file_index_to_file_name.p", "wb"))
pickle.dump(file_index_to_file_vector, open(file_path+"file_index_to_file_vector.p", "wb"))
pickle.dump(file_index_to_product_id, open(file_path+"file_index_to_product_id.p", "wb"))
#
# Step 5: Local Testing, for image similarity
#
url = 'https://m.media-amazon.com/images/I/51C17a0EFRL._AC_SX575_.jpg'
img_retrived = urlretrieve(url, 'img.jpg')

test_img = 'img.jpg'
topK = 4

test_vec = np.squeeze(module(load_img(test_img)))

basewidth = 224
img = Image.open(test_img)
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
img

path_images = "Fashion_data/categories"
path_dict = {}
for path in Path(path_images).rglob('*.jpg'):
  path_dict[path.name] = path

nns = t.get_nns_by_vector(test_vec, n=topK)
plt.figure(figsize=(20, 10))
for i in range(topK):
  x = file_index_to_file_name[nns[i]]
  x = path_dict[x+'.jpg']
  y = file_index_to_product_id[nns[i]]
  title = '\n'.join([str(j) for j in list(styles.loc[y].values[-5:])])
  plt.subplot(1, topK, i+1)
  plt.title(title)
  plt.imshow(mpimg.imread(x))
  plt.axis('off')
plt.tight_layout()
#
# Step 6: API Call
#

# to save the utils.py
%%writefile utils.py
### ----utils.py---- ###

import os
import tensorflow as tf
import tensorflow_hub as hub
from annoy import AnnoyIndex

root_path = 'ImgSim'

class Encoder:
  TRANSFER_LEARNING_FLAG = 0   # 1 if using transfer learning, 0 otherwise
  if TRANSFER_LEARNING_FLAG:
    encoder = tf.keras.models.load_model(os.path.join(root_path, 'bit_feature_extractor'))
  else:
    module_handle = "https://tfhub.dev/google/bit/m-r50x3/1"
    encoder = hub.load(module_handle)

class Indexer:
  dims = 6144
  topK = 6
  indexer = AnnoyIndex(dims, 'angular')
  indexer.load(os.path.join(root_path, 'indexer.ann'))

encoder = Encoder()
indexer = Indexer()
#
# Streamlit app - app.py
#
%%writefile app.py
### ----app.py---- ###

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from annoy import AnnoyIndex
import glob
import os
import tensorflow as tf
import tarfile
import pickle
from pathlib import Path
import time
from utils import encoder, indexer

root_path = 'ImgSim/'

start_time = time.time()
encoder = encoder.encoder
print("---Encoder--- %s seconds ---" % (time.time() - start_time))

topK = 6

start_time = time.time()
t = indexer.indexer
print("---Indexer--- %s seconds ---" % (time.time() - start_time))

# load the meta data
meta_data = pd.read_csv(os.path.join('styles.csv'))

# load the mappings
file_index_to_file_name = pickle.load(open(os.path.join(root_path ,'file_index_to_file_name.p'), 'rb'))
file_index_to_file_vector = pickle.load(open(os.path.join(root_path ,'file_index_to_file_vector.p'), 'rb'))
file_index_to_product_id = pickle.load(open(os.path.join(root_path ,'file_index_to_product_id.p'), 'rb'))

# load image path mapping
path_dict = {}
for path in Path('Fashion_data/categories').rglob('*.jpg'):
  path_dict[path.name] = path

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img

query_path = 'user_query.jpg'

st.title("Image Similarity App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save(query_path)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Top similar images...")

    start_time = time.time()
    test_vec = np.squeeze(encoder(load_img(query_path)))
    print("---Encoding--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    nns = t.get_nns_by_vector(test_vec, n=topK)
    print("---SimilarityIndex--- %s seconds ---" % (time.time() - start_time))

    img_files = []
    img_captions = []

    start_time = time.time()
    for i in nns:
      #image files
      img_path = str(path_dict[file_index_to_file_name[i]+'.jpg'])
      img_file = Image.open(img_path)
      img_files.append(img_file)
      #image captions
      item_id = file_index_to_product_id[i]
      img_caption = '\n'.join([str(j) for j in list(meta_data.loc[item_id].values[-5:])])
      img_captions.append(img_caption)
    print("---Output--- %s seconds ---" % (time.time() - start_time))

    st.image(img_files, caption=img_captions, width=200)

    #
    # to execute the app: streamlit run app.py
    # 
