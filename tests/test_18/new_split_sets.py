import sh

print(sh.cat("new_split_sets.py"))
print(sh.cat("new_unet.py"))

# base imports
from glob import glob
import itertools
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import os
# from osgeo import gdal
import pandas as pd
import rasterio
from random import shuffle
import sklearn.metrics as skmetrics
# %tensorflow_version 2.x
import tensorflow as tf
import pickle
import random
import sys
import datetime

random_seed = 31415
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

#tf.config.set_visible_devices([], 'GPU')  # use the CPU instead of GPU

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

"""### Loading in file links"""

# read in the CSV file that contains file links to the entire dataset
dataset_file = 'https://storage.googleapis.com/naip_public/dataset.csv'
df = pd.read_csv(dataset_file)
print(len(df))
df.head(5)

# use these as base directories for reading the dataset from cloud storage
image_base = '../../hw2_dataset/images'
target_base = '../../hw2_dataset/targets_v2'

# load in file links from cloud storage
samples = list()
for idx, row in df.iterrows():
    image_file = os.path.join(image_base, row['images'])
    target_file = os.path.join(target_base, row['targets'])

    samples.append((image_file, target_file))

#with rasterio.open(samples[0][0]) as src:
#    img = src.read()

#with rasterio.open(samples[0][1]) as src:
#    tgt = src.read()

#print(img.shape)
#print(tgt.shape)
#print(len(samples))

"""### Visualization"""

# NLCD color scheme
nlcd_color_mapping = {
    11: (70, 107, 159),
    12: (209, 222, 248),
    21: (222, 197, 255),
    22: (217, 146, 130),
    23: (235, 0, 0),
    24: (171, 0, 0),
    31: (179, 172, 159),
    41: (104, 171, 95),
    42: (28, 95, 44),
    43: (181, 197, 143),
    52: (204, 184, 121),
    71: (223, 223, 194),
    81: (220, 217, 57),
    82: (171, 108, 40),
    90: (184, 217, 235),
    95: (108, 159, 184),
}

# NLCD class mapping
nlcd_name_mapping = {
    11: 'Open Water',
    12: 'Perennial Ice/Snow',
    21: 'Developed, Open Space',
    22: 'Developed, Low Intensity',
    23: 'Developed, Medium Intensity',
    24: 'Developed, High Intensity',
    31: 'Barren Land',
    41: 'Deciduous Forest',
    42: 'Evergreen Forest',
    43: 'Mixed Forest',
    52: 'Shrub/Scrub',
    71: 'Grassland/Herbaceous',
    81: 'Pasture/Hay',
    82: 'Cultivated Crops',
    90: 'Woody Wetlands',
    95: 'Emergent Herbaceous Wetlands',
}

# %%

nlcd_mapping_reduced = {
    11: 0,  # Open Water -> water
    12: 1,  # Perennial Ice/Snow -> snow/ice
    21: 2,  # Developed, Open Space -> built area
    22: 2,  # Developed, Low Intensity -> built area
    23: 2,  # Developed, Medium Intensity -> built area
    24: 2,  # Developed, High Intensity -> built area
    31: 3,  # Barren Land -> bare
    41: 4,  # Deciduous Forest -> forest
    42: 4,  # Evergreen Forest -> forest
    43: 4,  # Mixed Forest -> forest
    52: 5,  # Shrub/Scrub -> shrub/scrub
    71: 6,  # Grassland/Herbaceous -> grass
    81: 7,  # Pasture/Hay -> crops
    82: 7,  # Cultivated Crops -> crops
    90: 8,  # Woody Wetlands -> wetlands
    95: 8,  # Emergent Herbaceous Wetlands -> wetlands
     0: 9, # Dummy class because there seem to be 0s in the target...
}

nlcd_color_mapping_reduced = {
    0: (70, 107, 159),
    1: (209, 222, 248),
    2: (235, 0, 0),
    3: (179, 172, 159),
    4: (28, 95, 44),
    5: (204, 184, 121),
    6: (223, 223, 194),
    7: (220, 217, 57),
    8: (108, 159, 184),
    9: (255, 255, 255),
}

nlcd_name_mapping_reduced = {
    0: 'Water',
    1: 'Snow/Ice',
    2: 'Built Area',
    3: 'Barren',
    4: 'Forest',
    5: 'Shrub/Scrub',
    6: 'Grass',
    7: 'Crops',
    8: 'Wetlands',
    9: 'None'
}

# %% training params

model_type = sys.argv[1]
learning_rate = 0.001
batch_size = 16
num_epochs = 800
num_classes = len(set(nlcd_mapping_reduced.values()))
num_filters = 64
dropout_ratio = 0.7
input_shape = (256, 256, 17)
patience = 150

# %%

# build a matplotlib colormap so we can visualize this data
colors = list()
for idx, class_val in enumerate(nlcd_color_mapping.keys()):
    red, green, blue = nlcd_color_mapping[class_val]

    color_vec = np.array([red/255, green/255, blue/255])
    colors.append(color_vec)

colors = np.stack(colors)
cmap = matplotlib.colors.ListedColormap(colors=colors, N=len(colors))

bounds = list(range(len(colors)))
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors))

# write a function so that we can display image/target/predictions
def display_image_target(display_list):
    plt.figure(dpi=200)
    title = ['Image', 'Target', 'Prediction']

    for idx, disp in enumerate(display_list):
        plt.subplot(1, len(display_list), idx+1)
        plt.title(title[idx], fontsize=6)
        plt.axis('off')

        if title[idx] == 'Image':
            arr = disp.numpy()
            rgb = np.stack([arr[:, :, 3], arr[:, :, 2], arr[:, :, 1]], axis=-1) / 3000.0
            plt.imshow(rgb)

        elif title[idx] == 'Target':
            tgt = disp.numpy().squeeze()
            plt.imshow(tgt, interpolation='none', norm=norm, cmap=cmap)

        elif title[idx] == 'Prediction':
            pred = np.argmax(disp, axis=-1) # argmax across probabilities to get class outputs
            plt.imshow(pred, interpolation='none', norm=norm, cmap=cmap)

    plt.show()
    plt.close()

"""### Tensorflow dataset loader"""
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# rotates, zooms, and flips depending on a random integer
def normalize_img(img):
    channel_wise_mean = np.mean(img, axis=(0, 1))
    channel_wise_std = np.std(img, axis=(0, 1))
    channel_wise_std[channel_wise_std == 0.0] = 1e-7 # np.finfo(float).eps
    zscore = (img - channel_wise_mean)/channel_wise_std
    return zscore
    # return (0.5 * (np.tanh(0.01 * zscore)))

def normalized_diff(b1, b2):
    return (b1-b2)/(b1+b2+1e-7)

def add_indices(_img):
    '''
    Band 1 Visible (0.43 - 0.45 µm) 30 m. -> 0 (coastal aerosol)
    Band 2 Visible (0.450 - 0.51 µm) 30 m. -> 1 (blue)
    Band 3 Visible (0.53 - 0.59 µm) 30 m. -> 2 (green)
    Band 4 Red (0.64 - 0.67 µm) 30 m. -> 3 (red)
    Band 5 Near-Infrared (0.85 - 0.88 µm) 30 m. -> 4 (nir)
    Band 6 SWIR 1(1.57 - 1.65 µm) 30 m. -> 5 (swir1)
    Band 7 SWIR 2 (2.11 - 2.29 µm) 30 m. -> 6 (swir2)
    
    so it turns out MNDWI and NDSI are the same thing, but designed to measure different things
        (Water 2020, 12, 1339; doi:10.3390/w12051339)
    '''
    
    _out = np.zeros(input_shape)
    _out[:,:,:8] = _img
    ndvi = normalized_diff(_img[:,:,4], _img[:,:,3]) # ndvi
    ndbi = normalized_diff(_img[:,:,5], _img[:,:,4]) # ndbi
    mndwi = normalized_diff(_img[:,:,2], _img[:,:,5]) # mndwi
    # ndsi = normalized_diff(_img[:,:,2], _img[:,:,5]) # ndsi
    msavi = (2 * _img[:,:,4] + 1 - np.sqrt ((2 * _img[:,:,4] + 1)**2 - 8 * (_img[:,:,4] - _img[:,:,3]))) / 2 # msavi
    gndvi = normalized_diff(_img[:,:,4], _img[:,:,2]) # gndvi
    evi = 2.5* (_img[:,:,4] - _img[:,:,3]) / (_img[:,:,4] + 6*_img[:,:,3] - 7.5*_img[:,:,1] + 1 + 1e-7) 
    ai = normalized_diff(_img[:,:,1], _img[:,:,2]) # aerosol index https://surfaceheat.sites.yale.edu/sites/default/files/files/Coastal%20Aerosol%20Band_1.pdf
    
    # and now for the more bizarre ones:
    bu = ndbi - ndvi # Built Up Index Bhatti (2014)
    dbsi = -mndwi - ndvi # Bare Soil Index Rasul et al (2018) -> focus was in dry climates

    # finally we assign the vars
    _out[:,:, 8] = ndvi
    _out[:,:, 9] = ndbi
    _out[:,:,10] = mndwi
    _out[:,:,11] = msavi
    _out[:,:,12] = gndvi 
    _out[:,:,13] = evi
    _out[:,:,14] = ai
    _out[:,:,15] = bu
    _out[:,:,16] = dbsi 
    return _out

def relabel_tgt(tgt):
    with rasterio.open(tgt) as src:
        # the transpose isn't absolutely necessary, but is more consistent with the rest of the code
        tgt_data = np.transpose(src.read(), axes=(1, 2, 0)).astype('uint8')
        tgt_out = np.zeros(tgt_data.shape)
        tgt_out[:] = tgt_data
        # arguably if we have some initial labels that are not zero and not from the NLCD
        # then this will catch them.
        for nlcd_label, reduced_label in nlcd_mapping_reduced.items():
            tgt_out[tgt_data == nlcd_label] = reduced_label
        illegal_labels = tgt_out[tgt_out >= num_classes]
        if illegal_labels.size != 0:
            print(f"Illegal label found in target! labels {illegal_labels}")
    return tgt_out

def read_sample(data_path: str) -> tuple:
    path = data_path.numpy()
    image_path, target_path = path[0].decode('utf-8'), path[1].decode('utf-8')
    
    with rasterio.open(image_path) as src:
        img = np.transpose(src.read(), axes=(1, 2, 0)).astype('uint16')
        # img[:,:,9] == 0 where invalid; good for ma 
        # img[:,:,8] == 1 where cloud is aka invalid
        mask = np.logical_or(img[:,:,9] == 0, img[:,:,8] == 1)
        img[mask==True] = 0 # zero it out.
        img = img[:,:,:8].astype('float32')
        img = add_indices(img)
        img = normalize_img(img)

    tgt_out = relabel_tgt(target_path)
    tgt_out[mask==True] = 9 # invalid label
#    with rasterio.open(target_path) as src:
#        tgt = np.transpose(src.read(), axes=(1, 2, 0)).astype(np.uint8)

#    tgt_out = np.zeros(tgt.shape, dtype=np.uint8)
#    for nlcd_label, reduced_label in nlcd_mapping_reduced.items():
#        tgt_out[tgt == nlcd_label] = reduced_label
#    for class_val, nlcd_val in enumerate(nlcd_color_mapping.keys()):
#        tgt_out[tgt == nlcd_val] = class_val

#    uniq, inv = np.unique(tgt, return_inverse=True)
#    tgt_out = np.array([nlcd_mapping_reduced[x] for x in uniq])[inv].reshape(tgt.shape)
    # print(f"tshape {tgt_out.shape}")
    #tgt_out = to_categorical(tgt_out, num_classes=num_classes)

    return (img, tgt_out)

@tf.function
def tf_read_sample(data_path: str) -> dict:
    # wrap custom dataloader into tensorflow
    [image, target] = tf.py_function(read_sample, [data_path], [tf.float32, tf.uint8])

    # explicitly set tensor shapes
    image.set_shape(input_shape)
    target.set_shape((256, 256, 1))

    return {'image': image, 'target': target}

@tf.function
def load_sample(sample: dict) -> tuple:
    image = tf.image.resize(sample['image'], (256, 256))
    target = tf.image.resize(sample['target'], (256, 256))

    # cast to proper data types
    image = tf.cast(image, tf.float32)
    target = tf.cast(target, tf.uint8)

    return image, target

@tf.function
def augment_sample(_img, _tgt) -> tuple:
    if tf.random.uniform([]) > 0.5: # random rotate
        num_rot = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        _img = tf.image.rot90(_img, k=num_rot)
        _tgt = tf.image.rot90(_tgt, k=num_rot)
    if tf.random.uniform([]) > 0.5: # random flip lr
        _img = tf.image.flip_left_right(_img)
        _tgt = tf.image.flip_left_right(_tgt)
#    if tf.random.uniform([]) > 0.5: # random flip ud
#        _img = tf.image.flip_up_down(_img)
#        _tgt = tf.image.flip_up_down(_tgt)
#    if tf.random.uniform([]) > 0.5:
#        fake_seed = np.random.randint(35000) #tf.random.uniform(shape=[], minval=0, maxval=30000, dtype=tf.int32).numpy()
#        crop_size = np.random.randint(200, 257) #tf.random.uniform(shape=[], minval=20, maxval=257, dtype=tf.int32).numpy()
#        _img = tf.image.resize(
#                tf.image.random_crop(_img, size=(crop_size, crop_size, input_shape[2]), seed=fake_seed), method='nearest', size=(256, 256))
#        _tgt = tf.image.resize(
#                tf.image.random_crop(_tgt, size=[crop_size , crop_size, 1], seed=fake_seed), method='nearest', size=(256, 256))

    return _img, _tgt

# create tensorflow dataset from file link
random.shuffle(samples)
num_samples = len(samples)
pickle.dump(samples, open(f"sample_ordering_{model_type}.pkl", "wb+"))

train_ratio = 0.8
valid_ratio = 0.195 # remaining 0.5% reserved for visualization/testing.

train_slice = slice(0, int(train_ratio * num_samples))
valid_slice = slice(int(train_ratio * num_samples), int((train_ratio+valid_ratio)*num_samples))
test_slice = slice(int((train_ratio+valid_ratio) * num_samples), num_samples+1)
print(f"Indices in samples list:\n\tTraining: {train_slice}\n\tValidation: {valid_slice}\n\tTesting: {test_slice}")

train_samples = samples[train_slice]
valid_samples = samples[valid_slice]
test_samples = samples[test_slice]

def make_ds(_sample_list):
    _ds = tf.data.Dataset.from_tensor_slices(_sample_list)
    # read in image/target pairs
    _ds = _ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # read in as tensors
    return _ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = make_ds(train_samples)
valid_ds = make_ds(valid_samples)

train_ds = train_ds.map(augment_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = train_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

#ds = make_ds(samples).cache().shuffle(len(samples), reshuffle_each_iteration=False, seed=random_seed)
#ds = ds.enumerate()
#train_ds = ds.filter(lambda i, data : i % 10 < 8).map(lambda i, data: data).batch(batch_size)
#valid_ds = ds.filter(lambda i, data : i % 10 >= 8).map(lambda i, data: data).batch(batch_size)

# %%

# model def
from new_unet import unet
model = unet(input_shape, 64, num_classes=num_classes, dropout_ratio=dropout_ratio, num_layers=4)

def get_pix_weights(samp, nclass=num_classes):
    res = np.zeros((nclass))
    for _, __tgt in samp:
        tgt_out = relabel_tgt(__tgt).astype('int64')
        res += np.bincount(tgt_out.flatten(), minlength=10)
    return (len(samp)*(256**2))/(res*10) # how sklearn does it

#cw = get_pix_weights(samples, num_classes)
## make the weight of the None class less than the most popular class, as it is not supposed to be there
#cw[-1] = np.min(cw) - np.min(cw)/4
#cw = cw/np.min(cw) # normalize
#cw_dict = dict(enumerate(cw))

#from logits=True means that we don't apply softmax to the end of the last layer
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=['accuracy'])
model.summary()

# tf.keras.utils.plot_model(model, to_file=f"{model_type}.png", show_shapes=True)

checkpoint_filepath = f"best_{model_type}.hdf5" #'weights.{epoch:02d}-{val_accuracy:.4f}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='auto',
    save_best_only=True)

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,
                               cooldown=0,
                               patience=patience,
                               min_lr=0.5e-7,
                               verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Model weights are saved at the end of every epoch, if it's the best seen so far.
# model.load_weights(checkpoint_filepath)
history = model.fit(train_ds, epochs=num_epochs, validation_data=valid_ds,
                    verbose=1,
                    callbacks=[model_checkpoint_callback, lr_reducer, tensorboard_callback])


model.save(f"just_in_case_{model_type}.hdf5")

pickle.dump(history.history, open(f"model_history_{model_type}.pkl","wb+"))

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

for i, (img, tgt) in enumerate(test_samples):
    print(f"Running image: {img}")
    img_data, tgt_data = load_sample(tf_read_sample((img, tgt)))
    predicted = model.predict(tf.expand_dims(img_data, axis=0))
    predicted = np.argmax(predicted, axis=-1)[0]
    print(img_data.numpy().shape, tgt_data.numpy().shape, predicted.shape)
    err = np.sum(tgt_data.numpy()[:,:,0] != predicted)/(256*256)
    print(f"\tPredicted Shape:{predicted.shape}\tError: {err}")
    np.savez_compressed(f"{'.'.join(img.split('.')[:-1]).split('/')[-1]}_{model_type}.npz", img_data.numpy(), tgt_data.numpy()[:,:,0], predicted)


