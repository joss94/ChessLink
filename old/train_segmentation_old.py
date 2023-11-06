
import os, shutil

os.environ["CUDA_VISIBLE_DEVICES"]="2" # first gpu

import argparse
import numpy as np
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
import numpy as np
import pandas as pd
import keras.backend as K
import keras
import keras.layers as layers

from sklearn.model_selection import train_test_split

import cv2
import glob

import tensorflow as tf
from matplotlib import pyplot as plt
import json

# Directories
data_augment_directory='./dataset_augment'
data_base_directory='./dataset2'
data_directory=data_augment_directory

masks_base_directory = './masks'
masks_augment_directory = './masks_augment_directory'
masks_directory = masks_augment_directory

model_directory = "./model"

# Constants
img_rows = 256
img_cols = 256
channels=3
batch_size = 32
rescale = 1.0 / 255
num_classes=2

# Square in [0;63]
def ROI_from_square(square, annots):
    rank = square % 8
    file = square - (rank * 9)
    square_corners = [
        annots["board"][rank * 9 + file],
        annots["board"][rank * 9 + file + 1],
        annots["board"][rank * 9 + file + 9],
        annots["board"][rank * 9 + file + 10]
    ]

    print(np.array(square_corners)[:, 0])
    min_x = np.min(np.array(square_corners)[:, 0])
    max_x = np.max(np.array(square_corners)[:, 0])
    min_y = np.min(np.array(square_corners)[:, 1])
    max_y = np.max(np.array(square_corners)[:, 1])

    return (min_x, min_y, max_x, max_y)


class PlotLearning(tf.keras.callbacks.Callback):

    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []


    def on_epoch_end(self, epoch, logs={}):

        window = 0

        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        smooth_metrics = {}
        for label in self.metrics:
            smooth_metrics[label] = []
            values = self.metrics[label]
            for i in range(len(values)):
                smooth_metrics[label].append(np.mean(values[i-min(i,window):i+1]))

        # Plotting
        metrics = [x for x in logs if x in ['loss', 'accuracy']]

        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        smooth_metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            smooth_metrics['val_' + metric],
                            label='val_' + metric)
            if metric=="loss":
                axs[i].set_ylim(0, 0.2)
            if metric=="accuracy":
                axs[i].set_ylim(0.5, 1)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.savefig("./train_hist.jpg")
        plt.close('all')














def create_df(data_dir, masks_dir, shuffle=True):
    path=masks_dir + '/*.jpg'
    masks_file_paths=glob.glob(path,recursive=True)

    image_file_paths = [os.path.join(data_dir, os.path.basename(path[:-9] + ".jpg")) for path in masks_file_paths]

    # run a check and make sure filename without extensions match
    df=pd.DataFrame({'image': image_file_paths, 'label':masks_file_paths}).astype(str)
    if shuffle:
        df=df.sample(frac=1.0, replace=False, weights=None, random_state=123, axis=0).reset_index(drop=True)
    return df

class jpgen():
    batch_index=0  #tracks the number of batches generated
    def __init__(self, df, train_split=None, test_split=None):
        self.train_split=train_split  # float between 0 and 1 indicating the percentage of images to use for training
        self.test_split=test_split
        self.df=df.copy() # create a copy of the data frame
        if self.train_split != None: # split the df to create a training df
            self.train_df, dummy_df=train_test_split(self.df, train_size=self.train_split, shuffle=True)
            if self.test_split !=None: # create as test set and a validation set
                t_split=self.test_split/(1.0-self.train_split)
                self.test_df, self.valid_df=train_test_split(dummy_df, train_size=t_split, shuffle=True)
                self.valid_gen_len=len(self.valid_df['image'].unique())# create var to return no of samples in valid generator
                self.valid_gen_filenames=list(self.valid_df['image'])# create list ofjpg file names in valid generator
            else: self.test_df=dummy_df
            self.test_gen_len=len(self.test_df['image'].unique())#create var to return no of test samples
            self.test_gen_filenames=list(self.test_df['image']) # create list to return jpg file paths in test_gen
        else:
            self.train_df=self.df
        self.tr_gen_len=len(self.train_df['image'].unique())  # crete variable to return no of samples in train generator

        for e in self.train_df.iloc:
            print(e['image'])

    def flow(self,  batch_size=32, image_shape=None,shuffle=True, subset=None ):
        # flows batches of jpg images and png masks to model.fit
        self.batch_size=batch_size
        self.image_shape=image_shape
        self.shuffle=shuffle
        self.subset=subset
        image_batch_list=[] # initialize list to hold a batch of jpg  images
        label_batch_list=[] # initialize list to hold batches of labels
        if self.subset=='training' or self.train_split ==None:
            op_df=self.train_df
        elif self.subset=='test':
            op_df=self.test_df
        else:
            op_df=self.valid_df
        if self.shuffle : # shuffle  the op_df then rest the index
            op_df=op_df.sample(frac=1.0, replace=False, weights=None, random_state=123, axis=0).reset_index(drop=True)
        #op_df will be either train, test or valid depending on subset
        # develop the batch of data
        while True:
            label_batch_list=[]
            image_batch_list=[]
            start=jpgen.batch_index * self.batch_size # set start value of iteration
            end=start + self.batch_size   # set end value of iteration to yield 1 batch of data of length batch_size
            sample_count=len(op_df['image'])
            for i in range(start, end): # iterate over one batch size of data
                j=i % sample_count # used to roll the images  back to the front if the end is reached
                k=j % self.batch_size
                path_to_image= op_df.iloc[j]['image']
                path_to_label= op_df.iloc[j] ['label']

                image=cv2.imread(path_to_image)
                image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image= np.float32(image) * rescale
                image=cv2.resize(image, self.image_shape)

                mask=cv2.imread(path_to_label).astype(np.uint8)
                mask= np.float32(mask) * rescale
                mask=cv2.resize(mask, self.image_shape)
                mask = mask[:, :, 0].reshape(self.image_shape[0], self.image_shape[1], 1).astype(np.uint8)

                # gt.sort(key=lambda x: (x[0], x[1]))
                label_batch_list.append(mask)
                image_batch_list.append(image)
            image_array=np.array(image_batch_list)
            label_array=np.array(label_batch_list)
            jpgen.batch_index +=1
            yield (image_array, label_array)


def augment_data():
    if not os.path.exists(data_augment_directory):
        os.mkdir(data_augment_directory)

    if not os.path.exists(masks_augment_directory):
        os.mkdir(masks_augment_directory)

    path=data_base_directory + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)

    for image_path in image_file_paths:
        print("Augmenting ", image_path)

        filename = os.path.basename(image_path[:-4])
        label_path = os.path.join(masks_base_directory, filename + "_mask.jpg")

        shutil.copy(image_path, os.path.join(data_augment_directory, f"{filename}.jpg"))
        shutil.copy(label_path, os.path.join(masks_augment_directory, f"{filename}_mask.jpg"))

        cv2.imwrite(os.path.join(data_augment_directory, f"{filename}_flipped.jpg"), cv2.flip(cv2.imread(image_path), 1))
        cv2.imwrite(os.path.join(masks_augment_directory, f"{filename}_flipped_mask.jpg"), cv2.flip(cv2.imread(label_path), 1))

def getEncoder(input_shape, lr):

    inputs = keras.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=300,
            decay_rate=0.9)
    # lr_schedule = lr
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_metric = get_lr_metric(opt)

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy", lr_metric], run_eagerly=True)
    model.summary()

    return model

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

def train_network(epochs, lr, use_existing):

    # augment_data()

    shuffle=False # if True shuffles the dataframe
    df=create_df(data_directory, masks_directory, shuffle) # create a dataframe with columns 'images' , 'labels'
                                                # where labels are the noisy images
    train_split=.7 # use 80% of files for training
    test_split=.02  # use 10% for test, automatically sets validation split at 1-train_split-test_split

    gen=jpgen(df, train_split=train_split, test_split=test_split) # create instance of generator class

    tr_gen_len=gen.tr_gen_len
    test_gen_len= gen.test_gen_len
    valid_gen_len=gen.valid_gen_len

    test_filenames=gen.test_gen_filenames # names of test file paths used for training

    train_steps=tr_gen_len//batch_size #  use this value in for steps_per_epoch in model.fit
    valid_steps=valid_gen_len//batch_size # use this value for validation_steps in model.fit
    test_steps=test_gen_len//batch_size  # use this value for steps in model.predict

    # instantiate generators
    image_shape=(img_rows, img_cols)

    train_gen=gen.flow(batch_size=batch_size, image_shape=image_shape, shuffle=True, subset='training')
    valid_gen=gen.flow(batch_size=batch_size, image_shape=image_shape, shuffle=True, subset='valid')
    test_gen=gen.flow(batch_size=batch_size, image_shape=image_shape, shuffle=True, subset='test')

    if (use_existing):
        model_path = os.path.join(model_directory, f"model.h5")
        model = load_model(model_path)
    else:
        model = getEncoder(input_shape = (img_rows, img_cols, channels), lr=lr)

    save_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_directory, 'model.h5'), monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    board_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    plot_cb = PlotLearning()

    history=model.fit(train_gen, epochs=epochs, steps_per_epoch=train_steps,validation_data=valid_gen,
                    validation_steps=valid_steps, verbose=1, shuffle=True, callbacks=[save_cb, plot_cb])

    # print(len(test_gen))
    # predictions=model.predict(test_gen, steps=test_steps)

    model.save(model_directory)

def convert_coords(x, y, image_shape):
    return (int(x * image_shape[1]), int((1.0 - y) * image_shape[0]))

def plot_chess_grid(image, grid):
    # low_grid = [grid[0], grid[8], grid[72], grid[80]]
    # plot_predicted_chess_grid(image, low_grid)
    # return


    low_grid = []
    for i, pt in enumerate(grid):
        if i < 9 or i >= 81 - 9 or i%9 == 0 or i%9 == 8:
            low_grid.append(pt)
    plot_predicted_chess_grid(image, low_grid)
    return

    for i in range(9):
        pt1 = convert_coords(grid[i*9][0], grid[i*9][1], image.shape)
        pt2 = convert_coords(grid[i*9+8][0], grid[i*9+8][1], image.shape)
        cv2.line(image, pt1, pt2, (0.0, 0.0, 255.0))

    for i in range(9):
        pt1 = convert_coords(grid[i][0], grid[i][1], image.shape)
        pt2 = convert_coords(grid[8*9+i][0], grid[8*9+i][1], image.shape)
        cv2.line(image, pt1, pt2, (0.0, 0.0, 255.0))

def plot_predicted_chess_grid(image, grid):

    # bottom_left = convert_coords(grid[0][0], grid[0][1], image.shape)
    # bottom_right = convert_coords(grid[8][0], grid[8][1], image.shape)
    # top_left = convert_coords(grid[23][0], grid[23][1], image.shape)
    # top_right = convert_coords(grid[31][0], grid[31][1], image.shape)

    # cv2.line(image, bottom_left, bottom_right, (0.0, 0.0, 255.0))
    # cv2.line(image, bottom_right, top_right, (0.0, 0.0, 255.0))
    # cv2.line(image, top_right, top_left, (0.0, 0.0, 255.0))
    # cv2.line(image, top_left, bottom_left, (0.0, 0.0, 255.0))

    for i in range(len(grid)):

        pt1 = convert_coords(grid[i][0], grid[i][1], image.shape)
        pt2 = convert_coords(grid[(i+1)%len(grid)][0], grid[(i+1)%len(grid)][1], image.shape)
        cv2.line(image, pt1, pt2, (0.0, 0.0, 255.0))

def plot_chess_pieces(image, pieces):
    for piece in pieces:
        piece_code = (ord(piece["piece"]) - 70) * 5
        print(piece_code)
        coords = piece["bbox"]
        for i in range(4):
            color = (piece_code, 255.0 - piece_code, piece_code/2)
            cv2.rectangle(image, convert_coords(coords[0], coords[1], image.shape), convert_coords(coords[2], coords[3], image.shape), color)


def check_annotations(data_name):
    with open(os.path.join(data_directory, f"{data_name}.json")) as f:
        annots = json.loads(f.read())

    image=cv2.imread(os.path.join(data_directory, f"{data_name}.jpg"))

    image=np.float32(image)# * rescale
    image=cv2.resize(image, (img_rows, img_cols))

    plot_chess_grid(image, annots["board"])
    plot_chess_pieces(image, annots["pieces"])

    cv2.imwrite("./annotation.jpg", image)

def check_roi(data_name):
    with open(os.path.join(data_directory, f"{data_name}.json")) as f:
        annots = json.loads(f.read())

    image=cv2.imread(os.path.join(data_directory, f"{data_name}.jpg"))

    for i in range(8):
        roi = ROI_from_square(i, annots)
        color = (0.0, 0.0, 255.0)
        cv2.rectangle(image, convert_coords(roi[0], roi[1], image.shape), convert_coords(roi[2], roi[3], image.shape), color)

    cv2.imwrite("./rois.jpg", image)

def test(data_name):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=200,
            decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_metric = get_lr_metric(opt)

    model_path = os.path.join(model_directory, f"model.h5")
    model = load_model(model_path, custom_objects={"lr": lr_metric })

    image=cv2.imread(os.path.join(data_directory, f"{data_name}.jpg"))
    # image=cv2.imread("./test_images/data_2.jpg")
    image_inf= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_inf=np.float32(image_inf) * rescale
    image_inf=cv2.resize(image_inf, (img_rows, img_cols))

    features = model.predict(image_inf.reshape(1, img_rows, img_cols, channels), batch_size=1)
    print(features.reshape(img_rows, img_cols, -1)[128, 128])
    print(features.reshape(img_rows, img_cols, -1)[0, 0])

    mask_color = np.array([[x[1]*255, x[1]*255, x[1]*255] for x in features.reshape(-1,2)])
    mask_color = (mask_color.reshape(img_rows, img_cols, 3)).astype(np.uint8)
    key = 0
    mask_color[0,0] = [255, 255, 255]
    while(key != ord('x')):
        # cv2.imshow("test3", cv2.resize(cv2.imread(os.path.join(masks_directory, f"{data_name}_mask.jpg")), (img_rows, img_cols)))
        # cv2.imshow("test2", image_inf)
        cv2.imshow("test", mask_color)
        key = cv2.waitKey()

    # cv2.imwrite("./test.jpg", image)


    return features

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize annotation')

    parser.add_argument('--use_existing', action='store_true',
                        help='Start from previous weights')

    parser.add_argument('--test', action='store_true',
                        help='True to test instead of train')

    parser.add_argument('--data_test', type=str, default="data_0",
                        help='Name of the data to test')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.visualize:
        check_annotations(args.data_test)
        check_roi(args.data_test)
    elif args.test:
        test(args.data_test)
    else:
        train_network(epochs = args.epochs, lr = args.lr, use_existing = args.use_existing)

if __name__ == "__main__":
    main()






