# You need to make sure that the system has the following built in.

# !pip install bing-image-downloader
# !mkdir images

# Delete the images: 
# !rm -r images

from bing_image_downloader import downloader
from PIL import Image
from pathlib import Path
import imghdr
import os
import matplotlib.pyplot as plt
from tensorflow import one_hot
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_DIR = "./images" # TODO CHECK
IMAGE_SIZE_H = 256
IMAGE_SIZE_W = 256
size = (IMAGE_SIZE_H, IMAGE_SIZE_W)
data_dir = BASE_DIR
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

search_inputs = ["dogs", "cats"] # TODO 
search_inputs_size = len(search_inputs)

# FILTERING FUNCTIONS:
def filtered_images(images):

    good_images = []
    bad_images = []
    for filename in images:
        try:
            img = Image.open(filename)
            # pixel distribution
            v = img.histogram()
            h,  w = img.size
            percentage_monochrome = max(v) / float(h * w)

            # filter bad and small images
            if ((percentage_monochrome > 0.8 or h < 300 or w < 300) or not (filename[-3:] == "png" or filename[-3:] == "jpg")):
                bad_images.append(filename)
            else:
                good_images.append(filename)

        except:
            pass
    
    print("Number of good images: {} \n".format(len(good_images)))
    print("Number of bad images: {} \n".format(len(bad_images)))    
    return good_images, bad_images


def remove_other_filetypes():
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
                os.remove(filepath)
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                os.remove(filepath)

def one_hot_ds(ds, depth):
    ds = ds.map(lambda x, y: (x, one_hot(y, depth=depth)))
    return ds

print("The pipeline starts...")

def train_model_based_on_inputs(search_inputs):

    # Download everything that is needed here. Pass the files through the filters and delete the bad images.

    for search in search_inputs:
        downloader.download(search, limit=300, output_dir="images")

    for search_input in search_inputs:
        get_class_path = lambda name: os.path.join(BASE_DIR, search_input)
        class_dir = get_class_path(search_input)
        after_filter_images = map(lambda f: os.path.join(class_dir, f), os.listdir(class_dir))
        # Filter the images using the functions
        g_images, b_images = filtered_images(after_filter_images)
        for filename in b_images:
            os.remove(filename)

    remove_other_filetypes()

    batch_size = 2
    train_ds = image_dataset_from_directory(
    BASE_DIR,
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=(IMAGE_SIZE_H, IMAGE_SIZE_W),
    batch_size=batch_size,
    shuffle=True)

    val_ds = image_dataset_from_directory(
    BASE_DIR,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=(IMAGE_SIZE_H, IMAGE_SIZE_W),
    batch_size=batch_size,
    shuffle=True)

    class_names = train_ds.class_names
    print(class_names)

    # Make the labels one hot embeddings.
    train_ds = one_hot_ds(train_ds, search_inputs_size)
    val_ds = one_hot_ds(val_ds, search_inputs_size)

    # Do some preprocessing:
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)

    # ERROR WITH TF AFTER MOVE FROM EXPERIMENTAL
    # # Prepare for the data augmentation and sample it:
    # data_augmentation = keras.Sequential(
    #     [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
    # )

    # FOR VALIDATION:
    # for images, labels in train_ds.take(1):
    #     plt.figure(figsize=(10, 10))
    #     first_image = images[0]
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         augmented_image = data_augmentation(
    #             tf.expand_dims(first_image, 0), training=True
    #         )
    #         plt.imshow(augmented_image[0].numpy().astype("int32"))
    #         plt.title(labels[0].numpy())
    #         plt.axis("off")


    # Create the model

    base_model = keras.applications.ResNet50V2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(IMAGE_SIZE_H, IMAGE_SIZE_W, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(256, 256, 3))
    # x = data_augmentation(inputs)  # Apply random data augmentation

    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(search_inputs_size, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    print(model.summary())

    # Compile and fit the model.
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.categorical_accuracy],
    )
    epochs = 1
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    model.save('./models', save_format='tf')
    print("The pipeline finished!")
    
    return model

