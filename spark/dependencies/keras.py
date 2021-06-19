import numpy as np
import pandas as pd

from keras.layers import Conv2D, Activation, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


def model_efficacy(predictions_y, test_gen, classes):
    predictions = np.array(list(map(lambda x: np.argmax(x), predictions_y)))

    conf_matrix = pd.DataFrame(
        confusion_matrix(test_gen.classes, predictions),
        columns=classes,
        index=classes
    )

    acc = accuracy_score(test_gen.classes, predictions)

    results_all = precision_recall_fscore_support(
        test_gen.classes, predictions,
        average='macro',
        zero_division=1
    )
    results_class = precision_recall_fscore_support(
        test_gen.classes,
        predictions,
        average=None, zero_division=1
    )

    # Organise the Results into a Dataframe
    metric_columns = ['Precision', 'Recall', 'F-Score', 'S']
    accuracy = pd.concat([pd.DataFrame(list(results_class)).T, pd.DataFrame(list(results_all)).T])
    accuracy.columns = metric_columns
    accuracy.index = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia', 'Total']

    return [conf_matrix, accuracy]


def add_conv2d_entry_layer(model):
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=(299, 299, 1)))
    model.add(BatchNormalization())
    return None


def add_conv2d_layer(model):
    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(BatchNormalization())
    return None


def add_average_pooling2d_layer(model):
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    return None


def add_exit_layers(model):
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))
    return None


def train_generator_from_dataframe(dataframe_keras_master, batch_size, classes):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=dataframe_keras_master,
        directory=None,
        x_col="origin",
        y_col="class_name",
        classes=classes,
        batch_size=batch_size,
        target_size=(299, 299),
        class_mode="categorical",
        subset="training",
        color_mode="grayscale",
        shuffle=True,
        validate_filenames=False
    )

    return [train_datagen, train_gen]


def test_generator_from_dataframe(dataframe_keras_master, batch_size, classes):
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=dataframe_keras_master,
        directory=None,
        x_col="origin",
        y_col="class_name",
        classes=classes,
        batch_size=batch_size,
        target_size=(299, 299),
        class_mode="categorical",
        subset="validation",
        color_mode="grayscale",
        shuffle=False,
        validate_filenames=False
    )

    return [test_datagen, test_gen]
