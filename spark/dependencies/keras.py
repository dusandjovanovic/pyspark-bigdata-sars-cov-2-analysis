import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
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


def model_callbacks():
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.5, min_lr=0.00001)
    early_stopping_monitor = EarlyStopping(patience=100, monitor='loss', mode='min')

    callbacks_list = [learning_rate_reduction, early_stopping_monitor]

    return callbacks_list


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
        shuffle=False,
        validate_filenames=False
    )

    return [test_datagen, test_gen]
