import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from time import time
import math
import random
import numpy
import json
import logging
from guppy import hpy;
from utils import convert_time
from sklearn import metrics
import joblib

h = hpy()
h.setrelheap()

logger = logging.getLogger(__name__)


def get_training_set_from_csv(csv_file):
    df = read_csv(csv_file)
    df["classification_type"] = "TRAIN"

    size = df.shape[0]
    sample_size = math.ceil(size / 5)
    test_index = random.sample(range(size), sample_size)

    df.classification_type[test_index] = "TEST"

    return df


def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    df["feature_vector"] = df.apply(row_to_vector, axis=1)
    return df


def row_to_vector(row):
    row_dict = row.to_dict()
    return string_to_vector(row_dict["feature_vector"])


def list_to_vector(vector_list):
    return numpy.array(vector_list, dtype=numpy.float32)


def string_to_vector(vector_string):
    return list_to_vector(json.loads(vector_string.replace("[[", "[").replace("]]", "]")))


def get_training_test_data_frames(df: pd.DataFrame):
    train_df = df[df.classification_type == 'TRAIN']
    test_df = df[df.classification_type == 'TEST']
    logger.debug("Split data into training size {} and testing size {}".format(len(train_df.index), len(test_df.index)))
    logger.debug(h.heap().bytype)

    return train_df, test_df


def export_model(svc_std, sc, le, local_file_name):
    logger.debug("Saving model to {}".format(local_file_name))
    with open(local_file_name, 'wb') as local_file:
        joblib.dump((svc_std, sc, le), local_file, compress=True)
        logger.debug("Dumped model to local file")
    logger.info("Saved model to {}".format(local_file_name))

def import_model(model_file_name):
    svc_std, sc, le = joblib.load(model_file_name)
    return svc_std, sc, le


def predict_batch(feature_vector_list, svc_std, sc, le):
    feature_vector_std = sc.transform(feature_vector_list)
    pred_list = svc_std.predict(feature_vector_std)
    labels = le.inverse_transform(pred_list)
    # label = le.classes_[pred]
    logger.debug("Got predictions {}, {}".format(pred_list, labels))
    return labels


def train_data_frame_with_logistic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, threads):
    logger.debug(h.heap().bytype)
    vectors = train_df.feature_vector.to_numpy()
    logger.debug("Completed feature vector to numpy")
    logger.debug(h.heap().bytype)
    X = numpy.stack(vectors)
    logger.debug("Stacked vectors")
    logger.debug(h.heap().bytype)
    labels = train_df.human_classification_label.to_numpy()
    logger.debug("Trained")
    logger.debug(h.heap().bytype)

    le = LabelEncoder()
    le.fit(labels)
    Y = le.transform(labels)
    x = X[:, :]
    y = Y[:]
    sc = StandardScaler()
    sc.fit(x)
    X_train_std = sc.transform(x)

    svc_std = LogisticRegression(verbose=1, n_jobs=threads, max_iter=500)
    tic = time()
    svc_std.fit(X_train_std, y)
    logger.info('trained svm in {} seconds'.format(convert_time(time() - tic)))
    logger.debug(h.heap().bytype)

    test_df_copy = test_df[test_df['human_classification_label'].isin(le.classes_)]
    test_vectors = test_df_copy.feature_vector.to_numpy()
    X_test = numpy.stack(test_vectors)
    labels_test = test_df_copy.human_classification_label.to_numpy()

    # Run predictions on testing set
    X_test_std = sc.transform(X_test)
    y_test = le.transform(labels_test)
    tic = time()
    y_pred = svc_std.predict(X_test_std)
    logger.info('predicted on {} points in {}'.format(len(y_pred), convert_time(time() - tic)))
    logger.debug(h.heap().bytype)
    f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    logger.info("Accuracy: {} F1 Score {}".format(accuracy_score, f1_score))
    return svc_std, sc, le, f1_score, accuracy_score



