import ml_tools
import logging
import os
import random
logger = logging.getLogger(__name__)


def add_aug_data(df, aug_df, label_col, min_count):
    print ("aug_df.shape")
    print (aug_df.shape)
    available_augs = aug_df[aug_df.orig_file.isin(df.FILE_NAME)]
    print ("available_augs.shape")
    print (available_augs.shape)
    labels = df.groupby(label_col).size().reset_index().rename(columns={0: 'cnt'})
    out_df = df
    for index, label in labels.iterrows():
        augs_for_label = available_augs[available_augs[label_col] == label[label_col]]
        augs_required = min_count - label["cnt"]
        if augs_required > 0:
            size = augs_for_label.shape[0]
            if size > augs_required:
                selected_index = random.sample(range(size), augs_required)
                augs_to_add = augs_for_label.iloc[selected_index, :]
            else:
                augs_to_add = augs_for_label
            out_df = out_df.append(augs_to_add)

    return out_df


def train_file(vectors_file, label_col, search=[], replace=[], del_col=[], del_val=[]):
    logging.basicConfig(level="DEBUG")
    logger.debug("A debug message")
    df = ml_tools.get_training_set_from_csv('C:/greg/reefscan_ml/tests/input/' + vectors_file)
    for i in range(len(search)):
        df[label_col] = df[label_col].replace(search[i], replace[i])

    for i in range(len(del_col)):
        df = df[df[del_col[i]] != del_val[i]]

    print(str(df.shape))
    train, test = ml_tools.get_training_test_data_frames(df)
    aug_data = ml_tools.read_csv("C:/greg/reefscan_ml/tests/input/reefscan-vectors1-aug.csv")

    print(str(train.shape))
    print(str(test.shape))

    train = add_aug_data(train, aug_data, label_col, 160)
    test = add_aug_data(test, aug_data, label_col, 40)

    print(str(train.shape))
    print(str(test.shape))

    svc_std, sc, le, f1_score, accuracy_score = ml_tools.train_data_frame_with_logistic_regression(train, test, 4,
                                                            label_column=label_col,
                                                            test_out_csv=label_col + "reefscan-train-test.csv",
                                                            train_out_csv=label_col + "reefscan-train.csv",
                                                            )
    ml_tools.export_model(svc_std, sc, le, "C:/greg/reefscan_ml/tests/models/model-reefscan-group.sav")

if __name__ == "__main__":
    train_file ("reefscan-vectors1.csv", 'GROUP_CODE', search=['SG', 'SP', 'AB', 'SC', 'HC'], replace=['OT', 'OT', 'OT', 'Coral', 'Coral'])
    train_file ("reefscan-vectors1.csv", 'AI_CODE_2021')
    train_file ("reefscan-vectors1.csv", 'KER_CODE')
    train_file ("reefscan-vectors1.csv", 'BENTHOS_DESC')







    # svc_std, sc, le, f1_score, accuracy_score = ml_tools.train_data_frame_with_logistic_regression(train, test, 4,
    #                                         label_column='human_classification_group_label', test_out_csv="aims-group-train-test.csv")
    # ml_tools.export_model(svc_std, sc, le, "c:/temp/model-group.sav")
    #
