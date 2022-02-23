import ml_tools
import logging
import os
logger = logging.getLogger(__name__)


def add_aug_data(df, aug_df, min_count):
    available_augs = aug_df[aug_df.orig_file.isin(df.FILE_NAME)]
    return df.append(available_augs)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    logger.debug("A debug message")
    # df = ml_tools.get_training_set_from_csv('c:/temp/rc.csv')
    df = ml_tools.get_training_set_from_csv('C:/greg/reefscan_ml/tests/input/reefscan-vectors1.csv')
    df = df[df.GROUP_CODE != 'IN']

    # df.human_classification_group_label = df.human_classification_group_label.replace('SG', 'OT')
    # df.human_classification_group_label = df.human_classification_group_label.replace('SP', 'OT')
    # df.model_classification_group_rc = df.model_classification_group_rc.replace('SG', 'OT')
    # df.model_classification_group_rc = df.model_classification_group_rc.replace('SP', 'OT')
    df.GROUP_CODE = df.GROUP_CODE.replace('SG', 'OT')
    df.GROUP_CODE = df.GROUP_CODE.replace('SP', 'OT')

    print(str(df.shape))
    train, test = ml_tools.get_training_test_data_frames(df)
    aug_data = ml_tools.read_csv("C:/greg/reefscan_ml/tests/input/reefscan-vectors1-aug.csv")

    print(str(train.shape))
    print(str(test.shape))

    train = add_aug_data(train, aug_data, 160)
    test = add_aug_data(test, aug_data, 40)

    print(str(train.shape))
    print(str(test.shape))


    # svc_std, sc, le, f1_score, accuracy_score = ml_tools.train_data_frame_with_logistic_regression(train, test, 4,
    #                                         label_column='human_classification_group_label', test_out_csv="aims-group-train-test.csv")
    # ml_tools.export_model(svc_std, sc, le, "c:/temp/model-group.sav")
    #
    svc_std, sc, le, f1_score, accuracy_score = ml_tools.train_data_frame_with_logistic_regression(train, test, 4,
                                                            label_column='GROUP_CODE',
                                                            test_out_csv="reefscan-group-train-test.csv",
                                                            train_out_csv="reefscan-group-train.csv",
                                                            )
    ml_tools.export_model(svc_std, sc, le, "C:/greg/reefscan_ml/tests/models/model-reefscan-group.sav")
