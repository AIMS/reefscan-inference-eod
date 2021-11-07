import ml_tools
import logging


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
    df=ml_tools.get_training_set_from_csv('c:/temp/rc.csv')
    print(str(df.shape))
    train, test = ml_tools.get_training_test_data_frames(df)
    print(str(train.shape))
    print(str(test.shape))

    svc_std, sc, le, f1_score, accuracy_score = ml_tools.train_data_frame_with_logistic_regression(train, test, 1)
    ml_tools.export_model(svc_std, sc, le, "c:/temp/model.sav")
