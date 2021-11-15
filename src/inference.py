import ml_tools
import pandas as pd

if __name__ == "__main__":
    # svc_std, sc, le = ml_tools.import_model("c:/temp/model-group.sav")
    # svc_std, sc, le = ml_tools.import_model("c:/temp/aims-latest.sav")
    svc_std, sc, le = ml_tools.import_model("c:/temp/model-reefscan-group.sav")
    print (len(le.classes_))
    print (le.classes_)
    df=ml_tools.read_csv('c:/temp/reefscan-vectors1.csv')
    # labels = pd.unique(df['human_classification_label'])
    # print(len(labels))
    # print(labels)
    pred = ml_tools.predict_batch(df["feature_vector"].tolist(), svc_std, sc, le)
    print(str(pred))
    df["predicted"] = pred
    print(str(df.shape))
    df=df.drop(columns=["feature_vector"])
    df.to_csv("c:/temp/predicted-reefscan-reefscan.csv")



