import ml_tools

if __name__ == "__main__":
    df=ml_tools.read_csv('c:/temp/rc-small.csv')
    print(str(df.shape))
    svc_std, sc, le = ml_tools.import_model("c:/temp/model.sav")
    pred = ml_tools.predict_batch(df["feature_vector"].tolist(), svc_std, sc, le)
    print(str(pred))
    df["predicted"] = pred
    print(str(df.shape))
    df=df.drop(columns=["feature_vector"])
    df.to_csv("c:/temp/predicted.csv")



