import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

base_folder = "C:/greg/reefscan_ml/tests/"


def report(name,orig_csv, test_result_csv, label_col, search=[], replace=[], del_col=[], del_val=[]):
    result_df = pd.read_csv(base_folder + "train-results/" +  test_result_csv)
    orig_df = pd.read_csv(base_folder + "input/" +  orig_csv)
    for i in range(len(search)):
        result_df[label_col] = result_df[label_col].replace(search[i], replace[i])
        orig_df[label_col] = orig_df[label_col].replace(search[i], replace[i])

    for i in range(len(del_col)):
        result_df = result_df[result_df[del_col[i]] != del_val[i]]
        orig_df = orig_df[orig_df[del_col[i]] != del_val[i]]

    training_labels = orig_df.groupby(label_col).size().reset_index().rename(columns={0: 'count_total'})
    human_count_labels = result_df.groupby(label_col).size().reset_index().rename(columns={0: 'human_count'})
    model_count_labels = result_df.groupby('predicted').size().reset_index().rename(columns={0: 'model_count'})
    model_count_labels = model_count_labels.rename(columns={"predicted": label_col})

    match = result_df[result_df[label_col] == result_df["predicted"]]
    match_count = match.groupby(label_col).size().reset_index().rename(columns={0: 'match_count'})

    labels = pd.merge(training_labels, human_count_labels, on=label_col, how="outer")
    labels = pd.merge(labels, model_count_labels, on=label_col, how="outer")
    labels = pd.merge(labels, match_count, on=label_col, how="outer")
    labels = labels.fillna(0)
    labels["match_percent"] = 100 * labels.match_count / labels.human_count

    labels = labels.sort_values(by="count_total")

    labels.to_csv(base_folder + "reports/" + name + ".csv")

    labels_for_cm = labels[(labels.human_count > 0) | (labels.model_count > 0)][label_col]

    cm = sklearn.metrics.confusion_matrix(result_df[label_col], result_df.predicted, normalize='true',
                                          labels=labels_for_cm)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels_for_cm, yticklabels=labels_for_cm)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plt.show()
    plt.savefig(base_folder + "reports/" + name + ".png")


report("reefscan_group", "reefscan-vectors1.csv", "reefscan-train-test.csv", 'GROUP_CODE', search=['SG', 'SP'], replace=['OT', 'OT'], del_col=['GROUP_CODE'], del_val=['IN'])
report("reefscan_ai", "reefscan-vectors1.csv", "reefscan-ai-train-test.csv", 'AI_CODE_2021', del_col=['GROUP_CODE'], del_val=['IN'])
report("reefscan_ker", "reefscan-vectors1.csv", "reefscan-ker-train-test.csv", 'KER_CODE', del_col=['GROUP_CODE'], del_val=['IN'])
report("reefscan_benthos", "reefscan-vectors1.csv", "reefscan-benthos-train-test.csv", 'BENTHOS_DESC', del_col=['GROUP_CODE'], del_val=['IN'])

