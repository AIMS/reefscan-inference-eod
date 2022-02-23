import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

base_folder = "C:/greg/reefscan_ml/tests/"


def report(orig_csv, label_col, search=[], replace=[], del_col=[], del_val=[]):
    result_df = pd.read_csv(base_folder + "train-results/" +  label_col + "reefscan-train-test.csv" )
    train_df = pd.read_csv(base_folder + "train-results/" +  label_col + "reefscan-train.csv")
    orig_df = pd.read_csv(base_folder + "input/" +  orig_csv)
    for i in range(len(search)):
        result_df[label_col] = result_df[label_col].replace(search[i], replace[i])
        orig_df[label_col] = orig_df[label_col].replace(search[i], replace[i])
        train_df[label_col] = train_df[label_col].replace(search[i], replace[i])

    for i in range(len(del_col)):
        result_df = result_df[result_df[del_col[i]] != del_val[i]]
        orig_df = orig_df[orig_df[del_col[i]] != del_val[i]]
        train_df = train_df[train_df[del_col[i]] != del_val[i]]

    orig_training_labels = orig_df.groupby(label_col).size().reset_index().rename(columns={0: 'count_total_before_aug'})
    train_human_count_labels = train_df.groupby(label_col).size().reset_index().rename(columns={0: 'train_human_count'})
    human_count_labels = result_df.groupby(label_col).size().reset_index().rename(columns={0: 'test_human_count'})
    model_count_labels = result_df.groupby('predicted').size().reset_index().rename(columns={0: 'test_model_count'})
    model_count_labels = model_count_labels.rename(columns={"predicted": label_col})

    match = result_df[result_df[label_col] == result_df["predicted"]]
    match_count = match.groupby(label_col).size().reset_index().rename(columns={0: 'match_count'})

    labels = pd.merge(orig_training_labels, train_human_count_labels, on=label_col, how="outer")
    labels = pd.merge(labels, human_count_labels, on=label_col, how="outer")
    labels = pd.merge(labels, model_count_labels, on=label_col, how="outer")
    labels = pd.merge(labels, match_count, on=label_col, how="outer")
    labels = labels.fillna(0)
    labels["match_percent"] = 100 * labels.match_count / labels.test_human_count

    labels = labels.sort_values(by="count_total_before_aug")

    labels.to_csv(base_folder + "reports/" + label_col + ".csv")

    labels_for_cm = labels[(labels.test_human_count > 0) | (labels.test_model_count > 0)][label_col]

    cm = sklearn.metrics.confusion_matrix(result_df[label_col], result_df.predicted, normalize='true',
                                          labels=labels_for_cm)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels_for_cm, yticklabels=labels_for_cm)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plt.show()
    plt.savefig(base_folder + "reports/" + label_col + ".png")
    plt.close()

    cal = pd.read_csv(base_folder + "train-results/" +  label_col + "reefscan-train-test.csv.cal.csv" )
    cal.plot(x="prob_pred", y="prob_true")
    plt.plot([0, 1], [0, 1], color='k', linestyle=':', linewidth=1)
    plt.savefig(base_folder + "reports/cal-" + label_col + ".png")
    plt.close()

    cal['ece'] = abs(cal.prob_true - cal.prob_pred)
    cal.boxplot(column="ece")
    plt.savefig(base_folder + "reports/calbox-" + label_col + ".png")
    plt.close()




report("reefscan-vectors1.csv", 'GROUP_CODE', search=['SG', 'SP', 'AB', 'SC', 'HC'], replace=['OT', 'OT', 'OT', 'Coral', 'Coral'])
report("reefscan-vectors1.csv", 'AI_CODE_2021')
# report("reefscan-vectors1.csv", 'KER_CODE')
# report("reefscan-vectors1.csv", 'BENTHOS_DESC')

