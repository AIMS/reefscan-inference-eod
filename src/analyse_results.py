import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = "c:/temp/rc2.csv"
df = pd.read_csv(csv_file)

print(df.head())
print(df.columns)
# df.human_classification_group_label = df.human_classification_group_label.replace('SG', 'OT')
# df.human_classification_group_label = df.human_classification_group_label.replace('SP', 'OT')
# df.model_classification_group_rc = df.model_classification_group_rc.replace('SG', 'OT')
# df.model_classification_group_rc = df.model_classification_group_rc.replace('SP', 'OT')
# df.GROUP_CODE = df.GROUP_CODE.replace('SG', 'OT')
# df.GROUP_CODE = df.GROUP_CODE.replace('SP', 'OT')

# labels=df.groupby(['human_classification_label', 'human_classification_group_label']).size().reset_index().rename(columns={0:'count'})
# labels=labels.rename(columns={'human_classification_label':'label', 'human_classification_group_label':'group'})
# print(labels)
# groups = pd.unique(labels.group)

# labels=df.groupby(['KER_CODE', 'GROUP_CODE']).size().reset_index().rename(columns={0:'count'})
# labels=labels.rename(columns={'KER_CODE':'label', 'GROUP_CODE':'group'})
# print(labels)
# groups = pd.unique(labels.group)

groups = pd.unique(df.model_classification_group_rc)

# groups = pd.unique(df.predicted.append(df.KER_CODE))


# print(df.shape)
# df=pd.merge(df, labels, left_on="predicted", right_on="label")
# print(df.shape)

# cm = sklearn.metrics.confusion_matrix(df.human_classification_label, df.predicted, normalize='true', labels=labels.label)
cm = sklearn.metrics.confusion_matrix(df.human_classification_group_label, df.model_classification_group_rc, normalize='true', labels=groups)
# cm = sklearn.metrics.confusion_matrix(df.human_classification_group_label, df.group, normalize='true', labels=groups)
# cm = sklearn.metrics.confusion_matrix(df.human_classification_group_label, df.predicted, normalize='true', labels=groups)
# cm = sklearn.metrics.confusion_matrix(df.KER_CODE, df.predicted, normalize='true', labels=groups)
# cm = sklearn.metrics.confusion_matrix(df.GROUP_CODE, df.group, normalize='true', labels=groups)


print(cm)

fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(cm, annot=True, fmt='.1f', xticklabels=labels.label, yticklabels=labels.label)
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=groups, yticklabels=groups)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

