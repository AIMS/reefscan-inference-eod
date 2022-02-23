import pandas as pd
import matplotlib.pyplot as plt

cal = pd.read_csv("C:/greg/reefscan_ml/tests/train-results/GROUP_CODEreefscan-train-test.csv.cal.csv")
cal.plot(x="prob_pred", y="prob_true")
plt.plot([0, 1], [0, 1], color='k', linestyle=':', linewidth=1)
plt.show(block=True)

cal['ece'] = abs(cal.prob_true - cal.prob_pred)
cal.boxplot(column="ece")
plt.show(block=True)
print(cal)

