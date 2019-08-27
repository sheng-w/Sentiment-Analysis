import pandas as pd
from sklearn.metrics import precision_recall_curve

data = pd.read_csv("prediction.csv")
data = data.sort_values(by=['pred'])
precision, recall, _ = precision_recall_curve(data['label'], data['pred'])

for i in range(len(recall)):
    print(recall[i], precision[i])
