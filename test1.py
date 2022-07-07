import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn.metrics import precision_recall_curve


y_true = np.array([0, 0, 1, 1, 0, 0, 1])
y_scores = np.array([0.3, 0.2, 0.65, 0.7, 0.4, 0.15, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

print("精度  -    ", precision)
print("召回率 -   ", recall)
print("阈值  -    ", thresholds)
target = precision + recall
print(target)
index = argmax(target)
print(thresholds[index])

