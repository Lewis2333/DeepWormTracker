import pandas as pd
import matplotlib.pyplot as plt

# 读取训练指标
train_metrics = pd.read_csv('results.csv')



fig, axs = plt.subplots(4, 1, figsize=(12, 16))

# 精度曲线
axs[0].plot(train_metrics['epoch'], train_metrics['metrics/precision(B)'], color='blue')
axs[0].set_title('Precision')

# 召回率曲线
axs[1].plot(train_metrics['epoch'], train_metrics['metrics/recall(B)'], color='green')
axs[1].set_title('Recall')

# mAP@0.5曲线
axs[2].plot(train_metrics['epoch'], train_metrics['metrics/mAP50(B)'], color='red')
axs[2].set_title('mAP@0.5')


plt.tight_layout()
plt.savefig('all_metrics.png')