"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_xsx.py
    @Time: 2025/7/19 22:20
    @Email: None
"""
from matplotlib import pyplot as plt

files = [
r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UnetHybridAttentionV23Ablation1\20250720_011505\att2_log.txt',
r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UnetHybridAttentionV23Ablation2\20250720_011507\att2_log.txt',
r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UNetHybridAttentionV23Ablation3\20250720_011502\att2_log.txt'
]
thresholdss = []
sparsityss = []
for file in files:
    f = open(file, 'r')
    lines = f.readlines()
    thresholds = []
    sparsitys = []
    for line in lines:
        if line.strip() != '':
            t = line[:-1]
            if 'Epoch' in t:
                thresholds.append(float(t.split(':')[1]))
            if 'sparsity' in t:
                sparsitys.append(float(t.split(':')[1]))
    thresholdss.append(thresholds)
    sparsityss.append(sparsitys)

epochs = list(range(1, len(thresholds) + 1))

plt.plot(epochs, thresholdss[0], color='red', label='fix 0.5')
plt.plot(epochs, thresholdss[1], color='blue', label='learnable')
plt.plot(epochs, thresholdss[2], color='green', label='fix 0.02 & 0.2')
plt.xlabel('epoch')
plt.ylabel('threshold')
plt.legend()
plt.show()

