import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math


# ref_color = (0, 1, 0)
# rho = np.linspace(0, 1, 100)
# phi = np.linspace(0, math.pi * 2, 500)
# RHO, PHI = np.meshgrid(rho, phi)
# h = (PHI-PHI.min()) / (PHI.max()-PHI.min()) # use angle to determine hue, normalized from 0-1
# h = np.flip(h)
# s = RHO               # saturation is set as a function of radias
# v = np.ones_like(RHO) # value is constant
# h,s,v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
# target_colors = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]
# target_colors = np.array(target_colors)
# target_colors = (target_colors).astype(np.uint8)

# for i in range(len(target_colors)):
#     plt.close()
#     plt.figure(figsize=(4, 4), facecolor=ref_color)
#     plt.text(0.15, 0.4, 'hello', color=target_colors[i], fontsize=80, fontweight='bold')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(f'/node_data_2/EyeExam/dataset/sensitivity/ocr-text-green/{i}.jpg')

# increasing font sizes: 5, 10, 15, ..., 80
# font_sizes = np.linspace(1, 100, 1000)

# for i, font_size in enumerate(font_sizes):
#     plt.close()
#     plt.figure(figsize=(4, 4))
#     plt.text(0.0, 0.0, 'hello', fontsize=font_size, fontweight='bold')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(f'/node_data_2/EyeExam/dataset/sensitivity/ocr-size/{i}.jpg')


import seaborn as sns
import pandas as pd
# Shares
# Greenland 4.52
# Mauritania 4.15
# Indonesia 3.06
# Ireland 2.33

countries = ['Mauritania', 'Greenland', 'Indonesia', 'Ireland']
shares = [4.15, 4.52, 3.06, 2.33]
data = {'countries': countries, 'shares': shares}
df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")
colors = sns.color_palette('Spectral', 1024)
for i in range(11):
    start = 0
    end = 2**i
    step = (end - start) // 4
    if step == 0:
        step = 1
    color_set = colors[start:end:step]
    ax = sns.barplot(x='shares', y='countries', data=df, palette=color_set, orient='h', hue='countries', legend=True, dodge=False)
    # ax.legend(["Mauritania", "Indonesia", "Ireland", "Greenland"])
    sns.move_legend(ax, "lower center", ncol=4, title=None, frameon=False,  bbox_to_anchor=(.5, 1))
    # for p in ax.patches:
    #     ax.annotate(f'{p.get_width():.2f}', (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
    #                 ha='center', va='center', fontsize=12, color='black', xytext=(10, 0),
    #                 textcoords='offset points')
    ax.set_yticklabels([])
    ax.set_xlabel('Shares')
    ax.set_ylabel('')
    # plt.title('Shares')
    plt.tight_layout()
    plt.savefig('/node_data_2/EyeExam/dataset/sensitivity/chart-color/{}.jpg'.format(i))
    plt.close()
