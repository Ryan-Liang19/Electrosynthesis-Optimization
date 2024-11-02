import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt


if __name__ == '__main__':
    Obj = np.empty((50, 0))
    F = np.empty((50, 0))
    Y = np.empty((50, 0))
    for num in [2, 0, 6, 12, 14, 8, 10]:
        df = pd.ExcelFile('experimental simulation.xlsx')
        t = np.array(df.parse(df.sheet_names[num]))
        y = np.array(df.parse(df.sheet_names[num+1]))
        t_budget = 900
        # BO alone: 0, 1; CAPBO: 2, 3; single obj: 4, 5; OVAT: 6, 7; HTS1: 8, 9; HTS2: 10, 11; DoE: 12, 13; CMA-ES: 14, 15

        count_set = []
        obj_set = []
        F_set = []
        y_set = []
        for i in range(0, 50):
            tt = t[:, i]
            yy = y[:, 4*i:4*i+4]
            for j in range(len(tt)):
                if tt[j] > t_budget:
                    break
            count_set.append(j)
            index = np.argmax(yy[:j, 3])
            while yy[index, 1] < 0.2 or yy[index, 2] < 0.85:
                yy[index, 3] = 0
                index = np.argmax(yy[:j, 3])
            obj_set.append(yy[index, 0])
            F_set.append(yy[index, 1])
            y_set.append(yy[index, 2])
            print('{0} / 50: {1}, {2}'.format(i, j, yy[index, 0]))

        print('Number of experiments: {0} +- {1}'.format(np.mean(count_set), np.std(count_set)))
        print('Highest current density: {0} +- {1}'.format(np.mean(obj_set), np.std(obj_set)))
        print('Corresponding Faraday efficiency: {0} +- {1}'.format(np.mean(F_set), np.std(F_set)))
        print('Corresponding yield: {0} +- {1}'.format(np.mean(y_set), np.std(y_set)))

        Obj = np.concatenate([Obj, np.array(obj_set).reshape(-1, 1)], axis=1)
        F = np.concatenate([F, np.array(F_set).reshape(-1, 1)], axis=1)
        Y = np.concatenate([Y, np.array(y_set).reshape(-1, 1)], axis=1)

    labels = ['Proposed BO', 'Classical BO', 'OVAT', 'DoE', 'CMA-ES', 'HTE & RS', 'HTE & PSO']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    bplot1 = axes[0].boxplot(Obj, notch=True, showfliers=False, patch_artist=True, labels=labels)
    axes[0].set_title('Current Density ' + r'$(mA/cm^2)$', fontsize=18)
    axes[0].tick_params(labelsize=14)
    bplot2 = axes[1].boxplot(Y, notch=True, showfliers=False, patch_artist=True, labels=labels)
    axes[1].set_title('Faraday Efficiency', fontsize=18)
    axes[1].tick_params(labelsize=14)
    axes[1].set_ylim((0.84, 0.94))
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    axes[1].axhline(0.85, c='#8983BF', linewidth=3, linestyle='dashed')
    bplot3 = axes[2].boxplot(F, notch=True, showfliers=False, patch_artist=True, labels=labels)
    axes[2].set_title('Yield', fontsize=18)
    axes[2].tick_params(labelsize=14)
    axes[2].set_ylim((0.19, 0.28))
    axes[2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    axes[2].axhline(0.2, c='#8983BF', linewidth=3, linestyle='dashed')
    colors = ['#F27970', '#BB9727', '#32B897', '#05B9E2', '#C76DA2', '#A0D0D0', '#B7F5DE']
    for bplot in (bplot1, bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    for i in range(3):
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(30)
    fig.tight_layout()
    plt.show()
    # plt.savefig('test.tif', dpi=300)
