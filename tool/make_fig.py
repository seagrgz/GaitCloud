#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.scale as scale
import numpy as np

def original(x):
    global margin
    return max(x+margin,0)

def tpmargin(x):
    global margin, k
    return max(x+margin,0)*k

def scaledtp(x):
    global margin, lamda, k
    if x+margin>0:
        y = np.power((np.power(margin,1/lamda)/margin)*(x+margin),lamda)
    else:
        y = 0
    return y*k

def expscale(x):
    global alpha
    return 1 - np.exp(alpha*x)

def rationalscale(x):
    global beta
    return x/(x+beta)

margin = 0.2
lamda = 1.2
k = 1
alpha = -0.2
beta = 10

#x = np.linspace(-0.5,1,1000)
#y1 = np.asarray([tpmargin(item) for item in x])
#y2 = np.asarray([scaledtp(item) for item in x])
#y3 = np.asarray([original(item) for item in x])

#plt.plot(x, y1, label='scaled', linewidth=5)
#plt.plot(x, y2, label='scaled', linewidth=5)
#plt.plot(x, y3, label='original', linewidth=5)
#plt.plot(x, np.repeat(0,1000), color='k', linestyle='--', linewidth=3)
#plt.plot(np.repeat(0,4), np.linspace(-0.2,1,4), color='k', linestyle='--', linewidth=3)
#plt.xlabel('distance')
#plt.ylabel('loss')
#plt.ylim(-0.2,1)

#x = np.linspace(0,50, 1000)
#y1 = np.asarray([expscale(item) for item in x])
#y2 = np.asarray([rationalscale(item) for item in x])

#plt.plot(x, y1, label='exp', linewidth=5)
#plt.plot(x, y2, label='rational', linewidth=5)
#plt.xlabel('depth')
#plt.ylabel('scale')
#plt.ylim(0,1)

def resolution():
    #dilution exp
    exps = ['Original-', 'Augment-']
    var = [[92.45, 85.68, 79.32, 70.56, 42.87],
            [90.4, 89.68, 88.84, 87.29, 77.55]]
    view = [[94.58, 70.03, 55.25, 39.83, 11.71],
            [93.52, 92.62, 91.09, 89.64, 72.79]]
    xaxis = [1, 2, 3, 4, 8]
    xlabels = ['/1', '/2', '/3', '/4', '/8']
    xname = 'TestSet'
    yname = 'Rank-1 accuracy (%)'
    colors = ['#2F7FC1', '#96C37D']
    markers = ['*', 's']
    anchor = (0.05,0.05)
    loc = 'lower left'
    linewidth = 2

    fig, axes = plt.subplots(figsize=(6,4), dpi=300)
    for i, group in enumerate(exps):
        axes.plot(xaxis, var[i], label='{}Variance'.format(group), linestyle='-', color=colors[i], linewidth=linewidth, marker=markers[i], markersize=4)
        axes.plot(xaxis, view[i], label='{}View'.format(group), linestyle='-.', color=colors[i], linewidth=linewidth, marker=markers[i], markersize=4)

    axes.set_xticks(xaxis, labels=xlabels)
    plt.legend(bbox_to_anchor=anchor, loc=loc)
    axes.set_xlabel(xname, fontsize=12)
    axes.set_ylabel(yname, fontsize=12)
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig('test_out.png')
    #plt.grid(True)
    plt.close()

def framenum():
    exps = ['GaitBase', 'LidarGait', 'GaitCloud', 'GaitCloud+']
    titles = ['Variance', 'Cross-view']
    var = [[45.73,63.72,75.96,85.61,86.37],
            [59.07,75.96,82.09,84.56,85.26],
            [44.57,69.94,83.46,90.54,91.38],
            [43.88,68.93,85.24,92.45,93.49]]
    view = [[33.23,45.73,59.84,74.62,75.18],
            [60.74,80.82,87.89,90.58,90.59],
            [45.04,71.91,89.2,94.51,95.25],
            [44.35,68.7,88.49,94.58,95.61]]
    xaxis = [2, 5, 10, 20, 30]
    xlabels = xaxis
    xname = 'Frame Number'
    yname = 'Rank-1 Accuracy (%)'
    #colors = ['#2F7FC1', '#96C37D']
    markers = ['*', 's', 'd', 'o']
    anchor = (0.95,0.05)
    loc = 'lower right'
    linewidth = 2

    fig, axes = plt.subplots(2, 1, figsize=(4.5,6), dpi=300)
    for i, group in enumerate(exps):
        axes[0].plot(xaxis, var[i], label=group, linewidth=linewidth, marker=markers[i], markersize=4)
        axes[1].plot(xaxis, view[i], label=group, linewidth=linewidth, marker=markers[i], markersize=4)
    plt.xticks(xaxis, labels=xlabels)
    plt.xlabel(xname, fontsize=12)
    plt.ylabel(yname, fontsize=12)
    for i, ax in enumerate(axes):
        ax.legend(bbox_to_anchor=anchor, loc=loc)
        ax.set_title(titles[i])
        ax.set_xlabel(xname, fontsize=12)
        ax.set_ylabel(yname, fontsize=12)
        ax.set_xticks(xaxis, labels=xlabels)
        #ax.set_ylim(0,100)
    plt.tight_layout()
    plt.savefig('test_out.png')
    plt.grid(True)
    plt.close()

def traindistribute():
    global lab_f, le_f
    attr = ['nm', 'bg', 'cl', 'cr', 'ub', 'uf', 'oc', 'nt']
    #attr_num = [180, 684, 176, 1665, 624, 404, 176, 188]
    attr_num = [658, 1846, 432, 5827, 1872, 1102, 588, 693]

    fig, axes = plt.subplots(figsize=(16,10))
    axes.bar(attr, attr_num, color='#96C37D')
    axes.set_xlabel('attributes', fontsize=16)
    axes.set_ylabel('number of samples', fontsize=16)
    axes.set_xticks(np.arange(len(attr)))
    plt.xticks(fontsize=16)
    axes.set_xticklabels(attr)
    plt.savefig('test_out.png')
    plt.close()

def losses():
    global lab_f, le_f
    loss_b = np.load('/home/sx-zhang/[2024-07-09_14:23:02.221595]_loss.npy')[1][:40000]
    loss_l = np.load('/home/sx-zhang/[2024-07-09_18:00:47.559497]_loss.npy')[1][:40000]
    loss_lh = np.load('/home/sx-zhang/[2024-07-09_14:17:22.681586]_loss.npy')[1][:40000]
    loss_b_r = np.load('/home/sx-zhang/[2024-07-11_18:19:33.668545]_loss.npy')[1][:40000]
    loss_l_r = np.load('/home/sx-zhang/[2024-07-10_08:12:42.624429]_loss.npy')[1][:40000]
    loss_lh_r = np.load('/home/sx-zhang/[2024-07-10_07:11:02.810357]_loss.npy')[1][:40000]
    iters = np.arange(len(loss_b))
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    axes[0].plot(iters, loss_b, color='#F3D266', label='baseline')
    axes[0].plot(iters, loss_l, color='#2F7FC1', label='baseline+LE')
    axes[0].plot(iters, loss_lh, color='#96C37D', label='baseline+LE+HCP')
    axes[1].plot(iters, loss_b_r, color='#F3D266', label='baseline')
    axes[1].plot(iters, loss_l_r, color='#2F7FC1', label='baseline+LE')
    axes[1].plot(iters, loss_lh_r, color='#96C37D', label='baseline+LE+HCP')
    def scale_func(in_array):
        out_array = []
        for values in in_array:
            if values < 30:
                out_array.append(values/3)
            else:
                out_array.append(values/100+10)
        return np.asarray(out_array)

    def inverse_func(out_array):
        in_array = []
        for values in out_array:
            if values < 10:
                in_array.append(values*3)
            else:
                in_array.append((values-10)*100)
        return np.asarray(in_array)
                
    axes[0].set_yscale(scale.FuncScale(axes[0].yaxis, (scale_func, inverse_func)))
    axes[1].set_yscale(scale.FuncScale(axes[1].yaxis, (scale_func, inverse_func)))
    major_ticks = np.concatenate((np.arange(0, 31, 3), np.array([30,200,400,600])))
    major_ticks_r = np.concatenate((np.arange(0, 31, 3), np.array([30,200,400,600,800,1000])))
    axes[0].yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    axes[1].yaxis.set_major_locator(ticker.FixedLocator(major_ticks_r))

    axes[0].set_xlabel('iterations', fontsize=lab_f)
    axes[0].set_ylabel('triplet loss', fontsize=lab_f)
    axes[0].legend(loc='upper right', fontsize=le_f)
    axes[1].set_xlabel('iterations', fontsize=lab_f)
    axes[1].set_ylabel('triplet loss', fontsize=lab_f)
    axes[1].legend(loc='upper right', fontsize=le_f)
    plt.savefig('test_out.png')
    plt.close()

def noise():
    var_accu = np.array([92.45,91.96,91.21,87.31,82.04,72.4,59.45,46.78,35.81,25.66,18.63])
    view_accu = np.array([94.58,94.47,93.27,89.37,80.63,67.66,53.54,40.6,30.07,22.38,16.48])
    noise = np.array([0,0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095])
    #noise = np.arange(0,0.1,0.005)
    noise = (3**0.5)*noise
    fig, axes = plt.subplots(figsize = (6,4), dpi=300) 
    plt.plot(noise, var_accu, label = 'Variance')
    plt.plot(noise, view_accu, label = 'Cross-view')
    plt.plot(np.repeat(0.03,4), np.linspace(0,100,4), color='k', linestyle='--', linewidth=2)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.85))
    plt.text(0.032, 5, '0.03m', fontsize=12, fontweight='bold')
    axes.set_xlabel('Noise(m)', fontsize=12)
    axes.set_ylabel('Rank-1 accuracy(%)', fontsize=12)
    plt.ylim(0,100)
    fig.tight_layout()
    plt.savefig('test_out.png')
    plt.close()

def data_vol():
    def gaitbase(t,w=44,h=64):
        return t*w*h/1e3

    def lidargait(t,w=64,h=64,c=3):
        return t*w*h*c/1e3

    def gaitcloud(t,h=64,l=40,w=40):
        return np.full_like(t, h*l*w/1e3)

    def gaitcloud_p(t,h=64,l=40,w=40,c=2):
        return h*l*w/1e3+t*h*w*c/1e3

    def lidargait_3d(t,h=64,l=40,w=40):
        return t*h*l*w/1e3

    frames = np.arange(1,31,1)
    fig, axes = plt.subplots(figsize=(6.8,4.8), dpi=300)
    plt.plot(frames, gaitbase(frames), label='GaitBase')
    plt.plot(frames, lidargait(frames), label='LidarGait')
    plt.plot(frames, gaitcloud(frames), label='GaitCloud')
    plt.plot(frames, gaitcloud_p(frames), label='GaitCloud+')
    plt.plot(frames, lidargait_3d(frames), label='3D-LidarGait')
    plt.legend(bbox_to_anchor=(0.5,0.95), loc='upper center')
    axes.set_xlabel('Frames', fontsize=12)
    axes.set_ylabel('Data Points (k)', fontsize=12)
    plt.xlim(0,31)
    plt.ylim(0,500)
    fig.tight_layout()
    plt.savefig('test_out.png')
    plt.close()

def comp_tradeoff():
    import matplotlib.pyplot as plt
    import numpy as np

    # Example data
    methods = ["GaitBase", "LidarGait", "GaitCloud", "GaitCloud+", "Naive3D"]
    groups = ["Accuracy", "MACs(G)", "Time(ms)"]
    accu_texts = [73, 87, 93, 94, 95]
    accu_values = [73.25, 87, 92.525, 93.515, 95]
    mac_texts = [44, 44, 58, 87, 1061]
    mac_values = [44.06, 44.16, 58.19, 86.97, 1060.55]
    time_texts = [1.4, 1.4, 2.5, 3.4, 24.8]
    time_values = [1.39, 1.38, 2.46, 3.42, 24.79]

    # X positions for each method
    x = np.arange(len(groups))  
    width = 0.15  # width of each bar
    bars = []
    values = []
    texts = []
    plt.rcParams.update({
        'font.size': 14,        # default text size
        'xtick.labelsize': 14,  # x tick labels
        'ytick.labelsize': 14,  # y tick labels
        'legend.fontsize': 16,  # legend
        })
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot grouped bars
    for i in range(len(methods)):
        values.append([accu_values[i],mac_values[i]*0.1,time_values[i]*4])
        texts.append([accu_texts[i],mac_texts[i],time_texts[i]])
        bars.append(ax.bar(x+(i-2)*width, values[-1], width, label=methods[i]))

    # Labels & ticks
    #ax.set_xlabel("Methods")
    #ax.set_ylabel("Values")
    #ax.set_title("Comparison of Accuracy, MACs, and Forward Time")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0)
    plt.legend(bbox_to_anchor=(0.9,0.75), loc='upper right')

    # Optional: Add values on top of bars
    for i, bargroup in enumerate(bars):
        for j, bar in enumerate(bargroup):
            height = bar.get_height()
            ax.annotate(texts[i][j],
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 2),  # 3 pts above bar
                textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    lab_f = 16
    le_f = 16
    #losses()
    #resolution()
    #noise()
    #data_vol()
    #framenum()
    comp_tradeoff()
