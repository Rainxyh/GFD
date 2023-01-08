import matplotlib.pyplot as plt
import numpy as np
from pyts.image import RecurrencePlot
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from data.gearbox_data import GearboxData

'''
读取时间序列的数据
怎么读取需要你自己写
X为ndarray类型数据
'''
def seq2img(idx, X):
    if idx == 1:
        recurrence_plot(X)
    elif idx == 2:
        markov_transition_field(X)
    elif idx == 3:
        gramian_angular_field(X)
    elif idx == 0:
        recurrence_plot(X)
        markov_transition_field(X)
        gramian_angular_field(X)

# 1.Recurrence Plot （递归图）--------------------------------------------------------------------------------
def recurrence_plot(X):
    # Recurrence plot transformation
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(X)

    # Show the results for the first time series
    plt.figure(figsize=(5, 5))
    plt.imshow(X_rp[0], cmap='binary', origin='lower')
    plt.title('Recurrence Plot', fontsize=16)
    plt.tight_layout()
    plt.show()

# 2.Markov Transition Field （马尔可夫变迁场）--------------------------------------------------------------------------------
def markov_transition_field(X):
    # MTF transformation
    mtf = MarkovTransitionField(image_size=24)
    X_mtf = mtf.fit_transform(X)

    # Show the image for the first time series
    plt.figure(figsize=(5, 5))
    plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
    plt.title('Markov Transition Field', fontsize=18)
    plt.colorbar(fraction=0.0457, pad=0.04)
    plt.tight_layout()
    plt.show()

# 3. Gramian Angular Field （格拉米角场）--------------------------------------------------------------------------------
def gramian_angular_field(X):
    # Transform the time series into Gramian Angular Fields
    gasf = GramianAngularField(image_size=256, method='summation')
    X_gasf = gasf.fit_transform(X)
    gadf = GramianAngularField(image_size=256, method='difference')
    X_gadf = gadf.fit_transform(X)

    # Show the results for the first time series
    axs = plt.subplots()
    plt.subplot(211)
    plt.imshow(X_gasf[0], cmap='rainbow', origin='lower')
    plt.title("GASF", fontsize=16)
    plt.subplot(212)
    plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
    plt.title("GADF", fontsize=16)

    #plt.axes((left, bottom, width, height)
    cax = plt.axes([0.7, 0.1, 0.02, 0.8])
    plt.colorbar(cax=cax)
    plt.suptitle('Gramian Angular Fields', y=0.98, fontsize=16)
    plt.tight_layout()
    plt.show()
