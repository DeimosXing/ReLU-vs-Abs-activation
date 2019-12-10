from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

train_abs = np.loadtxt('save_abs/trainloss.txt', dtype=np.float)
test_abs = np.loadtxt('save_abs/testloss.txt', dtype=np.float)
train_relu = np.loadtxt('save_relu/trainloss.txt', dtype =np.float)
test_relu = np.loadtxt('save_relu/testloss.txt', dtype =np.float)
xnew = np.linspace(0, 199, 200)
train_relu_spl = make_interp_spline(np.array(list(range(len(train_relu)))), train_relu, k=3)  # type: BSpline
train_relu_smooth = train_relu_spl(xnew)
train_abs_spl = make_interp_spline(np.array(list(range(len(train_abs)))), train_abs, k=3)
train_abs_smooth = train_abs_spl(xnew)
plt.plot(train_abs_smooth,color = 'C1',label = 'train_abs')
plt.plot(train_relu_smooth,color = 'C2',label = 'train_relu')
# plt.plot(test_abs,'C1--',label = 'test_abs')
# plt.plot(test_relu,'C2--',label = 'test_relu')
plt.text(180, 1.45, 'acc:75.64', color = 'C1')
plt.text(180, 1.05, 'acc:83.42', color = 'C2')
plt.title('Comparing Abs and ReLU Activation on Resnet20')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss(log scale)')
plt.legend()
plt.show()
