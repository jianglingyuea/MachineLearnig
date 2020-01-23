from matplotlib.pyplot import scatter
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

X, y = make_moons(n_samples=1000, noise=0.25, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=42)
print(X, y)
def make_plot(X, y, plot_name, file_name, XX=None, YY=None, preds=None):
    plt.figure()
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.08,
                     )
    plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5],
                cmap="Greys",
                vmin=0, vmax=.6)
    # 绘制正负样本
    markers = ['o' if i == 1 else 's' for i in y.ravel()]
    scatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, edgecolors='none', m=markers)
    plt.savefig(file_name)
make_plot(X, y, None, "dataset.svg")

