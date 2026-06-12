import numpy as np
import matplotlib.pyplot as plt

sizes = [5, 25, 50, 100]

for s in sizes:
    f1 = np.load(f"f1_h{s}.npy")
    plt.plot(f1, label=f"h={s}")

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Hidden Units Comparison")
plt.legend()
plt.savefig("hidden_units_plot.png")
plt.show()
