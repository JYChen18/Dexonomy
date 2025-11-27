# %%
import numpy as np
import matplotlib.pyplot as plt

miu = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
miu = np.array(miu)
per_old = [0.03, 0.11, 0.17, 0.24, 0.28, 0.30]
per_new = [0.16, 0.27, 0.37, 0.43, 0.49, 0.51]

miu = np.array(miu)
per_old = np.array(per_old)
per_new = np.array(per_new)

plt.plot(miu, per_old * 100, label="old")
plt.plot(miu, per_new * 100, label="new")
plt.xlabel("miu")
plt.ylabel("success percentage")
plt.legend()
plt.savefig("eval.png")
plt.show()
# %%
