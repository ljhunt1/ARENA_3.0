#%%
import numpy as np

a = np.array([[1,2, 3], [4, 5, 6]])

# %%
a.dtype
# %%
np.zeros([5, 3])
# %%
a = np.linspace(10, 20, num=5)
# %%
a[:, np.newaxis]
# %%
a
# %%
np.expand_dims(a, axis=1)
# %%
a = np.array([1, 5, 2, 6, 0, 100])