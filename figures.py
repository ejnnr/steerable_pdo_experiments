# The # %% lines denote cells, allowing this file to be run using
# the interactive mode of the Python VS Code extension. But you
# can also run it simply as a normal python script.

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.signal import correlate2d
from e2cnn.diffops.utils import discretize_homogeneous_polynomial as dp
from e2cnn import nn
from e2cnn import gspaces
# %%
sns.set_theme(context="paper")
os.makedirs("fig", exist_ok=True)
# %%
x = np.linspace(0, np.pi, 128)
xx, yy = np.meshgrid(x[32:], x[:64])
zz_x = np.sin(xx + yy) + 0.5 * (xx - 2)
zz_y = np.cos(xx - yy) + 0.5 * (yy - 2)
zz_mag = np.sqrt(zz_x**2 + zz_y**2)

plt.figure(figsize=(6, 4))
plt.gcf().set_facecolor((0.95, 0.95, 0.95))
plt.quiver(zz_x[::8, ::8], zz_y[::8, ::8], zz_mag[::8, ::8])
plt.axis("equal")
plt.axis("off")
plt.gcf().tight_layout()
plt.savefig("fig/vector_input.pdf", bbox_inches="tight", facecolor=plt.gcf().get_facecolor())

# %%

# the order is like that ... for reasons
# y_grad, x_grad = np.gradient(zz)
# gradient_mag = np.sqrt(x_grad**2 + y_grad**2)

# Laplacian of the divergence:
x_filter = dp([-2, -1, 0, 1, 2], np.array([0, 1, 0, 1])).reshape(5, 5)
# curl:
x_filter += dp([-2, -1, 0, 1, 2], np.array([1, 0])).reshape(5, 5)
y_filter = dp([-2, -1, 0, 1, 2], np.array([1, 0, 1, 0])).reshape(5, 5)
y_filter += dp([-2, -1, 0, 1, 2], np.array([0, -1])).reshape(5, 5)

out = correlate2d(zz_x, x_filter, mode="valid") + correlate2d(zz_y, y_filter, mode="valid")

plt.figure(figsize=(6, 4))
plt.imshow(out, origin="lower")
plt.axis("off")
plt.axis("equal")
plt.gcf().tight_layout()
plt.savefig("fig/scalar_output.pdf", bbox_inches="tight")

# %%
plt.imshow(x_filter, cmap="gray")
plt.axis("off")
plt.gcf().tight_layout()
plt.savefig("fig/laplacian_divergence_filter_x.pdf", bbox_inches="tight")
plt.imshow(y_filter, cmap="gray")
plt.axis("off")
plt.gcf().tight_layout()
plt.savefig("fig/laplacian_divergence_filter_y.pdf", bbox_inches="tight")
# %%
gs = gspaces.Rot2dOnR2(8)
in_type = nn.FieldType(gs, [gs.trivial_repr])
out_type = nn.FieldType(gs, [gs.regular_repr])
# %%
models = {}
for kernel_size in [3, 5]:
    models[("Kernel", kernel_size)] = nn.R2Conv(in_type, out_type, kernel_size)
    max_order = 2 if kernel_size == 3 else 3
    models[("FD", kernel_size)] = nn.R2Diffop(in_type, out_type, kernel_size, maximum_order=max_order)
    models[("RBF-FD", kernel_size)] = nn.R2Diffop(in_type, out_type, kernel_size, rbffd=True, maximum_order=max_order)
    smoothing = 1 if kernel_size == 3 else 1.3
    models[("Gauss", kernel_size)] = nn.R2Diffop(in_type, out_type, kernel_size, smoothing=smoothing, maximum_order=max_order)
# %%
filters = {}
for k, model in models.items():
    exp = getattr(model.basisexpansion, "block_expansion_('irrep_0', 'regular')")
    size = k[1]
    filters[k] = exp.sampled_basis.numpy().reshape(-1, 8, size, size)
# %%
methods = ["Kernel", "FD", "RBF-FD", "Gauss"]
for size in [3, 5]:
    fig, ax = plt.subplots(4, 6)
    vmin = min(np.min(filters[(method, size)]) for method in methods)
    vmax = max(np.max(filters[(method, size)]) for method in methods)
    for i, method in enumerate(methods):
        ax[i, 0].set_ylabel(method, size="large")
        for j in range(6):
            ax[i, j].imshow(
                filters[(method, size)][-j-1, 1],
                cmap="bwr",
                vmin=vmin,
                vmax=vmax,
            )
            ax[i, j].axis("equal")
            # ax[i, j].axis("off")
            ax[i, j].get_xaxis().set_ticks([])
            ax[i, j].get_yaxis().set_ticks([])
    # fig.tight_layout()
    fig.subplots_adjust(hspace=.5)
    fig.subplots_adjust(wspace=.5)
    fig.savefig(f"fig/stencils_{size}.pdf")