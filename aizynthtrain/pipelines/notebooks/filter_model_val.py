# %%
import pandas as pd
import matplotlib.pylab as plt
from IPython.display import Markdown

pd.options.display.float_format = "{:,.2f}".format
print_ = lambda x: display(Markdown(x))

# %% tags=["parameters"]
validation_metrics_filename = ""

# %% [markdown]
"""
## Statistics on filter model training
"""

# %%
val_data = pd.read_csv(validation_metrics_filename)
val_data.tail()

# %%
print_("Convergence of validation loss and accuracy")
fig = plt.figure()
ax = fig.gca()
ax2 = ax.twinx()
val_data.plot(x="epoch", y="val_loss", ax=ax, legend=False)
val_data.plot(x="epoch", y="val_accuracy", style="g", ax=ax2, legend=False)
_ = fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

# %%
print_("Metrics at the last epoch")
for key, val in val_data.iloc[-1].to_dict().items():
    if key == "epoch":
        continue
    print_(f"- {key} = {val:.2f}")
