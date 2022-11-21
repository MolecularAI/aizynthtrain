# %%
import json

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from IPython.display import Markdown

pd.options.display.float_format = "{:,.2f}".format
print_ = lambda x: display(Markdown(x))

# %% tags=["parameters"]
validation_metrics_filename = ""
onestep_report = ""
multistep_report = ""

# %% [markdown]
"""
## Statistics on expansion model training
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

# %% [markdown]
"""
## Evaluation of multi-step retrosynthesis capabilities
"""

# %%
if multistep_report:
    with open(multistep_report, "r") as fileobj:
        multistep_stats = json.load(fileobj)
else:
    multistep_stats = None

# %% [markdown]
"""
### Route finding capabilities
"""

# %%
if multistep_stats:
    pd_stats = pd.DataFrame(multistep_stats["finding"])
    print_(f"Average first solution time: {pd_stats['first solution time'].mean():.2f}")
    print_(f"Average number of solved target: {pd_stats['is solved'].mean()*100:.2f}%")
else:
    print_("Route finding capabilities not evaluated")


# %% [markdown]
"""
### Route recovery capabilities
"""

# %%
if multistep_stats:
    pd_stats = pd.DataFrame(multistep_stats["recovery"])
    display(pd_stats)

    print_(f"Average number of solved target: {pd_stats['is solved'].mean()*100:.2f}%")
    print_(f"Average found reference: {pd_stats['found reference'].mean()*100:.2f}%")
    print_(
        f"Average closest to reference: {pd_stats['closest to reference'].mean():.2f}"
    )
    print_(f"Average rank of closest: {pd_stats['rank of closest'].mean():.2f}")
else:
    print_("Route recovery capabilities not evaluated")


# %% [markdown]
"""
## Evaluation of one-step retrosynthesis capabilities
"""

# %%
stats = pd.read_json(onestep_report)
display(stats)

print_(f"Average found expected: {stats['found expected'].mean()*100:.2f}%")
print_(f"Average rank of expected: {stats['rank of expected'].mean():.2f}")
print_(f"Average ring broken when expected: {stats['ring broken'].mean()*100:.2f}%")
print_(f"Percentage of ring reactions: {stats['ring breaking'].mean()*100:.2f}%")
print_(f"Average non-applicable (in top-50): {stats['non-applicable'].mean():.2f}")
