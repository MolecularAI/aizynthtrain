# %%
import json
from collections import defaultdict, Counter

import pandas as pd
from IPython.display import Markdown

pd.options.display.float_format = "{:,.2f}".format
print_ = lambda x: display(Markdown(x))

# %% tags=["parameters"]
input_filename = ""
output_filename = ""

# %%
data = pd.read_csv(input_filename, sep="\t")

# %%
def make_stats_by_sources(data, extra_columns=None):
    extra_columns = extra_columns or []
    sources = sorted(set(data.source))
    source_stats = defaultdict(list)
    for source in sources:
        source_stats["source"].append(source)
        df = data[data.source == source]
        source_stats["nreactions"].append(len(df))
        source_stats["% of total"].append(len(df) / len(data) * 100)
        unique_hashes = set(df["PseudoHash"])
        source_stats["n unique reactions"].append(len(unique_hashes))
        source_stats["% unique"].append(len(unique_hashes) / len(df) * 100)
        has_changes = ~df.StereoChanges.isna()
        source_stats["n reactions with stereo changes in centre"].append(
            has_changes.sum()
        )
        source_stats["% with stereo changes in centre"].append(has_changes.mean() * 100)
        source_stats["% unclassified"].append(
            df["classification"].str.startswith("0.0 ").mean() * 100
        )
        for column in extra_columns:
            source_stats[column].append(df[column].sum())
        df2 = df[~df.date.isna()]
        dates = pd.to_datetime(df2.date, infer_datetime_format=True)
        source_stats["from"].append(dates.min())
        source_stats["to"].append(dates.max())

    if len(sources) > 1:
        source_stats["source"].append("all")
        source_stats["nreactions"].append(sum(source_stats["nreactions"]))
        source_stats["% of total"].append("100")
        source_stats["n unique reactions"].append(len(set(data["PseudoHash"])))
        source_stats["% unique"].append(
            source_stats["n unique reactions"][-1]
            / source_stats["nreactions"][-1]
            * 100
        )
        has_changes = ~data.StereoChanges.isna()
        source_stats["n reactions with stereo changes in centre"].append(
            has_changes.sum()
        )
        source_stats["% with stereo changes in centre"].append(has_changes.mean() * 100)
        for column in extra_columns:
            source_stats[column].append(data[column].sum())
        source_stats["% unclassified"].append(
            data["classification"].str.startswith("0.0 ").mean() * 100
        )
        source_stats["from"].append(min(source_stats["from"]))
        source_stats["to"].append(max(source_stats["to"]))
    return source_stats


# %% [markdown]
"""
## Statistics on extracted and validated reactions
"""

# %%
print_(f"Total number of extracted reactions = {len(data)}")
print("")
pd.DataFrame(
    make_stats_by_sources(
        data,
        extra_columns=[
            "HasChiralReagent",
            "StereoCentreCreated",
            "StereoCentreRemoved",
            "PotentialStereoCentre",
            "StereoOutside",
            "MesoProduct",
        ],
    )
)


# %%
sel = data.StereoChanges.isna()
print(
    f"Removing {sel.sum()} reactions where a stereo centre did not change during the reaction"
)
data = data[~sel]

# %% [markdown]
"""
## Categories of stereochemical reactions
Finding reactions for buckets:
    - Stereospecific
    - Reagent controlled
    - Substrate controlled
"""

# %%
ss_sel = ~(data["StereoCentreCreated"] | data["StereoCentreRemoved"])
ss_sel = ss_sel & (~data["MesoProduct"])

rc_sel = (
    data["StereoCentreCreated"]
    & (~data["PotentialStereoCentre"])
    & (~data["StereoOutside"])
    & (data["HasChiralReagent"])
)
rc_sel = rc_sel & (~data["MesoProduct"])

sc_sel = (
    data["StereoCentreCreated"]
    & (~data["PotentialStereoCentre"])
    & (data["StereoOutside"])
    & (~data["HasChiralReagent"])
)
sc_sel = sc_sel & (~data["MesoProduct"])

data = data.assign(
    IsStereoSpecific=ss_sel,
    IsReagentControlled=rc_sel,
    IsSubstrateControlled=sc_sel,
    StereoBucket="",
)
data.loc[ss_sel, "StereoBucket"] = "stereospecific"
data.loc[rc_sel, "StereoBucket"] = "reagent controlled"
data.loc[sc_sel, "StereoBucket"] = "substrate controlled"

# %%
sel = data["StereoBucket"] == ""
print(f"Removing {sel.sum()} reactions that does not fit into any category")
data = data[~sel]

# %%
print_("Statistics by source after filtering and categorizing")
pd.DataFrame(
    make_stats_by_sources(
        data,
        extra_columns=[
            "IsStereoSpecific",
            "IsReagentControlled",
            "IsSubstrateControlled",
        ],
    )
)

# %% [markdown]
"""
## Finalizing
"""

# %%
filename = output_filename.replace(".csv", "_all.csv")
print_(f"Saving all selected reactions with all columns to {filename}")
data.to_csv(filename, index=False, sep="\t")

# %%
hash_to_id = data.groupby("PseudoHash")["id"].apply(lambda x: list(x.values)).to_dict()
filename = output_filename.replace(".csv", "_ids.json")
with open(filename, "w") as fileobj:
    json.dump(hash_to_id, fileobj)
print_(f"Saving translation from reaction hash to reaction connect id to {filename}")

# %%
print_("Replacing classification with most common classification for each reaction")
most_common_classification_by_hash = (
    data.groupby("PseudoHash")["classification"]
    .apply(lambda x: Counter(x.values).most_common(1)[0][0])
    .to_dict()
)
data["classification"] = data.apply(
    lambda row: most_common_classification_by_hash[row["PseudoHash"]], axis=1
)


# %%
data.rename(columns={"rsmi_processed": "RxnSmilesClean"}, inplace=True)
data = data[
    ["RxnSmilesClean", "PseudoHash", "classification", "RingBreaker", "StereoBucket"]
]
data = data.drop_duplicates(subset="PseudoHash")
print_(
    f"Saving clean reaction SMILES, reaction hash, classification, "
    f"ring breaker flag and stereo bucket for selected reactions to {output_filename}",
)
data.to_csv(output_filename, index=False, sep="\t")
