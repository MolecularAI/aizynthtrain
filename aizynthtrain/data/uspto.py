""" Module containing class to transform UPSTO data prepared by reaction_utils pipelines
"""
import pandas as pd


class UsptoImporter:
    def __init__(self, data_path: str, confidence_cutoff: float = 0.1) -> None:
        self.data = self.transform_data(data_path, confidence_cutoff)

    @staticmethod
    def transform_data(filename: str, confidence_cutoff: float) -> pd.DataFrame:
        df = pd.read_csv(
            filename, sep="\t", usecols=["ID", "Year", "mapped_rxn", "confidence"]
        )
        sel = df["confidence"] < confidence_cutoff
        print(
            f"Removing {sel.sum()} with confidence of mapping less than {confidence_cutoff}"
        )
        df = df[~sel]
        return pd.DataFrame(
            {
                "id": df["ID"],
                "source": ["uspto"] * len(df),
                "date": [f"{year}-01-01" for year in df["Year"]],
                "rsmi": df["mapped_rxn"],
                "classification": ["0.0 Unrecognized"] * len(df),
            }
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Prepare USPTO data")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--transformed_path")
    args = parser.parse_args()

    importer = UsptoImporter(args.data_path)
    if args.transformed_path:
        importer.data.to_csv(args.transformed_path, sep="\t", index=False)
