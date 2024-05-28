import shutil
import sys
import os
from pathlib import Path

import requests
import tqdm


FILES_TO_DOWNLOAD = [
    {"url": "", "filename": "routes_for_eval.json"},
    {"url": "", "filename": "smiles_for_eval.txt"},
    {"url": "", "filename": "stock_for_eval_recov.txt"},
    {
        "url": "https://ndownloader.figshare.com/files/23086469",
        "filename": "stock_for_eval_find.hdf5",
    },
    {"url": "", "filename": "ref_routes_n1.json"},
    {"url": "", "filename": "ref_routes_n2.json"},
]


def _download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(
            total=total_size,
            desc=f"Downloading {os.path.basename(filename)}",
            unit="B",
            unit_scale=True,
        )
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


USPTO_CONFIG_PATH = Path(__file__).parent

FILES_TO_COPY = [
    "expansion_model_pipeline_config.yml",
    "ringbreaker_model_pipeline_config.yml",
    "template_pipeline_config.yml",
]

if __name__ == "__main__":
    project_path = Path(sys.argv[1])

    project_path.mkdir(exist_ok=True)

    for filename in FILES_TO_COPY:
        print(f"Copy {filename} to project folder")
        shutil.copy(USPTO_CONFIG_PATH / filename, project_path)

    for filespec in FILES_TO_DOWNLOAD:
        try:
            _download_file(filespec["url"], str(project_path / filespec["filename"]))
        except requests.HTTPError as err:
            print(f"Download failed with message {str(err)}")
            sys.exit(1)
