# aizynthtrain

aizynthtrain is a collection of routines, configurations and pipelines for training synthesis prediction models for the AiZynthFinder software.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Linux, Windows or macOS platforms are supported - as long as the dependencies are supported on these platforms.

* You have installed [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.9 - 3.10

The tool has been developed on a Linux platform.

## Installation

First clone the repository using Git.

Then execute the following commands in the root of the repository 

    conda env create -f env-dev.yml
    conda activate aizynthtrain
    poetry install
    
the `aizynthtrain` package is now installed in editable mode.

Now, you need to create a Jupyter Kernel for this environment

    conda activate aizynthtrain
    ipython kernel install --name "aizynthtrain" --user

## Usage

There is a number of example configurations and SLURM scripts in the `configs/uspto` folder that
was used to retrain the USPTO-based expansion model for AiZynthFinder.

The example configurations can be used as-is to reproduce the modelling, whereas the SLURM scripts
migt have to be adjusted to your system.

To use the given configurations, follow this procedure:

For the training and validation of the expansion models, you need a number of data files that can be downloaded from Zenodo. 
In addition the pipelines create a lof of artifcats and therefore, you will start with setting up a folder for your modelling

1. Create a new folder where the pipelines will be executed using

    python configs/uspto/setup_folder.py PATH_TO_YOUR_FOLDER

2. cd PATH_TO_YOUR_FOLDER

3. Perform the pipelines from the `rxnutils` package to download and prepare the USPTO datasets for modelling: https://molecularai.github.io/reaction_utils/uspto.html

The `rxnutils` package pipelines should have produced a file `uspto_data_mapped.csv` that will be used as the starting point of the template-extraction pipeline

4. Adapt and execute the `configs/uspto/template_pipeline.sh` SLURM script on your machine.

The template-extraction pipeline will produce a number of artifacts, the most important being
- `reaction_selection_report.html` that is the report of the reaction selection
- `template_selection_report.html` that is the report of the template selection
- `uspto_template_library.csv` and `uspto_ringbreaker_template_library.csv` that are the final reactions and reaction templates that will be used to train the expansion model.

5. Adapt and execute the `configs/uspto/expansion_pipeline.sh` SLURM script on your machine.

The expansion model pipeline will produce these important artifacts

- `uspto_expansion_model_report.html` that is the report of the model training and validation
- `uspto_keras_model.hdf5` the trained Keras model
- `uspto_unique_templates.csv.gz` the template library for AiZynthFinder

You can also execute the `configs/uspto/ringbreaker_pipeline.sh` to train a RingBreaker model.

### Testing

Tests uses the ``pytest`` package, and is installed by `poetry`

Run the tests using:

    pytest -v


## Contributing

We welcome contributions, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.


To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

Please use ``black`` package for formatting, and follow ``pep8`` style guide.


## Contributors

* [@SGenheden](https://www.github.com/SGenheden)

The contributors have limited time for support questions, but please do not hesitate to submit an issue (see above).

## License

The software is licensed under the Appache 2.0 license (see LICENSE file), and is free and provided as-is.

## References

1. Genheden S, Norrby PO, Engkvist O (2022) AiZynthTrain: robust, reproducible, and extensible pipelines for training synthesis prediction models. ChemRxiv. Prerint. https://doi.org/10.26434/chemrxiv-2022-kls5q
2. Kannas C, Thakkar A, Bjerrum E, Genheden S (2022) rxnutils – A Cheminformatics Python Library for Manipulating Chemical Reaction Data. ChemRxiv. https://doi.org/10.26434/chemrxiv-2022-wt440-v2
3. Genheden S, Thakkar A, Chadimova V, et al (2020) AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. J. Cheminf. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1
4. Thakkar A, Kogej T, Reymond J-L, et al (2019) Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain. Chem Sci. https://doi.org/10.1039/C9SC04944D
