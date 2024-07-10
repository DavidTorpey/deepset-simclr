# DeepSet SimCLR

This repository contains an implementation of the [DeepSet SimCLR](https://arxiv.org/abs/2402.15598)
model.

## Setup

### Dependencies

Create a virtualenv and install the dependencies:

```bash
python3.10 -m virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

### Dataset

This repository contains a dummy dataset, and should be replaced prior to running
the code. Simply add a new file for your dataset to the `src/data/` folder, and
add it as an additional option in the `src/data/data.py` file.

Ideally, the dataset should contain separate folders for each scan, where each
folder contains a multiple images, each representing a separate slice of that scan.
The code expects the file names of these images to have the following pattern:

```text
<str>-<int>.jpg
```

The `<int>` should be an integer representing the index of that slice in the scan.
This is used when sampling in order to sort the slices in the correct order.

## Training the Model

### Configuration File

A configuration file is provided when training the model. An example of such a
configuration file can be found at `configs/example.yaml`. Change the `data.dataset`
configuration property to the name of your dataset that you added to the
`src/data/data.py` file from above.

Further, if you enable `general.log_to_wandb`, ensure you specify the correct
project name when wandb in initialised in the `src/trainer.py` file.

### Execution

To train the model, run:

```bash
python -m src.train --config_path configs/example.yaml
```
