# Sensorium 2022 Challenge
Codebase for NeurIPS 2022 Sensorium challenge

## Quick URLs
- Sensorium 2022 webpage: [sensorium2022.net](https://sensorium2022.net/home)
- Starting kit: [github.com/sinzlab/sensorium](https://github.com/sinzlab/sensorium)
- Dataset: [gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022)
- Dataset whitepaper: [arxiv.org/abs/2206.08666](https://arxiv.org/abs/2206.08666)
- ICLR2021 paper (SOTA model) from the organizer: [openreview.net/forum?id=Tp7kI90Htd](https://openreview.net/forum?id=Tp7kI90Htd)

## File structure
The codebase repository should have the following structure. Check [.gitignore](.gitignore) for the ignored files.
```
sensorium2022/
    data/
        sensorium/
            static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
            static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
            ...
        franke2022/
        imagenet/
        README.md
    misc/
    src/
        sensorium/
            ...
    .gitignore
    LICENSE
    pyproject.toml
    README.md
    requirements.txt
    setup.py
    setup.sh
    ...
```
- [data](data/) store the datasets, please check [data/README.md](data/README.md) for more information.
- [misc](misc/) contains scripts and notebooks to generate various plots and figures used in the paper.
- [src/sensorium](src/v1t/) contains the code for the main Python package.

## Installation
- Create a new [conda](https://docs.conda.io/en/latest/miniconda.html) environment with Python 3.8.
  ```bash
  conda create -n sensorium python=3.8
  ```
- Activate `sensorium` virtual environment
  ```bash
  conda activate sensorium
  ```
- Install all dependencies and packages using `setup.sh` script.
  ```bash
  sh setup.sh
  ```

## Demo
- A demo to load the best model and run the Sensorium test set is available in [demo.ipynb](demo.ipynb) 

## Run model
- An example command to train a V1T core and Gaussian readout on the Sensorium dataset
  ```bash
  python train.py --data data/sensorium --output_dir runs/v1t_model --core vit --readout gaussian2d --ds_scale --behavior_mode 3 --epochs 400 --batch_size 16
  ```
- use the `--help` flag to see all available options

## Pull Requests (PRs)
- Always create a new branch to work on a/any features.
- When your branch is ready, create a PR on GitHub.
- Every PR should be reviewed and approved by another person.
- Once a PR has been approved, the submitter should select **`Squash and merge`** to merge the branch into `main` as a **single** commit.

## Reviewing PRs
Quality reviews are really important. You should spend time reviewing the code your peers write (not just fixing their mistakes without saying anything). If you can't review something because you don't understand what they're doing, there's something very wrong with their code, not you. Ask clarifying questions and suggest ways for them to make their code more interpretable. Request they put comments where comment are necessary.

If your code is being reviewed, don't be insulted or annoyed at requests to reformat/add comments. Err on the side of helping your colleagues understand your work. We should all have [Black](https://github.com/psf/black) installed:
- Install [Black](https://github.com/psf/black) `pip install black[jupyter]`, or, if you are using `zsh`: `pip install 'black[jupyter]'`
- Usage:
    - Command line:
      ```
      black <filename>
      ```
    - [PyCharm](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea)
    - [Visual Studio Code](https://black.readthedocs.io/en/stable/integrations/editors.html#visual-studio-code)
    - [SublimeText 3](https://black.readthedocs.io/en/stable/integrations/editors.html#sublimetext-3)