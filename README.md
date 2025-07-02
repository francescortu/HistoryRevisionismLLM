# History Revisionism

## Setup
The project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.
Make sure you have Poetry installed. If not, you can install it by following the instructions on
their [official documentation](https://python-poetry.org/docs/#installation).

```bash
git clone <this repository>
cd <this repository>
git submodule update --init --recursive
poetry install
```


## Description of the Repository
- `data`: Contains the data files used in the project. This includes the raw data, processed data, and any other relevant files.
    - `EUvsDISINFO`: Contains the data files for the EUvsDISINFO source
        - `input`: Contains the raw data files for the EUvsDISINFO source
        - `processed`: Contains the generated prompts for the EUvsDISINFO source
    - `manual_historical`: Contains the data files for the manual historical source
        - `input`: Contains the raw data files for the manual historical source
        - `processed`: Contains the generated prompts for the manual historical source
- `prompts`: Contains the prompt templates used for working with the LLMs in the different stages of the project.
    - `user_cases.py`: Contains the instruction to generate evaluation prompts from raw data.

