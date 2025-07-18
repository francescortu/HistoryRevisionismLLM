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
- `data`: Contains the dataset, the responses of the model and all the data used in the project. 
   - Each subfolder contains a specific dataset and source. For now we have:
        - `EUvsDISINFO`: Contains the dataset from the EUvsDisinfo source.
        - `manual_historical`: Contains the dataset from the manual selection
    - Each dataset folder contains:
        - `input`: Contains the input data for the dataset.
        - `processed`: Contains the dataset after processing. Basically, it contins the prompt with the different scenarios ready to be used to evaluate LLMs.
        - `responses`: Contains the responses of the LLMs to the prompts in the processed folder.
        - `judgements`: Contains the LLMs as a judgement scores for the responses in the responses folder.
    - Each file could be saved in multiple version using the --tag argument in almost of the scripts. In this way we can keep track of parallel experiments and different versions of the same dataset.


- `src`: Contains the source code of the project, and the main scripts to run the different experiments and part of the pipeline.

- `prompts`: Contains the prompt templates used for working with the LLMs in the 
different stages of the project.
    - `scenarios.py`: Contains the instruction to generate evaluation prompts from raw data.
    - `judge_instructions.py`: Contains the instructions for the LLM-as-a-judge to evaluate the responses of the LLMs.

- `easyroutine`: This is a submodule that contains a library used in the project. It is a Francesco's library and it should work out of the box. However, it is used as a submodule to give the possibility to update it in the future if needed in a easy way. It basically used just to handle the inference of the models used in a easier way.

## Launch the Experiments
The main scripts to run the different experiments are in the `src` folder.
**For running these scripts, you need to either have set the .env file with the required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY) and/or have enough GPUs available to run the experiments.**


### From raw data to processed dataset
To generate the processed dataset (prompts) from the raw data, you can use the `generate_prompts.py` script.
```bash
poetry run python src/DataProcessing/generate_prompts.py \
    --model_name <model_name> \ # for now the prompts are optimized for google/gemma-3-27b-it and seems work well. For other models you may need to adjust the prompts.
    --dataset_type <euvsdisinfo, historical> \ # dataset type to process - different datasets have different processing. The loading of the files is done automatically. To add a new dataset, you need to add a new case in the `generate_prompts.py` script. For our select dataset, we can use a similar pipeline of the historical dataset.
    --output_dir <path_to_output_dir> \  # /data/<dataset_name>/processed
    --tag <tag> \ # tag to use for the output files. It is used to keep track of the different versions of the dataset.
    --debug \ # if you want to run the script in debug mode, it will use a smaller subset of the data and a smaller batch size. It is useful for testing the script and debugging.
    --batch_size <batch_size> # batch size to use for the processing. Default
    --push_level no_push, implicit_push, explicit_push \ # apply push level to the question (i.e. implicit or explicit revisionism in the question). Default is no_push. You can use multiple push levels separated by commas. WARNING: each push level will generate a new set of prompts of size {lenght dataset} * {number of scenarios - 11}. So keep this in mind if you are using openai API, as it will generate a lot of requests and you will need to have enough credits to cover the cost.
```
It will generate the processed dataset in the `data/<dataset_name>/processed` folder, with the prompts ready to be used for evaluation.

### From processed dataset to LLM responses
For generating the responses of the LLMs to the prompts in the processed dataset, you can use the `generate_responses.py` script.
```bash
poetry run python src/Evaluation/get_response.py \
    --model <model to evaluate, all, debug> \ # model to use for generating the responses. If you want to evaluate all the models, you can use `all`. If you want to use a specific model, you can use its name. If you want to use the debug models, you can use `debug`.
    --debug \ # if you want to run the script in debug mode, it will use a smaller subset of the data and a smaller batch size. It is useful for testing the script and debugging.
    --tag <tag> \ # tag to use for the output files. It is used to keep track of the different versions of the dataset.
    --batch_size <batch_size> \ # batch size to use for the processing. Default is 8.
    --input_path <path_to_input_file> \ # path to the input file with the prompts. It should be in the `data/<dataset_name>/processed` folder.
    --output_dir <path_to_output_dir> \ # path to the output folder where the responses will be saved. It should be in the `data/<dataset_name>/responses` folder.
```
It will generate the responses of the LLMs to the prompts in the processed dataset, and save them in the `data/<dataset_name>/responses` folder, with the responses ready to be evaluated. 

### From LLM responses to LLM-as-a-judge judgements
For generating the judgements of the LLM-as-a-judge to the responses of the LLMs, you can use the `evaluate_responses.py` script.
```bash
poetry run python src/Evaluation/evaluate_responses.py \
    --model <model to use for the judgement, all, debug> \ # model to use for generating the judgements. If you want to evaluate all the models, you can use `all`. If you want to use a specific model, you can use its name. If you want to use the debug models, you can use `debug`. The idea is to use multiple judges to evaluate the responses of the LLMs.
    --debug \ # if you want to run the script in debug mode, it will use a smaller subset of the data and a smaller batch size. It is useful for testing the script and debugging.
    --tag <tag> \ # tag to use for the output files. It is used to keep track of the different versions of the dataset.
    --batch_size <batch_size> \ # batch size to use for the processing. Default is 8.
    --input_path <path_to_input_file> \ # path to the input file with the responses. It should be in the `data/<dataset_name>/responses` folder.
    --output_dir <path_to_output_dir> \ # path to the output folder where the judgements will be saved. It should be in the `data/<dataset_name>/judgements` folder.
    --judge_response_path <path_to_judge_response_file> \ # OPTIONAL. path to the file with the responses of the LLMs to the prompts. It should be in the `data/<dataset_name>/responses` folder. If it is passed, the script will skip the generation of the judgements and will use the responses in the file. This is useful if you want to use the judgments generated by a previous run of the script and just compute the aggreement scores.
```

## TODO
- [ ] Add the selected dataset to the pipeline
- [ ] Add the support for API LLMs for the generation of the prompts based on the scenarios.


