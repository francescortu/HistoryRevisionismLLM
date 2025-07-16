import sys
import os
import csv

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../src",
        )
    )
)
from dotenv import load_dotenv
from easyroutine.inference import (
    LiteLLMInferenceModel,
    LiteLLMInferenceModelConfig,
    VLLMInferenceModel,
    VLLMInferenceModelConfig,
)
from easyroutine.console import progress
import pandas as pd
from src.utils import OutputDataFrame
from argparse import ArgumentParser
from datetime import datetime

load_dotenv()
# get the openai api key from the .env file
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

assert OPENAI_API_KEY is not None, "OPENAI_API_KEY is not set in the .env file"
assert ANTHROPIC_API_KEY is not None, "ANTHROPIC_API_KEY is not set in the .env file"
assert XAI_API_KEY is not None, "XAI_API_KEY is not set in the .env file"

debug_models = [
    "gpt-4.1-nano-2025-04-14",
    "claude-3-5-haiku-20241022",
    "grok-3-mini",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-3-27b-it",
]

evaluation_models = [
    "gpt-4.1-2025-04-14",
    "claude-3-7-sonnet-20250219",
    "grok-4-0709",
    "google/gemma-3-27b-it",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

MODEL_ALIASES = {
    "gpt": "gpt-4.1-2025-04-14",
    "gpt-nano": "gpt-4.1-nano-2025-04-14",
    "claude": "claude-3-7-sonnet-20250219",
    "claude-haiku": "claude-3-5-haiku-20241022",
    "grok": "grok-4-0709",
    "grok-mini": "grok-3-mini",
    "gemma-27b": "google/gemma-3-27b-it",
    "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "deepseek-qwen-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


MODEL_CONFIG = {
    "gpt-4.1-2025-04-14": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="gpt-4.1-2025-04-14",
            openai_api_key=OPENAI_API_KEY,
        ),
    },
    "gpt-4.1-nano-2025-04-14": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="gpt-4.1-nano-2025-04-14",
            openai_api_key=OPENAI_API_KEY,
        ),
    },
    "claude-3-7-sonnet-20250219": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="claude-3-7-sonnet-20250219",
            anthropic_api_key=ANTHROPIC_API_KEY,
        ),
    },
    "claude-3-5-haiku-20241022": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="claude-3-5-haiku-20241022",
            anthropic_api_key=ANTHROPIC_API_KEY,
        ),
    },
    "grok-4-0709": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="grok-4-0709",
            xai_api_key=XAI_API_KEY,
        ),
    },
    "grok-3-mini": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="grok-3-mini",
            xai_api_key=XAI_API_KEY,
        ),
    },
    "google/gemma-3-27b-it": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="google/gemma-3-27b-it", n_gpus=2
        ),
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct", n_gpus=4
        ),
    },
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", n_gpus=2
        ),
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.3", n_gpus=2
        ),
    },
}


def init_model(model_name: str):
    """
    Initialize the model based on the model name.
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model {model_name} is not supported.")

    model_info = MODEL_CONFIG[model_name]
    model_interface = model_info["interface"]
    model_config = model_info["config"]

    return model_interface(model_config)


def load_prompts(file_path: str, debug: bool = False):
    """
    Load prompts from a file.
    """
    df = pd.read_csv(file_path)

    data = df.to_dict(orient="records")

    if debug:
        # For debugging, limit the number of prompts to a small subset
        data = data[:5]

    return data


def main():
    parser = ArgumentParser(
        description="Evaluate models on historical revisionism prompts."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to evaluate. Choose from: gpt, gpt-nano, claude, claude-haiku, grok, grok-mini, gemma-27b, llama4-maverick, deepseek-qwen-8b, mistral-7b, all, debug",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Load a small subset of the data for debugging purposes.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/manual_historical/processed/historical_processed_v1.csv",
        help="Path to the input CSV file containing prompts.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/manual_historical/responses/historical_processed_v1_evaluated",
        help="Path to the output CSV file where results will be saved.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing prompts."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Tag for the output files.",
    )
    # assert input_path exists
    args = parser.parse_args()
    if args.debug:
        print(
            "Debug mode is enabled. This will run only on a small subset of the data."
        )
        args.tag = "debug"
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file {args.input_path} does not exist.")

    # create output directory if it does not exist
    output_path = args.output_path + f"_{args.tag}.csv"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load input file
    prompts = load_prompts(args.input_path, debug=args.debug)

    output_df = OutputDataFrame(
        columns=[
            "id",
            "case_id",
            "Model",
            "Prompt",
            "Response",
            "Historical Event",
            "True Version",
            "False Version",
            "Country/Region",
            "Source",
            "Historical Period",
            "Push Level",
            "Scenario",
            "Dataset",
        ],
        path=output_path,
    )

    # create model list
    if args.model == "all":
        models = evaluation_models
    elif args.model == "debug":
        models = debug_models
    else:
        models = [MODEL_ALIASES.get(args.model, args.model)]

    # Load checkpoint if output file exists
    processed_keys = set()
    output_path = args.output_path + f"_{args.tag}.csv"
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_keys.add(
                        (
                            str(row.get("id")),
                            str(row.get("Model", row.get("model", ""))),
                        )
                    )
        except Exception as e:
            print(f"Warning: Could not load checkpoint file: {e}")

    print(f"Evaluating with models: {models}")
    for model_name in models:
        assert model_name is not None, f"Model {model_name} is not supported."
        print(f"Initializing model: {model_name}")
        model = init_model(model_name)
        for i in progress(
            range(0, len(prompts), args.batch_size),
            description=f"Processing prompts with {model_name}",
            total=len(prompts) // args.batch_size + 1,
        ):
            batch_index = [j for j in range(i, min(i + args.batch_size, len(prompts)))]
            batch_keys = [
                (str(prompts[j].get("id", j)), model_name) for j in batch_index
            ]
            to_process = [
                (j, key)
                for j, key in zip(batch_index, batch_keys)
                if key not in processed_keys
            ]
            if not to_process:
                continue
            filtered_batch_prompts = [prompts[j]["Prompt"] for j, _ in to_process]
            try:
                formatted_prompts = [
                    model.append_with_chat_template(message=prompt, role="user")
                    for prompt in filtered_batch_prompts
                ]
                responses = model.batch_chat(formatted_prompts)
            except Exception as e:
                print(f"Error during inference for model {model_name}: {e}")
                continue
            for idx, (j, key) in enumerate(to_process):
                id_ = prompts[j]["id"]
                case_id = prompts[j]["case_id"]
                historical_event = prompts[j]["Historical Event"]
                true_version = prompts[j]["True Version"]
                false_version = prompts[j]["False Version"]
                country_region = prompts[j]["Country/Region"]
                source = prompts[j]["Source"]
                historical_period = prompts[j]["Historical Period"]
                push_level = prompts[j]["Push Level"]
                scenario = prompts[j]["Scenario"]
                dataset = prompts[j]["Dataset"]
                try:
                    response_text = model.get_last_text_from_response(responses[idx])
                except Exception as e:
                    print(
                        f"Error extracting response for id={j}, model={model_name}: {e}"
                    )
                    continue
                output_df.add_rows(
                    [
                        {
                            "id": id_,
                            "case_id": case_id,
                            "Model": model_name,
                            "Prompt": filtered_batch_prompts[idx],
                            "Response": response_text,
                            "Historical Event": historical_event,
                            "True Version": true_version,
                            "False Version": false_version,
                            "Country/Region": country_region,
                            "Source": source,
                            "Historical Period": historical_period,
                            "Push Level": push_level,
                            "Scenario": scenario,
                            "Dataset": dataset,
                        }
                    ]
                )
                processed_keys.add(key)


if __name__ == "__main__":
    main()
