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
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

only_local_models = [
    # "google/gemma-3-27b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

MODEL_ALIASES = {
    "gpt": "gpt-4.1-2025-04-14",
    "gpt-nano": "gpt-4.1-nano-2025-04-14",
    "claude": "claude-3-7-sonnet-20250219",
    "claude-haiku": "claude-3-5-haiku-20241022",
    "grok": "grok-4-0709",
    "grok-mini": "grok-3-mini",
    "gemma-27b": "google/gemma-3-27b-it",
    "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "qwen-32b": "Qwen/Qwen3-32B",
    "deepseek-qwen-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseel-qewen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
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
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            n_gpus=8,
            gpu_memory_utilization=0.9,  # Adjusted for 8 GPUs
            max_model_len=5000,
            # must be a divisor of 40 and 1408 (num of attn heads)
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
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct", n_gpus=4
        ),
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", n_gpus=2
        ),
    },
    "Qwen/Qwen3-32B": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(model_name="Qwen/Qwen3-32B", n_gpus=2),
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
        help="Model name to evaluate. Choose from: gpt, gpt-nano, claude, claude-haiku, grok, grok-mini, gemma-27b, llama4-maverick, deepseek-qwen-8b, mistral-7b, all, debug, local",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Load a small subset of the data for debugging purposes.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/manual_historical/processed/data_18072025_v1.csv",
        help="Path to the input CSV file containing prompts.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file instead of treating it as checkpoint.",
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
    if args.output_path is None:
        output_path = (
            f"data/manual_historical/responses/data_responses_18072025_{args.tag}.csv"
        )
    else:
        if not args.output_path.endswith(".csv"):
            raise ValueError("Output path must end with .csv")
        output_path = args.output_path.rstrip(".csv") + f"_{args.tag}.csv"

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load input file
    prompts = load_prompts(args.input_path, debug=args.debug)

    # Initialize OutputDataFrame with existing data if not overwriting
    if not args.overwrite and os.path.exists(output_path):
        print(f"Loading existing output file: {output_path}")
        existing_df = pd.read_csv(output_path)
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
        # Load existing data into the OutputDataFrame
        output_df.df = existing_df
        print(f"Loaded {len(existing_df)} existing entries from checkpoint")
    else:
        if args.overwrite and os.path.exists(output_path):
            print(f"Overwriting existing output file: {output_path}")
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
    elif args.model == "local":
        models = only_local_models
    else:
        models = [MODEL_ALIASES.get(args.model, args.model)]

    print(f"Evaluating with models: {models}")
    for model_name in models:
        # Load checkpoint for each model specifically (only if not overwriting)
        processed_keys = set()
        if not args.overwrite and os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Only add to processed_keys if it matches the current model
                        if str(row.get("Model", row.get("model", ""))) == model_name:
                            processed_keys.add(
                                (
                                    str(row.get("id")),
                                    str(row.get("Model", row.get("model", ""))),
                                )
                            )
                print(
                    f"Found {len(processed_keys)} already processed entries for model {model_name}"
                )
            except Exception as e:
                print(f"Warning: Could not load checkpoint file: {e}")
        elif args.overwrite:
            print(f"Overwrite mode: not loading checkpoint for {model_name}")

        assert model_name is not None, f"Model {model_name} is not supported."
        print(f"Evaluating model: {model_name}")
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
