import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src", )))
from dotenv import load_dotenv
from easyroutine.inference import  LiteLLMInferenceModel, LiteLLMInferenceModelConfig, VLLMInferenceModel, VLLMInferenceModelConfig
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
        )
    },
    "gpt-4.1-nano-2025-04-14": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="gpt-4.1-nano-2025-04-14",
            openai_api_key=OPENAI_API_KEY,
        )
    },
    "claude-3-7-sonnet-20250219": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="claude-3-7-sonnet-20250219",
            anthropic_api_key=ANTHROPIC_API_KEY,
        )
    },
    "claude-3-5-haiku-20241022": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="claude-3-5-haiku-20241022",
            anthropic_api_key=ANTHROPIC_API_KEY,
        )
    },
    "grok-4-0709": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="grok-4-0709",
            xai_api_key=XAI_API_KEY,
        )
    },
    "grok-3-mini": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="grok-3-mini",
            xai_api_key=XAI_API_KEY, 
        )
    },
    "google/gemma-3-27b-it": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="google/gemma-3-27b-it",
            n_gpus=2
        ),
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            n_gpus=4
        ),
    },
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            n_gpus=2
        ),
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "interface": VLLMInferenceModel,
        "config": VLLMInferenceModelConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            n_gpus=2    
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
    
    parser = ArgumentParser(description="Evaluate models on historical revisionism prompts.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to evaluate. Choose from: gpt, gpt-nano, claude, claude-haiku, grok, grok-mini, gemma-27b, llama4-maverick, deepseek-qwen-8b, mistral-7b, all, debug"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Load a small subset of the data for debugging purposes."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/manual_historical/processed/historical_processed_v1.csv",
        help="Path to the input CSV file containing prompts."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/manual_historical/responses/historical_processed_v1_evaluated",
        help="Path to the output CSV file where results will be saved."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing prompts."
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
            
            
            # take the batch of prompts (column "Prompt")
            batch_prompts = [prompts[j]["Prompt"] for j in batch_index]
            formatted_prompts = [model.append_with_chat_template(message=prompt, role="user") for prompt in batch_prompts]
            
            responses = model.batch_chat(formatted_prompts)
            
            for j, response in enumerate(responses):
                case_id = prompts[batch_index[j]]["case_id"]
                historical_event = prompts[batch_index[j]]["Historical Event"]
                true_version = prompts[batch_index[j]]["True Version"]
                false_version = prompts[batch_index[j]]["False Version"]
                country_region = prompts[batch_index[j]]["Country/Region"]
                source = prompts[batch_index[j]]["Source"]
                historical_period = prompts[batch_index[j]]["Historical Period"]
                push_level = prompts[batch_index[j]]["Push Level"]
                scenario = prompts[batch_index[j]]["Scenario"]
                dataset = prompts[batch_index[j]]["Dataset"]

                output_df.add_rows([{
                    "id": output_df.current_index,
                    "case_id": case_id,
                    "Prompt": batch_prompts[j],
                    "Response": model.get_last_text_from_response(response),
                    "Historical Event": historical_event,
                    "True Version": true_version,
                    "False Version": false_version,
                    "Country/Region": country_region,
                    "Source": source,
                    "Historical Period": historical_period,
                    "Push Level": push_level,
                    "Scenario": scenario,
                    "Dataset": dataset
                }])
            
        
        
if __name__ == "__main__":
    main()