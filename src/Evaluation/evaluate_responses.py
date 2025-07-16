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
from prompts.judge_instructions import JUDGE_INSTRUCTIONS
from argparse import ArgumentParser
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
load_dotenv()
# get the openai api key from the .env file
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

assert OPENAI_API_KEY is not None, "OPENAI_API_KEY is not set in the .env file"
assert ANTHROPIC_API_KEY is not None, "ANTHROPIC_API_KEY is not set in the .env file"
assert XAI_API_KEY is not None, "XAI_API_KEY is not set in the .env file"



judge_models = [
    "gpt-4.1-2025-04-14",
    "claude-3-7-sonnet-20250219",
    # "grok-4-0709",
    "google/gemma-3-27b-it",
    # "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    # "mistralai/Mistral-7B-Instruct-v0.3",
]
debug_models = [
    "google/gemma-3-27b-it"
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


def load_csv(file_path: str, debug: bool = False):
    """
    Load prompts from a file.
    """
    df = pd.read_csv(file_path)

    data = df.to_dict(orient="records")

    if debug:
        # For debugging, limit the number of prompts to a small subset
        data = data[:5]

    return data


def add_judgment_prompt(responses, true_versions, false_versions, model_name):
    """
    Add a judgment instruction to give to the judge model in order to evaluate the responses.
    """
    judge_instruction = JUDGE_INSTRUCTIONS.get(model_name, None)
    if judge_instruction is None:
        raise ValueError(f"No judgment instruction found for model {model_name}.")

    if isinstance(responses, str):
        responses = [responses]

    return [
        judge_instruction.format(
            response=response, true_version=true_version, false_version=false_version
        )
        for response, true_version, false_version in zip(
            responses, true_versions, false_versions
        )
    ]


def post_process_judgment(judgment_text: str, model_name: str):
    """
    Post-process the judgment text to ensure it is in the correct format.
    Basically, we need to parse the judgment text
    """
    raise NotImplementedError(
        "Post-processing of judgment text is not implemented yet. "
        "You need to implement this function based on your specific requirements."
    )


def obtain_judgements(
    input_path: str, output_dir: str, tag: str, requested_models: str, debug: bool, batch_size:int 
):
    # launch llm-as-a-judge on the input file
    judjment_path = os.path.join(output_dir, f"responses_{tag}.csv")
    response_df = load_csv(input_path, debug=debug)

    judgment_df = OutputDataFrame(
        columns=[
            "id",
            "case_id",
            "Judge",  # This is the model that will judge the responses
            "Judgement",  # This is the judgement of the response
            "Model",  # This is the model that generated the response
            "Prompt",  # This is the prompt that was used to generate the response
            "Response",  # This is the response that was generated by the model
            "Historical Event",  # This is the historical event that the response is about
            "True Version",  # This is the true version of the historical event
            "False Version",  # This is the false version of the historical event
            "Country/Region",  # This is the country or region that the response is about
            "Source",  # This is the source of the response
            "Historical Period",  # This is the historical period that the response is about
            "Push Level",  # This is the push level of the response (Neutral, Implicit push, Explicit push)
            "Scenario",  # This is the scenario t
            "Dataset",
        ],
        path=judjment_path,
    )

    if requested_models == "all":
        models = judge_models
    elif requested_models == "debug":
        models = debug_models
    else:
        models = [MODEL_ALIASES.get(requested_models, requested_models)]

    # load checkpoint if output file exists
    processed_keys = set()
    if os.path.exists(judjment_path):
        try:
            with open(judjment_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_keys.add(
                        (
                            str(row.get("id")),
                            str(row.get("Model", row.get("model", ""))),
                            str(row.get("Judge", row.get("judge", ""))),
                        )
                    )
        except Exception as e:
            print(f"Warning: Could not load checkpoint file: {e}")
    print(f"Judging with models: {models}")

    for model_name in models:
        assert model_name is not None, f"Model {model_name} is not supported."
        print(f"Initializing model: {model_name}")
        for model in init_model(model_name):
            for i in progress(
                range(0, len(response_df), batch_size),
                description=f"Processing responses with {model_name}",
                total=len(response_df) // batch_size + 1,
            ):
                batch_index = [
                    j for j in range(i, min(i + batch_size, len(response_df)))
                ]
                batch_keys = [
                    (str(response_df[j].get("id", j)), model_name) for j in batch_index
                ]
                to_process = [
                    (j, key)
                    for j, key in zip(batch_index, batch_keys)
                    if key not in processed_keys
                ]
                if not to_process:
                    continue
                filtered_batch_responses = [
                    response_df[j]["Response"] for j, _ in to_process
                ]
                filtered_batch_true_versions = [
                    response_df[j]["True Version"] for j, _ in to_process
                ]
                filtered_batch_false_versions = [
                    response_df[j]["False Version"] for j, _ in to_process
                ]
                filtered_batch_responses = add_judgment_prompt(
                    filtered_batch_responses,
                    filtered_batch_true_versions,
                    filtered_batch_false_versions,
                    model_name,
                )
                try:
                    formatted_responses = [
                        model.append_with_chat_template(message=response, role="user")
                        for response in filtered_batch_responses
                    ]
                    judgments = model.batch_chat(formatted_responses)
                except Exception as e:
                    print(f"Error during inference for model {model_name}: {e}")
                    continue
                for idx, (j, key) in enumerate(to_process):
                    id_ = response_df[j]["id"]
                    case_id = response_df[j]["case_id"]
                    judge = model_name
                    try:
                        judgment_text = model.get_last_text_from_response(
                            judgments[idx]
                        )
                        judge_score, judge_explanation = post_process_judgment(
                            judgment_text, model_name
                        )
                    except Exception as e:
                        print(
                            f"Error extracting judgment for id={j}, model={model_name}: {e}"
                        )
                        continue
                    output_row = {
                        "id": id_,
                        "case_id": case_id,
                        "Judge": judge,
                        "Judgement": judge_score,
                        "Judgement Explanation": judge_explanation,
                        "Model": response_df[j]["Model"],
                        "Prompt": response_df[j]["Prompt"],
                        "Response": response_df[j]["Response"],
                        "Historical Event": response_df[j]["Historical Event"],
                        "True Version": response_df[j]["True Version"],
                        "False Version": response_df[j]["False Version"],
                        "Country/Region": response_df[j]["Country/Region"],
                        "Source": response_df[j]["Source"],
                        "Historical Period": response_df[j]["Historical Period"],
                        "Push Level": response_df[j]["Push Level"],
                        "Scenario": response_df[j]["Scenario"],
                        "Dataset": response_df[j]["Dataset"],
                    }
                    judgment_df.add_rows([output_row])
                    processed_keys.add(
                        (str(id_), str(response_df[j]["Model"]), str(judge))
                    )
    judgment_df.save()
    print(f"Judgment results saved to {judjment_path}")
    return judjment_path


def scoring_responses(
    judgements_path: str,
    output_dir: str,
    tag: str,
    debug: bool = False,
):
    score_path = os.path.join(output_dir, f"scores_{tag}.csv")
    if os.path.exists(score_path):
        print(f"Scores already computed. Loading from {score_path}")
        return pd.read_csv(score_path)
    print(f"Computing scores for judgments in {judgements_path}")
    judgment_df = pd.read_csv(judgements_path)
    if debug:
        print(
            "Debug mode is enabled. This will run only on a small subset of the data."
        )
        judgment_df = judgment_df.head(5)
    # Compute scores based on the judgments

    # Now we need to take, for each response (key by (id, model)) all the judgments
    # Then compute the score for that (id, model) pair, and finally average the scores for each model. The score are a number between 1 to 5. We also want to save the df with an additional column with the score for each response.
    score_df = OutputDataFrame(
        columns=[
            "id",
            "Model",
            "Majority Judgement", # This is the majority judgment for the response
            "PA", # This is the percentage of agreement for the response tot agrr over total judgments times 100
            "Average Judgement",  
            "Response",
            "Prompt",
            "Historical Event",
            "True Version",
            "False Version",
        ],
        path=score_path,
    )
    grouped = judgment_df.groupby(["id", "Model"])
    for (id_, model), group in progress(
        grouped,
        description="Computing scores",
        total=len(grouped),
    ):
        # For each group, we want to compute the score
        # The score is the average of the judgments
        judgments = group["Judgement"].tolist()
        if not judgments:
            print(f"No judgments found for id={id_}, model={model}. Skipping.")
            continue

        # Compute the majority judgment
        majority_judgment = max(set(judgments), key=judgments.count)
        # Compute the percentage of agreement
        total_judgments = len(judgments)
        agreement_count = judgments.count(majority_judgment)
        percentage_agreement = (agreement_count / total_judgments) * 100 if total_judgments > 0 else 0  
        # Compute the average judgment
        average_judgment = sum(judgments) / total_judgments if total_judgments > 0 else 0

        score_df.add_rows(
            [
                {
                    "id": id_,
                    "Model": model,
                    "Majority Judgement": majority_judgment,
                    "PA": percentage_agreement,
                    "Average Judgement": average_judgment,
                    "Response": group["Response"].iloc[0],
                    "Prompt": group["Prompt"].iloc[0],
                    "Historical Event": group["Historical Event"].iloc[0],
                    "True Version": group["True Version"].iloc[0],
                    "False Version": group["False Version"].iloc[0],
                }
            ]
        )
    score_df.save()
    
    print(f"Scores saved to {score_path}")
    return score_df
        



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
        default="data/manual_historical/responses/historical_processed_v1_evaluated_v1.csv",
        help="Path to the input CSV file containing prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/manual_historical/evaluation",
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
        "--judge_response_path",
        type=str,
        default=None,
        help="if provided, the script will only compute the evaluation metrics for the responses in this file.",
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.judge_response_path is None:
        judgements_path = obtain_judgements(
            input_path=args.input_path,
            output_dir=args.output_dir,
            tag=args.tag,
            requested_models=args.model,
            debug=args.debug,
            batch_size=args.batch_size,
        )
    else:
        judgements_path = args.judge_response_path
        if not os.path.exists(judgements_path):
            raise FileNotFoundError(f"Judgment file {judgements_path} does not exist.")

    scores = scoring_responses(
        judgements_path=judgements_path,
        output_dir=args.output_dir,
        tag=args.tag,
        debug=args.debug,
    )

    # launch metrics computatio
