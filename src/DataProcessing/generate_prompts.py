from easyroutine.inference import VLLMInferenceModel, VLLMInferenceModelConfig
import os
import sys

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from argparse import ArgumentParser
import pandas as pd
from easyroutine.console import progress
from datetime import datetime
from prompts.scenarios import get_output_specs, build_meta_prompt
from src.utils import OutputDataFrame

def init_model(
    model_name: str, n_gpus: int = 1, dtype: str = "bfloat16"
) -> VLLMInferenceModel:
    """
    Initialize the VLLM inference model with the given configuration.

    Arguments:
        model_name (str): Name of the model to initialize.
        n_gpus (int): Number of GPUs to use.
        dtype (str): Data type for the model.

    Returns:
        VLLMInferenceModel: An instance of the VLLM inference model.
    """
    config = VLLMInferenceModelConfig(model_name=model_name, n_gpus=n_gpus, dtype=dtype)
    return VLLMInferenceModel(config)


def load_euvsdisinfo(debug: bool = False) -> list:
    import json

    if debug:
        # Load a small subset of the data for debugging purposes
        print("Loading a small subset of the EUvsDisinfo dataset for debugging.")
        return json.load(
            open(
                "data/EUvsDISINFO/input/euvsdisinfo_historical_revisionism_cases_cleaned.json",
                "r",
            )
        )[:10]
    return json.load(
        open(
            "data/EUvsDISINFO/input/euvsdisinfo_historical_revisionism_cases_cleaned.json",
            "r",
        )
    )


def load_historical_dataset(file_path: str, debug: bool = False) -> list:
    """
    Load the additional historical revisionism dataset.

    Arguments:
        file_path (str): Path to the dataset file (CSV format).
        debug (bool): Whether to load a small subset of the data for debugging.

    Returns:
        list: A list of dictionaries containing the historical dataset.
    """
    df = pd.read_csv(file_path)

    # Convert DataFrame to list of dictionaries
    data = df.to_dict("records")

    if debug:
        print("Loading a small subset of the historical dataset for debugging.")
        return data[:10]

    return data




def process_case(
    case_id: int,
    case: dict,
    model: VLLMInferenceModel,
    df: OutputDataFrame,
    dataset_type: str = "euvsdisinfo",
    push_level: list = ["no_push", "implicit_push", "explicit_push"]
) -> None:
    """
    Process a single historical revisionism case using the provided model and DataFrame.

    Arguments:
        case_id (int): The ID of the case.
        case (dict): A single case from either dataset.
        model (VLLMInferenceModel): The inference model to use for processing.
        df (OutputDataFrame): The DataFrame to which results will be appended.
        dataset_type (str): The type of dataset ('euvsdisinfo' or 'historical').
    """
    # Check if we have the required country/region information
    if dataset_type == "euvsdisinfo":
        if "countries" not in case or not case["countries"]:
            print(f"Skipping case {case_id} due to missing countries.")
            return
        countries = ", ".join(case["countries"])
        historical_event = case["title"]
        true_version = case["response"]
        false_version = case["summary"]
        source = case.get("url", "")
        time_period = ""  # EUvsDisinfo doesn't have this field
    else:  # historical dataset
        if "Country/Region" not in case or not case["Country/Region"]:
            print(f"Skipping case {case_id} due to missing countries.")
            return
        countries = case["Country/Region"]
        historical_event = case["Historical Event"]
        true_version = case["True Version (Detailed)"]
        false_version = case["False Version (Regime's Narrative)"]
        source = ""  # Historical dataset doesn't have source field
        time_period = case.get("Historical Period", "")

    try:
        scenarios = get_output_specs()
        batched_request = []
        for pl in push_level:
            for scenario, output_spec in scenarios.items():
                meta_prompt = build_meta_prompt(
                    push_level=pl,
                    scenario=scenario,
                    output_spec=output_spec,
                    event_title=historical_event,
                    ground_truth_summary=true_version,
                    revisionist_summary=false_version,
                    countries=countries,
                    time_period=time_period,
                )
                chat_messages = model.append_with_chat_template(
                    message=meta_prompt, role="user"
                )
                batched_request.append(
                    {
                        "chat_messages": chat_messages,
                        "push_level": pl,
                        "scenario": scenario,
                    }
                )

        dataframe_rows = []

        if not batched_request:
            return

        responses = model.chat(
            [x["chat_messages"] for x in batched_request], use_tqdm=True
        )
        for response, request in zip(responses, batched_request):
            # remove ``` at the beginning and end of the response text if present
            response_text = response.outputs[0].text.strip()
            if response_text.startswith("```"):
                response_text = response_text[3:].strip()
            if response_text.endswith("```\n```"):
                response_text = response_text[:-3].strip()
            dataframe_rows.append(
                {
                    "case_id": case_id,
                    "Historical Event": historical_event,
                    "True Version": true_version,
                    "False Version": false_version,
                    "Country/Region": countries,
                    "Source": source,
                    "Historical Period": time_period,
                    "Push Level": request["push_level"],
                    "Scenario": request["scenario"],
                    "Prompt": response.outputs[0].text,
                    "Dataset": dataset_type,
                }
            )
        df.add_rows(dataframe_rows)
    except Exception as e:
        print(f"Error processing case {case_id}: {e}")
        return


def main():
    parser = ArgumentParser(description="Initialize and run the VLLM inference model.")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to initialize.",
        default="google/gemma-3-27b-it",
    )
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--tag",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Tag for the output files.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["euvsdisinfo", "historical", "both"],
        default="historical",
        help="Dataset type to process: 'euvsdisinfo', 'historical', or 'both'.",
    )
    parser.add_argument(
        "--historical_dataset_path",
        type=str,
        default="data/manual_historical/input/data_18072025.csv",
        help="Path to the historical dataset CSV file.",
    )
    parser.add_argument(
        "--push_level",
        type=str,
        choices=["no_push", "implicit_push", "explicit_push"],
        default="no_push",
        help="Push level to use for processing.",
    )

    args = parser.parse_args()

    if args.debug:
        print(
            "Debug mode is enabled. This will run only on a small subset of the data."
        )
        args.tag = "debug"

    model = init_model(model_name=args.model_name, n_gpus=args.n_gpus)
    print(f"Model {args.model_name} initialized with {args.n_gpus} GPUs.")
    
    if args.dataset_type == "euvsdisinfo":
        output_path = f"data/EUvsDISINFO/processed/euvsdisinfo_processed_{args.tag}.csv"
    elif args.dataset_type == "historical":
        output_path = f"data/manual_historical/processed/{args.historical_dataset_path.split('/')[-1].split('.csv')[0]}_{args.tag}.csv"
    else:  # both datasets
        output_path = f"data/processed/revisionism_processed_{args.tag}.csv"
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
    df = OutputDataFrame(
        columns=[
            "case_id",
            "Historical Event",
            "True Version",
            "False Version",
            "Country/Region",
            "Source",
            "Historical Period",
            "Push Level",
            "Scenario",
            "Prompt",
            "Dataset",
        ],
        path=output_path,
        add_id=True 
    )
    
    #push level to list
    if args.push_level == "no_push":
        args.push_level = ["no_push"]
    elif args.push_level == "implicit_push":
        args.push_level = ["implicit_push"]
    elif args.push_level == "explicit_push":
        args.push_level = ["explicit_push"]
    else:
        args.push_level = ["no_push", "implicit_push", "explicit_push"]

    # Process EUvsDisinfo dataset
    if args.dataset_type in ["euvsdisinfo", "both"]:
        euvsdisinfo = load_euvsdisinfo(debug=args.debug)
        print(f"Loaded {len(euvsdisinfo)} EUvsDisinfo cases.")

        for case_id, case in progress(
            enumerate(euvsdisinfo),
            description="Processing EUvsDisinfo cases",
            total=len(euvsdisinfo),
        ):
            process_case(case_id, case, model, df, dataset_type="euvsdisinfo", push_level=args.push_level)

    # Process historical dataset
    if args.dataset_type in ["historical", "both"]:
        try:
            historical_data = load_historical_dataset(
                args.historical_dataset_path, debug=args.debug
            )
            print(f"Loaded {len(historical_data)} historical dataset cases.")

            for case_id, case in progress(
                enumerate(historical_data),
                description="Processing historical dataset cases",
                total=len(historical_data),
            ):
                process_case(case_id, case, model, df, dataset_type="historical", push_level=args.push_level)
        except FileNotFoundError:
            print(
                f"Historical dataset file not found at {args.historical_dataset_path}"
            )
            if args.dataset_type == "historical":
                print("No datasets were processed.")
                return

if __name__ == "__main__":
    main()