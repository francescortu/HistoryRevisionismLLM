from easyroutine.inference import VLLMInferenceModel, VLLMInferenceModelConfig
import os
import sys

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from argparse import ArgumentParser
import pandas as pd
from easyroutine.console import progress
from datetime import datetime
from prompts.user_cases import get_scenario_descriptions, build_meta_prompt

def init_model(model_name: str, n_gpus: int = 1, dtype: str = 'bfloat16') -> VLLMInferenceModel:
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


def load_euvsdisinfo(debug:bool = False) -> list:
    import json
    import os
    if debug:
        # Load a small subset of the data for debugging purposes
        print("Loading a small subset of the EUvsDisinfo dataset for debugging.")
        return json.load(open("data/EUvsDISINFO/input/euvsdisinfo_historical_revisionism_cases_cleaned.json", "r"))[:10]
    return json.load(open("data/EUvsDISINFO/input/euvsdisinfo_historical_revisionism_cases_cleaned.json", "r"))

class OutputDataFrame:
    def __init__(self, columns: list, path: str = f"data/EUvsDISINFO/processed/euvsdisinfo_processed.csv"):
        self.columns = columns
        self.path = path
        self.df = pd.DataFrame(columns=self.columns)
        
        
    def add_rows(self, rows: list):
        """
        Add multiple rows to the DataFrame.
        
        Arguments:
            rows (list): A list of dictionaries, each representing a row of data.
        """
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        self.save()

    
    def save(self):
        """
        Save the DataFrame to a CSV file.
        """
        self.df.to_csv(self.path, index=False)
        
            

        

def process_case(id:int, case: dict, model: VLLMInferenceModel, df: OutputDataFrame) -> None:
    """
    Process a single EUvsDisinfo case using the provided model and DataFrame.
    
    Arguments:
        case (dict): A single case from the EUvsDisinfo dataset.
        model (VLLMInferenceModel): The inference model to use for processing.
        df (pd.DataFrame): The DataFrame to which results will be appended.
    """
    scenarios = get_scenario_descriptions()
    batched_request = []
    for push_level in ["no_push", "implicit_push", "explicit_push"]:
        for scenario, _ in scenarios.items():
            meta_prompt = build_meta_prompt(
                push_level=push_level,
                scenario=scenario,
                event_title=case["title"],
                ground_truth_summary=case["response"],
                revisionist_summary=case["summary"],
                countries=", ".join(case["countries"]),
            )
            chat_messages = model.append_with_chat_template(message=meta_prompt, role='user')
            batched_request.append({
                "chat_messages": chat_messages,
                "push_level": push_level,
                "scenario": scenario,
            })
            # response = model.chat(chat_messages, use_tqdm=True)
            
    dataframe_rows = []
    
    responses = model.chat([x["chat_messages"] for x in batched_request], use_tqdm=True)
    for response, request in zip(responses, batched_request):
            dataframe_rows.append({
                "ID": id,
                "Historical Event": case["title"],
                "True Version": case["response"],
                "False Version": case["summary"],
                "Country/Region": case["countries"],
                "Source": case["url"],
                "Push Level": request["push_level"],
                "Scenario": request["scenario"],
                "Prompt": response.outputs[0].text
            })
    df.add_rows(dataframe_rows)
            
    
    

def main():
    parser = ArgumentParser(description="Initialize and run the VLLM inference model.")
    parser.add_argument("--model_name", type=str,  help="Name of the model to initialize.", default="google/gemma-3-27b-it")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--tag", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Tag for the output files.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    
    args = parser.parse_args()
    
    if args.debug:
        print("Debug mode is enabled. This will run only ona small subset of the data.")
        args.tag = "debug"
    
    model = init_model(model_name=args.model_name, n_gpus=args.n_gpus)
    
    print(f"Model {args.model_name} initialized with {args.n_gpus} GPUs.")
    
    euvsdisinfo = load_euvsdisinfo(debug=args.debug)
    print(f"Loaded {len(euvsdisinfo)} EUvsDisinfo cases.")
    
    df = OutputDataFrame(
        columns=[
            "ID", "Historical Event", "True Version", "False Version", "Country/Region", "Source", "Push Level", "Scenario", "Prompt"
        ],
        path=f"data/EUvsDISINFO/processed/euvsdisinfo_processed_{args.tag}.csv"
    )
    for id,case in progress(enumerate(euvsdisinfo), description="Processing cases", total=len(euvsdisinfo)):
        process_case(id,case, model, df)
        
        
    
    
if __name__ == "__main__":
    main()