import os
import pandas as pd
import torch
from transformers import pipeline
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys # --- ADDED for sys.exit ---

# --- CONFIGURATION ---
SAMPLES_DIR = "samples"
METADATA_FILE = "metadata.csv"
RESULTS_CSV_FILE = "evaluation_results.csv"
CHART_FILE = "evaluation_summary.png"
ASR_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-greek"

def initial_setup_check():
    """
    --- ADDED DEBUGGING ---
    Performs initial checks for required files and directories before doing heavy work.
    """
    print("--- Initial Setup Check ---")
    current_directory = os.getcwd()
    print(f"Running script from: {current_directory}")

    # Check 1: Does the samples directory exist?
    if not os.path.isdir(SAMPLES_DIR):
        print(f"\nFATAL ERROR: The samples directory '{SAMPLES_DIR}' was not found.")
        print(f"Please make sure you are running this script from the repository root and that the '{SAMPLES_DIR}' folder exists.")
        sys.exit(1) # Exit the script
    print(f"[✓] Found samples directory: '{SAMPLES_DIR}'")

    # Check 2: Does the metadata file exist?
    if not os.path.isfile(METADATA_FILE):
        print(f"\nFATAL ERROR: The metadata file '{METADATA_FILE}' was not found.")
        print("Please create this file in the root directory. It should have two columns: 'filename' and 'ground_truth_text'.")
        sys.exit(1) # Exit the script
    print(f"[✓] Found metadata file: '{METADATA_FILE}'")
    print("---------------------------\n")
    return True

def setup_asr_pipeline():
    """Initializes and returns the ASR pipeline."""
    print("Loading ASR model... This might take a while on the first run.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_ID,
        device=device
    )
    print("ASR model loaded successfully.")
    return asr_pipeline

def evaluate_models():
    """Main function to run the evaluation."""
    
    # --- MOVED METADATA LOADING TO THE TOP ---
    print(f"Loading ground truth texts from '{METADATA_FILE}'...")
    metadata_df = pd.read_csv(METADATA_FILE)
    ground_truth_map = pd.Series(metadata_df.ground_truth_text.values, index=metadata_df.filename).to_dict()
    print(f"[✓] Loaded {len(ground_truth_map)} text entries.")

    # Find all model folders and audio files
    evaluation_tasks = []
    model_folders = [d for d in os.listdir(SAMPLES_DIR) if os.path.isdir(os.path.join(SAMPLES_DIR, d))]
    
    if not model_folders:
        print(f"WARNING: No model subdirectories found inside '{SAMPLES_DIR}'. Nothing to evaluate.")
        return None
        
    print(f"\nFound {len(model_folders)} models to evaluate: {', '.join(model_folders)}")

    for model_name in model_folders:
        model_path = os.path.join(SAMPLES_DIR, model_name)
        for filename in os.listdir(model_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(model_path, filename)
                evaluation_tasks.append((model_name, filename, audio_path))

    if not evaluation_tasks:
        print("WARNING: Found model folders, but no '.wav' files inside them. Nothing to evaluate.")
        return None

    # Now load the ASR model, since we know there's work to do
    asr_pipeline = setup_asr_pipeline()
    
    # Process each audio file
    results_list = []
    print(f"\nProcessing {len(evaluation_tasks)} audio files...")
    
    for model_name, filename, audio_path in tqdm(evaluation_tasks, desc="Evaluating"):
        if filename not in ground_truth_map:
            # --- IMPROVED WARNING ---
            tqdm.write(f"--> WARNING: Skipping '{filename}' for model '{model_name}' because it's not in {METADATA_FILE}.")
            continue
            
        ground_truth = ground_truth_map[filename].lower()
        
        transcription_result = asr_pipeline(audio_path)
        transcribed_text = transcription_result['text'].lower() if transcription_result['text'] else ""
        
        wer = jiwer.wer(ground_truth, transcribed_text)
        cer = jiwer.cer(ground_truth, transcribed_text)
        
        results_list.append({
            "model": model_name,
            "filename": filename,
            "ground_truth": ground_truth,
            "transcribed_text": transcribed_text,
            "wer": wer,
            "cer": cer
        })

    if not results_list:
        print("\nEvaluation finished, but no valid results were generated. Please check your filenames and metadata.")
        return None
        
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_CSV_FILE, index=False)
    print(f"\nDetailed results have been saved to '{RESULTS_CSV_FILE}'")
    
    return results_df

def create_summary_chart(df):
    """Calculates average scores and creates a bar chart."""
    if df is None or df.empty:
        print("Cannot create chart from empty results.")
        return
        
    summary_df = df.groupby('model')[['wer', 'cer']].mean().reset_index()
    print("\n--- Average Scores per Model ---")
    print(summary_df)
    
    plot_df = summary_df.melt(id_vars='model', var_name='metric', value_name='score')

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    bar_plot = sns.barplot(x='model', y='score', hue='metric', data=plot_df, palette="viridis")
    
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.3f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points')

    plt.title('TTS Model Intelligibility Comparison (Lower is Better)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Error Rate', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(CHART_FILE)
    print(f"Summary chart has been saved to '{CHART_FILE}'")
    plt.show()

if __name__ == "__main__":
    # --- ADDED INITIAL CHECK ---
    if initial_setup_check():
        results_dataframe = evaluate_models()
        create_summary_chart(results_dataframe)