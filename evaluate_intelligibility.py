import os
import pandas as pd
import torch
from transformers import pipeline
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import string
import librosa # --- NEW: Import librosa for audio resampling ---

# --- CONFIGURATION ---
SAMPLES_DIR = "samples"
METADATA_FILE = "metadata.csv"
RESULTS_CSV_FILE = "evaluation_results_final.csv" # Renamed for clarity
CHART_FILE = "evaluation_summary_final.png"     # Renamed for clarity
ASR_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-greek"
TARGET_SAMPLE_RATE = 16000 # --- NEW: Define the required sample rate for the ASR model ---

def initial_setup_check():
    """Performs initial checks for required files and directories."""
    print("--- Initial Setup Check ---")
    current_directory = os.getcwd()
    print(f"Running script from: {current_directory}")

    if not os.path.isdir(SAMPLES_DIR):
        print(f"\nFATAL ERROR: The samples directory '{SAMPLES_DIR}' was not found.")
        sys.exit(1)
    print(f"[✓] Found samples directory: '{SAMPLES_DIR}'")

    if not os.path.isfile(METADATA_FILE):
        print(f"\nFATAL ERROR: The metadata file '{METADATA_FILE}' was not found.")
        sys.exit(1)
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
    
    print(f"Loading ground truth texts from '{METADATA_FILE}'...")
    metadata_df = pd.read_csv(METADATA_FILE)
    ground_truth_map = pd.Series(metadata_df.ground_truth_text.values, index=metadata_df.filename).to_dict()
    print(f"[✓] Loaded {len(ground_truth_map)} text entries.")

    evaluation_tasks = []
    model_folders = [d for d in os.listdir(SAMPLES_DIR) if os.path.isdir(os.path.join(SAMPLES_DIR, d))]
    
    if not model_folders:
        print(f"WARNING: No model subdirectories found inside '{SAMPLES_DIR}'.")
        return None
        
    print(f"\nFound {len(model_folders)} models to evaluate: {', '.join(model_folders)}")

    for model_name in model_folders:
        model_path = os.path.join(SAMPLES_DIR, model_name)
        for filename in os.listdir(model_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(model_path, filename)
                evaluation_tasks.append((model_name, filename, audio_path))

    if not evaluation_tasks:
        print("WARNING: No '.wav' files found inside model folders.")
        return None

    asr_pipeline = setup_asr_pipeline()
    
    punctuation_to_remove = string.punctuation + ";"
    translator = str.maketrans('', '', punctuation_to_remove)

    results_list = []
    print(f"\nProcessing {len(evaluation_tasks)} audio files...")
    
    for model_name, filename, audio_path in tqdm(evaluation_tasks, desc="Evaluating"):
        if filename not in ground_truth_map:
            tqdm.write(f"--> WARNING: Skipping '{filename}' for model '{model_name}' because it's not in {METADATA_FILE}.")
            continue
        
        # --- MODIFIED: Load audio with Librosa to ensure correct sample rate ---
        try:
            # librosa.load will automatically resample the audio to the target rate
            audio_array, sample_rate = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)
        except Exception as e:
            tqdm.write(f"--> ERROR: Could not load or process audio file {audio_path}. Skipping. Error: {e}")
            continue

        # Get transcription by passing the resampled audio array directly
        transcription_result = asr_pipeline(audio_array)
        transcribed_text_raw = transcription_result['text'] if transcription_result['text'] else ""
        
        ground_truth_raw = ground_truth_map[filename]
        
        # Normalize both texts (lowercase, no punctuation)
        ground_truth_normalized = ground_truth_raw.lower().translate(translator)
        transcribed_text_normalized = transcribed_text_raw.lower().translate(translator)

        # Calculate WER and CER on the clean, normalized text
        wer = jiwer.wer(ground_truth_normalized, transcribed_text_normalized)
        cer = jiwer.cer(ground_truth_normalized, transcribed_text_normalized)
        
        results_list.append({
            "model": model_name,
            "filename": filename,
            "ground_truth": ground_truth_raw,
            "transcribed_text": transcribed_text_raw,
            "wer": wer,
            "cer": cer
        })

    if not results_list:
        print("\nEvaluation finished, but no valid results were generated.")
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
    print("\n--- Average Scores per Model (Normalized & Resampled) ---")
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

    plt.title('TTS Model Intelligibility Comparison (Normalized & Resampled, Lower is Better)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Error Rate', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(CHART_FILE)
    print(f"Summary chart has been saved to '{CHART_FILE}'")
    plt.show()

if __name__ == "__main__":
    if initial_setup_check():
        results_dataframe = evaluate_models()
        create_summary_chart(results_dataframe)