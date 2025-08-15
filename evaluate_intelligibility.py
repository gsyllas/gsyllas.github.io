import os
import pandas as pd
import torch
from transformers import pipeline
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURATION ---
# The main folder containing subfolders for each model
SAMPLES_DIR = "samples"
# The manifest file mapping audio filenames to their correct text
METADATA_FILE = "metadata.csv"
# The output file where detailed results will be stored
RESULTS_CSV_FILE = "evaluation_results.csv"
# The output image file for the summary chart
CHART_FILE = "evaluation_summary.png"
# The Hugging Face model to use for Automatic Speech Recognition (ASR)
ASR_MODEL_ID = "lightaime/whisper-large-v2-greek"

def setup_asr_pipeline():
    """Initializes and returns the ASR pipeline."""
    print("Loading ASR model... This might take a while on the first run.")
    # Check if a CUDA-enabled GPU is available and use it
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
    """
    Main function to run the evaluation. It finds models, processes audio,
    calculates scores, and saves the results.
    """
    # 1. Setup the ASR model
    asr_pipeline = setup_asr_pipeline()
    
    # 2. Load the ground truth texts from the metadata file
    try:
        metadata_df = pd.read_csv(METADATA_FILE)
        # Create a dictionary for quick lookups: {filename: text}
        ground_truth_map = pd.Series(metadata_df.ground_truth_text.values, index=metadata_df.filename).to_dict()
    except FileNotFoundError:
        print(f"ERROR: The file '{METADATA_FILE}' was not found. Please create it.")
        return

    # 3. Find all model folders and audio files to process
    evaluation_tasks = []
    model_folders = [d for d in os.listdir(SAMPLES_DIR) if os.path.isdir(os.path.join(SAMPLES_DIR, d))]
    
    print(f"\nFound {len(model_folders)} models to evaluate: {', '.join(model_folders)}")

    for model_name in model_folders:
        model_path = os.path.join(SAMPLES_DIR, model_name)
        for filename in os.listdir(model_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(model_path, filename)
                evaluation_tasks.append((model_name, filename, audio_path))

    # 4. Process each audio file and calculate scores
    results_list = []
    print(f"\nProcessing {len(evaluation_tasks)} audio files...")
    
    # tqdm creates a nice progress bar
    for model_name, filename, audio_path in tqdm(evaluation_tasks, desc="Evaluating"):
        if filename not in ground_truth_map:
            print(f"Warning: No ground truth text found for '{filename}' in {METADATA_FILE}. Skipping.")
            continue
            
        ground_truth = ground_truth_map[filename].lower()
        
        # Get transcription from the ASR model
        transcription_result = asr_pipeline(audio_path)
        transcribed_text = transcription_result['text'].lower()
        
        # Calculate scores
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

    # 5. Save detailed results to a CSV file
    if not results_list:
        print("No results to save. Evaluation might have failed or no files were found.")
        return
        
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_CSV_FILE, index=False)
    print(f"\nDetailed results have been saved to '{RESULTS_CSV_FILE}'")
    
    return results_df

def create_summary_chart(df):
    """Calculates average scores and creates a bar chart."""
    if df is None or df.empty:
        print("Cannot create chart from empty results.")
        return
        
    # Calculate the average WER and CER for each model
    summary_df = df.groupby('model')[['wer', 'cer']].mean().reset_index()
    print("\n--- Average Scores per Model ---")
    print(summary_df)
    
    # "Melt" the dataframe to make it suitable for plotting with seaborn
    plot_df = summary_df.melt(id_vars='model', var_name='metric', value_name='score')

    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    bar_plot = sns.barplot(x='model', y='score', hue='metric', data=plot_df, palette="viridis")
    
    # Add labels on top of the bars
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
    plt.tight_layout() # Adjust layout to make room for labels
    
    # Save the chart to a file
    plt.savefig(CHART_FILE)
    print(f"Summary chart has been saved to '{CHART_FILE}'")
    plt.show()


if __name__ == "__main__":
    results_dataframe = evaluate_models()
    create_summary_chart(results_dataframe)