import os
import pandas as pd
from speechbrain.inference.speaker import SpeakerRecognition
import torch
import torchaudio
from tqdm import tqdm
import sys

# --- CONFIGURATION: EDIT THESE VALUES ---
# The main parent folder containing all your audio samples
SAMPLES_DIR = "samples"

# The name of the folder with the original, ground-truth human audio
BASELINE_DIR_NAME = "baseline" # Or "_ASR_BASELINE" to match your other scripts

# The names of the two speaker-specific model folders you want to evaluate
MODEL_1_NAME = "lora_specific_speaker" 
MODEL_2_NAME = "speaker_specific"

# A name for our baseline-to-itself comparison in the final report
BASELINE_SELF_SIM_NAME = "_BASELINE_SELF_SIMILARITY"

# The output file where detailed similarity results will be stored
RESULTS_CSV_FILE = "similarity_evaluation_results.csv"

# The Hugging Face model ID for the speaker recognition model.
SPEAKER_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

def to_mono(signal):
    """
    --- NEW HELPER FUNCTION ---
    Averages the channels of a stereo signal to convert it to mono.
    """
    if signal.shape[0] > 1: # Check if number of channels is more than 1
        return torch.mean(signal, dim=0, keepdim=True)
    return signal

def calculate_similarity_for_models():
    """
    Main function to find corresponding audio files, calculate Speaker Similarity
    between the baseline, models, and the baseline itself, and save the results.
    """
    print("--- Starting Speaker Similarity (SIM-S) Evaluation ---")
    
    # Construct full paths to the directories
    base_path = os.getcwd()
    baseline_path = os.path.join(base_path, SAMPLES_DIR, BASELINE_DIR_NAME)
    model1_path = os.path.join(base_path, SAMPLES_DIR, MODEL_1_NAME)
    model2_path = os.path.join(base_path, SAMPLES_DIR, MODEL_2_NAME)

    # Sanity Checks: Verify that all folders exist
    for path, name in [(baseline_path, BASELINE_DIR_NAME), (model1_path, MODEL_1_NAME), (model2_path, MODEL_2_NAME)]:
        if not os.path.isdir(path):
            print(f"\nFATAL ERROR: The directory '{name}' was not found at path: {path}")
            sys.exit(1)
            
    # Load the Speaker Recognition model
    print(f"Loading speaker recognition model: {SPEAKER_MODEL_ID}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        speaker_model = SpeakerRecognition.from_hparams(source=SPEAKER_MODEL_ID, run_opts={"device":device})
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    except Exception as e:
        print(f"FATAL ERROR: Could not load the speaker recognition model. Error: {e}")
        sys.exit(1)
    print("Speaker recognition model loaded successfully.")

    # Get the list of audio files from the baseline directory
    try:
        baseline_files = [f for f in os.listdir(baseline_path) if f.endswith('.wav')]
        if not baseline_files:
            print(f"WARNING: No '.wav' files found in the baseline directory '{BASELINE_DIR_NAME}'. Nothing to process.")
            return None
    except FileNotFoundError:
        print(f"FATAL ERROR: The baseline directory '{BASELINE_DIR_NAME}' was not found.")
        sys.exit(1)
        
    print(f"Found {len(baseline_files)} audio files in the baseline directory to use as reference.")
    print(f"Comparing against models: '{MODEL_1_NAME}' and '{MODEL_2_NAME}'")

    results_list = []
    
    for filename in tqdm(baseline_files, desc="Calculating Similarity"):
        baseline_audio_path = os.path.join(baseline_path, filename)
        model1_audio_path = os.path.join(model1_path, filename)
        model2_audio_path = os.path.join(model2_path, filename)

        if not os.path.exists(model1_audio_path) or not os.path.exists(model2_audio_path):
            tqdm.write(f"--> WARNING: Skipping '{filename}'. Missing a corresponding file in one of the model folders.")
            continue

        try:
            # Load the audio files into tensors
            signal_base, fs_base = torchaudio.load(baseline_audio_path)
            signal_model1, fs_model1 = torchaudio.load(model1_audio_path)
            signal_model2, fs_model2 = torchaudio.load(model2_audio_path)
            
            # --- MODIFIED: Ensure all audio is mono before processing ---
            signal_base = to_mono(signal_base)
            signal_model1 = to_mono(signal_model1)
            signal_model2 = to_mono(signal_model2)

            # Generate the embeddings from the audio tensors
            embedding_base = speaker_model.encode_batch(signal_base)
            embedding_model1 = speaker_model.encode_batch(signal_model1)
            embedding_model2 = speaker_model.encode_batch(signal_model2)
            
            # Calculate the cosine similarity. .squeeze() removes unnecessary dimensions.
            sim_model1 = cosine_similarity(embedding_base.squeeze(), embedding_model1.squeeze()).item()
            sim_model2 = cosine_similarity(embedding_base.squeeze(), embedding_model2.squeeze()).item()
            sim_baseline = cosine_similarity(embedding_base.squeeze(), embedding_base.squeeze()).item()

            results_list.append({"model": MODEL_1_NAME, "filename": filename, "similarity": sim_model1})
            results_list.append({"model": MODEL_2_NAME, "filename": filename, "similarity": sim_model2})
            results_list.append({"model": BASELINE_SELF_SIM_NAME, "filename": filename, "similarity": sim_baseline})

        except Exception as e:
            tqdm.write(f"--> ERROR processing file '{filename}'. Skipping. Error: {e}")

    # Save results to a CSV and print a summary
    if not results_list:
        print("\nEvaluation finished, but no valid results were generated. Please check your filenames.")
        return None
        
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_CSV_FILE, index=False)
    print(f"\nDetailed similarity results have been saved to '{RESULTS_CSV_FILE}'")
    
    # Calculate and print the average similarity for each model
    summary_df = results_df.groupby('model')['similarity'].mean().reset_index()
    summary_df = summary_df.sort_values(by='similarity', ascending=False)
    
    print("\n--- Average Speaker Similarity Scores (Higher is Better) ---")
    print(summary_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    calculate_similarity_for_models()
    print("\n--- Speaker Similarity Evaluation Complete ---")
