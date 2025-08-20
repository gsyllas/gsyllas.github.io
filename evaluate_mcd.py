import os
import pandas as pd
import librosa
import pymcd
from tqdm import tqdm
import sys

# --- CONFIGURATION: EDIT THESE VALUES ---
# The main parent folder containing all your audio samples
SAMPLES_DIR = "samples"

# The name of the folder with the original, ground-truth human audio
BASELINE_DIR_NAME = "baseline" 

# The names of the two speaker-specific model folders you want to evaluate
# NOTE: These folders must be inside the SAMPLES_DIR
MODEL_1_NAME = "lora_specific_speaker" 
MODEL_2_NAME = "speaker_specific"

# The output file where detailed MCD results will be stored
RESULTS_CSV_FILE = "mcd_evaluation_results.csv"

# The sample rate to use for all audio processing. 
# This MUST be consistent. 22050Hz is common for TTS models like Parler-TTS.
# Check your model's configuration if you are unsure.
TARGET_SAMPLE_RATE = 22050

def calculate_mcd_for_models():
    """
    Main function to find corresponding audio files, calculate MCD between
    the baseline and each model, and save the results.
    """
    print("--- Starting MCD Evaluation ---")
    
    # 1. Construct full paths to the directories
    base_path = os.getcwd()
    baseline_path = os.path.join(base_path, SAMPLES_DIR, BASELINE_DIR_NAME)
    model1_path = os.path.join(base_path, SAMPLES_DIR, MODEL_1_NAME)
    model2_path = os.path.join(base_path, SAMPLES_DIR, MODEL_2_NAME)

    # 2. --- Sanity Checks: Verify that all folders exist ---
    for path, name in [(baseline_path, BASELINE_DIR_NAME), (model1_path, MODEL_1_NAME), (model2_path, MODEL_2_NAME)]:
        if not os.path.isdir(path):
            print(f"\nFATAL ERROR: The directory '{name}' was not found at path: {path}")
            print("Please check your CONFIGURATION section and file structure.")
            sys.exit(1)
    
    # 3. Get the list of audio files from the baseline directory to use as reference
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

    # 4. Loop through each baseline file, find the match in the model folders, and calculate MCD
    results_list = []
    
    for filename in tqdm(baseline_files, desc="Calculating MCD"):
        baseline_audio_path = os.path.join(baseline_path, filename)
        model1_audio_path = os.path.join(model1_path, filename)
        model2_audio_path = os.path.join(model2_path, filename)

        # Ensure the corresponding file exists in both model folders before processing
        if not os.path.exists(model1_audio_path):
            tqdm.write(f"--> WARNING: Skipping '{filename}'. Not found in model folder '{MODEL_1_NAME}'.")
            continue
        if not os.path.exists(model2_audio_path):
            tqdm.write(f"--> WARNING: Skipping '{filename}'. Not found in model folder '{MODEL_2_NAME}'.")
            continue

        try:
            # Load all three audio files, ensuring they are at the same sample rate
            y_base, sr_base = librosa.load(baseline_audio_path, sr=TARGET_SAMPLE_RATE)
            y_model1, sr_model1 = librosa.load(model1_audio_path, sr=TARGET_SAMPLE_RATE)
            y_model2, sr_model2 = librosa.load(model2_audio_path, sr=TARGET_SAMPLE_RATE)

            # Calculate MCD. pymcd automatically handles alignment (DTW). A lower score is better.
            mcd_model1 = pymcd.mcd(y_model1, y_base, sr=TARGET_SAMPLE_RATE)
            mcd_model2 = pymcd.mcd(y_model2, y_base, sr=TARGET_SAMPLE_RATE)

            # Append results to our list
            results_list.append({"model": MODEL_1_NAME, "filename": filename, "mcd": mcd_model1})
            results_list.append({"model": MODEL_2_NAME, "filename": filename, "mcd": mcd_model2})

        except Exception as e:
            tqdm.write(f"--> ERROR processing file '{filename}'. Skipping. Error: {e}")

    # 5. Save results to a CSV and print a summary
    if not results_list:
        print("\nEvaluation finished, but no valid results were generated. Please check your filenames.")
        return None
        
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_CSV_FILE, index=False)
    print(f"\nDetailed MCD results have been saved to '{RESULTS_CSV_FILE}'")
    
    # Calculate and print the average MCD for each model
    summary_df = results_df.groupby('model')['mcd'].mean().reset_index()
    print("\n--- Average MCD Scores (Lower is Better) ---")
    print(summary_df)
    
    return results_df


if __name__ == "__main__":
    calculate_mcd_for_models()
    print("\n--- MCD Evaluation Complete ---")
