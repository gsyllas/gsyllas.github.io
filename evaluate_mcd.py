
import os
import pandas as pd
import librosa
from pymcd.mcd import Calculate_MCD
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
    
    # Construct full paths to the directories
    base_path = os.getcwd()
    baseline_path = os.path.join(base_path, SAMPLES_DIR, BASELINE_DIR_NAME)
    model1_path = os.path.join(base_path, SAMPLES_DIR, MODEL_1_NAME)
    model2_path = os.path.join(base_path, SAMPLES_DIR, MODEL_2_NAME)

    # Sanity Checks: Verify that all folders exist
    for path, name in [(baseline_path, BASELINE_DIR_NAME), (model1_path, MODEL_1_NAME), (model2_path, MODEL_2_NAME)]:
        if not os.path.isdir(path):
            print(f"\nFATAL ERROR: The directory '{name}' was not found at path: {path}")
            print("Please check your CONFIGURATION section and file structure.")
            sys.exit(1)
    
    # Get the list of audio files from the baseline directory to use as reference
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

    # Instantiate the MCD calculator object once
    # Using 'dtw_sl' which incorporates speech/silence detection for better accuracy
    mcd_toolbox = Calculate_MCD(MCD_mode='dtw_sl')

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
            # The mcd_toolbox object's calculate_mcd method expects file paths.
            # It handles loading and resampling internally if needed, but it's best to be consistent.
            # We don't need to load with librosa first.
            mcd_model1 = mcd_toolbox.calculate_mcd(model1_audio_path, baseline_audio_path)
            mcd_model2 = mcd_toolbox.calculate_mcd(model2_audio_path, baseline_audio_path)

            results_list.append({"model": MODEL_1_NAME, "filename": filename, "mcd": mcd_model1})
            results_list.append({"model": MODEL_2_NAME, "filename": filename, "mcd": mcd_model2})

        except Exception as e:
            tqdm.write(f"--> ERROR processing file '{filename}'. Skipping. Error: {e}")

    # Save results to a CSV and print a summary
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
