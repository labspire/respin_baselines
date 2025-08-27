import os
import json
import argparse
from collections import defaultdict

def load_metadata(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def generate_kaldi_files(meta_json_path, output_dir):
    data = load_metadata(meta_json_path)

    base_dir = os.path.dirname(meta_json_path)

    wav_scp = []
    utt2spk = []
    text = []
    utt2lang = []
    utt2dialect = []
    utt2dur = []
    spk2utt_dict = defaultdict(list)

    for utt_id, info in data.items():
        wav_path = os.path.join(base_dir, info["wav_path"])
        speaker_id = "_".join(utt_id.split("_")[:5])  # Corrected speaker ID
        lang = info["lid"]
        dialect = info["dialect"]
        transcript = info["text"]
        # Remove commas and question marks, strip leading/trailing spaces, and collapse consecutive spaces
        transcript = transcript.replace(",", "").replace("?", "")
        transcript = " ".join(transcript.strip().split())
        duration = info["duration"]

        # Populate each line
        wav_scp.append(f"{utt_id}\t{wav_path}")
        utt2spk.append(f"{utt_id}\t{speaker_id}")
        text.append(f"{utt_id}\t{transcript}")
        utt2lang.append(f"{utt_id}\t{lang}")
        utt2dialect.append(f"{utt_id}\t{dialect}")
        utt2dur.append(f"{utt_id}\t{duration:.2f}")
        spk2utt_dict[speaker_id].append(utt_id)

    # Create spk2utt from utt2spk
    spk2utt = [f"{spk}\t{' '.join(utts)}" for spk, utts in spk2utt_dict.items()]

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Write files
    write_file(os.path.join(output_dir, "wav.scp"), sorted(wav_scp))
    write_file(os.path.join(output_dir, "utt2spk"), sorted(utt2spk))
    write_file(os.path.join(output_dir, "spk2utt"), sorted(spk2utt))
    write_file(os.path.join(output_dir, "text"), sorted(text))
    write_file(os.path.join(output_dir, "utt2lang"), sorted(utt2lang))
    write_file(os.path.join(output_dir, "utt2dialect"), sorted(utt2dialect))
    write_file(os.path.join(output_dir, "utt2dur"), sorted(utt2dur))

    print(f"Generated Kaldi files in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaldi files from RESPIN metadata JSON.")
    parser.add_argument("--meta_json", type=str, required=True, help="Path to meta JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output Kaldi files")
    args = parser.parse_args()

    generate_kaldi_files(args.meta_json, args.output_dir)
