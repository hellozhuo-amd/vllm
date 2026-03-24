## extract_responses.py
import json
import sys
import os

def extract_responses(json_file):
    with open(json_file, 'r') as f:
        results = json.load(f)

    if 'generated_texts' not in results:
        print("Error: 'generated_texts' not found. Make sure you ran with --save-detailed flag")
        return

    # Generate output filename
    base_name = os.path.splitext(json_file)[0]
    output_file = f"{base_name}_responses.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Found {len(results['generated_texts'])} responses\n\n")
        f.write("=" * 80 + "\n")

        for i, text in enumerate(results['generated_texts'], 1):
            f.write(f"\nResponse {i}:\n")
            f.write("-" * 80 + "\n")
            f.write(text + "\n")
            f.write("-" * 80 + "\n")





if __name__ == "__main__":
    if len(sys.argv) != 2:

        sys.exit(1)

    extract_responses(sys.argv[1])
