#!/bin/bash

# Exit on error
set -e

# Default values
TRAIN_FILE="unbalanced_nguni_train.csv"
VAL_FILE="unbalanced_nguni_val.csv"
OUTPUT_DIR="./data_unbalanced"
SHARD_SIZE=10000000 # 10^7

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Converting dataset to binary format..."
echo "Training file: $TRAIN_FILE"
echo "Validation file: $VAL_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Shard size: $SHARD_SIZE"

# Run the conversion script
python csv_to_binary.py \
	--train "$TRAIN_FILE" \
	--val "$VAL_FILE" \
	--output-dir "$OUTPUT_DIR" \
	--shard-size "$SHARD_SIZE"

echo "Conversion complete! Binary files are in $OUTPUT_DIR"

# Print the resulting files
echo -e "\nGenerated files:"
ls -lh "$OUTPUT_DIR"
