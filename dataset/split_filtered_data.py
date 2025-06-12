from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_val_split(
    df, val_size=0.2, stratify_col="language", random_state=42, shuffle_data=True
):
    try:
        # Initialize tqdm progress bar
        with tqdm(total=4, desc="Processing") as pbar:
            # Shuffle the dataframe if requested
            if shuffle_data:
                print("Shuffling data...")
                df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
                pbar.update(1)  # Update progress bar after shuffling

            # Perform stratified split
            print("Performing stratified split...")
            train_df, val_df = train_test_split(
                df,
                test_size=val_size,
                stratify=df[stratify_col],
                random_state=random_state,
            )
            pbar.update(1)  # Update progress bar after splitting

            # Verify the splits maintain language distribution (optional)
            print("Verifying distribution...")
            original_dist = df[stratify_col].value_counts(normalize=True)
            train_dist = train_df[stratify_col].value_counts(normalize=True)
            val_dist = val_df[stratify_col].value_counts(normalize=True)

            print("\nData Distribution Verification:")
            print("-" * 30)
            print(f"Original distribution:\n{original_dist}")
            print(f"\nTrain distribution:\n{train_dist}")
            print(f"\nValidation distribution:\n{val_dist}")
            pbar.update(1)  # Update progress bar after verification

            # Save the splits to CSV
            print("Saving files...")
            train_df.to_csv("filtered_data_train.csv", index=False)
            val_df.to_csv("filtered_data_val.csv", index=False)
            pbar.update(1)  # Update progress bar after saving files

            print("\nFiles saved:")
            print("- filtered_data_train.csv")
            print("- filtered_data_val.csv")

        return train_df, val_df

    except Exception as e:
        print(f"Error creating train/val split: {str(e)}")
        return None, None


# Example usage
print("Reading CSV file...")
combined_filtered_data = pd.read_csv(
    "combined_filtered_data.csv", dtype=str, low_memory=False
)
train_df, val_df = create_train_val_split(combined_filtered_data)
