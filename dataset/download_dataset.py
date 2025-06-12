from datasets import load_dataset
import pandas as pd


def prepare_nguni_dataset():
    # Load the dataset
    print("Loading dataset...")
    ds = load_dataset("anrilombard/sa-nguni-languages")

    # Convert to pandas DataFrame (using the 'train' split by default)
    df = pd.DataFrame(ds["train"])

    # List of Nguni languages we're interested in
    nguni_languages = ["isiZulu", "isiXhosa", "isiNdebele"]

    # Filter for only Nguni languages
    df_filtered = df[df["language"].isin(nguni_languages)].copy()

    # Basic data cleaning
    # Remove any rows with empty text
    df_filtered = df_filtered[df_filtered["text"].notna()]
    df_filtered = df_filtered[df_filtered["text"].str.strip() != ""]

    # Print statistics
    print("\nDataset Statistics:")
    print("-------------------")
    for lang in nguni_languages:
        count = len(df_filtered[df_filtered["language"] == lang])
        print(f"{lang}: {count} documents")

    # Save to CSV
    output_file = "nguni_languages_dataset.csv"
    df_filtered.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nDataset saved to {output_file}")

    return df_filtered


if __name__ == "__main__":
    try:
        df = prepare_nguni_dataset()
        print("\nSample of the dataset:")
        print("----------------------")
        print(df.head())
    except Exception as e:
        print(f"An error occurred: {str(e)}")
