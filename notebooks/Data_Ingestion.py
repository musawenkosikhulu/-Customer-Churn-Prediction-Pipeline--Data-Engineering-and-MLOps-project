import os
import kaggle

def download_kaggle_data(dataset_name, download_path='./data'):
    # Set the download path directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Get Kaggle credentials from environment variables
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        raise ValueError("Kaggle API credentials are not set correctly in GitHub secrets.")

    # Set up Kaggle API environment variables
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    # Download the dataset from Kaggle
    print(f"Downloading dataset: {dataset_name}")
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Download completed. Files are saved to: {download_path}")

if __name__ == "__main__":
    dataset = 'blastchar/telco-customer-churn'  # Replace with the dataset you want to download
    download_path = './data'  # Specify the path where you want to store the dataset
    download_kaggle_data(dataset, download_path)

