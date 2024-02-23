from google.cloud import storage
from dt_mine_rl.config import config
import argparse
import os


def download_data(source_blob_prefix, destination_directory):
    """Downloads all blobs with a specific prefix from the bucket."""
    bucket_name = "dt-mine-rl-project"

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_blob_prefix)

    print("blobs", blobs) 

    for blob in blobs:

        if blob.name.endswith("/"):
            continue
        destination_file_name = os.path.join(destination_directory, blob.name)
        blob.download_to_filename(destination_file_name)
        print(f"Descargado {blob.name} a {destination_file_name}.")

def main(args):

    env_key = args.env

    os.makedirs(os.dirname(os.path.join(config["common"]["path_data"], "VPT-models")), exist_ok=True)
    os.makedirs(os.dirname(os.path.join(config["common"]["path_data"], "embeddings","foundation-model-1x.weights",env_key)), exist_ok=True)

    download_data("VPT-models", config["common"]["path_data"])
    download_data("embeddings/foundation-model-1x.weights", config["common"]["path_data"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a DT agent on VPT embeddings for the BASALT Benchmark")
    parser.add_argument("--env", type=str, required=True, help="Environment to train")
    args = parser.parse_args()
    main(args)