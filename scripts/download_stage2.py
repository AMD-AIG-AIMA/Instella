import os
from huggingface_hub import snapshot_download, login
import time

# Create the target directory if it doesn't exist
os.makedirs("/data/datasets", exist_ok=True)

# You can uncomment and use your token if needed
# login("your_huggingface_token")

# Set up manual retry logic
max_retries = 100
retry_count = 0
success = False

while retry_count < max_retries and not success:
    try:
        print(f"Download attempt {retry_count + 1} of {max_retries}...")
        
        # Download with increased timeout
        repo_path = snapshot_download(
            repo_id="allenai/dolmino-mix-1124",
            local_dir="/data/datasets/dolmino-mix-1124",
            repo_type="dataset",
            etag_timeout=300  # Increased to 5 minutes
        )
        
        print(f"Dataset downloaded to: {repo_path}")
        print(f"Download complete!")
        success = True
        
    except Exception as e:
        retry_count += 1
        print(f"Download failed with error: {str(e)}")
        
        if retry_count < max_retries:
            wait_time = 10 * retry_count  # Increasingly longer waits
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print("Maximum retry attempts reached. Download failed.")
            raise
