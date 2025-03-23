import os
import subprocess

def download_images():
    base_s3_path = "s3://lin-2023-orion-crc/data"
    base_local_path = "./data"

    for i in range(11, 31):
        crc_folder = f"CRC{i:02d}"
        local_path = os.path.join(base_local_path, crc_folder)
        s3_path = f"{base_s3_path}/{crc_folder}/"

        # Create local directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)

        # Run AWS S3 copy command
        command = ["aws", "s3", "cp", s3_path, local_path, "--recursive"]

        print(f"Downloading from {s3_path} to {local_path}...")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Downloaded {crc_folder} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {crc_folder}: {e}")
            print("Standard Output:", e.stdout)
            print("Standard Error:", e.stderr)
            print(f"Check if the path exists: {s3_path}")
            print(f"Try running manually: {' '.join(command)}")

if __name__ == "__main__":
    download_images()
