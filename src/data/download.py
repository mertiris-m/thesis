import requests
from pathlib import Path
from tqdm import tqdm
import tarfile 

"""Downloading the datasets from their official source"""


def download_qm9(): # Downloads the QM9 Dataset to /data/QM9

    # URL for the dataset
    url = "https://springernature.figshare.com/ndownloader/files/3195389"
    
    # Get the project root
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "QM9" / "dsgdb9nsd.xyz"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"QM9 Dataset already exists at:\n{output_path.resolve()}")
        return True  # Return status

    # Download with progress bar
    print(f"Downloading QM9 dataset to:\n{output_path.resolve()}")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc="dsgdb9nsd.xyz",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("Download completed successfully!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
            print("Removed partial download")
        return False

if __name__ == "__main__":
    download_qm9()



def download_thermoML(): # Downloads the NIST ThermoML Dataset to /data/ThermoML
    
    # URLs for the dataset and schema
    data_url = "https://data.nist.gov/od/ds/mds2-2422/ThermoML.v2020-09-30.tgz"
    schema_url = "https://data.nist.gov/od/ds/mds2-2422/ThermoML.xsd"
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "ThermoML"
    archive_path = output_dir / "ThermoML.tgz"
    schema_path = output_dir / "ThermoML.xsd"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    xml_files = list(output_dir.glob("*.xml"))
    if xml_files and schema_path.exists():
        print(f"ThermoML Dataset already exists at:\n{output_dir.resolve()}")
        print(f"Found {len(xml_files)} XML files and schema")
        return True

    try:
        print("Downloading ThermoML schema...")
        with requests.get(schema_url, stream=True) as r:
            r.raise_for_status()
            with open(schema_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Schema saved to: {schema_path}")

        print(f"\nDownloading ThermoML dataset to:\n{archive_path}")
        with requests.get(data_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(archive_path, 'wb') as f, tqdm(
                desc="ThermoML.tgz",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract the .tgz file
        print("\nDownload complete, now extracting...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        
        # Clean up
        archive_path.unlink()
        print(f"\nExtraction complete! Files saved to:\n{output_dir.resolve()}")
        print(f"Total XML files: {len(list(output_dir.glob('*.xml')))}")
        return True
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if archive_path.exists():
            archive_path.unlink()
        if schema_path.exists():
            schema_path.unlink()
        print("Removed partial downloads")
        return False

if __name__ == "__main__":
    download_thermoML()



def download_esol(): # Downloads the ESOL Dataset to /data/ThermoML
    
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "ESOL"
    output_path = output_dir / "delaney-processed.csv"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Dataset already exists at:\n{output_path.resolve()}")
        return True

    print(f"Downloading ESOL dataset to:\n{output_path}")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc="delaney-processed.csv",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("Download completed successfully!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
            print("Removed partial download")
        return False

if __name__ == "__main__":
    download_esol()

