import os
import re
import shutil
import concurrent
from tqdm import tqdm
from glob import glob


def copy_file(source_path, destination_path):
    if not os.path.exists(destination_path) and not os.path.isdir(destination_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        try:
            shutil.copy(source_path, destination_path)
        except IsADirectoryError:
            pass
        except Exception as e:
            raise e

def copy_files_parallel(source_directory, destination_directory, filter_regex=None, num_threads=4, overwrite=False):
    
    source_files = glob(os.path.join(source_directory, '**', '*'), recursive=True)
    if filter_regex is not None:
        source_files = list(filter(
            lambda filepath: os.path.isfile(filepath) and re.search(filter_regex, filepath, re.IGNORECASE),
            source_files
        ))
        print(len(source_files))
    
    destination_files = [filepath.replace(source_directory, destination_directory) for filepath in source_files]
    
    if overwrite:
        shutil.rmtree(destination_directory)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {
            executor.submit(copy_file, source, dest): source
            for source, dest in zip(source_files, destination_files)
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(future_to_file), unit="file"):
            source = future_to_file[future]
            if future.exception() is not None:
                print(f"Error copying {source}: {future.exception()}")
