import gzip
import shutil
import os
import time

def unzip_gz_file(input_file, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_file)[0]  # Remove .gz extension

    start_time = time.time()
    input_size = os.path.getsize(input_file)

    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    end_time = time.time()
    output_size = os.path.getsize(output_file)

    print(f"File unzipped successfully!")
    print(f"Input file: {input_file} ({input_size:,} bytes)")
    print(f"Output file: {output_file} ({output_size:,} bytes)")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

# Usage
input_file = 'ILtrain_combined_1.db.gz'  # Replace with your .gz file name
unzip_gz_file(input_file)