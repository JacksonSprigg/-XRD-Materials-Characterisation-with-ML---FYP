import os
import time

def combine_gzip_files(input_prefix, num_parts, output_file):
    total_size = sum(os.path.getsize(f"training_data/simXRD_full_data/new/{input_prefix}_{i}.db.gz") for i in range(1, num_parts + 1))
    processed_size = 0
    start_time_total = time.time()

    with open(output_file, 'wb') as f_out:
        for i in range(1, num_parts + 1):
            part_file = f"training_data/simXRD_full_data/new/{input_prefix}_{i}.db.gz"
            file_size = os.path.getsize(part_file)
            
            print(f"Processing file {i} of {num_parts}: {part_file}")
            start_time = time.time()

            with open(part_file, 'rb') as f_in:
                # Read and write in chunks to avoid loading entire file into memory
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    processed_size += len(chunk)
                    progress = (processed_size / total_size) * 100
                    elapsed_time_total = time.time() - start_time_total
                    avg_speed = processed_size / elapsed_time_total
                    estimated_time_remaining = (total_size - processed_size) / avg_speed
                    print(f"\rProgress: {progress:.2f}% | Est. time remaining: {estimated_time_remaining:.2f} seconds", end="", flush=True)

            elapsed_time = time.time() - start_time
            print(f"\nFile {i} completed in {elapsed_time:.2f} seconds\n")

    total_elapsed_time = time.time() - start_time_total
    print(f"All files combined successfully in {total_elapsed_time:.2f} seconds!")

combine_gzip_files('ILtrain_combined', 2, 'ILtest_combined.db.gz')