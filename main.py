from SigLIP import SigLIPModel
import LOGS

import os
import sys
from PIL import Image
from time import perf_counter
import torch
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
from lancedb import connect
import threading


IMAGE_FOLDER = "IMAGES"
OUTPUT_FOLDER = "OUTPUT"
IMAGE_SCALE = 512  # Resize images to have this width while maintaining aspect ratio
USE_GPU_RESIZE = True  # set True to try GPU resize (needs torchvision and CUDA)
THREAD_COUNT = 6  # number of threads for CPU processing
LANCEDB_DIR = "lancedb_storage"
TABLE_NAME = "image_embeddings"
SEARCH_TOP_K = 100
EMBEDDING_DIM = 512  # Matches the truncated output from SigLIP.encode_image/encode_text

# Thread lock for safe DB writes
db_lock = threading.Lock()

def init_vector_table():
    db = connect(LANCEDB_DIR)
    schema = pa.schema([
        pa.field("filename", pa.string()),
        pa.field("query", pa.string()),
        pa.field("confidence", pa.float32()),
        pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])
    return db.open_table(TABLE_NAME) if TABLE_NAME in db.table_names() else db.create_table(TABLE_NAME, schema=schema)

def process_image(image_file: str, siglip: SigLIPModel, user_search: str, table):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        try:
            start_time = perf_counter()
            image = Image.open(image_path).convert("RGB")
            new_w = IMAGE_SCALE
            new_h = image.height * IMAGE_SCALE // image.width

            if USE_GPU_RESIZE and siglip.use_cuda:
                tensor = TF.pil_to_tensor(image).to(siglip.device, non_blocking=True)
                tensor = TF.resize(tensor, [new_h, new_w], antialias=True)
                image = TF.to_pil_image(tensor.cpu())
            else:
                image = image.resize((new_w, new_h))

            output = siglip.forward(image=image, texts=[user_search])
            # encode_image already returns a 1D CPU float tensor of size 512
            embedding = siglip.encode_image(image).numpy().astype("float32").tolist()
            end_time = perf_counter()
            LOGS.log_info(f"Processing time for image {image_file}: {end_time - start_time:.4f} seconds")
            for _, confidence in output:
                # Fix: thread-safe DB write
                with db_lock:
                    table.add([{
                        "filename": image_file,
                        "query": user_search,
                        "confidence": float(confidence),
                        "embedding": embedding,
                    }])
                LOGS.log_info(f"Image: {image_file}, Confidence for '{user_search}': {confidence:.2%}")
                if confidence > 0.15:
                    output_path = os.path.join(OUTPUT_FOLDER, image_file)
                    image.save(output_path)
                    LOGS.log_success(f"Image '{image_file}' saved to '{OUTPUT_FOLDER}' with confidence {confidence:.2%}")
        except Exception as e:
            LOGS.log_error(f"Failed to process image {image_file}: {e}")


def search_table(table, siglip: SigLIPModel, user_search: str, top_k: int = SEARCH_TOP_K):
    # encode_text already returns a 1D CPU float tensor of size 512
    query_vec = siglip.encode_text(user_search).numpy().astype("float32")
    df = table.search(query_vec).metric("cosine").limit(top_k).to_pandas()
    if df.empty:
        LOGS.log_warning("No results found in vector DB.")
    else:
        # Group by filename to avoid showing duplicates, show best match per file
        seen_files = set()
        for _, row in df.iterrows():
            if row['filename'] not in seen_files:
                seen_files.add(row['filename'])
                LOGS.log_info(f"Match: file={row['filename']}, distance={row['_distance']:.4f}")
                Image.open(os.path.join(IMAGE_FOLDER, row['filename'])).save(os.path.join(OUTPUT_FOLDER, f"search_result_{row['filename']}"))

if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        user_search = args[0]
    except IndexError:
        LOGS.log_error("No search term provided. Please provide a search term as a command-line argument.")
        LOGS.log_info("Usage: python main.py <search_term> [--search-only]")
        sys.exit(1)
    search_only = "--search-only" in args

    LOGS.log_info("Initializing SigLIP model...")
    try:
        SigLip = SigLIPModel(use_compile=False)
        LOGS.log_success("SigLIP model initialized.")
    except Exception as e:
        LOGS.log_error(f"Failed to initialize SigLIP model: {e}")
        sys.exit(1)

    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        LOGS.log_success(f"Output folder '{OUTPUT_FOLDER}' is ready.")
    except Exception as e:
        LOGS.log_error(f"Failed to create output folder '{OUTPUT_FOLDER}': {e}")
        sys.exit(1)

    table = init_vector_table()

    if search_only:
        LOGS.log_info("Search-only mode: querying existing embeddings.")
        search_table(table, SigLip, user_search)
        sys.exit(0)

    LOGS.log_info(f"Processing images in folder: {IMAGE_FOLDER} for search term: '{user_search}'")
    image_files = os.listdir(IMAGE_FOLDER)
    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = [executor.submit(process_image, image_file, SigLip, user_search, table) for image_file in image_files]
        for future in as_completed(futures):
            future.result()