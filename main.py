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
EMBEDDING_DIM = 1152  # SigLIP2-large-patch16-512 has 1152-dim embeddings


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


def process_image(image_file: str, siglip: SigLIPModel, table):
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

            # Only embed & index the image. Text-based searching is handled by `search_table`.
            embedding = siglip.encode_image(image).numpy().astype("float32").tolist()
            end_time = perf_counter()
            LOGS.log_info(f"Indexing time for image {image_file}: {end_time - start_time:.4f} seconds")
            # Thread-safe DB write
            with db_lock:
                table.add([{
                    "filename": image_file,
                    "query": "",
                    "confidence": 0.0,
                    "embedding": embedding,
                }])
            # LOGS.log_success(f"Indexed image '{image_file}' in vector DB.")
        except Exception as e:
            LOGS.log_error(f"Failed to index image {image_file}: {e}")


def search_table(table, siglip: SigLIPModel, user_search: str, top_k: int = SEARCH_TOP_K):
    """Search by cosine similarity; report ranking, not fake probabilities."""
    # encode_text returns a normalized 1D CPU float tensor
    query_vec = siglip.encode_text(user_search).numpy().astype("float32")
    df = table.search(query_vec).metric("cosine").limit(top_k).to_pandas()
    if df.empty:
        LOGS.log_warning("No results found in vector DB.")
    else:
        # Group by filename to avoid showing duplicates, show best match per file
        seen_files = set()
        rank = 1
        for _, row in df.iterrows():
            if row['filename'] in seen_files:
                continue
            seen_files.add(row['filename'])

            # For cosine metric, similarity ~= 1 - distance
            similarity = 1.0 - float(row['_distance'])
            if similarity < 0.1:
                break       # skip low-similarity results
            LOGS.log_info(f"Rank {rank}: file={row['filename']}, cosine_sim={similarity:.4f}, distance={row['_distance']:.4f}")

            rank += 1
            image = Image.open(os.path.join(IMAGE_FOLDER, row['filename']))
            new_w, new_h = IMAGE_SCALE, image.height * IMAGE_SCALE // image.width
            image.resize((new_w, new_h)).save(os.path.join(OUTPUT_FOLDER, f"search_result_{row['filename']}"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index or search images using SigLIP2 embeddings")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index-only", action="store_true", help="Index images into the vector DB")
    group.add_argument("--search-only", action="store_true", help="Search existing index (requires a search term)")
    parser.add_argument("search_term", nargs="?", help="Search term for --search-only")
    parser.add_argument("--recreate-db", action="store_true", help="Delete and recreate the vector DB before indexing")
    args = parser.parse_args()

    if args.search_only and not args.search_term:
        LOGS.log_error("Search term required for --search-only")
        sys.exit(1)

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

    if args.recreate_db and os.path.exists(LANCEDB_DIR):
        import shutil
        try:
            shutil.rmtree(LANCEDB_DIR)
            LOGS.log_info("Removed existing vector DB storage.")
        except Exception as e:
            LOGS.log_error(f"Failed to remove existing DB dir: {e}")

    table = init_vector_table()

    if args.search_only:
        LOGS.log_info("Search-only mode: querying existing embeddings.")
        search_table(table, SigLip, args.search_term)
        sys.exit(0)

    if args.index_only:
        LOGS.log_info(f"Index-only mode: processing images in folder: {IMAGE_FOLDER}")
        start_time = perf_counter()
        image_files = os.listdir(IMAGE_FOLDER)
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(process_image, image_file, SigLip, table) for image_file in image_files]
            for future in as_completed(futures):
                future.result()
        end_time = perf_counter()
        LOGS.log_success(f"Completed processing {len(image_files)} images in {end_time - start_time:.2f} seconds.")