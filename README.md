> [!IMPORTANT]
> Within few days the model will be swapped for DINOv3 (model accuracy reasons)

# VectorView - SigLIP Image Embedding Pipeline 🖼️ ➡️ 🚀

This project implements an image processing pipeline that leverages the SigLIP ( Sigmoid Loss for Language Image Pre-Training ;) ) model from Hugging Face Transformers to generate image embeddings. These embeddings are then stored in a LanceDB database, enabling efficient similarity search and other downstream tasks. The pipeline supports multi-threading for faster processing and GPU acceleration for image resizing and model inference. This project solves the problem of efficiently converting images into a vector representation suitable for semantic search and analysis.

## 🚀 Key Features

- **Image Encoding:** Converts images into fixed-size vector representations (embeddings) using the SigLIP model.
- **Similarity Search:** Stores image embeddings in a LanceDB database for efficient similarity search.
- **Multi-threading:** Processes multiple images concurrently to improve performance.
- **CUDA Optimizations:** The pipeline includes CUDA-optimized preprocessing and inference paths. On a CUDA-enabled GPU a single image typically takes about 0.2–0.3s to process (resize + encode), making the pipeline fast for large-scale, high-throughput workflows.
- **General optimizations:** Efficient image loading, resizing, and batching to minimize latency and maximize throughput. For folder of 400 images, the entire processing and indexing takes about 25 seconds. (laptop with RTX 4070 GPU)
- **Configurable:** Offers various configuration options, such as image scale, thread count, and database location.
- **Logging:** Provides informative messages to the user through a custom logging system.
- **Easy Integration:** Designed to be easily integrated into larger applications that require image and text understanding.

## 🛠️ Tech Stack

*   **Frontend:** (None explicitly, but could be integrated with a frontend for search)
*   **Backend:** Python
*   **Database:** LanceDB
*   **AI Model:** SigLIP (Hugging Face Transformers)
*   **Image Processing:** PIL (Pillow), TorchVision
*   **Concurrency:** `concurrent.futures` (ThreadPoolExecutor), `threading`

## 📦 Getting Started

### Prerequisites

- Python 3.12
- CUDA-enabled GPU (optional, but recommended for batter performance)
- Pip package manager

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/MatriX1232/VectorView.git
    cd VectorView
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running Locally

1.  Configure the application:

    Modify the configuration variables in `main.py` to suit your needs. These include:

    -   `IMAGE_FOLDER`: The path to the folder containing the images to process.
    -   `OUTPUT_FOLDER`: The path to the folder where output files (if any) will be stored.
    -   `IMAGE_SCALE`: The desired scale for resizing images.
    -   `USE_GPU_RESIZE`: A boolean indicating whether to use GPU acceleration for image resizing.
    -   `THREAD_COUNT`: The number of threads to use for parallel processing.
    -   `SEARCH_TOP_K`: The number of results to return for similarity searches.
    -   `EMBEDDING_DIM`: The dimensionality of the image embeddings.
    -   `LANCEDB_DIR`: The directory where the LanceDB database will be stored.
    -   `TABLE_NAME`: The name of the table in the LanceDB database to store image embeddings.

2.  Run the application:

    ```bash
    python main.py
    ```

    This will process the images in the specified `IMAGE_FOLDER`, encode them using the SigLIP model, and store the resulting embeddings in the LanceDB database.

### TUI (Interactive)

An interactive text UI is available via `tui.py`. It uses `prompt_toolkit` and provides simple commands to index images and run text searches.

Run:

```bash
python tui.py
```

Commands:
- `index` — index all images in `IMAGE_FOLDER` (compute embeddings and store in LanceDB).
- `search` — prompt for a text query and run the vector search (results are saved to `OUTPUT`).
- `exit` — quit the TUI.


## 📝 License

This project is licensed under the [MIT License](LICENSE).
