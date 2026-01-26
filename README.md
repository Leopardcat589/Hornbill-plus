1. Environment Installation
Before you begin, ensure you meet the following setup requirements:

Operating System: Windows  11

1.1 Install Python and Conda
It's recommended to use Miniconda or Anaconda to manage the Python environment.

bash
# For Windows, download and run the installer from:
# https://docs.conda.io/en/latest/miniconda.html
# For Linux:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
1.2 Set Up the Python Environment
bash
# Clone the Hornbill+ repository
git clone [URL of the Hornbill+ GitHub repository]
cd Hornbill-plus

# Create and activate a dedicated Conda environment
conda create -n hornbill python=[Specify version, e.g., 3.8]
conda activate hornbill

# Install core dependencies from requirements.txt (if provided)
pip install -r requirements.txt

# Alternatively, install key packages manually (example - please verify)
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install [Other potential packages: scipy, tensorboard, pyarrow]
2. Download and Prepare the Dataset
The model is trained and evaluated on electrochemical signal data from 18 pesticides at various concentrations.

Dataset Description: The dataset consists of Differential Pulse Voltammetry (DPV) signals. Each sample is a time-series current response corresponding to a specific pesticide and concentration level. The dataset is split across 18 pesticides, with [number] concentrations per pesticide and [number] replicates.

Download Data: Please download the dataset from [Provide link to dataset, e.g., Zenodo, Figshare, or Google Drive].

Data Structure: After downloading and extracting, organize your data as follows:

text
Hornbill-plus/
├── data/
│   ├── raw/           # Raw .csv or .txt DPV signal files
│   ├── processed/     # Preprocessed numpy arrays (.npy files)
│   └── splits/        # Train/Val/Test indices (if provided)
├── notebooks/         # Jupyter notebooks for exploration
└── src/               # Source code
3. Step-by-Step Instructions
Disclaimer: This is a research prototype. Paths in configuration files may need adjustment for your local setup. Please raise an issue on GitHub for any bugs encountered.

3.1 Data Preprocessing
(Approximate runtime: ~[X] minutes using a CPU)
The raw electrochemical signals require cropping, normalization, and formatting.

bash
# Navigate to the source directory
cd src/data_preprocessing

# Run the main preprocessing script
python preprocess_dpv_signals.py \
    --raw_data_path ../data/raw/ \
    --output_path ../data/processed/ \
    --crop_range 0.4 0.9  # Crop to relevant voltage window (V)
    --normalize standard   # Normalization method
Note: Preprocessed data is also available at [Link to processed data]. You can use it directly to skip this step.

3.2 Feature Extraction & Model Training
(Approximate runtime: ~[X] hours using a GPU, ~[Y] hours using a CPU)
The core model uses a Pyramidal Attention mechanism to learn multi-scale features from the dual-enzyme DPV sequences.

★ Train the Pyramidal Attention Model:

bash
# Ensure you are in the project root and the 'hornbill' environment is active
cd src/model

# Train the model. Hyperparameters are defined in a config file.
python train.py --config ../configs/pyramidal_attention_train.json

# For GPU training, specify the device ID
python train.py --gpu_id 0 --config ../configs/pyramidal_attention_train.json
Configuration: Key parameters like the number of attention scales (S), layers (N), learning rate, and batch size are set in the JSON config file (configs/ directory).

3.3 Evaluation and Testing
(Approximate runtime: ~[X] minutes)
Evaluate the trained model on the held-out test set to reproduce the paper's results (e.g., 93.75% accuracy).

★ Test the Model and Generate Metrics:

bash
# Test the model
python test.py \
    --config ../configs/pyramidal_attention_test.json \
    --checkpoint ../checkpoints/best_model.pth  # Path to saved model

# Generate the comprehensive evaluation results
python evaluate.py \
    --results_dir ../results/ \
    --plot_roc True  # Generates ROC curves and calculates AUC
    --plot_confusion True  # Generates confusion matrices
Output: Results, including accuracy tables, ROC curves (AUC ~0.79), and confusion matrices, will be saved in the results/ directory.

3.4 Running Inference on New Data
Use the trained model to predict pesticide types and concentrations from new DPV signal files.

bash
python inference.py \
    --model_checkpoint ../checkpoints/best_model.pth \
    --input_file ./path_to_your_new_dpv_data.csv \
    --output_file ./prediction_results.json
To complete this guide accurately, I recommend you directly check the README.md or requirements.txt files in the Hornbill-plus repository for the exact commands and dependencies. If the repository is private or the link has changed, please verify the URL.
