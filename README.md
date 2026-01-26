### 1. Environment Installation
Ensure your system meets the following requirements:
- **Operating System**: Windows 11
- **Hardware**: NFC-enabled smartphone (for data collection) and a PC with GPU (recommended for training)
- **Software**: Git, Conda

#### 1.1 Install Conda
##### Linux
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh

##### Windows
    Download installer from https://docs.conda.io/en/latest/miniconda.html

#### 1.2 Set Up Environment
##### Clone repository
    git clone https://github.com/Leopardcat589/Hornbill-plus.git
    
    cd Hornbill-plus

##### Create Conda environment
    conda env create -f environment.yaml
    
    conda activate hornbill-plus

### 2. Download Datasets
The dataset contains DPV signals for 18 pesticides across 5 concentration gradients (10⁻⁷ to 10⁻¹⁵ mol/L), with 8 replicates per concentration.

#### Download dataset 
        wget https://github.com/Leopardcat589/Hornbill-plus
  
        tar -xzf hornbill_dataset_v1.tar.gz -C ./data/

### 3. Run Inference (Quick Test)

    python src/model/inference.py \
        --model_path checkpoints/best_model.pth \
        --input_file data/raw/example_sample.csv \
        --output_format json
#### Project Structure
        
        Hornbill-plus/
        ├── data/
        │   ├── processed/              # Preprocessed numpy arrays
        │   │   ├── train.npy
        │   │   ├── train_labels.npy
        │   │   ├── test.npy
        │   │   └── test_labels.npy
        │   └── raw/                    # Raw CSV files
        │       ├── pesticide_A/
        │       └── pesticide_B/
        ├── checkpoints/                # Model weights
        ├── configs/                    # Configuration files
        ├── results/                    # Evaluation outputs
        ├── src/
        │   ├── data_preprocessing/     # Data processing scripts
        │   ├── model/                  # Core model code
        │   ├── analysis/               # Visualization tools
        │   └── experiments/            # Experimental scripts
        ├── environment.yaml            # Conda environment
        └── README.md                   # This file

#### 3.1 Data Preprocessing

    cd src/data_preprocessing
    python preprocess_dpv.py \
        --raw_dir ../../data/raw/ \
        --output_dir ../../data/processed/ \
        --crop_range 0.4 0.9 \
        --normalize_method standard
Note: Skip this step if using provided preprocessed .npy files.

#### 3.2 Model Training

    cd src/model
    python train.py --config ../../configs/pyramidal_attention_config.json
    Configuration (configs/pyramidal_attention_config.json):

        json
    {
        "num_scales": 4,
        "num_attention_layers": 4,
        "hidden_dim": 128,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 1000,
        "early_stop_patience": 50
    }

#### 3.3 Model Evaluation

    python test.py \
        --config ../../configs/pyramidal_attention_config.json \
        --checkpoint ../../checkpoints/best_model.pth

    python evaluate.py \
        --results_dir ../../results/ \
        --plot_roc True \
        --plot_confusion True \
        --analyze_mrl True
