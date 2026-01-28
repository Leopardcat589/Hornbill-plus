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
        
        Hornbill-plus/ClassificationAlgorithm
        ├── Data/
        │   ├── 3 mixing ratios             
        │   ├── 9 combinations  
        │   ├── Fen-A   
        │   ├── Fen-B
             ......   
        ├── Code/                
        │   ├── data_provider             
        │   ├── exp  
        │   ├── layers   
        │   ├── models
        │   ├── pip             
        │   ├── scripts 
        │   ├── utils   
        │   ├── Pesticide_TEST.ts
        │   ├── Pesticide_TRAIN.ts
        │   ├── ROC.py
        │   ├── requirements.txt   
        │   ├── run.py
        └── README.md                   # This file

#### 3.1 Data Preprocessing

    cd ClassificationAlgorithm/Data
    python preprocess_dpv.py \
        --raw_dir ../../data/raw/ \
        --output_dir ../../data/processed/ \
        --crop_range 0.4 0.9 \
        --normalize_method standard

#### 3.2 Model Training

    cd ClassificationAlgorithm/Code/scripts
    python train.py --config classification\Pyraformer.sh
    Configuration (configs/classification\Pyraformer.sh):

        json
    {
        python -u run.py \
          --task_name classification \
          --is_training 1 \
          --root_path ./dataset/EthanolConcentration/ \
          --model_id EthanolConcentration \
          --model $model_name \
          --data UEA \
          --e_layers 3 \
          --batch_size 4 \
          --d_model 128 \
          --d_ff 256 \
          --top_k 3 \
          --des 'Exp' \
          --itr 1 \
          --learning_rate 0.001 \
          --train_epochs 100 \
          --patience 10
    }

#### 3.3 Model Evaluation

    python Pesticide_Test.ts \
        @problemName Pesticide
        @timeStamps false
        @missing false
        @univariate false
        @dimensions 3
        @equalLength true
        @seriesLength 630                    # Number of types
        @classLabel true  Ace7 Ace9 Ace11 Ace13 Ace15 Ani7 Ani9 Ani11 Ani13 Ani15 Chl7 Chl9 Chl11 Chl13 Chl15 Dim7 Dim9 Dim11 Dim13 Dim15 Gly7 Gly9 Gly11 Gly13 Gly15 Ipr7 Ipr9 Ipr11 Ipr13 Ipr15 Met7 Met9 Met11 Met13 Met15 Pro7 Pro9 Pro11 Pro13 Pro15 Tri7 Tri9 Tri11 Tri13 Tri15 Bro7 Bro9 Bro11 Bro13 Bro15 Pho7 Pho9 Pho11 Pho13 Pho15 Fen7 Fen9 Fen11 Fen13 Fen15 Pri7 Pri9 Pri11 Pri13 Pri15 Thi7 Thi9 Thi11 Thi13 Thi15 Prop7 Prop9 Prop11 Prop13 Prop15 Thio7 Thio9 Thio11 Thio13 Thio15 Tsum7 Tsum9 Tsum11 Tsum13 Tsum15 Phos7 Phos9 Phos11 Phos13 Phos15   
        @data                              # data from Pesticides

    python Pesticide_Train.ts \
        @problemName Pesticide
        @timeStamps false
        @missing false
        @univariate false
        @dimensions 3
        @equalLength true
        @seriesLength 90                    # Number of types
        @classLabel true  Ace7 Ace9 Ace11 Ace13 Ace15 Ani7 Ani9 Ani11 Ani13 Ani15 Chl7 Chl9 Chl11 Chl13 Chl15 Dim7 Dim9 Dim11 Dim13 Dim15 Gly7 Gly9 Gly11 Gly13 Gly15 Ipr7 Ipr9 Ipr11 Ipr13 Ipr15 Met7 Met9 Met11 Met13 Met15 Pro7 Pro9 Pro11 Pro13 Pro15 Tri7 Tri9 Tri11 Tri13 Tri15 Bro7 Bro9 Bro11 Bro13 Bro15 Pho7 Pho9 Pho11 Pho13 Pho15 Fen7 Fen9 Fen11 Fen13 Fen15 Pri7 Pri9 Pri11 Pri13 Pri15 Thi7 Thi9 Thi11 Thi13 Thi15 Prop7 Prop9 Prop11 Prop13 Prop15 Thio7 Thio9 Thio11 Thio13 Thio15 Tsum7 Tsum9 Tsum11 Tsum13 Tsum15 Phos7 Phos9 Phos11 Phos13 Phos15    
        @data                              # data from Pesticides
