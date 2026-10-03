# LNGCN
A Distance-Aware Dynamics Network for Protein-Protein Interaction Prediction

## 1. Quick Start

### Environment Setup
```bash
# Create environment
conda env create -f environment.yml
```
### Workflow

#### 1.1 Feature Extraction
```bash
# ESM2 feature extraction 
python features/esm2/esm2fea.py
(Note: The esm2_t33_650M_UR50D.pt pre-trained weights need to be downloaded from the official ESM-2 repository before running the code.)

# ESM-if1 feature extraction
python features/esm_if1/extract_structure_features-esmif1.py

# DSSP secondary structure feature extraction
python features\dssp\robust_batch_process_linux.py

# FreeSASA relative solvent accessibility calculation
python features\freesasa\run_rsa_linux.py

# Build adjacency matrix
python features/adj-matrix-rsa.py
```

#### 1.2 Model Training
```bash
# Train on balanced data (5-fold)
python model/main/main_model.py

# Train on imbalanced data
python model/imbalance/imbalance_train.py
```

#### 1.3 Generalization Evaluation
```bash
# Evaluate generalization on yeast data (using models saved from balanced-data training)
python model/generalization/generalization_evaluation.py
# Evaluate performance on imbalanced human data (using models saved from balanced-data training)
python model/imbalance/imbalance_human_evaluation.py
```

#### 1.4 Ablation Experiments
```bash
# Run ablation experiments to validate feature importance
python model/ablation_experiments/ablation_human/run_ablation.py
```

#### 1.5 Model Calibration
```bash
# Calibrate trained models
python model/calibration_and_predictor/calibrate_models.py
```

#### 1.6 Prediction
```bash
# Run prediction using the calibrated predictor
python model/calibration_and_predictor/ppi_predictor.py
```

## 2. Protein Interaction Pair Datasets
### Data Sources
STRING Database(https://string-db.org/)
UniProt(https://www.uniprot.org/)
### Dataset Details
  Contains 92,337 positive and 92,337 negative human PPI samples. This dataset serves as the primary human benchmark from which the C1–C4 evaluation settings are constructed.
#### C1: Random edge-level split (`C1`)  
  Conventional random interaction-level benchmark using five-fold cross-validation. Protein identities may be shared across the training, validation, and test subsets because the split is performed at the interaction-edge level. This setting is mainly used to evaluate predictive performance under the standard random-split condition.
#### C2: One-unseen-protein split (`C2`)  
  Evaluates generalization when each test interaction contains one protein observed during training and one previously unseen protein. Three independent constrained train/validation/test partitions are provided, with an approximate 8:1:1 ratio.
#### C3: Protein-disjoint split (`C3`)  
  Uses protein-disjoint partitions in which both proteins in each test interaction are absent from the training set. Three independent constrained train/validation/test partitions are provided, with an approximate 8:1:1 ratio.
#### C4: Protein-, homology-, and sequence-controlled split (`C4`)  
  Provides the most stringent generalization setting. In addition to protein-level separation, sequence- and structure-level relatedness across subsets is controlled using MMseqs2- and Foldseek-based exclusion components. Cross-subset protein pairs satisfying the predefined sequence-similarity criterion of ≥30% sequence identity and ≥80% bidirectional coverage are excluded. Three independent constrained train/validation/test partitions are provided, with an approximate 8:1:1 ratio.
#### Imbalanced human protein–protein interactions (`human_imbalance`)  
  Contains 3,000 positive and 29,724 negative samples, corresponding to an approximately 1:10 positive-to-negative ratio. This dataset is used to evaluate LNGCN under class imbalance, including five-fold evaluation and candidate-ranking analyses.
#### Yeast protein–protein interactions (`yeast`)  
  Contains positive and negative yeast PPI samples and is used as an independent external dataset for human-to-yeast transfer evaluation. Models trained on human data are directly evaluated on this dataset without additional retraining or fine-tuning on yeast samples.
#### Ablation experiment data (`ablation_experiments`)  
  Uses the predefined training, validation, and test subsets from C1 Fold 5 of the balanced human benchmark. These data are used to evaluate the contributions of individual LNGCN components under an identical data partition.
#### Calibration dataset (`calibration`)  
  Contains 3,000 positive and 3,000 negative protein pairs and is used to evaluate and fit the probability-calibration strategies applied to LNGCN outputs.

## 3. Protein Feature Extraction
### 3.1 ESM2 Sequence Feature Extraction
Path: `features/esm2/esm2fea.py`
Source: https://github.com/facebookresearch/fair-esm
### 3.2 ESM-IF1 Structure Feature Extraction
Path: `features/esm_if1/extract_structure_features-esmif1.py`
Source: https://github.com/facebookresearch/fair-esm
### 3.3 DSSP Secondary Structure Features
Path: `features/dssp/`
Source: https://github.com/PirxLab/DSSP
### 3.4 FreeSASA Relative Solvent Accessibility
Path: `features/freesasa/`
Source: https://github.com/mittinatten/freesasa
### 3.5 Protein Adjacency Matrix Generation
Path: `features/adj-matrix-rsa.py`
Processing workflow: Provide the extracted feature paths and specific protein IDs; the script outputs NPZ-format graph data for each corresponding protein, including node features and edge information.

## Reference Resources
· ESM2 sequence features (https://github.com/facebookresearch/fair-esm)
· ESM-IF1 structure features (https://github.com/facebookresearch/fair-esm)
· DSSP secondary structure (https://github.com/PirxLab/DSSP)
· FreeSASA relative accessibility (https://github.com/mittinatten/freesasa)
· STRING database (https://string-db.org/)
· UniProt database (https://www.uniprot.org/)
· Negatome database (https://mips.helmholtz-muenchen.de/proj/ppi/negatome)
· AlphaFold database (https://alphafold.ebi.ac.uk/)
· DGL graph neural network (https://www.dgl.ai/)

## License and Citation
If you use this project in your research, please cite this weblink and thanks Yueming Xiao, Yifan Zheng, Yu Hua, Jiahua Peng, Jinliang Liu, Yuan Qu, Jizhuang Xu, Rao Fu, Qiuting Qian, Make Zhao, Xinxin Zhang, Jingjing Zhao, Yifei Yao, Martin Kosar*, Yuehai Ke*, Ying Chi* with Department of Pharmacy of the Second Affiliated Hospital of Zhejiang University School of Medicine, and Zhejiang University-University of Edinburgh Institute (ZJE), Zhejiang University, No.866 Yu Hang Tang road, Hangzhou, 310058, Zhejiang Province, China.
