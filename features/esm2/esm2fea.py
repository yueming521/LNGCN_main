import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from Bio import SeqIO
import pickle
import esm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ESMFeatureExtractor")

def load_esm_model(model_path, gpu_id=0):
    """Load the ESM model onto the specified GPU"""
    logger.info(f"GPU {gpu_id} - Loading ESM model: {model_path}")
    try:
        # Load model
        model_data = torch.load(model_path, map_location=f"cuda:{gpu_id}", weights_only=False)
        
        # Extract model name
        model_location = Path(model_path)
        model_name = model_location.stem
        
        # Check regression weights
        regression_data = None
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        try:
            regression_data = torch.load(regression_location, map_location=f"cuda:{gpu_id}")
        except Exception:
            pass  # If no regression data exists, that's fine
        
        # Load model core
        from esm.pretrained import load_model_and_alphabet_core
        model, alphabet = load_model_and_alphabet_core(model_name, model_data, regression_data)
        model.eval()
        
        # Move model to the selected GPU
        if torch.cuda.is_available():
            model = model.cuda(gpu_id)
            logger.info(f"GPU {gpu_id} - Model loaded")
            
        return model, alphabet
    except Exception as e:
        logger.error(f"GPU {gpu_id} - Model load failed: {str(e)}")
        logger.info(f"GPU {gpu_id} - Attempting to load pretrained model...")
        
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda(gpu_id)
            
        return model, alphabet

def read_fasta_files(fasta_dir):
    sequences = []
    ids = []
    file_paths = []
    
    logger.info(f"Reading FASTA files from directory: {fasta_dir}")
    fasta_files = list(Path(fasta_dir).glob("*.fasta")) + list(Path(fasta_dir).glob("*.fa"))
    
    if not fasta_files:
        logger.warning(f"No FASTA files found in {fasta_dir}")
        return [], [], []
    
    for fasta_file in tqdm(fasta_files, desc="Reading FASTA files"):
        try:
            for record in SeqIO.parse(str(fasta_file), "fasta"):
                sequences.append(str(record.seq))
                ids.append(record.id)
                file_paths.append(str(fasta_file))
        except Exception as e:
            logger.error(f"Error reading file {fasta_file}: {str(e)}")
    
    logger.info(f"Successfully read {len(sequences)} sequences")
    return sequences, ids, file_paths

@torch.no_grad()
def extract_features(model, alphabet, sequences, ids, batch_size=1, gpu_id=0):
    """Extract features using the ESM-2 model"""
    all_results = []
    
    # Convert sequences and IDs into ESM input format
    data = list(zip(ids, sequences))
    
    # Use batch_converter
    batch_converter = alphabet.get_batch_converter()
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch_idx = i // batch_size + 1
        batch_data = data[i:i + batch_size]
        logger.info(f"GPU {gpu_id} - Processing batch {batch_idx}/{(len(data) + batch_size - 1) // batch_size} ({len(batch_data)} sequences)")
        
        try:
            # Convert sequences to tokens with batch_converter
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.cuda(gpu_id)
                
            # Run model to get representations, disable contact map computation to save memory
            results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
            
            # Process features for each sequence
            for j, (seq_id, seq) in enumerate(batch_data):
                # Get the last layer representation
                token_representations = results["representations"][model.num_layers][j]
                
                # Remove special start/end tokens, keep only amino acid embeddings
                seq_len = len(seq)
                seq_representations = token_representations[1:seq_len+1].cpu().numpy()
                
                # Compute average representation for the sequence
                mean_representation = seq_representations.mean(axis=0)
                
                # Save result
                all_results.append({
                    "id": seq_id,
                    "sequence": seq,
                    "per_residue": seq_representations,
                    "mean": mean_representation,
                })
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} - Error processing batch {batch_idx}: {str(e)}")
            logger.exception("Detailed traceback:")
    
    logger.info(f"GPU {gpu_id} - Successfully extracted features for {len(all_results)} sequences")
    return all_results

def process_split(gpu_id, seq_chunk, id_chunk, model_path):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Load model
    model, alphabet = load_esm_model(model_path, gpu_id)
    
    # Extract features
    return extract_features(model, alphabet, seq_chunk, id_chunk, batch_size=1, gpu_id=gpu_id)

# 定义进程目标函数 - 替换lambda
def process_gpu0(result_queue, sequences, ids, model_path):
    result = process_split(0, sequences, ids, model_path)
    result_queue.put(result)

def process_gpu1(result_queue, sequences, ids, model_path):
    result = process_split(1, sequences, ids, model_path)
    result_queue.put(result)

def main():
    parser = argparse.ArgumentParser(description="Extract protein sequence features using ESM-2 (low memory)")
    parser.add_argument("--model_path", type=str, 
                        default="LNGCN_main/features/esm2/esm2_t33_650M_UR50D.pt",
                        help="Path to the ESM model")
    parser.add_argument("--fasta_dir", type=str, 
                        default="LNGCN_main/features/esm2/fasta",
                        help="Directory containing FASTA files")
    parser.add_argument("--output_dir", type=str, 
                        default="LNGCN_main/features/esm2/output",
                        help="Directory for feature outputs")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        logger.error("At least 2 GPUs are required to run this script")
        return
    
    # Create output directory and ensure it exists
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {args.output_dir}")
        
        # Test if output directory is writable
        test_file = os.path.join(args.output_dir, "test_write.txt")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info("Output directory is writable")
        except Exception as e:
            logger.error(f"Output directory is not writable: {str(e)}")
            return
    except Exception as e:
        logger.error(f"Failed to create output directory: {str(e)}")
        return
    
    # Read FASTA files
    sequences, ids, _ = read_fasta_files(args.fasta_dir)
    
    if not sequences:
        logger.error("No sequences found, exiting")
        return
    
    # Split data into two parts
    mid_point = len(sequences) // 2
    sequences_1, sequences_2 = sequences[:mid_point], sequences[mid_point:]
    ids_1, ids_2 = ids[:mid_point], ids[mid_point:]
    
    logger.info(f"Data split into two parts: GPU 0 handles {len(sequences_1)} sequences, GPU 1 handles {len(sequences_2)} sequences")
    
    # Process data on two GPUs
    import torch.multiprocessing as mp
    
    # Use spawn to avoid CUDA initialization issues
    mp.set_start_method('spawn', force=True)
    
    # Create shared queue for results
    result_queue = mp.Queue()
    
    p1 = mp.Process(target=process_gpu0, args=(result_queue, sequences_1, ids_1, args.model_path))
    p2 = mp.Process(target=process_gpu1, args=(result_queue, sequences_2, ids_2, args.model_path))
    
    p1.start()
    p2.start()
    
    # Collect results
    results1 = result_queue.get()
    results2 = result_queue.get()
    
    # Wait for processes to complete
    p1.join()
    p2.join()
    
    # Merge results
    all_results = results1 + results2
    
    # 保存特征
    output_file = os.path.join(args.output_dir, "esm_features.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)
    
    logger.info(f"Features saved to: {output_file}")
    
    # Also save as numpy arrays
    mean_embeddings = np.array([r["mean"] for r in all_results])
    ids_array = np.array([r["id"] for r in all_results])
    sequences_array = np.array([r["sequence"] for r in all_results])
    
    np.save(os.path.join(args.output_dir, "esm_mean_embeddings.npy"), mean_embeddings)
    np.save(os.path.join(args.output_dir, "esm_ids.npy"), ids_array)
    np.save(os.path.join(args.output_dir, "esm_sequences.npy"), sequences_array)
    
    logger.info(f"Feature dimensions: {mean_embeddings.shape}")
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()