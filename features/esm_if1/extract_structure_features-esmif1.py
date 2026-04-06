import os
import gc
import torch
import numpy as np
import esm
import logging
import biotite.structure as bs
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import h5py

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"protein_feature_extraction_{timestamp}.log")
    
    logger = logging.getLogger("protein_feature_extractor")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def extract_and_save_features_combined(pdb_dir, output_file, chain_id=None, force_cpu=False):
    logger = setup_logger()
    logger.info(f"Starting processing directory: {pdb_dir}")
    logger.info(f"Output file: {output_file}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Loading ESM-IF1 model...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    
    device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)

    pdb_files = list(Path(pdb_dir).glob('*.pdb')) + list(Path(pdb_dir).glob('*.cif'))
    logger.info(f"Found {len(pdb_files)} PDB/CIF files")

    with h5py.File(output_file, 'w') as hf:
        metadata_group = hf.create_group('metadata')
        metadata_group.attrs['num_proteins'] = len(pdb_files)
        metadata_group.attrs['creation_date'] = datetime.now().isoformat()
        
        proteins_group = hf.create_group('proteins')
        feature_size = None
        
        success_count = 0
        chain_count = 0
        
        for i, pdb_path in enumerate(tqdm(pdb_files, desc="Processing PDB/CIF files")):
            try:
                pdb_id = pdb_path.stem
                logger.info(f"Processing file {i+1}/{len(pdb_files)}: {pdb_id}")

                if chain_id:
                    chains_to_process = [chain_id]
                else:
                    chains_to_process = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                
                protein_group = proteins_group.create_group(pdb_id)
                protein_chains_found = False
                
                for ch_id in chains_to_process:
                    try:
                        structure = esm.inverse_folding.util.load_structure(str(pdb_path), ch_id)
                        
                        coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
                        
                        if coords is None or seq is None or len(seq) == 0:
                            logger.debug(f"Unable to extract valid coords or sequence from file {pdb_id} chain {ch_id}; chain may not exist")
                            continue
                        
                        logger.info(f"Successfully extracted chain {ch_id} from file {pdb_id}, sequence length: {len(seq)}")
                        
                        if isinstance(coords, torch.Tensor):
                            coords_tensor = coords.clone().detach().to(device)
                        else:
                            coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
                        
                        with torch.no_grad():
                            rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords_tensor)
                        
                        chain_group = protein_group.create_group(f"chain_{ch_id}")
                        features_data = rep.cpu().numpy()
                        chain_group.create_dataset('features', data=features_data)
                        chain_group.create_dataset('sequence', data=np.array(list(seq), dtype='S1'))
                        chain_group.create_dataset('coords', data=coords)
                        
                        if feature_size is None:
                            feature_size = features_data.shape[-1]
                        
                        chain_group.attrs['seq_length'] = len(seq)
                        chain_group.attrs['feature_dim'] = features_data.shape[-1]
                        
                        logger.info(f"Feature shape: {rep.shape}, saved to file")
                        
                        protein_chains_found = True
                        chain_count += 1
                        
                    except Exception as e:
                        if "Chain not found" in str(e) or "No chain" in str(e):
                            logger.debug(f"Chain {ch_id} not found in structure {pdb_id}")
                        else:
                            logger.error(f"Error processing chain {ch_id}: {str(e)}")
                        continue
                    
                    if 'coords_tensor' in locals():
                        del coords_tensor
                    if 'rep' in locals():
                        del rep
                    gc.collect()
                    
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                if protein_chains_found:
                    success_count += 1
                else:
                    del proteins_group[pdb_id]

            except Exception as e:
                logger.error(f"Error processing file {pdb_path.name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        metadata_group.attrs['success_count'] = success_count
        metadata_group.attrs['chain_count'] = chain_count
        metadata_group.attrs['feature_dim'] = feature_size
    
    logger.info(f"Processing complete! Successfully processed {success_count}/{len(pdb_files)} protein structures, total {chain_count} chains")
    logger.info(f"All features saved to file: {output_file}")
    logger.info(f"Feature dimension: {feature_size}")

if __name__ == "__main__":
    pdb_dir = "LNGCN_main/features/esm_if1/pdb"  
    output_file = "LNGCN_main/features/esm_if1/output/test.h5"

    # Run feature extraction and save
    extract_and_save_features_combined(
        pdb_dir=pdb_dir, 
        output_file=output_file,
        chain_id=None, 
        force_cpu=True 
    )