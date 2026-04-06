import freesasa
import os
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_complete.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class BatchCompleteProcessor:
    def __init__(self, id_file, pdb_dir, output_dir, max_workers=4):
        self.id_file = id_file
        self.pdb_dir = pdb_dir
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.failed_files = []
        self.processed_count = 0
        self.total_files = 0
        self.lock = threading.Lock()
        self.fail_file = os.path.join(output_dir, 'fail.txt')
        
    def get_naccess_classifier(self):
        try:
            classifier = freesasa.Classifier()
            naccess_classifier = classifier.getStandardClassifier('naccess')
            return naccess_classifier
        except Exception as e:
            logging.warning(f"Could not get NACCESS classifier: {e}")
            return None
    
    def setup_naccess_parameters(self):
        params = freesasa.Parameters({
            'algorithm': freesasa.LeeRichards,
            'probe-radius': 1.4,
            'n-slices': 20
        })
        return params
    
    def load_id_list(self):
        try:
            with open(self.id_file, 'r', encoding='utf-8') as f:
                ids = [line.strip() for line in f if line.strip()]
            return ids
        except Exception as e:
            logging.error(f"Error loading ID file {self.id_file}: {e}")
            return []
    
    def process_single_pdb(self, pdb_id):
        rsa_file = os.path.join(self.output_dir, f"{pdb_id}_complete.rsa")
        if os.path.exists(rsa_file):
            with self.lock:
                self.processed_count += 1
            return True, pdb_id, "Already processed (skipped)"

        pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_file):
            error_msg = f"PDB file not found: {pdb_file}"
            logging.error(error_msg)
            with self.lock:
                self.failed_files.append(pdb_id)
                self.processed_count += 1
            return False, pdb_id, "PDB file not found"
        try:
            naccess_classifier = self.get_naccess_classifier()
            params = self.setup_naccess_parameters()
            if naccess_classifier:
                structure = freesasa.Structure(pdb_file, naccess_classifier)
            else:
                structure = freesasa.Structure(pdb_file)
            result = freesasa.calc(structure, params)
            self.generate_complete_output_files(structure, result, pdb_id, naccess_classifier)
            with self.lock:
                self.processed_count += 1
                if self.processed_count % 50 == 0:
                    logging.info(f"Processed {self.processed_count}/{self.total_files} files")
            return True, pdb_id, None
        except Exception as e:
            error_msg = f"Error processing {pdb_id}: {str(e)}"
            logging.error(error_msg)
            with self.lock:
                self.failed_files.append(pdb_id)
                self.processed_count += 1
            return False, pdb_id, str(e)
    
    def generate_complete_output_files(self, structure, result, pdb_id, classifier):
        # 1. Generate complete RSA file
        rsa_file = os.path.join(self.output_dir, f"{pdb_id}_complete.rsa")
        self.generate_complete_rsa(structure, result, rsa_file, classifier)
        
        # 2. Generate complete atom file
        atoms_file = os.path.join(self.output_dir, f"{pdb_id}_atoms_complete.txt")
        self.generate_complete_atoms(structure, result, atoms_file, classifier)
        
        # 3. Generate complete residue file
        residues_file = os.path.join(self.output_dir, f"{pdb_id}_residues_complete.txt")
        self.generate_complete_residues(structure, result, residues_file)
        
        # 4. Generate statistics report
        stats_file = os.path.join(self.output_dir, f"{pdb_id}_statistics.txt")
        self.generate_statistics_report(structure, result, stats_file, classifier, pdb_id)
    
    def generate_complete_rsa(self, structure, result, output_file, classifier):
        residues = {}
        for i in range(structure.nAtoms()):
            chain_id = structure.chainLabel(i)
            residue_number = structure.residueNumber(i)
            residue_name = structure.residueName(i)
            key = (chain_id, residue_number, residue_name)
            if key not in residues:
                residues[key] = []
            residues[key].append(i)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("REM  FreeSASA - COMPLETE output (ALL residues)\n")
            f.write("REM  Lee-Richards algorithm, probe radius 1.40A\n")
            f.write("REM  NACCESS atomic radii used\n")
            f.write("REM  Includes residues with SASA = 0.00 (completely buried)\n")
            f.write("REM\n")
            f.write(f"REM  Total accessible surface area: {result.totalArea():.2f} A^2\n")
            f.write(f"REM  Total residues: {len(residues)}\n")
            f.write("REM\n")
            f.write("RES  Chain Residue    Num  Total_SASA\n")
            for (chain_id, residue_number, residue_name), atom_indices in sorted(residues.items()):
                total_area = sum(result.atomArea(i) for i in atom_indices)
                f.write(f"RES  {chain_id:>1} {residue_name:>3} {residue_number:>4} {total_area:>10.2f}\n")
            f.write("END\n")

    def generate_complete_atoms(self, structure, result, output_file, classifier):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Complete atom-by-atom SASA details (ALL atoms)\n")
            f.write("# Includes atoms with SASA = 0.00 (completely buried)\n")
            f.write("# Chain Residue ResNum AtomName    SASA   Radius  Class\n")
            for i in range(structure.nAtoms()):
                atom_area = result.atomArea(i)
                chain_id = structure.chainLabel(i)
                residue_name = structure.residueName(i)
                residue_number = structure.residueNumber(i)
                atom_name = structure.atomName(i)
                if classifier:
                    try:
                        radius = classifier.radius(residue_name, atom_name)
                        classification = classifier.classify(residue_name, atom_name)
                        f.write(f"{chain_id:>5} {residue_name:>7} {residue_number:>6} {atom_name:>8} {atom_area:>8.2f} {radius:>7.2f} {classification:>8}\n")
                    except:
                        f.write(f"{chain_id:>5} {residue_name:>7} {residue_number:>6} {atom_name:>8} {atom_area:>8.2f}    N/A      N/A\n")
                else:
                    f.write(f"{chain_id:>5} {residue_name:>7} {residue_number:>6} {atom_name:>8} {atom_area:>8.2f}    N/A      N/A\n")
    
    def generate_complete_residues(self, structure, result, output_file):
        residues = {}
        for i in range(structure.nAtoms()):
            chain_id = structure.chainLabel(i)
            residue_number = structure.residueNumber(i)
            residue_name = structure.residueName(i)
            key = (chain_id, residue_number, residue_name)

            if key not in residues:
                residues[key] = []
            residues[key].append(i)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Complete residue-by-residue SASA summary (ALL residues)\n")
            f.write("# Includes residues with SASA = 0.00 (completely buried)\n")
            f.write("# Chain Residue ResNum    Total_SASA  Num_Atoms  Avg_SASA  Status\n")
            total_sasa = 0
            surface_residues = 0
            buried_residues = 0
            for (chain_id, residue_number, residue_name), atom_indices in sorted(residues.items()):
                residue_area = sum(result.atomArea(i) for i in atom_indices)
                avg_area = residue_area / len(atom_indices) if atom_indices else 0
                total_sasa += residue_area
                if residue_area > 0.01:
                    status = "Surface"
                    surface_residues += 1
                else:
                    status = "Buried"
                    buried_residues += 1
                f.write(f"{chain_id:>5} {residue_name:>7} {residue_number:>6} {residue_area:>12.2f} {len(atom_indices):>9} {avg_area:>8.2f} {status:>8}\n")
            f.write(f"\n# Summary:\n")
            f.write(f"# Total SASA: {total_sasa:.2f} A^2\n")
            f.write(f"# Number of residues: {len(residues)}\n")
            f.write(f"# Surface residues (SASA > 0.01): {surface_residues}\n")
            f.write(f"# Buried residues (SASA ≤ 0.01): {buried_residues}\n")
    
    def generate_statistics_report(self, structure, result, output_file, classifier, pdb_id):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"FreeSASA Complete Analysis Statistics - {pdb_id}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Structure Information:\n")
            f.write(f"  Total atoms: {structure.nAtoms()}\n")
            f.write(f"  Total SASA: {result.totalArea():.2f} A^2\n\n")
            atoms_with_sasa = 0
            atoms_buried = 0
            sasa_values = []
            for i in range(structure.nAtoms()):
                sasa = result.atomArea(i)
                sasa_values.append(sasa)
                if sasa > 0.001:
                    atoms_with_sasa += 1
                else:
                    atoms_buried += 1
            f.write("Atom Statistics:\n")
            f.write(f"  Atoms with SASA > 0: {atoms_with_sasa} ({atoms_with_sasa/structure.nAtoms()*100:.1f}%)\n")
            f.write(f"  Completely buried atoms: {atoms_buried} ({atoms_buried/structure.nAtoms()*100:.1f}%)\n")
            f.write(f"  Average SASA per atom: {sum(sasa_values)/len(sasa_values):.2f} A^2\n")
            f.write(f"  Maximum atom SASA: {max(sasa_values):.2f} A^2\n\n")
            residues = {}
            for i in range(structure.nAtoms()):
                chain_id = structure.chainLabel(i)
                residue_number = structure.residueNumber(i)
                residue_name = structure.residueName(i)
                key = (chain_id, residue_number, residue_name)
                if key not in residues:
                    residues[key] = []
                residues[key].append(i)
            surface_residues = 0
            buried_residues = 0
            residue_sasa_values = []
            for (chain_id, residue_number, residue_name), atom_indices in residues.items():
                residue_area = sum(result.atomArea(i) for i in atom_indices)
                residue_sasa_values.append(residue_area)
                if residue_area > 0.01:
                    surface_residues += 1
                else:
                    buried_residues += 1
            f.write("Residue Statistics:\n")
            f.write(f"  Total residues: {len(residues)}\n")
            f.write(f"  Surface residues (SASA > 0.01): {surface_residues} ({surface_residues/len(residues)*100:.1f}%)\n")
            f.write(f"  Buried residues (SASA ≤ 0.01): {buried_residues} ({buried_residues/len(residues)*100:.1f}%)\n")
            f.write(f"  Average SASA per residue: {sum(residue_sasa_values)/len(residue_sasa_values):.2f} A^2\n")
            f.write(f"  Maximum residue SASA: {max(residue_sasa_values):.2f} A^2\n\n")
            if classifier:
                f.write("NACCESS Classifier Information:\n")
                f.write("  Successfully loaded NACCESS atomic radii\n")
                f.write("  Equivalent to --radii=naccess option\n\n")
            
            f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def save_failed_files(self):
        if self.failed_files:
            with open(self.fail_file, 'w', encoding='utf-8') as f:
                f.write("# Failed PDB IDs\n")
                f.write(f"# Total failed: {len(self.failed_files)}\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("#\n")
                for pdb_id in self.failed_files:
                    f.write(f"{pdb_id}\n")
            logging.info(f"Failed files saved to: {self.fail_file}")
    
    def process_all_pdbs(self):
        id_list = self.load_id_list()
        if not id_list:
            logging.error(f"No IDs found in {self.id_file}")
            return
        self.total_files = len(id_list)
        logging.info(f"Found {self.total_files} IDs to process")
        logging.info(f"Using {self.max_workers} threads")
        logging.info(f"Output directory: {self.output_dir}")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_pdb, pdb_id): pdb_id 
                for pdb_id in id_list
            }
            for future in as_completed(future_to_id):
                pdb_id = future_to_id[future]
                try:
                    success, processed_id, error = future.result()
                    if not success:
                        logging.error(f"Failed to process {processed_id}: {error}")
                except Exception as exc:
                    logging.error(f"Exception processing {pdb_id}: {exc}")
                    with self.lock:
                        self.failed_files.append(pdb_id)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.generate_final_report(elapsed_time)
        self.save_failed_files()
    
    def generate_final_report(self, elapsed_time):
        successful = self.total_files - len(self.failed_files)
        report_file = os.path.join(self.output_dir, 'processing_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FreeSASA Complete Batch Processing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"ID file: {self.id_file}\n")
            f.write(f"PDB directory: {self.pdb_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Number of threads: {self.max_workers}\n\n")
            f.write("Results:\n")
            f.write(f"  Total IDs: {self.total_files}\n")
            f.write(f"  Successful: {successful}\n")
            f.write(f"  Failed: {len(self.failed_files)}\n")
            f.write(f"  Success rate: {successful/self.total_files*100:.1f}%\n\n")
            f.write(f"Processing time: {elapsed_time:.2f} seconds\n")
            f.write(f"Average time per file: {elapsed_time/self.total_files:.2f} seconds\n\n")
            f.write("Parameters used:\n")
            f.write("  Algorithm: Lee-Richards\n")
            f.write("  Probe radius: 1.4 A\n")
            f.write("  Atomic radii: NACCESS compatible (--radii=naccess)\n")
            f.write("  Output format: Complete (includes all atoms and residues)\n")
        logging.info(f"Processing completed!")
        logging.info(f"Total: {self.total_files}, Successful: {successful}, Failed: {len(self.failed_files)}")
        logging.info(f"Processing time: {elapsed_time:.2f} seconds")
        logging.info(f"Report saved to: {report_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='FreeSASA Complete Batch Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Batch processor based on freesasa_complete_output.py:
- Reads ID list from yourid.txt
- Processes corresponding PDB files in pdb directory
- Generates complete output (including all atoms and residues)
- Directly saves in youroutput directory without creating subfolders

Examples:
  python batch_complete_processor.py
  python batch_complete_processor.py --threads 8
        """
    )
    parser.add_argument('--id-file', default='yourid.txt', help='ID列表文件 (默认: yourid.txt)')
    parser.add_argument('--pdb-dir', default='pdb', help='PDB文件目录 (默认: pdb)')
    parser.add_argument('--output-dir', default='youroutput', help='输出目录 (默认: youroutput)')
    parser.add_argument('--threads', '-t', type=int, default=6, help='线程数 (默认: 6)')
    args = parser.parse_args()
    if not os.path.exists(args.id_file):
        print(f"Error: ID file '{args.id_file}' not found")
        sys.exit(1)
    if not os.path.exists(args.pdb_dir):
        print(f"Error: PDB directory '{args.pdb_dir}' not found")
        sys.exit(1)
    processor = BatchCompleteProcessor(
        id_file=args.id_file,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        max_workers=args.threads
    )
    try:
        processor.process_all_pdbs()
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        processor.save_failed_files()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        processor.save_failed_files()
if __name__ == "__main__":
    main()
