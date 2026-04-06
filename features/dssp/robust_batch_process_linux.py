import os
import sys
import subprocess
import threading
import time
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

class RobustDSSPBatchProcessor:
    def __init__(self, protein_ids_file, pdb_dir, output_base_dir, max_workers=4, force_reprocess=False):
        self.protein_ids_file = protein_ids_file
        self.pdb_dir = Path(pdb_dir)
        self.output_base_dir = Path(output_base_dir)
        self.max_workers = max_workers
        self.force_reprocess = force_reprocess
        self.dssp_dir = self.output_base_dir / "DSSP"
        self.standard_mmcif_dir = self.output_base_dir / "standard_mmCIF"
        self.experimental_mmcif_dir = self.output_base_dir / "experimental_mmCIF"
        for dir_path in [self.dssp_dir, self.standard_mmcif_dir, self.experimental_mmcif_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_base_dir / "processing_state.json"
        self.log_file = self.output_base_dir / "processing.log"
        self.failed_dssp = []
        self.failed_standard_mmcif = []
        self.failed_experimental_mmcif = []
        self.lock = threading.Lock()
        self.total_proteins = 0
        self.processed_count = 0
        self.success_dssp = 0
        self.success_standard = 0
        self.success_experimental = 0
        self.processed_proteins = set()
        self.should_stop = False
        self.mkdssp_cmd = shutil.which('mkdssp')
        self.use_wsl_wrapper = False
        if self.mkdssp_cmd is None and os.name == 'nt' and shutil.which('wsl'):
            self.mkdssp_cmd = 'mkdssp'
            self.use_wsl_wrapper = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping safely...")
        self.should_stop = True
        self.save_state()

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        sys.stdout.flush() 
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def save_state(self):
        state = {
            'processed_proteins': list(self.processed_proteins),
            'processed_count': self.processed_count,
            'success_dssp': self.success_dssp,
            'success_standard': self.success_standard,
            'success_experimental': self.success_experimental,
            'failed_dssp': self.failed_dssp,
            'failed_standard_mmcif': self.failed_standard_mmcif,
            'failed_experimental_mmcif': self.failed_experimental_mmcif,
            'timestamp': time.time()
        }
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.processed_proteins = set(state.get('processed_proteins', []))
                self.processed_count = state.get('processed_count', 0)
                self.success_dssp = state.get('success_dssp', 0)
                self.success_standard = state.get('success_standard', 0)
                self.success_experimental = state.get('success_experimental', 0)
                self.failed_dssp = state.get('failed_dssp', [])
                self.failed_standard_mmcif = state.get('failed_standard_mmcif', [])
                self.failed_experimental_mmcif = state.get('failed_experimental_mmcif', [])
                self.log(f"State loaded: {len(self.processed_proteins)} proteins already processed")
                return True
            except Exception as e:
                self.log(f"Failed to load state: {e}")
                return False
        return False

    def check_existing_files(self):
        existing_dssp = set()
        existing_standard = set()
        existing_experimental = set()
        
        # Check DSSP files
        if self.dssp_dir.exists():
            for file in self.dssp_dir.glob("*.dssp"):
                protein_id = file.stem
                existing_dssp.add(protein_id)
        
        # Check standard mmCIF files
        if self.standard_mmcif_dir.exists():
            for file in self.standard_mmcif_dir.glob("*_standard.cif"):
                protein_id = file.stem.replace("_standard", "")
                existing_standard.add(protein_id)
        
        # Check experimental mmCIF files
        if self.experimental_mmcif_dir.exists():
            for file in self.experimental_mmcif_dir.glob("*_experimental.cif"):
                protein_id = file.stem.replace("_experimental", "")
                existing_experimental.add(protein_id)
        self.log(f"Existing files found: DSSP={len(existing_dssp)}, standard mmCIF={len(existing_standard)}, experimental mmCIF={len(existing_experimental)}")
        return existing_dssp, existing_standard, existing_experimental

    def build_mkdssp_command(self, input_file, output_file, format_type):
        # Build mkdssp command (direct call on Linux, optional WSL fallback on Windows)
        if not self.mkdssp_cmd:
            return None
        input_path = str(input_file)
        output_path = str(output_file)
        if format_type == "dssp":
            cmd = [self.mkdssp_cmd, input_path, output_path]
        elif format_type == "standard_mmcif":
            cmd = [self.mkdssp_cmd, '--output-format', 'mmcif', input_path, output_path]
        elif format_type == "experimental_mmcif":
            cmd = [self.mkdssp_cmd, '--output-format', 'mmcif', '--write-other', input_path, output_path]
        else:
            return None
        if self.use_wsl_wrapper:
            cmd = ['wsl'] + cmd
        return cmd

    def fix_pdb_file(self, input_file, output_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            fixed_lines = []
            # If file does not start with HEADER, insert a synthetic header line.
            # This helps mkdssp recognize it as a PDB file instead of parsing as mmCIF.
            if not lines or not lines[0].startswith('HEADER'):
                current_date = time.strftime("%d-%b-%y").upper()
                fixed_lines.append(f"HEADER    GENERATED BY SCRIPT                       {current_date}   XXXX              \n")

            for line in lines:
                # Fix DBREF record issues
                if line.startswith('DBREF'):
                    if '_HUMAN' in line or '_MOUSE' in line or any(f'_{org}' in line for org in ['HUMAN', 'MOUSE', 'RAT', 'YEAST']):
                        line = 'REMARK   1 ' + line[11:].strip() + '\n'

                # Fix ATOM/HETATM record issues
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    if len(line) >= 80:
                        try:
                            occupancy = line[54:60].strip()
                            if occupancy and not occupancy.replace('.', '').replace('-', '').isdigit():
                                line = line[:54] + '  1.00' + line[60:]
                            temp_factor = line[60:66].strip()
                            if temp_factor and not temp_factor.replace('.', '').replace('-', '').isdigit():
                                line = line[:60] + ' 20.00' + line[66:]
                        except:
                            pass
                fixed_lines.append(line)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return True
        except Exception as e:
            self.log(f"Failed to repair PDB file {input_file}: {e}")
            return False

    def load_protein_ids(self):
        protein_ids = []
        with open(self.protein_ids_file, 'r') as f:
            for line in f:
                protein_id = line.strip()
                if protein_id:
                    protein_ids.append(protein_id)
        return protein_ids

    def process_single_format(self, protein_id, format_type):
        pdb_file = self.pdb_dir / f"{protein_id}.pdb"
        if not pdb_file.exists():
            return False, f"PDB file not found: {pdb_file}"

        fixed_pdb_file = self.pdb_dir / f"{protein_id}_fixed.pdb"
        if not self.fix_pdb_file(pdb_file, fixed_pdb_file):
            return False, "PDB repair failed"

        if format_type == "dssp":
            output_file = self.dssp_dir / f"{protein_id}.dssp"
        elif format_type == "standard_mmcif":
            output_file = self.standard_mmcif_dir / f"{protein_id}_standard.cif"
        elif format_type == "experimental_mmcif":
            output_file = self.experimental_mmcif_dir / f"{protein_id}_experimental.cif"
        else:
            return False, f"Unknown format type: {format_type}"

        cmd = self.build_mkdssp_command(fixed_pdb_file, output_file, format_type)
        if not cmd:
            return False, "mkdssp command not found. Please install DSSP (mkdssp) and ensure it is in PATH."

        if output_file.exists() and output_file.stat().st_size > 0:
            fixed_pdb_file.unlink(missing_ok=True)
            return True, "File already exists"

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return False, "Processing timed out"

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            fixed_pdb_file.unlink(missing_ok=True)
            return False, f"mkdssp failed: {err}"

        if not output_file.exists() or output_file.stat().st_size == 0:
            fixed_pdb_file.unlink(missing_ok=True)
            return False, "Output file is empty or missing"

        fixed_pdb_file.unlink(missing_ok=True)
        return True, "Success"
    def process_single_protein(self, protein_id):
        if self.should_stop:
            return None
        results = {
            'protein_id': protein_id,
            'dssp': self.process_single_format(protein_id, "dssp"),
            'standard_mmcif': self.process_single_format(protein_id, "standard_mmcif"),
            'experimental_mmcif': self.process_single_format(protein_id, "experimental_mmcif")
        }
        return results

    def update_progress(self, results):
        if results is None:
            return
        with self.lock:
            protein_id = results['protein_id']
            self.processed_proteins.add(protein_id)
            self.processed_count += 1
            if results['dssp'][0]:
                self.success_dssp += 1
            else:
                self.failed_dssp.append((protein_id, results['dssp'][1]))
            if results['standard_mmcif'][0]:
                self.success_standard += 1
            else:
                self.failed_standard_mmcif.append((protein_id, results['standard_mmcif'][1]))
            if results['experimental_mmcif'][0]:
                self.success_experimental += 1
            else:
                self.failed_experimental_mmcif.append((protein_id, results['experimental_mmcif'][1]))
            progress = (self.processed_count / self.total_proteins) * 100
            self.log(f"Progress: {self.processed_count}/{self.total_proteins} ({progress:.1f}%) | "
                     f"DSSP: {self.success_dssp} | standard mmCIF: {self.success_standard} | "
                     f"experimental mmCIF: {self.success_experimental}")
            if self.processed_count % 50 == 0:
                self.save_state()

    def save_failed_records(self):
        # Save DSSP failures
        if self.failed_dssp:
            with open(self.output_base_dir / "183failed_dssp.txt", 'w', encoding='utf-8') as f:
                f.write("# Proteins failed in DSSP format generation\n")
                f.write("# Format: ProteinID\tErrorMessage\n")
                for protein_id, error in self.failed_dssp:
                    f.write(f"{protein_id}\t{error}\n")
        
        # Save standard mmCIF failures
        if self.failed_standard_mmcif:
            with open(self.output_base_dir / "183failed_standard_mmcif.txt", 'w', encoding='utf-8') as f:
                f.write("# Proteins failed in standard mmCIF generation\n")
                f.write("# Format: ProteinID\tErrorMessage\n")
                for protein_id, error in self.failed_standard_mmcif:
                    f.write(f"{protein_id}\t{error}\n")
        
        # Save experimental mmCIF failures
        if self.failed_experimental_mmcif:
            with open(self.output_base_dir / "183failed_experimental_mmcif.txt", 'w', encoding='utf-8') as f:
                f.write("# Proteins failed in experimental mmCIF generation\n")
                f.write("# Format: ProteinID\tErrorMessage\n")
                for protein_id, error in self.failed_experimental_mmcif:
                    f.write(f"{protein_id}\t{error}\n")

    def run(self):
        self.log("Starting robust batch DSSP processing...")
        self.log(f"Max worker threads: {self.max_workers}")
        if not self.mkdssp_cmd:
            self.log("Error: mkdssp command not found. Install dssp/mkdssp and ensure it is in PATH.")
            return
        if self.use_wsl_wrapper:
            self.log("Windows detected, calling mkdssp via wsl wrapper.")
        else:
            self.log(f"Using local mkdssp: {self.mkdssp_cmd}")
        if not self.force_reprocess:
            self.load_state()
        else:
            self.log("Force reprocess mode enabled, skipping state loading")
        protein_ids = self.load_protein_ids()
        self.total_proteins = len(protein_ids)
        self.log(f"Total proteins: {self.total_proteins}")
        existing_dssp, existing_standard, existing_experimental = self.check_existing_files()
        if self.force_reprocess:
            remaining_proteins = []
            for protein_id in protein_ids:
                dssp_file = self.dssp_dir / f"{protein_id}.dssp"
                standard_mmcif_file = self.standard_mmcif_dir / f"{protein_id}_standard.cif"
                experimental_mmcif_file = self.experimental_mmcif_dir / f"{protein_id}_experimental.cif"
                if (
                    not dssp_file.exists() or dssp_file.stat().st_size == 0 or
                    not standard_mmcif_file.exists() or standard_mmcif_file.stat().st_size == 0 or
                    not experimental_mmcif_file.exists() or experimental_mmcif_file.stat().st_size == 0
                ):
                    remaining_proteins.append(protein_id)
            self.log(f"Force reprocess mode, files to process: {len(remaining_proteins)}/{len(protein_ids)}")
        else:
            remaining_proteins = [pid for pid in protein_ids if pid not in self.processed_proteins]
            self.log(f"Remaining to process: {len(remaining_proteins)}")
        if not remaining_proteins:
            self.log("All proteins are already processed!")
            return
        start_time = time.time()
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_protein = {
                    executor.submit(self.process_single_protein, protein_id): protein_id 
                    for protein_id in remaining_proteins
                }
                for future in as_completed(future_to_protein):
                    if self.should_stop:
                        break
                    try:
                        results = future.result()
                        self.update_progress(results)
                    except Exception as e:
                        protein_id = future_to_protein[future]
                        self.log(f"Exception while processing {protein_id}: {e}")
                        with self.lock:
                            self.processed_count += 1
                            error_msg = f"Processing exception: {str(e)}"
                            self.failed_dssp.append((protein_id, error_msg))
                            self.failed_standard_mmcif.append((protein_id, error_msg))
                            self.failed_experimental_mmcif.append((protein_id, error_msg))
        except KeyboardInterrupt:
            self.log("Interrupt received, stopping safely...")
            self.should_stop = True
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.save_state()
        self.log("Batch processing completed!")
        self.log(f"Total elapsed time: {elapsed_time:.1f} seconds")
        if self.processed_count > 0:
            self.log(f"Average per protein: {elapsed_time/self.processed_count:.2f} seconds")
        self.log("Success summary:")
        self.log(f"  DSSP: {self.success_dssp}/{self.total_proteins} ({self.success_dssp/self.total_proteins*100:.1f}%)")
        self.log(f"  standard mmCIF: {self.success_standard}/{self.total_proteins} ({self.success_standard/self.total_proteins*100:.1f}%)")
        self.log(f"  experimental mmCIF: {self.success_experimental}/{self.total_proteins} ({self.success_experimental/self.total_proteins*100:.1f}%)")
        self.log("Failure summary:")
        self.log(f"  DSSP failures: {len(self.failed_dssp)}")
        self.log(f"  standard mmCIF failures: {len(self.failed_standard_mmcif)}")
        self.log(f"  experimental mmCIF failures: {len(self.failed_experimental_mmcif)}")
        self.save_failed_records()
        self.log("Failure records saved")

def main():
    protein_ids_file = r"/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest/test.txt"
    pdb_dir = r"/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest/pdb"
    output_base_dir = r"/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest/dssp"
    max_workers = 4 
    if not os.path.exists(protein_ids_file):
        print(f"Error: protein ID file does not exist: {protein_ids_file}")
        return
    if not os.path.exists(pdb_dir):
        print(f"Error: PDB directory does not exist: {pdb_dir}")
        return
    force_reprocess = True
    processor = RobustDSSPBatchProcessor(protein_ids_file, pdb_dir, output_base_dir, max_workers, force_reprocess)
    processor.run()

if __name__ == "__main__":
    main()
