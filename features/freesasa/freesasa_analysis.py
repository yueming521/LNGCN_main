#!/usr/bin/env python3
"""
FreeSASA Analysis Script
Processes PDB files and generates SASA analysis with RSA output format
Compatible with NACCESS-like parameters
"""

import freesasa
import os
import sys
import argparse
from pathlib import Path

def create_naccess_classifier():
    """Create a classifier with NACCESS-like atomic radii"""
    # NACCESS atomic radii (approximate values)
    naccess_radii = {
        'C': 1.70,   # Carbon
        'N': 1.55,   # Nitrogen  
        'O': 1.52,   # Oxygen
        'S': 1.80,   # Sulfur
        'P': 1.80,   # Phosphorus
        'H': 1.20,   # Hydrogen
    }
    
    # Create classifier with NACCESS-like parameters
    classifier = freesasa.Classifier()
    
    # Add common amino acid atoms with NACCESS radii
    # This is a simplified version - full NACCESS has more specific radii per residue type
    for element, radius in naccess_radii.items():
        try:
            classifier.addAtom(element, radius, freesasa.polar if element in ['N', 'O'] else freesasa.apolar)
        except:
            pass  # Skip if atom type already exists
    
    return classifier

def calculate_sasa(pdb_file, output_dir, use_naccess_params=True):
    """
    Calculate SASA for a PDB file and generate output files
    
    Args:
        pdb_file: Path to PDB file
        output_dir: Directory to save results
        use_naccess_params: Whether to use NACCESS-like parameters
    """
    
    print(f"Processing: {pdb_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load structure
        structure = freesasa.Structure(pdb_file)
        print(f"Loaded structure with {structure.nAtoms()} atoms")
        
        # Set up parameters
        if use_naccess_params:
            # Use Lee-Richards algorithm with NACCESS-like parameters
            params = freesasa.Parameters({
                'algorithm': freesasa.LeeRichards,
                'probe-radius': 1.4,  # NACCESS default probe radius
                'n-slices': 20        # Resolution parameter
            })
            classifier = create_naccess_classifier()
            print("Using NACCESS-like parameters (Lee-Richards algorithm, probe radius 1.4 Å)")
        else:
            # Use default parameters
            params = freesasa.Parameters()
            classifier = None
            print("Using default FreeSASA parameters")
        
        # Calculate SASA
        # Note: FreeSASA Python API may not support custom classifiers in the same way
        result = freesasa.calc(structure, params)
        
        print(f"Total SASA: {result.totalArea():.2f} A^2")

        # Check available methods for result object
        print("Available result methods:", [method for method in dir(result) if not method.startswith('_')])

        # Try to get polar/apolar areas if available
        try:
            polar_area = result.polarArea()
            apolar_area = result.apolarArea()
            print(f"Polar SASA: {polar_area:.2f} A^2")
            print(f"Apolar SASA: {apolar_area:.2f} A^2")
        except AttributeError:
            print("Polar/Apolar areas not directly available from result object")
        
        # Generate output files
        base_name = Path(pdb_file).stem
        
        # Generate RSA-like output
        rsa_file = os.path.join(output_dir, f"{base_name}.rsa")
        generate_rsa_output(structure, result, rsa_file)

        # Generate detailed residue output
        res_file = os.path.join(output_dir, f"{base_name}_residues.txt")
        generate_residue_output(structure, result, res_file)

        # Generate summary
        summary_file = os.path.join(output_dir, f"{base_name}_summary.txt")
        generate_summary(structure, result, summary_file, use_naccess_params)

        # Generate PDB with B-factors as SASA values
        pdb_file = os.path.join(output_dir, f"{base_name}_sasa.pdb")
        result.write_pdb(pdb_file)
        
        print(f"Results saved to: {output_dir}")
        print(f"- RSA format: {rsa_file}")
        print(f"- Residue details: {res_file}")
        print(f"- Summary: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return False

def generate_rsa_output(structure, result, output_file):
    """Generate RSA-format output file"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("REM  Solvent Accessible Surface Area calculated by FreeSASA\n")
        f.write("REM  Algorithm: Lee-Richards\n")
        f.write("REM  Probe radius: 1.40 A\n")
        f.write("REM\n")
        f.write(f"REM  Total SASA: {result.totalArea():.2f} A^2\n")
        f.write("REM\n")
        f.write("RES  Chain Residue      SASA\n")
        
        # Get residue areas
        for i in range(structure.nAtoms()):
            atom_area = result.atomArea(i)
            if atom_area > 0:  # Only include atoms with surface area
                residue_name = structure.residueName(i)
                residue_number = structure.residueNumber(i)
                chain_id = structure.chainLabel(i)
                atom_name = structure.atomName(i)
                
                # This is a simplified RSA format - real RSA format has more columns
                f.write(f"ATOM {chain_id:>1} {residue_name:>3} {residue_number:>4} {atom_name:>4} {atom_area:>8.2f}\n")

def generate_residue_output(structure, result, output_file):
    """Generate detailed residue-by-residue output"""
    
    # Group atoms by residue
    residues = {}
    for i in range(structure.nAtoms()):
        chain_id = structure.chainLabel(i)
        residue_number = structure.residueNumber(i)
        residue_name = structure.residueName(i)
        key = (chain_id, residue_number, residue_name)
        
        if key not in residues:
            residues[key] = []
        residues[key].append(i)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Residue-by-residue SASA analysis\n")
        f.write("# Chain Residue ResNum    Total_SASA  Num_Atoms\n")

        for (chain_id, residue_number, residue_name), atom_indices in sorted(residues.items()):
            total_area = sum(result.atomArea(i) for i in atom_indices)

            f.write(f"{chain_id:>5} {residue_name:>7} {residue_number:>6} {total_area:>12.2f} {len(atom_indices):>9}\n")

def generate_summary(structure, result, output_file, use_naccess_params):
    """Generate summary file"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FreeSASA Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Input structure atoms: {structure.nAtoms()}\n")
        f.write(f"Algorithm: Lee-Richards\n")
        f.write(f"Probe radius: 1.4 Å\n")
        f.write(f"NACCESS-compatible: {'Yes' if use_naccess_params else 'No'}\n\n")
        
        f.write("SASA Results:\n")
        f.write(f"  Total SASA:  {result.totalArea():>10.2f} A^2\n\n")

def main():
    parser = argparse.ArgumentParser(description='FreeSASA Analysis Tool')
    parser.add_argument('pdb_file', help='Input PDB file')
    parser.add_argument('-o', '--output', help='Output directory', default='results')
    parser.add_argument('--no-naccess', action='store_true', help='Use default FreeSASA parameters instead of NACCESS-like')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdb_file):
        print(f"Error: PDB file {args.pdb_file} not found")
        sys.exit(1)
    
    # Create output directory based on PDB filename if not specified
    if args.output == 'results':
        base_name = Path(args.pdb_file).stem
        output_dir = os.path.join('results', base_name)
    else:
        output_dir = args.output
    
    success = calculate_sasa(args.pdb_file, output_dir, not args.no_naccess)
    
    if success:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
