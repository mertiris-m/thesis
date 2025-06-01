import os

"""QM9 dataset preprocess"""

qm9_edit_root = os.path.join(os.path.dirname(__file__), '../../data/QM9_edit')
os.makedirs(qm9_edit_root, exist_ok=True)


def xyz_text_extractor(xyz_text):
    """
    Safely extracts properties from QM9 .xyz text with validation.
    Returns: (smiles, dipole_moment, U, H, G, Cv) or None if invalid
    """

    lines = [line.strip() for line in xyz_text.split('\n') if line.strip()]
    
    # Basic validation
    if len(lines) < 3:  # Need at least: atom count + props + 1 atom
        print("Invalid file: too few lines")
        return None
    
    # # Get number of atoms from first line
    # try:
    #     num_atoms = int(lines[0])
    # except ValueError:
    #     print("Invalid atom count")
    #     return None
    
        """Index: 1) gdbg tag
              2) iter count
              3) A Rotational constant
              4) B Rotational constant
              5) C Rotational constant
              6) Dipole moment
              7) Isotropic polarizability
              8) Energy of HOMO
              9) Energy of LUMO
              10) Energy Gap
              11) Electronic spatial extent
              12) Zero point vibrational energy
              13) Internal energy at 0K
              14) Internal energy at 298.15K, U
              15) Enthalpy at 298.15K, H
              16) Freeenergy at 298.15K, G
              17) Heat capacity at 298.15K, Cv
              
              https://doi.org/10.1038/sdata.2014.22
              
              ------check if index is correct------
              
              """

    # Property line validation, is always line 2 
    prop_line = lines[1].split()
    if len(prop_line) < 17:  # Need at least 17 properties
        print(f"Invalid property line (only {len(prop_line)} values)")
        return None
    
    # SMILES extraction (second-to-last line)
    try:
        smiles = lines[-2].split()[0]
    except (IndexError, ValueError):
        print("Invalid SMILES line")
        return None
    

    try:
        # Indices are fixed per QM9 spec
        dipole_moment = float(prop_line[5])    # Index 6 
        U = float(prop_line[13])      # Index 14
        H = float(prop_line[14])       # Index 15
        G = float(prop_line[15])       # Index 16
        Cv = float(prop_line[16])      # Index 17
        
        # # Physics-based validation
        # if dipole < 0 or abs(U0) > 1e6 or abs(H) > 1e6:
        #     print("Unphysical values detected")
        #     return None
            
        return smiles, dipole_moment, U, H, G, Cv
        
    except (IndexError, ValueError) as e:
        print(f"Value error: {str(e)}")
        return None
    


def process_qm9_directory(input_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    valid_files = 0
    skipped_files = 0
    
    with open(output_csv, 'w', encoding='utf-8') as f_out:
        f_out.write("SMILES,Dipole_moment,U,H,G,Cv\n")
        
        for filename in sorted(os.listdir(input_dir)):
            if not filename.endswith('.xyz'):
                continue
                
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f_in:
                xyz_text = f_in.read()
            
            props = xyz_text_extractor(xyz_text)
            
            if props:
                smiles, dipole_moment, U, H, G, Cv = props
                f_out.write(f"{smiles},{dipole_moment:.6f},{U:.6f},{H:.6f},{G:.6f},{Cv:.6f}\n")
                valid_files += 1
            else:
                skipped_files += 1
    
    print(f"Processed {valid_files} valid files, skipped {skipped_files} invalid files")

