import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle
from scipy import stats


""" QM9 dataset preprocess """

root = Path.cwd().parent.parent if Path.cwd().name == "data" else Path.cwd()
qm9_path = root / 'data' / 'QM9' / 'dsgdb9nsd.xyz'
qm9_edit_dir = root / 'data' / 'QM9_edit'
output_csv = qm9_edit_dir / 'QM9_edit.csv'


os.makedirs(qm9_edit_dir, exist_ok=True)



def xyz_text_extractor(xyz_text): # Extracts properties from QM9, returns smiles,
                                  # dipole_moment, U, H, G, Cv or None if invalid

    lines = [line.strip() for line in xyz_text.split('\n') if line.strip()]
    
    if len(lines) < 3:  # Need at least: atom count + props + 1 atom
        print("Invalid file: too few lines")
        return None
    
        """Index:1) gdbg tag
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
    
    # SMILES extraction, second-to-last line
    try:
        smiles = lines[-2].split()[0]
    except (IndexError, ValueError):
        print("Invalid SMILES line")
        return None
    

    try:
        # Indices are fixed in place
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
                # 6 decimal percision on QM9
                valid_files += 1
            else:
                skipped_files += 1
    
    print(f"Processed {valid_files} valid files, skipped {skipped_files} invalid files")



""" Data exploration """

plt.style.use('seaborn-v0_8-paper')  


def load_data(csv_path):

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} molecules")
    return df




def visualize_histogram(df, bins=120):

    numerics = df.select_dtypes(include='number')
    columns = numerics.columns
    num_cols = len(columns)

    # Layout configuration
    cols = 3
    rows = math.ceil(num_cols / cols)

    # Plotting
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.hist(df[col], bins=bins, alpha=0.8)
        ax.set_title(f'Histogram of {col}', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    plt.show()



def outlier_detection(df, rate=1.5): # Outlier removal using IQR method
    
    numerics = df.select_dtypes(include=['number']).columns
    outlier_mask = pd.Series(False, index=df.index)

    for col in numerics:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - rate*iqr
        upper = q3 + rate*iqr

        col_outliers = (df[col] < lower) | (df[col] > upper)
        outlier_mask = outlier_mask | col_outliers
        print(f"Column {col} has {col_outliers.sum()} outliers. Lower cuttof: {lower:.3f}. Upper cutoff: {upper:.3f}")

    df_clean = df[~outlier_mask].drop_duplicates()
    df_outliers = df[outlier_mask].drop_duplicates()

    return df_clean, df_outliers



def visualize_outliers_boxplot(df):
    numerics = df.select_dtypes(include=['number'])
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([df[col] for col in numerics.columns], vert=True, labels=numerics.columns)
    plt.title('Boxplots of Numeric Columns')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    plt.show()


def visualize_outliers_histograms(df_clean, df_outliers):

    numerics = df_clean.select_dtypes(include=['number']).columns
    num_cols = len(numerics)

    # Determine subplot grid size
    cols = 2
    rows = math.ceil(num_cols / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numerics):
        ax = axes[i]
        ax.hist(df_clean[col], bins=100, alpha=0.8, label='Clean')

        if not df_outliers.empty:
            ax.hist(df_outliers[col], bins=30, alpha=0.8, label='Outliers', color='red')

        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    plt.show()