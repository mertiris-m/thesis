import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm
import json
from mordred import Calculator, descriptors


""" QM9 dataset preprocess """

root = Path.cwd().parent.parent if Path.cwd().name == "data" else Path.cwd()
qm9_path = root / 'data' / 'QM9' / 'dsgdb9nsd.xyz'
qm9_edit_dir = root / 'data' / 'QM9_edit'
output_csv = qm9_edit_dir / 'QM9_edit.csv'

os.makedirs(qm9_edit_dir, exist_ok=True)


def xyz_text_extractor(xyz_text):
    """
    Extracts properties from QM9 .xyz files.
    Returns smiles, atom_no, dipole_moment, U, H, G, Cv or None if invalid
    """
    
    # Extracts properties from QM9, returns smiles,
                                  # dipole_moment, U, H, G, Cv or None if invalid

    lines = [line.strip() for line in xyz_text.split('\n') if line.strip()]
    
    if len(lines) < 3:  # Need at least: atom count + props + 1 atom
        print("Invalid file: too few lines")
        return None
    
    # Atom count is in line 1, not necessary
    try:
        atom_no = int(lines[0])
    except ValueError:
        print("Invalid atom count")
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
              16) Free energy at 298.15K, G
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
            
        return smiles, atom_no, dipole_moment, U, H, G, Cv
        
    except (IndexError, ValueError) as e:
        print(f"Value error: {str(e)}")
        return None
    


def process_qm9_directory(input_dir, output_csv): # Saves a .csv from text extractor to data/QM9_edit/QM9_edit.csv
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    valid_files = 0
    skipped_files = 0
    
    with open(output_csv, 'w', encoding='utf-8') as f_out:
        f_out.write("SMILES,Atom_No,Dipole_moment,U,H,G,Cv\n")
        
        for filename in sorted(os.listdir(input_dir)):
            if not filename.endswith('.xyz'):
                continue
                
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f_in:
                xyz_text = f_in.read()
            
            props = xyz_text_extractor(xyz_text)
            
            if props:
                smiles, atom_no, dipole_moment, U, H, G, Cv = props
                f_out.write(f"{smiles},{atom_no},{dipole_moment},{U},{H},{G},{Cv}\n")
                # 6 decimal percision on QM9
                valid_files += 1
            else:
                skipped_files += 1
    
    print(f"Processed {valid_files} valid files, skipped {skipped_files} invalid files")



""" Data exploration """

plt.style.use('seaborn-v0_8-paper')  


def load_data(csv_path): # Reads .csv

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} molecules")
    return df




def visualize_histogram(df, bins=120): # Histogram visualization of numeric columns 

    numerics = df.select_dtypes(include='number')
    columns = numerics.columns
    num_cols = len(columns)

    # Layout 
    cols = 3
    rows = math.ceil(num_cols / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12 , 4 * rows))
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



def outlier_detection(df, rate=1.5): # Outlier removal using IQR method, excluding atom number
    
    numerics = df.select_dtypes(include=['number']).columns
    # Exclude Atom_No
    numerics = [col for col in numerics if col != 'Atom_No']
    outlier_mask = pd.Series(False, index=df.index)

    for col in numerics:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - rate*iqr
        upper = q3 + rate*iqr

        col_outliers = (df[col] < lower) | (df[col] > upper)
        outlier_mask = outlier_mask | col_outliers
        print(f"Column {col} has {col_outliers.sum()} outliers. Lower cuttof: {lower}. Upper cutoff: {upper}")

    df_clean = df[~outlier_mask].drop_duplicates()
    df_outliers = df[outlier_mask].drop_duplicates()

    return df_clean, df_outliers



def visualize_outliers_boxplot(df): # Outlier removal visualization
    numerics = df.select_dtypes(include=['number'])
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([df[col] for col in numerics.columns], vert=True, labels=numerics.columns)
    plt.title('Boxplots of Numeric Columns')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    plt.show()


def visualize_outliers_histograms(df_clean, df_outliers):  # Histogram visualization with outliers
    numerics = df_clean.select_dtypes(include=['number']).columns
    num_cols = len(numerics)

    cols = 3
    rows = math.ceil(num_cols / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numerics):
        ax = axes[i]
        ax.hist(df_clean[col], bins=100, alpha=0.8, label='Clean')

        if not df_outliers.empty:
            ax.hist(df_outliers[col], bins=100, alpha=0.8, label='Outliers', color='red')

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


def similarity_check(df, cols): # Does a simmilarity check for >=2 cols in dataframe

    print("\nColumn Differences:")
    print("-" * 30)

    for a, b in combinations(cols, 2):
            abs_diff = np.abs(df[a] - df[b])
            rel_diff = abs_diff / np.abs(df[a].mean())
        
            print(f"Comparing {a} vs {b}:")
            print(f"  Max absolute difference: {abs_diff.max():.6f}")
            print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
            print(f"  Max relative difference: {rel_diff.max():.2%}")
            print(f"  Mean relative difference: {rel_diff.mean():.2%}")
            print("-" * 30)


    # def scale_qm9_edit(df, cols=['U', 'H', 'G', 'Cv']):

    #     scaler = StandardScaler()
    #     df[cols] = scaler.fit_transform(df[cols])

    #     return df
    

# standardscaler will not work for reversing the process using VAE-BO
# so better to use minmaxscaler. However every dataset has a different
# minimum and maximum. Solution -> fit different scalers for each dataset
# and use the last (experimental) scaler for the reverse.
# So the limitation is the minimum and maximum values of the exp dataset.


def qm9_scaler(df, scaler_dir): # Scales QM9 dataframe using MinMaxScaler

    cols = ['Dipole_moment', 'U', 'Cv']
    scalers = {}
    
    os.makedirs(scaler_dir, exist_ok=True)

    for col in cols:
        scaler = MinMaxScaler()
        df[col + '_scaled'] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
        print(f"{col} column maximum value: {df[col].max()}")
        print(f"{col} column minimum value: {df[col].min()}")

        # Save each scaler
        joblib.dump(scaler, os.path.join(scaler_dir, f'{col}_QM9_scaler.pkl'))

    return df, scalers



""" ------------- SMILES to Graph conversion ------------- """

def get_atom_features(atom):
    """
    Generates a feature vector for an atom.
    Features are one-hot encoded where necessary.
    """

    node_features = []
    
    # List of possible values for hybridization and geometries
    possible_hybridizations = [Chem.rdchem.HybridizationType.S,
                               Chem.rdchem.HybridizationType.SP,
                               Chem.rdchem.HybridizationType.SP2,
                               Chem.rdchem.HybridizationType.SP2D,
                               Chem.rdchem.HybridizationType.SP3,
                               Chem.rdchem.HybridizationType.SP3D,
                               Chem.rdchem.HybridizationType.SP3D2]
    
    possible_geometries = [Chem.rdchem.ChiralType.CHI_ALLENE,
                           Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
                           Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
                           Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
                           Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                           Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                           Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL]
    
    # 1. Atomic Number
    node_features.append(atom.GetAtomicNum())
    
    # 2. One-hot encode hybridization
    hybridization = atom.GetHybridization()
    hybridization_encoding = [1.0 if h == hybridization else 0.0 for h in possible_hybridizations]
    node_features.extend(hybridization_encoding)

    # 3. One-hot encode geometry
    geometry = atom.GetChiralTag()
    geometry_encoding = [1.0 if g == geometry else 0.0 for g in possible_geometries]
    node_features.extend(geometry_encoding)

    # 4. Numerical features
    node_features.append(atom.GetValence(Chem.ValenceType.EXPLICIT)) # Returns the explicit valence of the atom.
    node_features.append(atom.GetDegree()) # Returns the explicit degree of the atom.
                                      # The degree of an atom is defined to be its number of directly-bonded neighbors.
                                      # The degree is independent of bond orders, but is dependent on whether or not Hs are explicit in the graph.
    node_features.append(atom.GetFormalCharge()) # Returns the formal charge for the molecule
    node_features.append(atom.GetNumRadicalElectrons()) # Returns the number of radical electrons for this atom
    node_features.append(atom.GetTotalNumHs()) # Returns the total number of Hs (hydrogens) (implicit and explicit) that this atom is bound to 
    
    # 5. Boolean features
    node_features.append(float(atom.GetIsAromatic())) # Returns whether the atom is aromatic.
    node_features.append(float(atom.IsInRing())) # Returns whether or not the bond is in a ring of any size.

    return node_features


def get_bond_features(bond):
    """
    Generates a feature vector for a bond.
    Features are one-hot encoded.
    """

    edge_features = []

    # List of interesting values for bond types
    possible_bond_types = [Chem.rdchem.BondType.SINGLE,
                           Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE,
                           Chem.rdchem.BondType.DATIVE,
                           Chem.rdchem.BondType.IONIC,
                           Chem.rdchem.BondType.AROMATIC]
    
    # 1. One-hot encode bond type
    bt = bond.GetBondType()
    bond_type_encoding = [1.0 if b == bt else 0.0 for b in possible_bond_types]
    edge_features.extend(bond_type_encoding)

    # 2. Boolean features
    edge_features.append(float(bond.GetIsAromatic()))
    edge_features.append(float(bond.IsInRing()))
    edge_features.append(float(bond.GetIsConjugated())) # Returns whether or not the bond is considered to be conjugated.

    return edge_features


def get_feature_names():
    """
    Generates and returns the node and edge feature names.
    """
    
    node_names = []
    node_names.append('AtomicNum')
    possible_hybridizations = ['S',
                               'SP',
                               'SP2',
                               'SP2D',
                               'SP3',
                               'SP3D',
                               'SP3D2']
    node_names.extend([f'Hybridization_{h}' for h in possible_hybridizations])
    possible_geometries = ['ALLENE',
                           'OCTAHEDRAL',
                           'SQUAREPLANAR',
                           'TETRAHEDRAL',
                           'TETRAHEDRAL_CCW',
                           'TETRAHEDRAL_CW',
                           'TRIGONALBIPYRAMIDAL']
    node_names.extend([f'Chiral_{g}' for g in possible_geometries])
    node_names.extend(['ExplicitValence ',
                       'Degree',
                       'FormalCharge',
                       'NumRadicalElectrons',
                       'TotalNumHs',
                       'IsAromatic',
                       'IsInRing'])
    

    edge_names = []
    possible_bond_types = ['SINGLE',
                           'DOUBLE',
                           'TRIPLE',
                           'DATIVE',
                           'IONIC',
                           'AROMATIC']
    edge_names.extend([f'BondType_{bt}' for bt in possible_bond_types])
    edge_names.extend(['IsAromatic', 'IsInRing', 'IsConjugated'])


    mordred_calculator = Calculator(descriptors, ignore_3D=True)
    global_feature_names = [str(d) for d in mordred_calculator.descriptors]

    return node_names, edge_names, global_feature_names


def save_feature_names(results_dir, filename):
    """
    Saves the node, edge and global feature names into a json.
    """

    os.makedirs(results_dir, exist_ok=True)
    
    print("Extracting feature names...")
    node_names, edge_names, global_names = get_feature_names()
    
    feature_names = {
        'node_features': node_names,
        'edge_features': edge_names,
        'global_features': global_names
    }
    
    names_filename = f'{filename}.json'
    names_filepath = os.path.join(results_dir, names_filename)
    
    with open(names_filepath, 'w') as f:
        json.dump(feature_names, f, indent=4)
        
    print(f"Node, edge and global feature names saved to '{names_filepath}'")


# ------------- Graph Conversion Functions without Globals ------------- 

def smiles_to_graph(smiles_string, target_values):
    """
    Generates a graph with atom and bond features from 
    get_atom_features, get_bond_features and appends the
    target values (property value) to the y tensor of the
    PyTorch Geometric Data class.
    """

    mol = Chem.MolFromSmiles(smiles_string)

    if mol is None:
        return None
    
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    
    if mol.GetNumBonds() > 0:
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feats = get_bond_features(bond)
        
            edge_indices.extend([(i, j), (j, i)])
            edge_attrs.extend([bond_feats, bond_feats]) # Same features for both directions
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        _, edge_names, _ = get_feature_names()
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_names)))
    
    y = torch.tensor(np.array(target_values, dtype=np.float32)).view(1, -1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data


def convert_smiles_to_graph(df, cols, output_dir, filename):
    """
    Converts a dataframe of SMILES strings to a PyTorch Geometric
    data object and saves it.

    For each row, it generates a graph using the SMILES
    string and its corresponding target (property) values.
    Saves to a .pt containing all graphs from the dataframe
    in the output directory.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'SMILES' column.
        cols (List[str]): A list of column names for the target (property) values.
        output_dir (str): The path to the directory to save the file.
        filename (str): The filename excluding the extension .pt.
    """

    os.makedirs(output_dir, exist_ok=True) 
    dataset_filename = f'{filename}.pt'
    dataset_filepath = os.path.join(output_dir, dataset_filename)
    
    print(f"\nConverting SMILES to Graph...")
    graph_list = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        target_values = [row[cols]] 

        graph = smiles_to_graph(row['SMILES'], target_values)
        
        if graph is not None:
            graph_list.append(graph)
    
    torch.save(graph_list, dataset_filepath)
    
    print(f"\nSuccessfully converted {len(graph_list)}/{len(df)} SMILES strings.")
    print(f"\nDataset saved successfully to '{dataset_filepath}'")


# ------------- Graph Conversion Functions with Globals -------------


def save_global_features(df, output_dir, filename, batch_size=5000): # made with A.I.
    """
    Calculates Mordred descriptors for all SMILES in a DataFrame in batches
    and saves them to a single JSON file in the results directory.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'SMILES' column.
        output_dir (str): The path to the directory to save the file.
        filename (str): The filename excluding the extension .json.
        batch_size (int): Number of molecules to process in each batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'{filename}.json'
    output_filepath = os.path.join(output_dir, output_filename)
    
    print(f"Calculating global features in batches of {batch_size}...")
    
    calc = Calculator(descriptors, ignore_3D=True)
    num_batches = (len(df) // batch_size) + 1
    
    all_descriptors = []
    failed_indices = []

    for i in tqdm(range(num_batches), desc="Calculating Mordred Descriptors"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        df_batch = df.iloc[start_idx:end_idx]

        if df_batch.empty:
            continue
            
        mols_batch = [Chem.MolFromSmiles(s) for s in df_batch['SMILES']]
        
        # calc.map is robust and returns results or error objects
        results = calc.map(mols_batch, quiet=True)
        
        for idx, desc_series in enumerate(results):
            original_index = df_batch.index[idx]
            if isinstance(desc_series, Exception):
                # If calculation failed, record the index and append NaNs
                failed_indices.append(original_index)
                all_descriptors.append([np.nan] * len(calc.descriptors))
            else:
                # Convert descriptor values to a list of floats, handling potential non-float types
                values = [float(v) if isinstance(v, (int, float)) else np.nan for v in desc_series.values()]
                all_descriptors.append(values)
                
    if failed_indices:
        print(f"\nWarning: Mordred calculation failed for {len(failed_indices)} molecules.")
        print(f"Indices of failed molecules: {failed_indices}")
    
    print(f"\nSaving {len(all_descriptors)} global feature vectors to '{output_filepath}'...")
    with open(output_filepath, 'w') as f:
        json.dump(all_descriptors, f)
        
    print("Global features saved successfully.")
    


def smiles_to_graph_with_globals(smiles_string, target_values, global_features):
    """
    Generates a graph with atom and bond features from 
    get_atom_features, get_bond_features and appends the
    target values (property value) to the y tensor of the
    PyTorch Geometric Data class. Also uses global features.
    """

    mol = Chem.MolFromSmiles(smiles_string)

    if mol is None:
        return None
    
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    u = torch.tensor(global_features, dtype=torch.float).view(1,-1)
    
    if mol.GetNumBonds() > 0:
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feats = get_bond_features(bond)
            edge_indices.extend([(i, j), (j, i)])
            edge_attrs.extend([bond_feats, bond_feats])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        _, edge_names, _ = get_feature_names()
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_names)))
        
    y = torch.tensor(np.array(target_values, dtype=np.float32)).view(1, -1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
    
    return data



def convert_smiles_to_graph_with_globals(df, cols, output_dir, filename,  global_feat_path):
    """
    Converts a dataframe of SMILES strings to a PyTorch Geometric
    data object and saves it.

    For each row, it generates a graph using the SMILES
    string and its corresponding target (property) values.
    Saves to a .pt containing all graphs from the dataframe
    in the output directory.

    Adds global features from a pre-calculated json file.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'SMILES' column.
        cols (List[str]): A list of column names for the target (property) values.
        global_feat_path (str): Path to the global features JSON.
        output_dir (str): The path to the directory to save the file.
        filename (str): The filename excluding the extension .pt.
    """

    os.makedirs(output_dir, exist_ok=True) 
    dataset_filename = f'{filename}.pt'
    dataset_filepath = os.path.join(output_dir, dataset_filename)

    print(f"Loading pre-calculated global features from '{global_feat_path}'...")
    try:
        with open(global_feat_path, 'r') as f:
            all_global_features = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Global features file not found!")
        print("Please run `calculate_and_save_global_features` first.")
        return

    if len(all_global_features) != len(df):
        print("ERROR: Mismatch between number of molecules in DataFrame and global features file.")
        return
    
    print("\nConverting SMILES to Graph with global features...")
    graph_list = []

    # Use tqdm to iterate with the index
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        target_values = [row[cols]] 
        
        # Get the corresponding global features for this molecule by index
        global_features = all_global_features[idx]

        # Handle potential NaNs from failed calculations by replacing them with 0
        global_features = [0.0 if np.isnan(v) else v for v in global_features]
        
        graph = smiles_to_graph_with_globals(row['SMILES'], target_values, global_features)
        
        if graph is not None:
            graph_list.append(graph)
    
    torch.save(graph_list, dataset_filepath)
    
    print(f"\nSuccessfully converted {len(graph_list)}/{len(df)} SMILES strings.")
    print(f"Dataset saved successfully to '{dataset_filepath}'")