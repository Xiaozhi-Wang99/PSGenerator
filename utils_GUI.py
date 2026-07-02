import csv
import pandas as pd
import numpy as np
import time
import os
import math
import tempfile
import subprocess
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.data import Data, Batch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors as desc, Crippen, Lipinski, Draw
from PyQt5.QtWidgets import QApplication
from joblib import load
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Predictor:
    @classmethod
    def calc_fp(self, mols, radius=3, bit_len=2048):
        ecfp = self.calc_ecfp(mols, radius=radius, bit_len=bit_len)
        phch = self.calc_physchem(mols)
        fps = np.concatenate([ecfp, phch], axis=1)
        return fps

    @classmethod
    def calc_ecfp(cls, mols, radius=3, bit_len=2048):
        fps = np.zeros((len(mols), bit_len))
        for i, mol in enumerate(mols):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, fps[i, :])
            except:
                pass
        return fps

    @classmethod
    def calc_physchem(cls, mols):
        prop_list = ['logP', 'HBA', 'HBD', 'Rotable', 'Amide',
                     'Bridge', 'Hetero', 'Heavy', 'Spiro', 'FCSP3', 'Ring',
                     'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'MW']
        fps = np.zeros((len(mols), 17))
        props = Property()
        for i, prop in enumerate(prop_list):
            props.prop = prop
            fps[:, i] = props(mols)
        return fps


class Property:
    def __init__(self, prop='MW'):
        self.prop = prop
        self.prop_dict = {'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'MW': desc.MolWt
                          }

    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                scores[i] = self.prop_dict[self.prop](mol)
            except:
                continue
        return scores


class CrossAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1, return_attn_importance=False):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.return_attn_importance = return_attn_importance

    def forward(self, Q, K, V):
        Q_proj = self.query_proj(Q)  # [B, Nq, D]
        K_proj = self.key_proj(K)  # [B, Nk, D]
        V_proj = self.value_proj(V)  # [B, Nk, D]

        scale = 1.0 / math.sqrt(K_proj.size(-1))
        attn_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) * scale  # [B, Nq, D] * [B, D, Nk] -> [B, Nq, Nk]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V_proj)  # [B, Nq, Nk] * [B, Nk, D] -> [B, Nq, D]

        if self.return_attn_importance:
            return output, attn_weights
        else:
            return output


class PSGCN_DoubleCrossAttn(nn.Module):
    def __init__(self, gcn_in=131, gcn_h1=256, gcn_h2=64, fp_dim=2065, d_model=128, n_emb1=1024, n_emb2=256, dropout=0.3):
        super(PSGCN_DoubleCrossAttn, self).__init__()
        self.conv1 = GCNConv(gcn_in, gcn_h1)
        self.conv2 = GCNConv(gcn_h1, gcn_h2)
        self.dropout = nn.Dropout(dropout)

        self.gcn_proj = nn.Linear(1, d_model)
        self.fp_proj = nn.Linear(1, d_model)

        self.cross_attn1 = CrossAttention(d_model, dropout)  # Q: GCN->FP
        self.cross_attn2 = CrossAttention(d_model, dropout)  # Q: FP->GCN

        self.fc1 = nn.Linear((fp_dim + gcn_h2) * d_model, n_emb1)
        self.fc2 = nn.Linear(n_emb1, n_emb2)
        self.fc3 = nn.Linear(n_emb2, 2)

    def forward(self, data):
        x, edge_index, batch, mol_fingerprints = data.x, data.edge_index, data.batch, data.mol_fingerprints
        x1 = F.relu(self.conv1(x.float(), edge_index))
        x1 = self.conv2(x1.float(), edge_index)

        gcn_graph_feat = gap(x1, batch)  # [B, 64]
        gcn_graph_feat = gcn_graph_feat.unsqueeze(-1)#[B, 64, 1]
        gcn_graph_feat_proj = self.gcn_proj(gcn_graph_feat)  #[B, 64, 1]-[1,D]->[B, 64, d_model]

        fp_feat = self.fp_proj(mol_fingerprints.unsqueeze(-1)) #[B, 2065, 1]-[1,D]->[B, 2065, D]

        gcn_updated = self.cross_attn1(gcn_graph_feat_proj, fp_feat, fp_feat) # Q: GCN, K/V: FP                  [B, 64, D]
        fp_updated = self.cross_attn2(fp_feat, gcn_graph_feat_proj, gcn_graph_feat_proj) # Q: FP, K/V: GCN       [B, 2065, D]

        fused = torch.cat([fp_updated, gcn_updated], dim=1)  # [B, 2129, D]
        fused = torch.flatten(fused, start_dim=1)
        out1 = F.relu(self.fc1(fused))
        features = F.relu(self.fc2(self.dropout(out1)))
        out = self.fc3(self.dropout(features))
        return out


def one_hot_encoding_unk(value, known_list):
    encoding = [0] * (len(known_list) + 1)
    index = known_list.index(value) if value in known_list else -1
    encoding[index] = 1
    return encoding


class featurization_parameters:
    def __init__(self):
        self.max_atomic_num = 100
        self.atom_features = {'atomic_num': list(range(self.max_atomic_num)),
                              'total_degree': [0, 1, 2, 3, 4, 5],
                              'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
                              'total_numHs': [0, 1, 2, 3, 4],
                              'hybridization': [Chem.rdchem.HybridizationType.SP,
                                                Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3,
                                                Chem.rdchem.HybridizationType.SP3D,
                                                Chem.rdchem.HybridizationType.SP3D2]}

        self.atom_fdim = sum(len(known_list) + 1 for known_list in self.atom_features.values()) + 3
        self.bond_fdim = 6


feature_params = featurization_parameters()


def atom_features(atom: Chem.rdchem.Atom):
    if atom is None:
        atom_feature_vector  = [0] * feature_params.atom_fdim
    else:
        atom_feature_vector  = one_hot_encoding_unk(atom.GetAtomicNum() - 1, feature_params.atom_features['atomic_num']) + \
            one_hot_encoding_unk(atom.GetTotalDegree(), feature_params.atom_features['total_degree']) + \
            one_hot_encoding_unk(atom.GetFormalCharge(), feature_params.atom_features['formal_charge']) + \
            one_hot_encoding_unk(int(atom.GetTotalNumHs()), feature_params.atom_features['total_numHs']) + \
            one_hot_encoding_unk(int(atom.GetHybridization()), feature_params.atom_features['hybridization']) + \
            [1 if atom.IsInRing()else 0]+ \
            [1 if atom.GetIsAromatic() else 0]+\
            [atom.GetMass() * 0.01]
    return atom_feature_vector


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        bond_feature_vector  = [0] * feature_params.bond_fdim
    else:
        bt = bond.GetBondType()
        bond_feature_vector  = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
    return bond_feature_vector


def process_single_smiles(data_row, smiles_col_name):
    smiles = data_row[smiles_col_name]
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    xs = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float32)
    edge_indices, edge_attrs = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([e, e])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, 6)
    mol_fingerprints = torch.tensor(data_row.drop(smiles_col_name).values.astype(float), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol_fingerprints=mol_fingerprints.unsqueeze(0), smiles=smiles)


def smiles_data_process(dataset, smiles_col_name='canonical_smiles', progress_callback=None):
    processed_data = []
    total = len(dataset)
    for i, (idx, row) in enumerate(dataset.iterrows()):
        d = process_single_smiles(row, smiles_col_name)
        if d is not None:
            processed_data.append(d)
        if progress_callback and i % 5 == 0: # 5 molecule update state
            progress = 30 + int((i / total) * 40)
            progress_callback(progress)
    return processed_data


def collate_fn(data_list):
    return Batch.from_data_list(data_list)


class PredictionController:
    def __init__(
        self,
        model_path='model_path/1_PS_GCN_best_model.pt',
        scaler_path='model_path/Classification_standard_scaler.joblib'
    ):
        self.model = None
        self.scaler = None
        self.is_ready = False
        self.load_error_msg = ""

        if not os.path.exists(model_path):
            self.load_error_msg = f"Model file not found! \nExpected path: {os.path.abspath(model_path)}"
            return
        if not os.path.exists(scaler_path):
            self.load_error_msg = f"Standard normalizer file not found! \nExpected path: {os.path.abspath(scaler_path)}"
            return

        try:
            self.scaler = load(scaler_path)
            self.model = PSGCN_DoubleCrossAttn().to(DEVICE)
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval()
            self.is_ready = True
        except Exception as e:
            self.model = None
            self.scaler = None
            self.is_ready = False
            self.load_error_msg = f"Error: {str(e)}"

    def clean_smiles(self, smiles_list):
        valid_smiles = []
        for smi in smiles_list:
            if not isinstance(smi, str):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol:
                parts = Chem.GetMolFrags(mol, asMols=True)
                largest_part = max(parts, default=None, key=lambda m: m.GetNumAtoms())
                if largest_part:
                    canonical_smi = Chem.MolToSmiles(largest_part, isomericSmiles=True)
                    valid_smiles.append(canonical_smi)
        unique_smiles_list = list(dict.fromkeys(valid_smiles))
        unique_mols = [Chem.MolFromSmiles(smi) for smi in unique_smiles_list]
        return unique_smiles_list, unique_mols

    def run_prediction(self, smiles_list, output_path=None, status_callback=None, progress_callback=None, **kwargs):
        if not self.is_ready:
            raise RuntimeError(f"Predictions not yet ready:\n{self.load_error_msg}")

        if status_callback:
            status_callback("Cleaning SMILES...")

        unique_smiles, mols = self.clean_smiles(smiles_list)
        if not unique_smiles:
            raise ValueError("No valid molecules.")
        num_unique = len(unique_smiles)
        if status_callback:
            status_callback(f"Processing {num_unique} molecules...")

        if progress_callback:
            progress_callback(10)
        fingerprints = Predictor.calc_fp(mols)
        if progress_callback:
            progress_callback(30)

        if self.scaler:
            if hasattr(self.scaler, 'feature_names_in_'):
                scaled_fingerprints = self.scaler.transform(pd.DataFrame(fingerprints, columns=self.scaler.feature_names_in_))
            else:
                scaled_fingerprints = self.scaler.transform(fingerprints)
        else:
            scaled_fingerprints = fingerprints
        smiles_df = pd.DataFrame(unique_smiles, columns=['canonical_smiles'])
        fingerprints_df = pd.DataFrame(scaled_fingerprints)
        combined_df = pd.concat([smiles_df, fingerprints_df], axis=1)

        processed_data = smiles_data_process(combined_df, progress_callback=progress_callback)
        data_loader = DataLoader(processed_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

        all_predictions = []
        all_type1_scores = []
        total_batches = len(data_loader)

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                data_batch = data_batch.to(DEVICE)
                out = self.model(data_batch)

                prob = F.softmax(out, dim=1)
                pred = prob.argmax(dim=1)
                type1_score = prob[:, 1]

                all_predictions.extend(pred.cpu().numpy())
                all_type1_scores.extend(type1_score.cpu().numpy())

                if progress_callback:
                    prog = 70 + int((i + 1) / total_batches * 30)
                    progress_callback(min(prog, 99))

        predictions = np.array(all_predictions)
        type1_scores = np.array(all_type1_scores)

        if output_path:
            save_df = pd.DataFrame({
                "SMILES": unique_smiles,
                "Prediction_Label": ["Type I" if x == 1 else "Non-type I" for x in predictions],
                "TypeI_score": type1_scores
            })
            save_df.to_csv(output_path, index=False)

        count_type1 = int(np.sum(predictions))
        percentage_type1 = (count_type1 / num_unique * 100) if num_unique > 0 else 0

        return {
            "total_unique": num_unique,
            "count_type1": count_type1,
            "percentage_type1": percentage_type1,
            "single_result": predictions[0] if num_unique == 1 else None,
            "single_score": float(type1_scores[0]) if num_unique == 1 else None
        }


def process_scaffold_smi(smi):
    """Converts [R] or [*] to RDKit-compatible attachment points [*:N]
    1. [*:0], [*:1] format，directly input
    2. [R] or [*]，need to process
    Returns the processed SMILES.
    """
    smi_for_parsing = smi.replace('[R]', '[*]')

    mol = Chem.MolFromSmiles(smi_for_parsing)
    if not mol:
        mol = Chem.MolFromSmiles(smi_for_parsing, sanitize=False)
        if not mol:
            raise ValueError(f"Invalid SMILES syntax: {smi}")

    start_offset = 1
    attachment_count = start_offset

    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]

    if not dummy_atoms:
        raise ValueError("No attachment points found.")

    for atom in dummy_atoms:
        atom.SetAtomMapNum(attachment_count)
        attachment_count += 1

    temp_smi = Chem.MolToSmiles(mol, isomericSmiles=True)

    for i in range(start_offset, attachment_count):
        old_tag = f'[*:{i}]'
        new_tag = f'[*:{i - start_offset}]'
        temp_smi = temp_smi.replace(old_tag, new_tag)

    return temp_smi


def run_generation(scaffold_smi, output_dir, num_random, num_decor, status_callback=None, **kwargs):
    PYTHON_EXECUTABLE = "python"
    GENERATION_SCRIPT_PATH = "sample_scaffolds.py"
    MODEL_PATH = "model_path/model.trained.90"

    if not os.path.exists(GENERATION_SCRIPT_PATH):
        raise FileNotFoundError(f"Generation script not found at: {GENERATION_SCRIPT_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Generation model not found at: {MODEL_PATH}")

    if status_callback:
        status_callback("Step 1/3: Preparing scaffold...")

    processed_smi = process_scaffold_smi(scaffold_smi)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi') as tmp_file:
        tmp_file.write(processed_smi)
        scaffold_input_path = tmp_file.name

    output_prefix = "Generated_molecules"
    raw_output_path = os.path.join(output_dir, f"{output_prefix}_repeat_dropna.csv")
    unique_output_path = os.path.join(output_dir, f"{output_prefix}_unique.csv")
    top10_img_path = os.path.join(output_dir, f"{output_prefix}_unique_top10.png")

    if status_callback:
        status_callback("Step 2/3: Starting generation process (this will take a while)...")

    env = os.environ.copy()
    if DEVICE == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"

    command = [
        PYTHON_EXECUTABLE,
        "-u",
        GENERATION_SCRIPT_PATH,
        "-m", MODEL_PATH,
        "-i", scaffold_input_path,
        "--output-dir", output_dir,
        "--output-prefix", output_prefix,
        "--of", "csv",
        "-r", str(num_random),
        "-n", str(num_decor),
        "-d", "multi",
        "--device", str(DEVICE),
    ]

    print(f"Executing command: {' '.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
    )

    output_lines = []

    for line in process.stdout:
        output_lines.append(line)
        if status_callback:
            status_callback(line.strip())

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("Generation failed:\n" + "".join(output_lines))

    os.remove(scaffold_input_path)

    if process.returncode != 0:
        print("--- STDERR ---")
        print(process.stderr)
        print("--- STDOUT ---")
        print(process.stdout)
        raise RuntimeError(f"Generation script failed with error:\n{process.stderr}")

    if status_callback:
        status_callback("Step 3/3: Processing generated molecules...")

    if not os.path.exists(unique_output_path):
        raise FileNotFoundError(
            f"Expected output file not found: {unique_output_path}\nScript output:\n{process.stdout}")

    df_unique = pd.read_csv(unique_output_path)

    if df_unique.empty:
        return {"unique_count": 0, "top1_smiles": "N/A", "output_csv_path": unique_output_path,
                "raw_output_path": raw_output_path}

    unique_count = len(df_unique)
    top1_smiles = df_unique.iloc[0]['smiles']

    if not os.path.exists(top10_img_path):
        top10_smiles = df_unique['smiles'].head(10).tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in top10_smiles if Chem.MolFromSmiles(smi)]
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=[s[:30] for s in top10_smiles])
            img.save(top10_img_path)

    return {
        "unique_count": unique_count,
        "top1_smiles": top1_smiles,
        "output_csv_path": unique_output_path,
        "raw_output_path": raw_output_path
    }


def run_generation_and_prediction(predictor_obj, scaffold_smi, output_dir, num_random, num_decor, status_callback=None,
                                  progress_callback=None, **kwargs):
    gen_result = run_generation(
        scaffold_smi=scaffold_smi,
        output_dir=output_dir,
        num_random=num_random,
        num_decor=num_decor,
        status_callback=status_callback
    )

    if gen_result['unique_count'] == 0:
        if status_callback:
            status_callback("No Molecules Generated. Prediction Skipped.")
        return {**gen_result, "count_type1": 0, "percentage_type1": 0}

    csv_path = gen_result['output_csv_path']
    df = pd.read_csv(csv_path)

    smiles_list = df['smiles'].tolist()

    if status_callback:
        status_callback("==================================================\nGeneration Done. \nStarting Prediction...")

    pred_output_path = csv_path.replace("_unique.csv", "_unique_with_predictions.csv")

    pred_result = predictor_obj.run_prediction(
        smiles_list=smiles_list,
        output_path=pred_output_path,
        status_callback=status_callback,
        progress_callback=progress_callback
    )

    combined_result = {
        **gen_result,
        **pred_result,
        "prediction_csv_path": pred_output_path
    }
    return combined_result