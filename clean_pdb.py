def clean(input_pdb):
    
    import MDAnalysis as mda
    import subprocess
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import warnings
    from openmm.app import PDBFile, Modeller, ForceField
    from openmm import unit
    import numpy as np
    import sys
    warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

    
    from openmm.app import PDBFile as OMM_PDBFile
    from pdbfixer import PDBFixer

    #input_pdb = sys.argv[1]
    output = input_pdb.rsplit(".", 1)[0]
    pdb_all = mda.Universe(input_pdb)

    water = {"HOH", "WAT", "TIP3", "SOL"}

    lig_residues = []

    for res in pdb_all.residues:
        
        if res.atoms.select_atoms("protein").n_atoms > 0:
            continue

        
        if res.resname.strip() in water:
            continue

        
        if res.atoms.n_atoms < 5:
            continue

        lig_residues.append(res)

    if not lig_residues:
        raise ValueError("No ligand detected.")

    #print("ligand_list", lig_residues)

    lenth_lig = []
    for i in lig_residues:
        lenth_lig.append(i.atoms.n_atoms)
    #print("lig atom counts:", lenth_lig)

    arr = np.array(lenth_lig, dtype=int)
    max_index = int(np.argmax(arr))
    #print("IDX", max_index)

    lig = lig_residues[max_index]

    
    try:
        chain_id = lig.atoms.chainIDs[0].strip()  
    except Exception:
        chain_id = ""

    segid = lig.atoms.segids[0]
    #print("Ligand chainID:", chain_id if chain_id else "NA", "| segid:", segid)

    chain = None
    chain_label = None

    if chain_id:
        
        try:
            chain = pdb_all.select_atoms(f"chainID {chain_id}")
        except Exception:
            chain = None

        if chain is None or chain.n_atoms == 0:
            try:
                chain = pdb_all.select_atoms(f"chainid {chain_id}")
            except Exception:
                chain = None

        if chain is not None and chain.n_atoms > 0:
            chain_label = chain_id

    if chain is None or chain.n_atoms == 0:
        chain = pdb_all.select_atoms(f"segid {segid}")
        chain_label = segid

    if chain is None or chain.n_atoms == 0:
        raise RuntimeError("Chain selection returned 0 atoms. Your PDB likely lacks usable chain IDs/segids.")

    chain.write(f"{output}_chain_{chain_label}.pdb")

    chain_protein_only = chain.select_atoms("protein")
    protein_pdb_path = f"{output}_chain_{chain_label}_protein.pdb"
    chain_protein_only.write(protein_pdb_path)

    lig_code = lig.resname
    #print("lig_code:", lig_code)

    
    lig.atoms.write("ligand.pdb")

    
    fixed_pdb_path = f"{output}_chain_{chain_label}_protein_fixed_H.pdb"

    try:
        
        fixer = PDBFixer(filename=protein_pdb_path)

        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        with open(fixed_pdb_path, "w") as f:
            OMM_PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

        #print("doneYOOOOOOOO (fixed + added H):", fixed_pdb_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    
    lig_n = Chem.MolFromPDBFile("ligand.pdb", removeHs=False, sanitize=False)
    lig_n = Chem.RemoveHs(lig_n, sanitize=False)

    with open("ligand.pdb", "r") as f:
        list_coords = []
        for line in f:
            if line.startswith("HETATM"):
                list_coords.append(line)   
            elif line.startswith("CONECT"):
                list_coords.append(line)   

    with open("ligand_correct.pdb", "w") as f:
        f.writelines(list_coords)

    subprocess.run(
        ["obabel", "-i", "pdb", "ligand_correct.pdb", "-o", "sdf", "-O", "ligand.sdf", "-xk"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    suppl = Chem.SDMolSupplier("ligand.sdf", removeHs=False)
    mol = suppl[0]
    if mol is None:
        raise ValueError("RDKit failed to read SDF. Your SDF is broken.")

    smiles = Chem.MolToSmiles(mol, canonical=True)
    #print(smiles)

    template = Chem.MolFromSmiles(smiles)
    lig_n = AllChem.AssignBondOrdersFromTemplate(template, lig_n)

    lig_n = Chem.AddHs(lig_n, addCoords=True)
    Chem.MolToPDBFile(lig_n, "ligand_H_good.pdb")

    lig_H = Chem.AddHs(lig_n, addCoords=True)
    Chem.MolToPDBFile(lig_H, "ligand_H_rawww.pdb")

    for atom in lig_H.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(lig_code)
            info.SetResidueNumber(1)

            
            ch = str(chain_label) if chain_label is not None else "A"
            if len(ch) == 0:
                ch = "A"
            info.SetChainId(ch[0])

            atom.SetMonomerInfo(info)
        else:
            info.SetResidueName(lig_code)
            info.SetResidueNumber(1)
            ch = str(chain_label) if chain_label is not None else "A"
            if len(ch) == 0:
                ch = "A"
            info.SetChainId(ch[0])

    Chem.MolToPDBFile(lig_H, "ligand_H_raw.pdb")

    with open("ligand_H_raw.pdb") as fin, open("ligand_H.pdb", "w") as fout:
        for line in fin:
            if line.startswith("ATOM"):
                line = "HETATM" + line[6:]
            fout.write(line)

    prot_H = mda.Universe(fixed_pdb_path)
    lig_H_u = mda.Universe("ligand_H.pdb")

    complex_u = mda.Merge(prot_H.atoms, lig_H_u.atoms)
    complex_out = f"{output}_chain_{chain_label}_complex_H.pdb"
    complex_u.atoms.write(complex_out)
