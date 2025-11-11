import numpy as np
import MDAnalysis as mda

def get_H_bond(input_pdb, ligand_name):
    #Complete_code_for_H-bond_ and_Salt_Bridge
    pdb=mda.Universe(input_pdb)
    lig_name=ligand_name
    all_atoms=pdb.select_atoms(f"protein or resname {lig_name}")  #select protein or ligand atom
    all_atoms_P=pdb.select_atoms("protein")                       #select only atoms of protein
    all_atoms_L=pdb.select_atoms(f"resname {lig_name}")           #select ligand atoms only
    all_H=all_atoms.select_atoms("element H")
    protein_idx=pdb.select_atoms("protein").indices.tolist()       
    #print(protein_idx)
    #print(type(protein_idx))
    ligand_idx=pdb.select_atoms(f"resname {lig_name}").indices.tolist()
    #print(ligand_idx)
    heavy_atoms_P=all_atoms_P.select_atoms("element O or element N or element S")     #selecting electronegative atoms
    heavy_atoms_L=all_atoms_L.select_atoms("element O or element N or element S")     
    hydrogen_bonds=[]
    dH_max=1.7               #maximum bond length limit for donor atom and hydrogen   
    d_a_max=4.0              #maximum donor acceptor distance
    h_a_max=2.7              #maximum hydrogen acceptor distance
    angle_min=100             #bond angle limit for hydrogen bond

    def label(a):
        return f"{a.resid}-{a.resname}-{a.name}"
    for H in all_H:
        H_position=H.position
        if H.index in protein_idx:                #cosidering if protein is having hydrogen and donor group and ligand is having acceptor group
            donor_group=heavy_atoms_P
            acceptor_group=heavy_atoms_L
        elif H.index in ligand_idx:               #cosidering if ligand is having hydrogen and donor group and protein is having acceptor group
            donor_group=heavy_atoms_L
            acceptor_group=heavy_atoms_P

        distance_same=np.linalg.norm(donor_group.positions-H_position, axis=1)    #distances of all donor atoms to all hydrogens 
        donor_index=np.argmin(distance_same)    #(index of each hydrogen having min distance from heavy atom(donor atoms))
        donor_atom=donor_group[donor_index]     #donor_atom now represents the specific donor atom (e.g., LYS-NZ, ARG-NE, etc.) that is closest to the current hydrogen.
        dH_distance=distance_same[donor_index]     #the numerical value of the minimum distance (the donor–hydrogen distance)

        if dH_distance>dH_max:
            continue
        #else:
            #print(H,dH_distance,donor_atom)
        
        donor_atom_position=donor_atom.position    
        dH_vector=(H_position-donor_atom_position)/np.linalg.norm(H_position-donor_atom_position)  # this donor hydrogen bond vector is calculated for bond angle caculations

        for acceptor in acceptor_group:
            acceptor_atom_position=acceptor.position  
            donor_acceptor_distance=np.linalg.norm(acceptor_atom_position-donor_atom_position) 
            if donor_acceptor_distance>d_a_max:
                continue

            hydrogen_acceptor_distance=np.linalg.norm(acceptor_atom_position-H_position)
            
            if hydrogen_acceptor_distance>h_a_max:
                continue
            
            h_a_vector=(acceptor_atom_position-H_position)/np.linalg.norm(acceptor_atom_position-H_position)    #for calculating angle
            cos_theta = np.clip(np.dot(dH_vector, h_a_vector), -1.0, 1.0)         
            angle = float(np.degrees(np.arccos(cos_theta)))
            angle=180-angle
            if angle >= angle_min:
                hydrogen_bonds.append({
                    "donor_atom": label(donor_atom),
                    "hydrogen":   label(H),
                    "acceptor_atom": label(acceptor),
                    "angle": angle
                })

    #print(hydrogen_bonds)
    return hydrogen_bonds

def salt_bridge(input_pdb, ligand_name):

    pdb=mda.Universe(input_pdb)
    lig_name=ligand_name

    all_atoms=pdb.select_atoms(f"protein or resname {lig_name}")
    all_atoms_P=pdb.select_atoms("protein")
    all_atoms_L=pdb.select_atoms(f"resname {lig_name}")

    protein_positive = all_atoms_P.select_atoms(
        "(resname LYS and name NZ) or "
        "(resname ARG and (name NH1 or name NH2 or name NE)) or "
        "((resname HIP or resname HSP) and (name ND1 or name NE2))"
    )                                                                     # selecting positive amino acids and possiblr atoms in those which can carry positive charge 
    protein_negative=all_atoms_P.select_atoms("(resname ASP and (name OD1 or name OD2)) or" "(resname GLU and (name OE1 or name OE2)) or" "(name OXT)")    # selecting negative amino acids and possiblr atoms in those which can carry negative charge 

    ligand_positive=all_atoms_L.select_atoms(f"(resname {lig_name} and element N)")    # possible ligand positive part 
    ligand_negative=all_atoms_L.select_atoms(f"(resname {lig_name} and element O) or" f"(resname {lig_name} and element S)")  # possible ligand negative part 
    ligand_hydrogens=all_atoms_L.select_atoms(f"resname {lig_name} and element H")
    ligand_carbons=all_atoms_L.select_atoms(f"resname {lig_name} and element C")

    ligand_nitrogens_positons=ligand_positive.positions




    pairs=[]
    cutoff_salt_bridge=3.2              

    def label(a):
        return f"{a.resid}-{a.resname}-{a.name}"


    for N in protein_positive:                                #considering positive part is protein and ligand having negative charge
        N_position=N.position
        O_ligand_position_all=ligand_negative.positions         # position of all oxygen on ligand 
        dist=np.linalg.norm(N_position-O_ligand_position_all, axis=1)    #distances of all nitrogens of protein to all oxygen of ligand 
        #print(O_ligand_position_all)
        #print(N_position)
        #print(dist)
        for i,j in enumerate(dist):                                        #i is index of each distance(O-N) we calculated above
            if j <= cutoff_salt_bridge:
                ligand_oxygen=ligand_negative[i]
                ligand_oxygen_position=ligand_oxygen.position                    
                dist_O_X_ligand=np.linalg.norm(ligand_oxygen_position-all_atoms_L.positions, axis=1)   #distance of oxygen from all atoms of ligand
                #print(dist_O_X_ligand)
                indices_2_3 = np.argsort(dist_O_X_ligand)[1:3]       #to find index of 2nd and 3rd minimum distances
                #print(indices_2_3)
                d_2= dist_O_X_ligand[indices_2_3[0]]                 #calculating the these minimum distances so that we can estimate number bonds oxygen forming so that we can estimate charge on oxygen
                d_3= dist_O_X_ligand[indices_2_3[1]]
                #print(d_2 , d_3)

                min_d_2_d_3=min(d_2,d_3)

                #print(min_d_2_d_3)

                if d_2<1.6 and d_3<1.6:                              #setting minimum bond length for oxygen 
                    continue

                elif min_d_2_d_3 <1.30:                        #This is for oxygen having double bond and no charge as above condition include both but this is to filter out double bond specially
                    continue

                else:
                    dist_N_O= np.linalg.norm(N_position-ligand_oxygen_position)        #distance between protein nitrogen and ligand oxygen if there exists a salt bridge
                    pairs.append({"Protein_N": label(N),
                    "Ligand_O":   label(ligand_oxygen),
                    "distance": dist_N_O})



    for O in protein_negative:         #Considering Protein having negative charge and ligand is having positive charge 
        O_protein_position= O.position               #
        dist_O_N=np.linalg.norm(O_protein_position-ligand_nitrogens_positons, axis=1)  #distances of all protein oxygen and all ligand nitrogens
        for k,l in enumerate(dist_O_N):             # k is index of the distances we calculated above and l is distance
            if l<= cutoff_salt_bridge:
                #print(k,l)
                nitrogen_position_ligand= ligand_nitrogens_positons[k]      #Position of nitrogen which satisfies cutoff
                ligand_nitrogen=ligand_positive[k]
                dist_N_X_ligand=np.linalg.norm(nitrogen_position_ligand-all_atoms_L.positions, axis=1)  #Distance of that nitrogen to all atoms of ligand
                indices_2_3_4_5 = np.argsort(dist_N_X_ligand)[1:5]     #to find the indeex of 2nd,3rd,4thand 5th closet atom distances to calculate no. of bond nitrogen having to estimate if it is charged or not
                #print(indices_2_3_4_5)
                d_2_N= dist_N_X_ligand[indices_2_3_4_5[0]]
                d_3_N= dist_N_X_ligand[indices_2_3_4_5[1]]
                d_4_N= dist_N_X_ligand[indices_2_3_4_5[2]]
                d_5_N= dist_N_X_ligand[indices_2_3_4_5[3]]
                #print(d_2_N, d_3_N,d_4_N,d_5_N)

                distances_N=dist_N_X_ligand[indices_2_3_4_5]     #list of those distances
                #print(distances_N)
                no_of_bonds_N=0
                no_of_N_H_bonds=0

                for D in distances_N:
                    if D <= 1.55:
                        no_of_bonds_N+=1                      #for loop for no. of bonds
                    
                    else:
                        continue
                
                for D in distances_N:                          #This is to separate out N-H bonds as those are shortest bond and we don't want them to be considered in double bond condition
                    if D<=1.05:
                        no_of_N_H_bonds+=1
                    else:
                        continue

                #print(no_of_N_H_bonds)

                #print(no_of_bonds_N) 


                if no_of_bonds_N>3: 
                                                        #if no. of bonds >3 means nitrogen is positively charged
                    pairs.append({"Protein_O": label(O),
                    "Ligand_N":   label(ligand_nitrogen),
                    "distance": l})

                elif no_of_bonds_N==3:
                    if no_of_bonds_N-no_of_N_H_bonds==1 and d_4_N<1.3:                   # This means total bonds of nitrogen =3 and N-H bonds are 2 so if other bond is double bond then N is positively charged
                        pairs.append({"Protein_O": label(O),
                        "Ligand_N":   label(ligand_nitrogen),
                        "distance": l})

                    elif no_of_bonds_N-no_of_N_H_bonds==2 and d_3_N<1.3:               # This means total bonds of nitrogen =3 and N-H bonds are 1 so if other bond can be 2 single bonds but this has already filtered out in first condition or also it can have one double bond and one single bond then N is positively charged and we will append in that case.
                        pairs.append({"Protein_O": label(O),
                        "Ligand_N":   label(ligand_nitrogen),
                        "distance": l})

                    elif no_of_bonds_N-no_of_N_H_bonds==3 and d_2_N<1.3:            # This means total bonds of nitrogen =3 and N-H bonds are 0 so if any of the bond is double bond then N is positively charged and here we are setting limits for shortest bond distance if that satisfies double bond criteria then N is positively charged.
                        pairs.append({"Protein_O": label(O),
                        "Ligand_N":   label(ligand_nitrogen),
                        "distance": l})

                    else:
                        continue



                    #print(distances_N,"D")                                 #if no. of bond =3
                
                elif no_of_bonds_N==2:                        #If no. of bonds =2 and min distance less than 1.5 then nitrogen is having triple bond and positively charged
                    min_dist=np.min(distances_N)
                    if min_dist<=1.15:
                        pairs.append({"Protein_O": label(O),
                        "Ligand_N":   label(ligand_nitrogen),
                        "distance": l})
                    else:
                        continue
                    
                else:
                    continue


    return pairs



        
