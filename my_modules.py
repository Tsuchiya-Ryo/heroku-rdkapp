import string, random, os, base64
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, rdCoordGen, MolStandardize
from rdkit.Chem.AllChem import ComputeGasteigerCharges
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from rdkit.Chem.AtomPairs import Sheridan, Pairs
from datetime import datetime


def random_string(n):
    randlist = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return ''.join(randlist)

def label(mol, flag="num", Set=None, N=None):
    for i, atom in enumerate(mol.GetAtoms()):
        if flag == "num":
            label = str(i)
        elif flag == "charge":
            label = str(round(Set[i], N))
        elif flag == "type":
            label = Set[i]
        atom.SetProp("atomLabel", label)
    return mol


def calc_specs(mol, img_dir, labelnum_on_structure=True):

    bondcount = Chem.RemoveHs(mol)
    Chem.Kekulize(bondcount)
    single = 0
    double = 0
    triple = 0

    if bondcount.GetNumBonds() >= 1:
        for i in bondcount.GetBonds():
            if str(i.GetBondType()) == "SINGLE":
                single += 1
            if str(i.GetBondType()) == "DOUBLE":
                double += 1
            if str(i.GetBondType()) == "TRIPLE":
                triple += 1
    else:
        pass

    spec = {}
    spec["MolWt"] = round(Descriptors.MolWt(mol), 3)
    spec["SingleBond"] = single
    spec["DoubleBond"] = double
    spec["TripleBond"] = triple
    spec["RotBond"] = Descriptors.NumRotatableBonds(mol)
    spec["AromaticAtom"] = len(mol.GetAromaticAtoms())
    spec["HeteroAtom"] = Descriptors.NumHeteroatoms(mol)
    spec["FracSP3"] = round(Descriptors.FractionCSP3(mol), 3)
    spec["LogP"] = round(Descriptors.MolLogP(mol), 3)
    spec["TPSA"] = round(Descriptors.TPSA(mol), 3)
    spec["HA"] = Descriptors.NumHAcceptors(mol)
    spec["HD"] = Descriptors.NumHDonors(mol)

    # DRAW
    mol.RemoveAllConformers()
    rdCoordGen.AddCoords(mol)
    view = rdMolDraw2D.MolDraw2DSVG(400,400)
    option = view.drawOptions()
    option.addAtomIndices=labelnum_on_structure
    view.DrawMolecule(mol)
    view.FinishDrawing()
    img = view.GetDrawingText()
    # img = Draw.MolToImage(mol, useSVG=True, size=(300,300))

    filename = "{0:%Y%m%d%H%M%S}".format(datetime.now()) + random_string(3) + ".svg"
    save_path = os.path.join(img_dir, filename)

    return spec, img, filename, save_path

def type_mapping(mol, img_dir):
    types = Sheridan.AssignPattyTypes(mol)
    typemol = label(mol, flag="type", Set=types)
    type_img = Draw.MolToImage(typemol, useSVG=True, size=(300,300))
    type_filename = "{0:%Y%m%d%H%M%S}".format(datetime.now()) + random_string(3) + "_type.png"
    type_save_path = os.path.join(img_dir, type_filename)

    types_dict = {}
    for i, t in enumerate(types):
        types_dict[i] = t

    return type_img, type_filename, type_save_path, types_dict

# GASTEIGER CHARGE MAPPING
def calc_gasteiger(mol, img_dir):
    gmol = Chem.RemoveHs(mol)
    ComputeGasteigerCharges(gmol)
    charges = []
    hcharges = []
    atoms = gmol.GetAtoms()
    for atom in atoms:
        charge = float(atom.GetProp("_GasteigerCharge"))
        hcharge = float(atom.GetProp("_GasteigerHCharge"))
        charges.append(charge)
        hcharges.append(hcharge)
    gimg = SimilarityMaps.GetSimilarityMapFromWeights(gmol, weights=charges, colorMap="bwr", size=(400,400))

    gfilename = "{0:%Y%m%d%H%M%S}".format(datetime.now()) + random_string(3) + "_charge.png"
    gsave_path = os.path.join(img_dir, gfilename)

    return gmol, charges, gimg, gfilename, gsave_path

# CHARGE VALUE MAPPING
def charge_value_mapping(mol, charges, ndigits, img_dir):
    charge_mol = label(mol, flag="charge", Set=charges, N=ndigits)
    charge_img = Draw.MolToImage(charge_mol, useSVG=True, size=(400,400))
    charge_filename = "{0:%Y%m%d%H%M%S}".format(datetime.now()) + random_string(3) + "_chargevalue.png"
    charge_save_path = os.path.join(img_dir, charge_filename)

    charge_dict = {}
    for i, ch in enumerate(charges):
        charge_dict[i] = round(ch, ndigits)

    return charge_img, charge_filename, charge_save_path, charge_dict

def label_mapping(mol, img_dir):
    lmol = label(mol, flag="num")
    label_img = Draw.MolToImage(lmol, useSVG=True, size=(300,300))
    label_filename = "{0:%Y%m%d%H%M%S}".format(datetime.now()) + random_string(3) + "_label.png"
    label_save_path = os.path.join(img_dir, label_filename)

    return label_img, label_filename, label_save_path

def standardize(mol):
    norm = MolStandardize.normalize.Normalizer()
    lfc = MolStandardize.fragment.LargestFragmentChooser()
    uc = MolStandardize.charge.Uncharger()
    mol_st = uc.uncharge(lfc.choose(norm.normalize(mol)))
    return mol_st

def smiles_or_inchi(STR):
    if Chem.MolFromSmiles(STR):
        mol = Chem.MolFromSmiles(STR)
    elif Chem.MolFromInchi(STR):
        mol = Chem.MolFromInchi(STR)
    else:
        mol = None
    return mol

def calc_bits(molA, molB, num_bits, molsPerRow):
    bitinfoA = {}
    bitinfoB = {}
    bits = {}
    fpA = AllChem.GetMorganFingerprintAsBitVect(molA, 2, 2048, bitInfo=bitinfoA)
    fpB = AllChem.GetMorganFingerprintAsBitVect(molB, 2, 2048, bitInfo=bitinfoB)

    common_bits = list(set(bitinfoA.keys()) & set(bitinfoB.keys()))

    bits["MolA_bits"] = list(bitinfoA.keys())
    bits["MolB_bits"] = list(bitinfoB.keys())
    bits["common_bits"] = common_bits
    bits["MolA_bits_count"] = len(bitinfoA.keys())
    bits["MolB_bits_count"] = len(bitinfoB.keys())
    bits["common_bits_count"] = len(common_bits)

    molA_tupples = ((molA, bit, bitinfoA) for bit in common_bits[:num_bits])
    common_bit_img = base64.b64encode(Draw.DrawMorganBits(molA_tupples, useSVG=False, molsPerRow=molsPerRow, legends=["bit No.{}".format(str(i)) for i in common_bits[:num_bits]])).decode("ascii")

    coefs = {}
    coefs["Tanimoto"] = round(DataStructs.FingerprintSimilarity(fpA, fpB, metric=DataStructs.TanimotoSimilarity), 3)
    coefs["Dice"] = round(DataStructs.FingerprintSimilarity(fpA, fpB, metric=DataStructs.DiceSimilarity), 3)

    return bits, bitinfoA, bitinfoB, common_bits, common_bit_img, coefs

def similarity_map(molA, molB, img_dir):
    sim_filename=None
    weight = SimilarityMaps.GetAtomicWeightsForFingerprint(molB, molA, SimilarityMaps.GetMorganFingerprint)
    if not weight:
        pass
    else:
        sim_image = SimilarityMaps.GetSimilarityMapFromWeights(molA, weight, size=(400,400))
        sim_filename = "{0:%Y%m%d%H%M%S}".format(datetime.now()) + random_string(3) + "_sim.png"
        sim_save_path = os.path.join(img_dir, sim_filename)
    return sim_image, sim_filename, sim_save_path

def grid_image(molA, molB, img_dir):

    molA.RemoveAllConformers()
    molB.RemoveAllConformers()
    rdCoordGen.AddCoords(molA)
    rdCoordGen.AddCoords(molB)
    view = rdMolDraw2D.MolDraw2DSVG(500,250,250,250)
    option = view.drawOptions()
    option.padding=0.2
    option.legendfontSize=18
    view.DrawMolecules([molA,molB], legends=["MoleculeA", "MoleculeB"])
    view.FinishDrawing()
    mols_image = view.GetDrawingText()

    # mols_image = Draw.MolsToGridImage([molA, molB], molsPerRow=2, subImgSize=(250,250), legends=['Molecule_A', 'Molecule_B'])
    mols_filename = "{0:%Y%m%d%H%M%S}".format(datetime.now()) + random_string(3) + "_mols.svg"
    mols_save_path = os.path.join(img_dir, mols_filename)
    return mols_image, mols_filename, mols_save_path


if __name__ == "__main__":
    random_string(n)
    label(mol, flag="num", Set=None, N=None)
    calc_specs(mol, img_dir)
    calc_gasteiger(mol)
    type_mapping(mol, img_dir)
    charge_value_mapping(mol, charges, ndigits, img_dir)
    label_mapping(mol, img_dir)
    standardize(mol)
    smiles_or_inchi(STR)
    calc_bits(molA, molB, num_bits, molsPerRow)
    similarity_map(molA, molB, img_dir)
    grid_image(molA, molB)