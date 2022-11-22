from pymatgen.core.surface import SlabGenerator, Structure, Lattice
from pymatgen.core.periodic_table import Element

import random, math
import numpy as np

def get_repeat_from_min_lw(slab, min_lw):
    """
    Modified version of algorithm from adsorption.py for determining the super cell 
        matrix of the slab given min_lw. This will location the smallest super slab 
        cell with min_lw by including square root 3x3 transformation matrices
    """
    
    xlength = np.linalg.norm(slab.lattice.matrix[0])
    ylength = np.linalg.norm(slab.lattice.matrix[1])
    xrep = np.ceil(min_lw / xlength)
    yrep = np.ceil(min_lw / ylength)
    rtslab = slab.copy()
    rtslab.make_supercell([[1,1,0], [1,-1,0], [0,0,1]])
    rt_matrix = rtslab.lattice.matrix
    xlength_rt = np.linalg.norm(rt_matrix[0])
    ylength_rt = np.linalg.norm(rt_matrix[1])
    xrep_rt = np.ceil(min_lw / xlength_rt)
    yrep_rt = np.ceil(min_lw / ylength_rt)

    xrep = xrep*np.array([1,0,0]) if xrep*xlength < xrep_rt*xlength_rt else xrep_rt*np.array([1,1,0]) 
    yrep = yrep*np.array([0,1,0]) if yrep*ylength < yrep_rt*ylength_rt else yrep_rt*np.array([1,-1,0]) 
    zrep = [0,0,1]
    return [xrep, yrep, zrep]

def get_HEO_rutile_lattice(metals):
    """
    Build the lattice of a theoretical HEO based on the average lattices (a and c) of the 
        list of metals that make up a rutile. The a and c of each MO2 is guessed based 
        on the atomic radius of the metal with linear fitting against the lattice a and c
        
        metals:: List of metals (strings)
    """
    
    c = np.mean([1.1700734279862302*Element(el).atomic_radius+1.4515684135250173 for el in metals])
    a = np.mean([1.1817107611718527*Element(el).atomic_radius+3.046503305784393 for el in metals])
    return Lattice([[a,0,0], [0,a,0], [0,0,c]])

def get_HEO_slab(metals, miller_index, min_slab_size, min_vacuum_size, 
                 lll_reduce=True, max_normal_search=1):
    
    """
    Create a high entropy oxide with a set of metals. Strategy is to maximize 
        entropy by making a structure as symmetrically random as possible 
        thus making it the config with the largest number of ensembles.
        
        metals:: List of metals (strings)
    """
    
    
    # Create a generic rutile lattice
    HeO2 = Structure(get_HEO_rutile_lattice(metals), 
                    ['He', 'He', 'O', 'O', 'O', 'O'],
                    [[0.5, 0.5, 0.5], [0., 0., 0.], [0.200764, 0.799236, 0.5], 
                     [0.799236, 0.200764, 0.5], [0.700764, 0.700764, 0.],
                     [0.299236, 0.299236, 0.]])

    # get supercell of slab
    slabgen = SlabGenerator(HeO2, miller_index, min_slab_size, min_vacuum_size, lll_reduce=lll_reduce, 
                            max_normal_search=max_normal_search, center_slab=True, primitive=True)    
    slab = slabgen.get_slabs()[0]
    repeat = get_repeat_from_min_lw(slab, 8)
    slab.make_supercell(repeat)
    
    # get the bond length between M-M
    Hesite = [site for site in slab if site.species_string == 'He'][0]
    iHesite = slab.index(Hesite)
    min_blength = min([slab.get_distance(nn.index, iHesite) for nn in 
                       slab.get_neighbors(Hesite, 5, include_index=True) if nn.species_string != 'O'])*1.1
    
    # get the metallic coordination number
    m_nn = len([nn for nn in slab.get_neighbors(Hesite, min_blength, include_index=True) 
                if nn.species_string != 'O'])
    
    # make a list of all metals we want in the slab so we can keep track of which ones have already
    # been placed. This way we make sure all metals are as equally distributed as possible in the slab
    n = math.ceil(48/5)
    total_metals = []
    for m in metals:
        total_metals.extend([m]*n)
        
    # start replacing He atoms with metals, the algorithm will check if any neighboring metals exist for 
    # current site and if it does, choose a different metal. This ensures maximum entropy by making an ensemble
    # as randomized and unordered as possible that this type of arrangement will have the largest ensemble
    HEO = slab.copy()
    for i, site in enumerate(HEO):
        if site.species_string == 'He':
            m = random.sample(total_metals, 1)[0]
            neighbor_metals = [nn.species_string for nn in 
                               HEO.get_neighbors(site, min_blength, include_index=True) if nn.species_string != 'O']

            while m in neighbor_metals:
                m = random.sample(total_metals, 1)[0]
                if len(total_metals) < m_nn+(n*len(metals)-slab.composition.as_dict()['He']):
                    break
            HEO.replace(i, m)
            del total_metals[total_metals.index(m)]

    return HEO