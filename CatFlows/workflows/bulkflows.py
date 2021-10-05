from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.core.periodic_table import Element

from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
)

#from fireworks import LaunchPad
#from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.dft_settings.settings import (
    set_bulk_magmoms,
)

#from CatFlows.workflows.surface_energy import SurfaceEnergy_WF
#from CatFlows.workflows.wulff_shape import WulffShape_WF
#from CatFlows.workflows.slab_ads import SlabAds_WF
#from CatFlows.adsorption.adsorbate_configs import OH_Ox_list


# Bulk structure workflow method
class BulkFlows:
    """
    BulkFlow is a general method to automatize DFT workflows to find the Equilibrium Bulk
    Structure with the right magnetic moments and Ordering.

    Args:
        bulk_structure
        conventional_standard

    Returns:
        The launchpad ready for execution!
    """
    def __init__(self, bulk_structure, conventional_standard=True):

        # Bulk structure
        self.bulk_structure = self._read_cif_file(bulk_structure)
        self.original_bulk_structure = self.bulk_structure.copy()
        # Convetional standard unit cell
        if conventional_standard:
            self.bulk_structure = self._get_conventional_standard()
        # Decorate with oxidations states 
        self.bulk_structure = self._get_oxidation_states()
        # Decorate the bulk structure with sites properties
        self.bulk_structure = self._get_wyckoffs_positions()
        # Get magmoms for metals 
        self.magmoms_dict = self._get_metals_magmoms()
        print(self.magmoms_dict)

    def _read_cif_file(self, bulk_structure, primitive=False):
        """Parse CIF file with PMG"""
        struct = CifParser(bulk_structure).get_structures(primitive=primitive)[0]
        return struct

    def _get_oxidation_states(self):
        """Parse CIF file with PMG"""
        oxid_transformer = AutoOxiStateDecorationTransformation()
        struct_new = oxid_transformer.apply_transformation(self.bulk_structure)
        return struct_new

    def _get_conventional_standard(self):
        """Convert Bulk structure to conventional standard"""
        SGA = SpacegroupAnalyzer(self.bulk_structure)
        bulk_structure = SGA.get_conventional_standard_structure()
        return bulk_structure

    def _get_wyckoffs_positions(self):
        """Decorates the bulk structure with wyckoff positions"""
        bulk_structure = self.bulk_structure.copy()
        SGA = SpacegroupAnalyzer(bulk_structure)
        bulk_structure.add_site_property("bulk_wyckoff", SGA.get_symmetry_dataset()["wyckoffs"])
        bulk_structure.add_site_property("bulk_equivalent", SGA.get_symmetry_dataset()["equivalent_atoms"].tolist())
        return bulk_structure

    def _get_metals_magmoms(self):
        """Returns dict with metal symbol and magmoms assigned"""
        bulk_structure = set_bulk_magmoms(self.bulk_structure)
        metals_symb = [metal for metal in self.original_bulk_structure.species if Element(metal).is_metal]
        magmoms_list = bulk_structure.site_properties["magmom"]
        magmoms_dict = {}
        for metal, magmom in zip(metals_symb, magmoms_list):
            magmoms_dict.update({str(metal): magmom})
        return magmoms_dict