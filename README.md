# MO Wulff Workflow

![workflow](img/MO_Workflow.png)

A starting point for a general Metal oxide Surfaces Workflow able to build the Wulff construction from DFT calculations automatized by Pymatgen, FireWorks and Atomate.

# Goals

The goal of this repository is to make accessible the code to everyone in group and
improve it in a collaborative way. Hopefully, ending on a great tool for many applications
in our group and more.

# Javi ToDo List

- [x] Metal oxide surfaces as non-dipolar, symmetric and Stoichiometric
- [x] Optimize both oriented bulk + slab model (VASP)
- [x] Surface Energy task after optimization
- [x] Wulff shape Analysis (as separeted Task)
- [x] Decorate bulk structure with magnetic moments (MAGMOM) based on crystal field theory
- [x] Bulk magnetic orders (NM, AFM, FM)
	- [x] Enumerate and decorate bulk structure with magmom and orderings
	- [x] General implementation of Optimization + Deformation + EOS_fitting for each magnetic order
	- [x] Bulk Stability Analysis (after single-point)
- [x] Surface Pourbaix Diagram Analysis
- [ ] Reactivity OER
	- [ ] Single site WNA
        - [ ] Double site WNA
        - [ ] I2M
- [ ] FrequenciesFW and ThermoCorrections
- [ ] Reactivity OER (WNA - Single site)
- [ ] SP with Hybrid Functional (worth it?)


# Richard ToDo List

- [x] Basic rotational DoF for OH and OOH (not generalized for other 85 molecules yet).
- [ ] Metal oxide surfaces with dipolar moment (oc21-dataset)
- [ ] Non-stoichiometric slab models (Terminations) + gamma_hkl (with chemical potential)
- [ ] Include GGA+U method
- [ ] **IDEA** Include defects in slabs

# Yuri ToDo List

- [x] Generate unique hashes to identify/query jobs in database
- [x] Look into passing job info from parent to child fireworks
- [x] Add ContinueOptimizeFW to refactor branch
- [x] Transfer of magnetic moments across parents to childs
- [x] WAVECAR (copy, delete, gzip) across parents to childs
- [ ] WAVECAR Transfer across different filesystems (really?)
- [x] Add compatibility with 2 nodes
- [ ] Hunting for bugs!

# Yuri/Javi ToDo List

- [x] Clean-up code for ADS_SLAB generation and add to refactor
- [x] Refactor branch merging with kubernetes_test
- [ ] Benchmark RuO2 and IrO2
	- [x] IrO2 Manually tested and getting close to good values!
- [x] PARSING ERROR!!!!! (was the container!)
- [ ] Get vasp ouputs for fake_vasp
