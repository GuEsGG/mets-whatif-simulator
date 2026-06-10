# MetS What-If Simulation Module

Patient-level what-if simulation for metabolic syndrome risk, from the paper
"A What-If Simulation Framework for Individual Metabolic Syndrome Risk".

## Contents
- metsim.py: the simulation module (MetSimulator).
- simulator_artifacts.pkl: trained sex-stratified models, isotonic calibrators,
  simulable-variable lists, and plausibility constraints. Contains no individual data.
- Simulation_Train_Test.ipynb: full training/evaluation/figures pipeline.

## Usage

    from metsim import MetSimulator
    sim = MetSimulator("simulator_artifacts.pkl")
    res = sim.simulate_patient(
        {"WC": 101, "BMI": 30, "athero_index": 3.1},
        sex="female", has_lab=True,
        modifications={"WC": 93, "athero_index": 2.5})
    print(res["baseline_risk"], res["modified_risk"])
    sim.plot_simulation(res)

## Data
The Tlalpan 2020 cohort data are not included due to participant confidentiality
and institutional restrictions (Instituto Nacional de Cardiologia Ignacio Chavez).
