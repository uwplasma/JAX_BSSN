# JAX-BSSN - Numerical Relativity in Python with JAX

This is a Python/JAX implementation of the NR1 numerical relativity code forked from 20K on Github.
This code uses the BSSN formulation of the ADM 3+1 decomposition of general relativity. The evolution
equations support nonzero shift terms with a single-variable Gamma-driver shift gauge, while the included
wave initial data still default to zero shift. Periodic finite-difference stencils are implemented, with
optional super-Gaussian boundary filters that damp selected faces toward flat space.

## Features

- BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation for 3+1 numerical relativity
- JIT-compiled finite difference operators for performance
- Clean, readable Python implementation
- Modular design with separate components for:
  - BSSN evolution equations
  - Finite difference derivatives
  - Initial data setup
  - Kreiss-Oliger dissipation
  - Error analysis

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python main.py
```

## Structure


- `bssn.py` - BSSN evolution equations and field definitions (FULLY OPERATIONAL)
- `derivatives.py` - Finite difference operators (FULLY OPERATIONAL)
- `tensor_algebra.py` - Tensor operations (Christoffel symbols, etc.) (FULLY OPERATIONAL)
- `errors.py` - Constraint violation analysis (HAMILTONIAN AND MOMENTUM ERRORS ARE FULLY OPERATIONAL)
- `kreiss_oliger.py` - Kreiss-Oliger dissipation (NOT TESTED)
- `main.py` - Main simulation loop and setup (NOT TESTED)
- `init.py` - Initial data setup (gravitational waves, etc.) (NOT TESTED)
- `tests/`  - Unit tests for tensor algebra, derivatives and MMS for BSSN evolution

## Usage

The code simulates gravitational wave evolution using the BSSN formulation. Key parameters can be modified in `main.py`:

- Grid size and resolution
- Evolution time and timestep
- Initial data type (gravitational waves, black holes, etc.)
- Boundary conditions

## Performance

All computationally intensive operations are JIT-compiled with JAX for near-C++ performance while maintaining Python's readability and ease of use.
