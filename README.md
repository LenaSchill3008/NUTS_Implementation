# NUTS Implementation

This project provides a **Python implementation of the No-U-Turn Sampler (NUTS)**, a variant of Hamiltonian Monte Carlo (HMC) for efficiently sampling from complex probability distributions. It also includes a **benchmark framework** to evaluate NUTS against a baseline algorithm (Random-Walk Metropolis).

---

## Project Structure

```bash
nuts/
├── core/
│   └── density.py           # Base class for target distributions
├── sampler/
│   ├── nuts.py              # NUTS sampler implementation
│   └── rwm.py               # Random-Walk Metropolis baseline
├── benchmarks/
│   ├── models.py            # Benchmark models (Standard Normal, Correlated Gaussian, Banana)
│   ├── metrics.py           # Evaluation metrics (ESS, autocorrelation, timing)
│   └── nuts_benchmark.py    # Runs full benchmark suite
├── examples/                # (Optional) example scripts, e.g., gaussian_example.py
├── main.py                  # Entry point; runs benchmarks by default
├── requirements.txt         # Python dependencies
```

---

## Benchmark Models

1. **Standard Normal (1D)** – basic sanity check  
2. **Correlated Gaussian (2D)** – demonstrates NUTS handling correlated variables  
3. **Banana-shaped distribution (2D)** – demonstrates NUTS handling nonlinear geometry  

These models allow testing both the **strengths** and **limitations** of NUTS.

---

## Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Benchmarks
By default, running main.py executes the full benchmark suite:

```bash
python main.py
```

Output includes: 
- Runtime of each sampler
- Effective Sample Size (ESS) for each dimension
- Step size (epsilon) for NUTS
- Acceptance rate for RWM

---

## References
Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(47), 1593–1623.