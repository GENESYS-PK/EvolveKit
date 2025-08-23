<img src="docs/resources/logo.svg" style="width: 100%; height: auto;" />

**EvolveKit** is an upcoming Python library designed to simplify solving complex optimization problems using evolutionary algorithms. Please note that EvolveKit is currently under active development and has not yet been released under an open-source license.

# :star: Features
* **Simple & Flexible** - easy to use for quick setups, yet fully customizable for advanced control.
* **Supports Heterogeneous Representation** - chromosomes within the same individual can use different gene types.
* **Built-in Statistics** – automatically record key metrics.
* **Inspectors** – log metrics, visualize trends, or halt evolution with custom callbacks.

# :hammer_and_wrench: Installation
To get started, first clone the repository to your local machine using Git:

```bash
git clone https://github.com/GENESYS-PK/EvolveKit.git
```

Make sure you're inside the project directory before proceeding.<br>
Next, install all required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

:test_tube: Examples

EvolveKit has many built-in examples. They are stored in `evolvekit/examples` directory. To run a basic example, navigate to the root of the repository and execute:

```bash
python -m evolvekit.examples.basic.main
```

To explore more advanced functionality, run:

```bash
python -m evolvekit.examples.advanced.main
```
