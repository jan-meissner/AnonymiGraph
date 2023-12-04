# Building Documentation

To build the documentation:

1. [Build and install](https://github.com/pyg-team/pytorch_geometric/blob/master/.github/README.md) from source.
1. CD into docs:
   ```bash
   cd docs
   ```
2. Install Requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the docs:
   ```bash
   make html
   ```
4. View the docs:
   Recommended: Use the LiveServer vscode extension, then simply navigate to docs/_build/html/index.html right click on it and open it with LiveServer.


The documentation is now available to view by opening `docs/build/html/index.html`.