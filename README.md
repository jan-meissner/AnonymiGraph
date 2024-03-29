[![codecov](https://codecov.io/gh/jan-meissner/AnonymiGraph/graph/badge.svg?token=xQucrvH23K)](https://codecov.io/gh/jan-meissner/AnonymiGraph)
[![Python CI](https://github.com/jan-meissner/AnonymiGraph/actions/workflows/CI.yml/badge.svg)](https://github.com/jan-meissner/AnonymiGraph/actions/workflows/CI.yml)

**[Documentation](https://jan-meissner.github.io/AnonymiGraph/)** |

# AnonymiGraph

Anonymization and Privacy Preserving Algorithms for Graphs in Python.

## Table of Contents

- [Codespaces: Development Without Installations](#codespaces-development-without-installations)
- [Installation](#installation)
  - [From Git Repository](#from-git-repository)
  - [Local Installation](#local-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Codespaces: Development Without Installations

GitHub Codespaces simplifies development by providing a cloud-based environment accessible through your browser. No need to worry about any installation. To use it:

1. Navigate to the GitHub repository.
2. Click on the 'Code' button and select 'Open with Codespaces'.
3. Once the environment is set up, you're ready to go - no additional setup required!

## Installation

### From Git Repository

To install the package from the Git repository (not available on PyPI), run the following command:

```bash
pip install git+https://github.com/username/repository.git
```

### Local Installation

For local development, you can install the project by following these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd repository
   ```
3. Install the package:
   ```bash
   pip install -e .[full,test,dev]
   ```
4. Set up pre-commits:
   ```bash
   pre-commit install
   ```
5. Run pytest to check if your setup worked
   ```bash
   pytest .
   ```

## Usage

Provide instructions on how to use the project after installation.

## License

Include information about the project's license here.
