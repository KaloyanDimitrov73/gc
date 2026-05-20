<div align="center">
<h2 align="center">HubLink: Leveraging Language Models for Enhanced Scholarly Information Retrieval on Research Knowledge Graphs<br/>
<i>Replication Package</i></h2><br/>
</div>

<div align="center">
<p>
    <a href='../../wikis/home'><img src='https://img.shields.io/badge/Wiki-Page-Green'></a>
<img src='https://img.shields.io/badge/Master-Thesis-orange'>
    <img src="https://img.shields.io/badge/python-3.12-blue">
</p>

</div>

<div align="center">
<h3>👋 First time Viewing this Project?</h3>

We recommend reviewing the <a href="#where-do-i-find---a-navigation-guide-to-the-replication-package">Navigation Guide to the Replication Package</a>. <br/> This guide will help you quickly find key elements in the replication package.

</div>

# Table of Contents
<ol>
    <li><a href="#about-the-project">About the Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#folder-structure-and-files">Folder Structure and Files</a></li>
    <li><a href="#description-of-folders">Description of Folders</a></li>
    <li><a href="#where-do-i-find---a-navigation-guide-to-the-replication-package">Where do I find? - A Navigation Guide to the Replication Package</a></li>
    <li><a href="#contact">Contact</a></li>
</ol>


# About the Project

With the application of Large Language Models (LLMs) to academic search a potential emerged to reduce barriers to accessing information and speed up literature research tasks. However, LLMs face substantial limitations when faced with complex knowledge tasks that require deep and responsible reasoning. To solve this issue, Knowledge Graph Question Answering approaches intend to enrich the LLM with external information from a Knowledge Graph (KG) without retraining. But the application of KGQA approaches in the scholarly domain is still an underdeveloped area. We argue, that the application of LLMs can help researchers to find relevant information in a KG to speed up their research process and to improve the quality of their research. In our thesis, we propose HubLink, a novel approach that uses an LLM to reason on subgraphs without needing additional training or fine-tuning. It uses Hubs, which are special concepts utilized in the retrieval approach that hold particular significance for a specific domain or question. Hubs consolidate information and serve as pivotal points when searching for relevant data. To enrich answers with further information and ensure transparency, Hubs are additionally linked to source information. 

To evaluate the effectiveness of the HubLink approach, we apply the retriever on the ORKG by applying it on the approach idea of [Kaplan et al. (2024)](https://publikationen.bibliothek.kit.edu/1000171637) called KARAGEN. This approach idea is designed to enhance and ease knowledge access in software architecture research by combining the strengths of KGs and LLMs. It involves generating and populating the ORKG with the help of LLM-based classification and retrieving knowledge by processing user queries through an LLM, querying the ORKG for relevant data, and enhancing it further with LLMs for tasks like summarization and classification. 

To realize this implementation, we developed the Scholarly Question Answering (SQA) system, a modular and flexible system that allows the user to configure and run experiments with different LLMs, datasets, pipelines, pipes, and knowledge bases. The SQA system provides the following capabilities:

1.  **Flexible Configuration Management:** Utilizes easily modifiable JSON-based configuration files to define and serialize parameters for all system components. This ensures straightforward reproducibility and systematic modification of experiments.
2.  **Experiment Execution and Evaluation:** Automates the process of conducting experiments using pipelines defined in configuration files. It evaluates performance using a suite of relevant metrics (e.g., retrieval recall, answer relevance, factuality), stores detailed outcomes along with reproducibility information, and facilitates results visualization through generated diagrams for easier analysis.
3.  **Data Ingestion:** Provides modules for loading publication data and QA pairs from standard JSON and CSV formats, serving as the foundation for knowledge base creation and evaluation datasets.
4.  **Modular RAG Pipeline:** Implements a fully customizable RAG pipeline architecture comprising pre-retrieval, retrieval, post-retrieval, and generation stages. This modularity allows easy interchanging, configuration, and testing of different algorithms or models at each stage of the workflow.
5.  **Scientific Text Extraction:** Integrates functionality to extract structured information from the text of publications by leveraging an LLM.
6.  **Knowledge Graph Integration:** Supports the modular construction and integration of diverse KGs by providing a unified interface.
7.  **Semi-Automated KGQA Pair Generation:** Incorporates strategies such as graph clustering and subgraph extraction to assist in the generation of relevant QA pairs directly from the underlying KGs.
8.  **Command-Line Interface (CLI):** Offers a CLI application for managing configuration files, triggering data ingestion, executing individual experiments, and conducting interactive QA sessions.

The architecture for the SQA system can be seen in the figure below and the code for the SQA system is provided in this repository. 

![Overview-Architecture](./assets/overview-Architecture.svg)


# Installation
This will walk you through the steps required to install and set up the project with a `pyproject.toml` configuration.

## Requirements
This project requires Python 3.12 - 3.13 for use. **Note:** Because the current version of [Microsoft GraphRag](https://microsoft.github.io/graphrag/get_started/) requires a Python version equal to or below 3.12, we recommend using Python 3.12 if you intend to work with Microsoft GraphRag. However, you can still run the code with Python 3.13 if you do not intend to work with Microsoft GraphRag as it is an optional dependency. 


### Prerequisites: Install and Initialize Git LFS
Large files are stored in Git LFS. If you don't have Git LFS installed, follow the installation instruction [here](https://github.com/git-lfs/git-lfs/wiki/Installation) to install it

After installing Git LFS, initialize it by running the following command in the terminal:

```bash
git lfs install
```

### Cloning the project repository
With an open terminal, clone the project’s repository by executing the following command:

```bash
git clone https://gitlab.kit.edu/kit/kastel/sdq/stud/abschlussarbeiten/masterarbeiten/marco-schneider/ma-schneider-implementation.git
```

> **Note:** If you are using a windows system, you may encounter an error with the file paths being too long. If this is the case, you have too configure your Windows to allow long paths. This can be done by opening a Powershell with administrator rights and running the commands: `reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1` and `git config --system core.longpaths true`. Restart your system and try the clone process again.

Once the cloning process is complete, navigate into the project directory:

```bash
cd ma-schneider-implementation
```

Then, pull the large files managed by Git LFS:

```bash
git lfs pull
``` 

### (optional but recommended) Create virtual environment
You can optionally choose to run the project in a virtual environment. It is strongly recommended doing this if you have multiple python projects that you would like to run on your device. This short guide will show you how to get a virtual python environment running with pythons `venv` module.

1. Navigate your terminal into the `sqa-system` folder (where pyproject.toml is located).

```bash
cd sqa-system
```

2. Create the virtual environment.

```bash
python -m venv venv
```

Now a new folder has been created in the projects root directory that includes the necessary files for the virtual environment. The folder should be called `venv`. After the preparation for the environment is done, the environment needs to be activated.

3. Activate the virtual environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux**

```bash
source venv/bin/activate
```	

Your terminal should now show a `(venv)` next to the path. If that is the case the virtual environment has been activated. If you encounter a terminal permission error take a look at [this post](https://stackoverflow.com/questions/56199111/visual-studio-code-cmd-error-cannot-be-loaded-because-running-scripts-is-disabl) or use CMD instead of powershell.

### Installing the Requirements
This project uses a `pyproject.toml` file to manage its dependencies, you can install everything with a single pip command. Make sure your terminal is located in the `sqa-system` directory (where pyproject.toml is located). Execute either of the following commands. It should then proceed to download and install all dependencies required to run the project.

> **Note**: Because currently the Microsoft GraphRAG retriever is not working with the codecarbon package, you either have to choose to install the codecarbon package or the Microsoft GraphRAG retriever.

Install the following if you are interested to run the project without the Microsoft GraphRAG retriever:
```bash
pip install .[codecarbon]
```

If you plan to use Microsoft GraphRag (and you are on Python 3.12 or below), run the following command. You can also first run the above command to install with the codecarbon package, run the experiments, delete the codecarbon package and then run the following command to install with Microsoft GraphRAG. This ensures that the emission tracking is working for the experiments that are not using Microsoft GraphRAG.

```bash
pip install .[graphrag]
```

> **Note:** It is possible that you may have to install gfortran to install the gensim package. On linux you can do this by running `sudo apt install build-essential gfortran python3-dev python3-pip`

### Creating an Account on Weight & Biases and Logging In
To run the experiments, the SQA system uses Weight & Biases (W&B) to track the experiments on the dashboard. You can create a free account on [Weight & Biases](https://wandb.auth0.com/login).

After the account has been created, run the following command in the terminal to login:

```bash
wandb login
```

🥳 That's it! You can now use the SQA-system. To replicate the experiments read the tutorial [here](./sqa-system/experiments/README.md#replicating-the-experiments).

# Folder Structure and Files
```
├── master_implementation/      # Root directory of the project
│   ├── Architekturreview/      # Documentation of the architecture review
│   ├── assets/                 # Additional assets for the project
│   ├── codereview/             # Documentation of the code review
│   ├── sqa-system/             # The SQA system including the implementation of HubLink
│   │   ├── data/               # Data folder storing raw data, configs, databases etc.
│   │   ├── experiments/        # Here all experiments are documented
│   │   ├── notebooks/          # Jupyter notebooks for the project
│   │   ├── scripts/            # Additional scripts for the project
│   │   ├── sqa_system/         # Source code of the backend
│   │   │   ├── app/            # User interaction code
│   │   │   │   ├── cli/        # Command line interface code
│   │   │   ├── core/           # Core functionality needed across the application
│   │   │   │   ├── base/       # Base classes for the application
│   │   │   │   ├── config/     # Configuration logic
│   │   │   │   │   ├── config_manager/ # Managers for each configuration
│   │   │   │   │   ├── factory/    # Factory for the configurations
│   │   │   │   │   └── models/ # Data models for the configurations
│   │   │   │   ├── data/       # Data handling logic
│   │   │   │   │   ├── data_loader/    # Data loading logic
│   │   │   │   │   ├── extraction/     # Data Extraction logic
│   │   │   │   │   ├── models/         # Data models
│   │   │   │   ├── language_model/ # Language model initialization and access logic
│   │   │   │   └── logging/    # Logging functionality
│   │   │   ├── experimentation/    # The code to conduct experiments
│   │   │   │   ├── evaluation/ # Evaluators to calculate the performance of the system
│   │   │   │   ├── file_evaluator/ # Implementations that allow to evaluate the files even after the experiment
│   │   │   │   └── utils/      # Utility functions for the experimentation
│   │   │   ├── knowledge_base/     # Code for the different knowledge bases
│   │   │   │   ├── knowledge_graph/    # Code for knowledge graphs (KG)
│   │   │   │   └── vector_store/   # Code for the vector stores
│   │   │   │       ├── chunking/   # Code for the chunking strategies
│   │   │   │       └── storage/    # Code for the storage of the vectors
│   │   │   ├── pipe/            # Code for the pipes that are used in the pipeline
│   │   │   │   ├── base/           # Base class for the pipes
│   │   │   │   ├── factory/        # Factory for the pipes
│   │   │   │   ├── generation/     # Code for the generation pipe
│   │   │   │   ├── post_retrieval/ # Code for the post retrieval pipes
│   │   │   │   ├── pre_retrieval/  # Code for the pre retrieval pipes
│   │   │   │   └── retrieval/      # Code for the retrieval pipes
│   │   │   ├── pipeline/     # Code for the RAG pipeline
│   │   │   ├── qa_generator/   # Code for the question answering generator
│   │   │   │   ├── base/    # Base classes for the qa generator
│   │   │   │   ├── qa_dataset_graph_converter/ # Code for converting QA datasets between graph variants
│   │   │   │   ├── strategies/    # Implementations of QA generation strategies
│   │   │   │   │   ├── clustering_strategy/    # Code for the clustering strategy
│   │   │   │   │   └── publication_subgraph_strategy/  # Code for the subgraph strategy
│   │   │   │   └── utils/   # Utility functions for the qa generator
│   │   │   ├── retrieval/  # Cod for the retrieval logic and all retriever implementations
│   │   │   │   ├── base/   # Base classes for the retrieval
│   │   │   │   ├── factory/    # Factory for the retrieval
│   │   │   │   └── implementations/   # Retriever implementations (including HubLink)
│   │   │   ├── tests/        # Code for the tests
```

> **Note:** A more detailed description of the folder structure and files can be found in the [SQA System Folder Structure](../../wikis/pages/sqa_system/folder_structure) wiki page. 

# Where do I find? - A Navigation Guide to the Replication Package

> **Where do I find the HubLink implementation?**

The implementation of HubLink is located in the `retrieval` folder of the SQA system. [Here](./sqa-system/sqa_system/retrieval/implementations/HubLink/) is a direct link to the implementation.

> **Where do I find the experiments?**

The experiments are located in the `experiments` folder of the SQA system. [Here](./sqa-system/experiments/). There is a `README.md` file located at the top level of the folder that provides an overview of the experiments folder and where to find each of the experiments that we have conducted and analyzed in the master thesis.

> **How can I replicate the experiments?**

We provide a detailed description of how to replicate the experiments in the `README.md` file located in the `experiments` folder. [Here](./sqa-system/experiments/README.md#replicating-the-experiments).

> **Where do I find the artifacts of the KGQA Retrieval Taxonomy?**

We created the KGQA Retrieval Taxonomy by applying the our proposed taxonomy construction process which synthesizes taxonomy information from the literature and incrementally refines the taxonomy. The artifacts that document the construction process are located in the `kgqa_retrieval_taxonomy` folder of the `assets/taxonomy_construction/` folder. [Here](./assets/taxonomy_construction/kgqa_retrieval_taxonomy/). You will also find `README.md` files in those folders that provide an overview of the artifacts and where to find them.

> **Where do I find the Artifacts of the Taxonomy Construction Process?**

We provide template files that can be used to conduct the taxonomy construction process for the creation of a new taxonomy. The artifacts are located in the `template_taxonomy_construction` folder of the `assets/taxonomy_construction/` folder. [Here](./assets/taxonomy_construction/template_taxonomy_construction/). 

> **Where do I find the generation of the KGQA Datasets?**

The generation of the KGQA datasets is located in the `experiments/qa_datasets` folder. [Here](./sqa-system/experiments/qa_datasets/README.md). There is a `README.md` file located at the top level of the folder that provides an overview of the generation process.

> **Where do I find the Question Templates used for the KGQA Dataset Generation?**

The templates are provided in the [templates.md](./sqa-system/experiments/qa_datasets/templates.md) file located in the `experiments/qa_datasets` folder. The file contains the templates that have been used to generate the question-answer pairs. The file also includes updated templates which we have corrected manually and LLM based to improve the quality of the questions.

> **Where do I find the selection of KGQA baseline approaches?**

The selection of KGQA baseline approaches is documented in the [assets](./sqa-system/assets/baseline_retriever_search/) folder. 

> **I just want to try out the SQA system, where do I start?**

If your intention is to just try out the functionality of the SQA system, we recommend running the HubLink approach in interactive mode on the ORKG with our prepared pipeline configuration. This can be done by following the instructions in the [Using the CLI](../../wikis/pages/user_guide/using_the_cli) section of the wiki.


# Contact
For any questions or feedback, feel free to contact me at marco.schneider@student.kit.edu