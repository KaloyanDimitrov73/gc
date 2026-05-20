This folder contains the experimental setup. All experiments that have been conducted are listed in the subfolders. The experiments followed a GQM plan that is shown in the [section](#goal-question-metric-plan) below. In the following we explain the structure and content of the folders.

### Table of Contents

- [Folder Structure](#folder-structure)
  - [1_experiment](#1_experiment)
  - [2_experiment](#2_experiment)
  - [qa_datasets](#qa_datasets)
- [Replicating the Experiments](#replicating-the-experiments)
- [Goal-Question-Metric Plan](#goal-question-metric-plan)
- [Contributions & Templates used in the ORKG](#contributions--templates-used-in-the-orkg)
  - [Contribution Templates](#contribution-templates)
  - [Supporting Templates](#supporting-templates)

### **Folder Structure:**

```
├── 1_experiment/ # The root directory of the first experiment
├── 2_experiment/ # The root directory of the second experiment
├── qa_datasets/ # The KGQA datasets and their creation scripts
├── testing/ # Debugging scripts and test runs
├── evaluator_configs.json # The configuration files for all the evaluators applied during the experiments
├── llm_evaluator_configs.json # The configuration files for the LLM evaluators applied after the experiments
├── README.md # This file
```

The folder structure differs from the [GQM plan](#goal-question-metric-plan). We organized the evaluation into two folders which we refer to as 1_experiment and 2_experiment. The questions of the GQM plan are answered in the following folders:
- **1_experiment**: The first experiment includes the _parameter selection process_ and the _final comparison_ of the retrievers. All questions of the GQM plan are answered with the results provided in the _final comparison_ folder but **Q5**.
- **2_experiment**: The second experiment includes test runs for four different graph variants of the ORKG. It is solely used to answer **Q5** of the GQM plan.
Furthermore, this folder contains the datasets used for the experiments and their creation scripts. Moreover the `evaluator_configs.json` contains all the configurations for each evaluator that calculates the evaluation metrics that have been used in each of our experiments. The `llm_evaluator_configs.json` contains the configurations for the LLM evaluators that are used to calculate the LLM-as-a-Judge metrics. We divided those to reduce runtime because the evaluation of the LLM based metrics takes significantly longer than the other metrics. 

#### **1_experiment**:
The first experiment includes the _parameter selection process_ and the _final comparison_ of the retrievers. The parameter selection process is used to find the final configuration for each retriever. The final comparison is used to compare the final configurations of each retriever with each other. The folder structure is as follows:

```
├── 1_experiment/      # Root directory of the first experiment
│   ├── base_configs/      # Base configuration files for each retriever
│   │   ├── HubLink/
│   │   |   └── base_config.json
│   │   ├── DiFaR/
│   │   |   └── base_config.json
│   │   ├── ../ # The other Retrievers
│   ├── runs/ 
│   │   ├── 0_test_runs/ # Prior debugging runs
│   │   ├── 1_parameter_selection/ # The parameter selection process
│   │   |   ├── HubLink/
│   │   |   |   ├── results/ # The outputs of the evaluation run
│   │   |   |   |   ├── [Hash ID]/
│   │   |   |   |   |   ├── configs/ # The exact config file for the run
│   │   |   |   |   |   ├── predictions/ # The predictions csv files for each configuration
│   │   |   |   |   |   ├── weave_export.csv # The extracted CSV file from the Weave dashboard
│   │   |   |   |   |   └── experiment.log # The log file of the run
│   │   |   |   |   ├── result_visualizations/ # Plots and tables that visualize the results for each parameter range tested
│   │   |   |   |   └── visualize.py # The script that generates the visualizations based on the results
│   │   |   |   ├── run_hublink.py # The script that runs the evaluation for HubLink
│   │   |   |   ├── tuning_parameters.json # The parameter ranges used for the parameter selection process
│   │   |   ├── ../ # The other retrievers
│   │   ├── 2_final_comparison/ # The comparison of the final configurations per retriever
|   │   |   ├── HubLink/
│   │   |   |   ├── results/ # The outputs of the evaluation run
|   │   |   |   ├── hublink_config.json # The final configuration for HubLink
│   │   |   |   └── run_hublink.py # The script that runs the evaluation for HubLink
│   │   |   ├── ../ # The other retrievers
│   │   |   ├── result_visualizations/ # Plots and tables that visualize the final results comparing all retrievers with each other
│   │   |   ├── run_llm_evaluation.py # The script that calculates the LLM-as-a-Judge metrics
│   │   |   ├── visualize.py # The script that generates the visualizations based on the results
```

In our final comparison, we applied four versions of HubLink:

- HubLink (T): The primary configuration, selected through the parameter selection process and used in previous experiments. It used the graph traversal strategy of HubLink.

- HubLink (D): The same configuration as version (T) but instead uses the direct retrieval strategy of HubLink.

- HubLink (F): A configuration specifically optimized for reduced runtime, which employs the direct retrieval strategy and limits hub consideration to 10 per question.

- HubLink (O): An open-source implementation sharing parameters with HubLink (T), but utilizing the mxbai-embed-large embedding model and the Qwen2.5-14B LLM.


#### **2_experiment**:
The second experiment includes test runs for four different graph variants of the ORKG. Note that because the graph variant `deep_distributed` is already run as part of the first experiment, we do not run it again and instead copied the results. The graph variants are:

* **GV1** This graph variant stores data in long paths. Additionally, the information is semantically separated and distributed across different contributions.
* **GV2** This graph variant stores data in long paths. All information is collected within a single contribution.
* **GV3** This graph variant stores data in short paths. The information is semantically separated and distributed across different contributions.
* **GV4** This graph variant stores data in short paths. All information is collected within a single contribution.

The folder structure is as follows:


```
├── 2_experiment/      # Root directory of the second experiment
│   ├── graph_configs/      # The configuration files for the different graph variants
│   ├── runs/   # The runs of each of the retrievers using the final configuration from the parameter selection process
|   │   ├── HubLink/
│   │   |   ├── results/ # The outputs of the evaluation run
|   │   |   ├── hublink_config.json # The final configuration for HubLink
│   │   |   ├── run_hublink_deep_centralized.py # Runs on the deep centralized graph
│   │   |   ├── run_hublink_flat_distributed.py # Runs on the flat distributed graph
│   │   |   └── run_hublink_flat_centralized.py # Runs on the flat centralized graph
│   │   ├── ../ # The other retrievers
│   │   ├── result_visualizations/ # Plots and tables that visualize the results for each graph variant
│   │   └── visualize.py # The script that generates the visualizations based on the results
```

#### **qa_datasets**:
The `qa_datasets` folder contains the datasets used for the experiments and their creation scripts. The datasets are divided into two folders `full` and `reduced`. The `full` folder contains the full datasets that have been used for the experiments while the `reduced` folder contains the reduced dataset that has been used for the parameter selection process. 

Additionally, this folder contains validation and analysis scripts that were used to create the dataset. Moreover, we created four different variants of the KGQA dataset. One for each of the four graph variants of the ORKG that we created (**GV-1-4**). The creation of the different datasets was done by first creating the dataset for the variant **GV1** and then using conversion script to create the other three datasets. 

The folder structure is as follows:

```
├── qa_datasets/      # Root directory of the datasets
|   ├── generation/ # Includes the notebook that was used to generate the dataset
|   |   └── qa_dataset_generation.ipynb # The notebook that was used to generate the dataset for variant GV1
|   ├── qa_datasets/ # The folder that contains the generated datasets
|   |   ├── full/ # The full datasets for all graph variants
|   |   |   ├── orkg_qa_dataset_conversion/ # The folder that contains the conversion scripts for the datasets
|   |   |   |   ├── convert_qa_dataset_deep_to_deep_distributed.py # The script that converts the dataset from GV1 to GV2
|   |   |   |   ├── convert_qa_dataset_deep_to_flat_distributed.py # The script that converts the dataset from GV1 to GV3
|   |   |   |   └── convert_qa_dataset_deep_to_flat_centralized.py # The script that converts the dataset from GV1 to GV4
|   |   |   ├── deep_centralized_graph_dataset.csv # The dataset for the variant GV2
|   |   |   ├── deep_distributed_graph_dataset.csv # The dataset for the variant GV1
|   |   |   ├── flat_centralized_graph_dataset.csv # The dataset for the variant GV4
|   |   |   └── flat_distributed_graph_dataset.csv # The dataset for the variant GV3
|   |   ├── reduced/ # The reduced dataset that was used for the parameter selection process
|   |   |   └── reduced_deep_distributed_graph_dataset.csv # The reduced dataset for the variant GV1
|   |   └── dataset_analysis.ipynb  # Notebook that shows data distribution of the dataset
|   ├── validation/ # Folder containing the validation scripts
|   |   ├── llm_comments_full_dataset/ # Folder that contains the LLM based feedback to the full dataset
|   |   ├── llm_comments_reduced_dataset/ # Folder that contains the LLM based feedback to the reduced dataset
|   |   └── llm_based_validation.ipynb # Notebook with the scripts that were used to validate the dataset using an LLM
```

### Replicating the Experiments

**Preparing Ollama**
Running the experiments, especially the parameter selection process, requires access to a local Ollama instance. To setup your local Ollama instance, please follow the instructions provided in the [Ollama GitHub](https://github.com/ollama/ollama?tab=readme-ov-file). Once you have Ollama ready, you can run the following command to download the models that we used for our experiments:

```bash
ollama pull qwen2.5:14b
ollama pull llama3.1
ollama pull mxbai-embed-large
ollama pull granite-embedding
ollama pull text-embedding-3-large
```

Now you will have all the models that are required to run the experiments.

**Preparing the Data**
Our experiments were conducted on the ORKG and have been added [here](https://sandbox.orkg.org/u/811081a0-4b25-4c76-b596-dae31bdfcff4?contentType=Paper). If you press this link, you will be redirected to the ORKG sandbox to the profile of the author. Under "Papers", there should be 153 papers listed, which are the ones we used for our experiments. If the papers are not listed or the list is incomplete, this is not an issue, as the SQA system can easily add the papers back.

However, there is one issue if the papers need to be added back. The ORKG distributes new IDs to each triple of the newly added papers. This means that the IDs of the golden triples in our KGQA Datasets [[../blob/main/sqa-system/experiments/qa_datasets/qa_datasets/full]] need to be updated to reflect the new IDs. Specifically for this scenario, we have prepared scripts that can be used to update the IDs of the golden triples in the KGQA Datasets with the new IDs. These scripts have been directly incorporated into the SQA system meaning that executing a simple script will both, ensure that all the papers are added to the ORKG and that the IDs of the golden triples are updated. 

This script is located in this directory `./prepare_data_for_replication.py`. To run the script, you need to have the SQA system installed. You can do this by following the instructions on the [Getting Started](../user_guide/getting_started.md) page. Once you have the SQA system installed, you can run the script by executing the following command in your terminal:

_Navigate your terminal to the `sqa-system` directory and run the following command:_
```bash
python ./prepare_data_for_replication.py
```

> Note: If this is your first time interacting with the ORKG, you will be prompted in the terminal to enter your ORKG credentials. These are needed to be able to upload data to the ORKG.

This process takes some time. Now, the SQA system will check whether each of the papers that were used during our experimentation is present on the ORKG and ensure that the IDs of the golden triples in the KGQA datasets are updated to reflect the new IDs.

**Running the Experiments**

Once finished, you can now run the experiments. Each experiment can be started with a corresponding `run.py` script. For this navigate into either the `./1_experiment` or `./2_experiment` directory and find the corresponding `run.py` script for the experiment that you want to run. All directories are similar to the following structure:

```
├── [Name of the Retriever]
│   ├── run_[name of the retriever].py
│   ├── ...
```

Therefore, you have to navigate into the directory of the retriever that you want to run and execute the run script over the terminal using

```bash
python run_[name of the retriever].py
```
This will start the experiment and the SQA system will take care of the rest. The results will be stored in the `results` folder of the retriever that you are running. 

> Note: If this is your first time running an experiment, you are likely prompted to enter your OpenAI Api key and your Weave API key. 

To update the visualizations (tables and plots), you find in each experiment folder a `visualize.py` script. This script will automatically update the visualizations based on the results of the experiment. After the experiment is done, run this script to see the data updated. 

_Note: you can also move the `visualize.py` script directly into the folder of the experiment you just ran and execute it there. In this case, the visualizations are only based on the results of the experiment you just ran and not all previous experiments also located in the folder._

### **Goal-Question-Metric Plan:**

**Ret1** Assess the relevance and robustness of retrieved contexts in scholarly literature search.
* **Q1** To what extent does the HubLink retrieval algorithm improve context relevance and accuracy compared to baseline KGQA methods in scholarly literature search?
    * **M1.1** Precision, **M1.2** Recall, **M1.3** F1, **M1.4** Hit@k, **M1.5** EM@k, **M1.6** MRR@k, **M1.7** MAP@k
* **Q2** How does retrieval performance vary with the logical complexity of operations required by different scholarly questions?
    * **(see Q1.1)**
* **Q3** How does retrieval performance vary across distinct scholarly literature-search use cases?
    * **(see Q1.1)**
* **Q4** What impact does the presence or absence of explicit type information in questions have on the retrieval performance?
    * **(see Q1.1)**
* **Q5** How robust is the proposed approach to structural and lexical variability across alternative knowledge graph schema representations?
    * **(see Q1.1)**
* **Q6** How efficient is the proposed approach considering runtime and language model tokens required when compared to baseline KGQA methods?
    * **M1.8** Runtime per question, **M1.9** LLM tokens per question
* **Q7** How does the proposed approach compare with regard to the environmental impact when compared to baseline KGQA methods?
    * **M1.10** Absolute carbon emissions, **M1.11** Relative carbon emissions, **M1.12** Delta carbon emissions

**GeT1** Evaluate how well the generated answer aligns semantically and factually with reference answers.
* **Q8** How semantically and factually consistent are the generated answers of the proposed approach when compared to answers generated by baseline KGQA approaches?
    * **M2.1** BLEU, **M2.2** ROGUE, **M2.3** Semantic Similarity, **M2.4** String Similarity, **M2.5** Bert Precision, **M2.6** Bert Recall, **M2.7** Bert F1, **M2.8** Factual Correctness Precision, **M2.9** Factual Correctness Recall, **M2.10** Factual Correctness F1

**GeT2** Evaluate how well the generated answer aligns with the intent and content of the question.
* **Q9** To what extent do the answers generated by HubLink reflect the semantic intent of scholarly questions when compared to baseline KGQA approaches?
    * **M3.1** Answer Relevancy
* **Q10** To what extent do the generated answers follow the instructional expectations of scholarly questions when compared to baseline KGQA approaches?
    * **M3.2** Instruction Following

**GeT3** Evaluate how well the generated answer aligns with the retrieved context.
* **Q11** To what extent are generated answers of HubLink faithful to the retrieved context and free from unsupported claims when compared to baseline KGQA approaches?
    * **M4.1** Faithfulness



### Contributions & Templates used in the ORKG

The following templates have been used in our experiments to store the data in the ORKG. The templates are divided into two categories: **Contribution Templates** and **Supporting Templates**. The contribution templates are used to store the data of the contributions in the ORKG while the supporting templates are helpers that are used to store the data in the contribution templates. 

Note that we used the *_1* and *_2* suffixes to differentiate between the different graph variants without introducing too much variability in the naming of the contribution itself. This is because the name of the contribution is provided to the retriever. Therefore, changing the name too much could negatively effect the performance.

#### Contribution Templates

**Classifications_1**

This contribution has been used for the **GV3** graph variant of the ORKG. 

```
- Research Level [Text] (0..*)
- Paper Class [Text] (0..*)
- Threat to Validity [Text] (0..*)
- Provides Input Data [Text] (0..1)
- Uses Tool Support [Text] (0..1)
- Has Threats to Validity Guideline [Boolean] (0..1)
- Research Object [Text] (0..*)
- Evaluation Method [Text] (0..*)
- Has Evaluation Guideline [Boolean] (0..1)
- Evaluation Property [Text] (0..*)
- Evaluation Sub-Property [Text] (0..*)
- Provides Replication Package [Boolean] (0..1)
- Replication Package Link [Text] (0..1)
```

**Classifications_2**

This contribution has been used for the **GV1** graph variant of the ORKG. 

```
- Research Level [Research Level] (0..1)
- Paper Class [Paper Class] (0..1)
- Evidence [Evidence] (0..1)
- Validity [Validity] (0..1)
- First Research Object [Research Object] (0..1)
- Second Research Object [Research Object] (0..1)
```

**Evidence 1**
This contribution has been used for the **GV4** graph variant of the ORKG:

```
- Provides Input Data [Text] (0..1)
- Uses Tool Support [Text] (0..1)
- Provides Replication Package [Boolean] (0..1)
- Replication Package Link [Text] (0..1)
```


**Evidence 2**
This contribution has been used for the **GV2** graph variant of the ORKG:

```
- Evidence [Evidence] (0..1)
```

**First Research Object 1**
This contribution has been used for the **GV4** graph variant of the ORKG:

```
- Research Object [Text] (0..*)
- Evaluation Method [Text] (0..*)
- Has Evaluation Guideline [Boolean] (0..1)
- Evaluation Property [Text] (0..*)
- Evaluation Sub-Property [Text] (0..*)
```

**First Research Object 2**
This contribution has been used for the **GV2** graph variant of the ORKG:

```
- First Research Object [Research Object] (0..1)
```

**Second Research Object 1**
This contribution has been used for the **GV4** graph variant of the ORKG:

```
- Research Object [Text] (0..*)
- Evaluation Method [Text] (0..*)
- Has Evaluation Guideline [Boolean] (0..1)
- Evaluation Property [Text] (0..*)
- Evaluation Sub-Property [Text] (0..*)
```

**Second Research Object 2**
This contribution has been used for the **GV2** graph variant of the ORKG:

```
- Second Research Object [Research Object] (0..1)
```

**Paper Class 1**
This contribution has been used for the **GV4** graph variant of the ORKG:

```
- paper Class [Text] (0..*)
```

**Paper Class 2**
This contribution has been used for the **GV2** graph variant of the ORKG:

```
- Paper Class [Paper Class] (0..1)
```

**Research Level 1**
This contribution has been used for the **GV4** graph variant of the ORKG:

```
- research level [Text] (0..*)
```

**Research Level 2**
This contribution has been used for the **GV2** graph variant of the ORKG:

```
- Research Level [Research Level] (0..1)
```

**Validity 1**
This contribution has been used for the **GV4** graph variant of the ORKG:

```
- Referenced Threats to Validity Guideline [Boolean] (0..1)
- threat to validity [Text] (0..*)
```

**Validity 2**
This contribution has been used for the **GV2** graph variant of the ORKG:

```
- Validity [Validity] (0..1)
```


#### Supporting Templates

[Research Level](https://sandbox.orkg.org/template/R766844)
```
- Primary Research [Boolean] (1..1)
- Secondary Research [Boolean] (1..1)
```

[Paper Class](https://sandbox.orkg.org/template/R725428)
```
- Evaluation Research [Boolean] (1..1)
- Philosophical Paper [Boolean] (1..1)
- Opinion Paper [Boolean] (1..1)
- Proposal of Solution [Boolean] (1..1)
- Personal Experience Paper [Boolean] (1..1)
- Validation Research [Boolean] (1..1)
```

[Evidence](https://sandbox.orkg.org/template/R814947/)
```
- Uses Input Data [Provision] (1..1)
- Uses Tool Support [Provision] (1..1)
- Provides Replication Package [Boolean] (1..1)
- Replication Package Link [Text] (0..1)
```

[Validity](https://sandbox.orkg.org/template/R766907)
```
- Threats to Validity [Validity Threats] (0..1)
- Referenced Threats to Validity Guideline [Boolean] (1..1)
```

[Validity Threats](https://sandbox.orkg.org/template/R768697/)
```
- External Validity [Boolean] (0..1)
- Internal Validity [Boolean] (0..1)
- Construct Validity [Boolean] (0..1)
- Confirmability [Boolean] (0..1)
- Repeatability [Boolean] (0..1)
```

[Property](https://sandbox.orkg.org/template/R766948)
```
- Name [Text] (1..1)
- Sub-Property [Property] (0..*)
```

[Evaluation](https://sandbox.orkg.org/template/R766945)
```
- Evaluation Method [Entity with Description] (0..*)
- Has Guideline [Boolean] (1..1)
- Properties [Property] (0..*)
```

[Research Object](https://sandbox.orkg.org/template/R766944)
```
- Object [Entity with Description] (1..*)
- Evaluation [Evaluation] (0..*)
```

[Entity with Description](https://sandbox.orkg.org/template/R766892)
```
- Name [Text] (1..1)
- Description [Text] (0..1)
```

[Provision](https://sandbox.orkg.org/template/R848276)
```
- Used [Boolean] (1..1)
- None [Boolean] (1..1)
- Available [Boolean] (1..1)
```
