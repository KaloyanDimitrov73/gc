1. **[C1] Flexible Configuration Management:** Utilizes easily modifiable JSON-based configuration files to define and serialize parameters for all system components. This ensures straightforward reproducibility and systematic modification of experiments.

2. **[C2] Experiment Execution and Evaluation:** Automates the process of conducting experiments using pipelines defined in configuration files. It evaluates performance using a suite of relevant metrics (e.g., retrieval recall, answer relevance, factuality), stores detailed outcomes along with reproducibility information, and facilitates results visualization through generated diagrams for easier analysis.

3. **[C3] Data Ingestion:** Provides modules for loading publication data and \gls{qa} pairs from standard JSON and CSV formats, serving as the foundation for knowledge base creation and evaluation datasets.

4. **[C4] Modular RAG Pipeline:** Implements a fully customizable \gls{rag} pipeline architecture comprising pre-retrieval, retrieval, post-retrieval, and generation stages. This modularity allows easy interchanging, configuration, and testing of different algorithms or models at each stage of the workflow.

5. **[C5] Scientific Text Extraction:** Integrates functionality to extract structured information from the text of publications by leveraging an \gls{llm}.

6. **[C6] Knowledge Graph Integration:** Supports the modular construction and integration of diverse \glspl{kg} by providing a unified interface.

7. **[C7] Semi-Automated KGQA Pair Generation:** Incorporates strategies such as graph clustering and subgraph extraction to assist in the generation of relevant \gls{qa} pairs directly from the underlying \glspl{kg}.

8. **[C8] Command-Line Interface (CLI):** Offers a CLI application for managing configuration files, triggering data ingestion, executing individual experiments, and conducting interactive \gls{qa} sessions.
