This folder contains the data of the SQA system and the data that we used to conduct our experiments. It is organized as follows:

```
├── data/
│   ├── cache/ # This directory contains all the cached data from the SQA system.
│   ├── configs/ # The default configuration files that are loaded when using the CLI of the SQA system.
│   ├── evaluation_results/ # When running a experiment using the CLI of the SQA system, the results are stored in this directory.
│   ├── external/ # This directory contains the external publication data that we used to conduct our experiments.
│   ├── file_paths/ # This directory contains the file paths that are managed by the `FilePathManager` class.
│   ├── knowledge_base/ # This directory contains the serialized knowledge graphs or vector stores.
│   ├── paper_extraction/ # This directory contains the extracted structured data from the full text of the papers.
│   ├── prompts/ # This directory contains all prompts used in the SQA system managed by the `PromptProvider` class.

```

### Experimental Data
In the `data/external/` directory, you can find the data that we used to conduct our experiments. 
This data has been created using the [combine_exports.py](../scripts/paper_scrapping/) script provided in the `scripts/paper_scrapping/` directory.


#### Explanation of the Data
To evaluate how well HubLink performs in the context of scholarly literature search, we require a dataset of scientific publications. 
For this purpose, we use the dataset provided by [1]. This dataset was originally created to analyze how SWA research objects are evaluated and how replication packages are provided. 
It was created through a literature search and annotations were extracted according to a specific schema. 
In their study, a total of 153 publications were included according to the following inclusion and exclusion criteria:

1.  Papers presented at ECSA and ICSA conferences between 2017 and 2021.
2.  Comprehensive technical papers, excluding short papers, experience reports, and opinion pieces.
3.  Papers focusing on evaluation research, validation studies, solution proposals, and philosophical discussions.

| **Data Item** | **Description** | **Type** |
| :------------------------ | :------------------------------------------------------------------ | :----------- |
| Paper Class               | A general classification of the publication.                        | Meta Data    |
| Research Level            | Distinguishes on whether the research is collected firsthand.       | Meta Data    |
| Kind                      | Classifies whether the paper can be seen as a full research paper.  | Meta Data    |
| Research Object           | The investigated object(s) of research.                           | Content Data |
| Tool Support              | Indicates whether the paper employed a tool.                        | Content Data |
| Input Data                | Indicates whether the paper used specific input data.               | Content Data |
| Replication Package       | Indicates whether the paper provides a dedicated replication package. | Content Data |
| Threats to Validity (TtV) | The threads to validity that are named in the paper.                | Content Data |
| TtV Guideline             | Indicates whether the paper references TtV guidelines.              | Content Data |
| Evaluation Method (EM)    | The applied evaluation method.                                      | Content Data |
| EM Guidelines             | Indicates whether the paper referenced guidelines for EM.             | Content Data |
| Property                  | The property that is evaluated with a EM for a research object.   | Content Data |


The schema employed for this annotation process is presented in the table above. Each publication is annotated according to its research objects, research level, paper class, and validity information. The table also categorizes data as either metadata or content data. The differentiation between metadata and content data presented in the table above is based on the information provided by [1].

According to [2], metadata is defined as "data that provides information about other data". In the context of the data schema defined in the table above, two types of metadata are present: *descriptive* and *preservation*. Descriptive metadata provides information for finding or understanding a resource, such as the title, authors, and publication year. Preservation metadata, a subtype of administrative metadata, encompasses information regarding the long-term management of a file [2].

In addition to metadata, the dataset also contains content data, which we define as information *that is contained within* the scientific artifact. This implies that to obtain the desired content data, the text of the publication must be read.

The extracted annotations in the dataset are based on the content of the publications and have been scientifically validated. Combined with the corresponding metadata, this dataset is well suited for our experiments, as it allows us to formulate questions regarding both metadata and content. We therefore compiled the data and consolidated them into a single JSON file to be used during our experiments. The resulting dataset consists of 153 publications with the annotations shown in the table above. In addition to the annotations, the dataset also includes metadata such as title, authors, DOI, and publication year.

-----

[1]: Konersmann, Marco, Angelika Kaplan, Thomas Kuhn, Robert Heinrich, Anne Koziolek, Ralf Reussner, Jan Jurjens, et al. “Evaluation Methods and Replicability of Software Architecture Research Objects.” In 2022 IEEE 19th International Conference on Software Architecture (ICSA), 157–68. Honolulu, HI, USA: IEEE, 2022. https://doi.org/10.1109/ICSA53651.2022.00023.

[2]: Riley, Jenn. Understanding Metadata: What Is Metadata, and What Is It For. NISO Primer Series. Baltimore, MD: National Information Standards Organization, 2017.