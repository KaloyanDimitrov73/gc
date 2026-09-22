
This folder contains the generation, validation and the `CSV` files for the KGQA datasets that we used in our experiments.

The creation was done in several steps:

1. Preparation of Question Templates
2. Automatic Generation using the Subgraph and Cluster strategies of the SQA system
3. Manual and LLM based refinement of the questions and answers

### Table of Contents
- [Dataset Variants](#dataset-variants)
- [Dataset Structure](#dataset-structure)
- [Question Diversity](#question-diversity)
- [Meta Data and Content Data in the Dataset](#meta-data-and-content-data-in-the-dataset)
- [Restrictions](#restrictions)
- [Classification According to KGQA Retrieval Taxonomy](#classification-according-to-kgqa-retrieval-taxonomy)
- [Template Questions](#template-questions)
- [Dataset Creation](#dataset-creation)


### Dataset Variants

We created a total of four different dataset variants. One for each of the following graph variants:

1.  **GV1** This graph variant stores data in long paths. Additionally, the information is semantically separated and distributed across different contributions. 
2.  **GV2** This graph variant stores data in long paths. All information is collected within a single contribution.
3.  **GV3** This graph variant stores data in short paths. The information is semantically separated and distributed across different contributions. 
4.  **GV4** This graph variant stores data in short paths. All information is collected within a single contribution. 

However, at the core, each dataset contains the same exact information. 
The only difference are the golden ground truth triples which differ for each graph variant. 
This is because the information in each variant is the same but stored in different ways.


### Dataset Structure

The dataset consists of kgqa pairs. Each pair includes the question itself along with the corresponding *ground-truth* data. This data is then used during the evaluation to determine whether the answer and the data retrieved by the retriever are correct. In this context, the ground-truth represents one possible valid answer to the question, as well as the set of triples needed to answer it. In addition to the question and the ground-truth, each kgqa pair also includes further metadata. This includes a *topic entity* that can serve as an entry point into the graph for the retriever, the dois of the papers referenced in the question and the template used to generate the question. The pairs are also classified according to use case, semi-typed nature, and the categories defined in the taxonomy.

### Question Diversity

To ensure that the questions in the qa dataset exhibit a high degree of variability that allows meaningful conclusions about the capabilities of a kgqa system, we structured the dataset considering multiple dimensions. First, we use *use cases* to map each question to a realistic application scenario. During the creation phase, we assigned questions to each of the six use cases to ensure a balanced distribution across all categories. To further reflect the retrieval capabilities required for each question, we incorporated the *retrieval operation* category from the kgqa retrieval taxonomy. Each question is assigned to a specific retrieval operation, ensuring that operations are covered evenly. Additionally, the dataset distinguishes between *untyped* and *semi-typed* questions. This distinction is meant to assess how well a retriever can handle synonyms or missing type information. The following characteristics apply:

* **Semi-Typed Questions** Each question in the dataset targets specific triples in the kg, with each triple consisting of a subject, predicate, and object. Depending on the triple, either the subject or the predicate may carry type information about the object. For example, if the question is asking to retrieve a triple like ('Research Object Entity', 'Name', 'Reference Architecture'), it specifies that the reference architecture is a research object.

* **Untyped Questions** In contrast, *untyped* questions do not include the type information. Using the same example, the question would simply ask about the reference architecture without stating that it is a research object. This increases the difficulty of retrieval, because the retriever must infer the object type based solely on the object name.

Because we have four different types of graphs, each question needs to be compatible with all four variants. With regard to the semi-typed questions, we needed to ensure that the type information is relevant for the triples of all variants without specifically targeting one of them. The table below shows the *placeholder* which indicates the restriction in the question. For each placeholder, specific *triples* are requested which have a different formatting depending on the variant. Each triple includes the type of the requested value either as its predicate or subject. To correctly indicate the type for all variants, the *typed string* is used. This string is added to the semi-typed question to indicate the type of information requested. We chose this string by differencing the strings in the triples to their common substring. For example the string 'uses tool support' and 'tool support' becomes 'tool support'. The table below shows the placeholders, the triples that are requested, and the typed string that is used in the question.

| Placeholder | Triple Representations [Variant] |  Typed String | 
| ---------	| -------	|  ----- |
| -	| (Paper, doi, 'value') [GV1, GV2, GV3, GV4] | doi |
| paper title	| - | paper with the title |
| author name | (authors list, has list element, 'value') [GV1, GV2, GV3, GV4] | author |
| -	| (Paper, venue, 'value') [GV1, GV2, GV3, GV4] | venue	|
| -	| (Paper, research field, 'value') [GV1, GV2, GV3, GV4] | research field  |
| year	| (Paper, publication year, 'value') [GV1, GV2, GV3, GV4]	| publication year  |
| research level	| (Contribution, research level, 'value') [GV3, GV4], (Research Level, 'value', 'boolean') [GV1, GV2]| research level |
| paper class name	| (Contribution, paper class, 'value') [GV3, GV4], (Paper Class, 'value', 'boolean') [GV1, GV2] | paper class |
| Threat to Validity | (Contribution, threat to validity, 'value') [GV3, GV4], (Threat to Validity, 'value', 'boolean') [GV1, GV2]	| threat to validity 
| -	| (Paper, Uses Input Data, 'value') [GV3, GV4], (Input Data, 'value', 'boolean') [GV1, GV2]	| input data | 
| -	| (Evidence, Replication Package Link, 'value') [GV1, GV2], (Contribution, Replication Package Link, 'value') [GV3, GV4] | replication package link | 
| -	| (Contribution, Uses Tool Support, 'value') [GV3, GV4], (Tool Support, 'value', 'boolean') [GV1, GV2] | tool support |
| - | (Validity, Referenced Threats to Validity Guideline, 'value') [GV1, GV2], (Contribution, Referenced Threats to Validity Guideline, 'value') [GV3, GV4]	| threats to validity guideline |
| research object name	| (Contribution, Research Object, 'value') [GV3, GV4], (Research Object Entity, Name, 'value') [GV1, GV2]	| research object |
| evaluation method name | (Contriubtion, Evaluation method, 'value') [GV3, GV4], (Evaluation Method Entity, Name, 'value') [GV1, GV2]| evaluation method |
| -	| (Evaluation, Has Guideline, 'value') [GV1, GV2], (Contribution, Has Evaluation Guideline, 'value')  [GV3, GV4]	| evaluation guideline |
| property name	| (Contribution, Evaluation Property, 'value') [GV3, GV4], (Property, Name, 'value')  [GV1, GV2]	| property |
| sub-property name	| (Contribution, Evaluation Sub-Property, 'value') [GV3, GV4], (Property, Name, 'value') [GV1, GV2]	| property|
| Provides Replication Package	| (Contribution, Provides Replication Package, 'boolean') [GV1, GV2, GV3, GV4]	| provides replication package |

### Meta Data and Content Data in the Dataset

The dataset has the following meta data and content data:

| Name |  Data Type | 
| ---------	| -------	| 
| Doi	| Descriptive Metadata | 
| Title	| Descriptive Metadata	| 	|
| Authors | Descriptive Metadata	|
| Venue	| Descriptive Metadata	|
| Research Field	| Descriptive Metadata	|
| Publication Year	| Descriptive Metadata	|
| Research Level	| Descriptive Metadata 	|
| Paper Class	| Descriptive Metadata	|
| Threat to Validity	| Content Data	|
| Uses Input Data	| Content Data	|
| Replication Package Link	| Preservation Metadata	|
| Uses Tool Support	| Content Data |
| Has Threats to Validity Guideline	| Content Data	|
| Research Object	| Content Data	|
| Evaluation Method	|Content Data	|
| Has Evaluation Guideline	| Content Data	|
| Evaluation Property	| 	Content Data	|
| Evaluation Sub Property	| Content Data	|
| Provides Replication Package	| Descriptive Metadata 	|

According to [1], metadata is defined as "data that provides information about other data". 
In the context of the Label dataset, two types of metadata are present: descriptive metadata and preservation metadata. 
Descriptive metadata provides information for finding or understanding a resource, such as the title, authors, and publication year. 
Preservation metadata is a subtype of administrative metadata information about the long term management of a file.

Besides metadata, the Label dataset also contains content data. 
We define content data as the information that is contained in the scientific artifact. 
This means, that to get the desired content data, the text of the scientific artifact has to be read.

[1] Riley Jenn, Understanding metadata: what is metadata, and what is it for, 978-1-937522-72-8, 2017, https://www.niso.org/publications/understanding-metadata-2017


### Restrictions

To keep the dataset manageable in terms of complexity, certain constraints were applied during its creation. 
First, the number of golden triples needed to answer a question was limited to a maximum of 10. 
This limitation ensures that questions do not require information spread across the entire dataset. 
Without this restriction, answering broad questions would require extensive aggregation, 
resulting in runtimes that are too long for the scope of a master thesis, which involves experiments on thousands of questions.



Furthermore, it is essential that the retriever parameters are set in such a way that the retrievers can fully answer the given questions. 
By limiting the number of golden triples, we can ensure that this is the case. 
The chosen threshold of *10* represents a compromise between the efficiency of the retrievers and the expressiveness of the questions. 
A higher threshold would require increasing retriever parameters, such as maximum depth or width, leading to longer run-times. 
A lower threshold, on the other hand, would reduce the informativeness of the questions or even make some of them unanswerable. 
Therefore, selecting a limit of ten triples strikes a balance between maintaining acceptable runtimes and preserving the relevance and feasibility of the questions.

The following table shows the distribution. The columns **1** through **6** refer to the **Use Cases**.

| **Retrieval Operation** | **1** | **2** | **3** | **4** | **5** | **6** |
|:------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| aggregation             |   4   |   4   |   4   |   4   |   4   |   4   |
| basic                   |   4   |   4   |   4   |   1   |   4   |   1   |
| comparative             |   4   |   4   |   4   |   4   |   4   |   4   |
| counting                |   4   |   4   |   4   |   4   |   4   |   4   |
| negation                |   0   |   0   |   4   |   4   |   4   |   4   |
| ranking                 |   4   |   4   |   4   |   4   |   4   |   4   |
| relationship            |   4   |   4   |   4   |   4   |   4   |   4   |
| superlative             |   0   |   0   |   4   |   4   |   4   |   4   |

*Table: Distribution of question templates across the use case and retrieval operation dimensions.*

### Template Questions

To utilize the semi-automatic kgqa generation, template questions are required. 
For the dataset, a total of 170 different template questions were created manually. 
As mentioned previously, these questions were diversified along the dimensions of the use case and retrieval operation. 
The precise distribution of these template questions is shown in the table above. For most combinations of use case and retrieval operation, four template questions were generated, although there are some exceptions. 
Specifically, for the use cases four and six, only two suitable examples were found for the basic operation. 
Additionally, for use cases one and two, no suitable template questions were identified in the negation and superlative categories that were meaningful and below the threshold of ten golden triples.

Regarding the semi-typed nature of the questions, the dataset contains 86 semi-typed and 84 untyped questions. 
Generally, an equal number of semi-typed and untyped questions were created for each combination of use case and retrieval operation, with the exception of the individual questions for use cases four and six as mentioned above.

The full list of the templates that we used for the creation is available in the [templates.md](./templates.md) file. 

Okay, here is the Markdown version of your LaTeX code:

### Classification According to KGQA Retrieval Taxonomy

The distribution of the classification following the kgqa retrieval taxonomy is as follows: Within the *Condition Type* category, 133 questions were classified as *Named Entity* and 37 as *Named Entity, Temporal*. This indicates that two types of constraints must be considered by the retriever: either solely the consideration of a named entity or additionally a temporal constraint.

For the *Answer Format* category, the dataset comprises 61 *enumerative*, 58 *simple*, and 51 *explanatory* questions. The classification according to *Graph Representation* shows that 152 questions belong to the *multi-fact* type, meaning they require multiple triples for their answer, while 18 questions are categorized as *single-fact*, requiring only one triple.

In the *Answer Type* category, 86 questions expect an answer of type *Named Entity*, 20 questions require a quantitative answer, two questions anticipate a *Boolean* answer, and two additional questions have an *Other Type* of response. Furthermore, the dataset includes questions with more complex answer types: 24 questions expect either a description combined with a quantitative value (*Description, Quantitative*) or a description with a named entity (*Description, Named Entity*). Additionally, 13 questions require answers of the type *Description, Quantitative, Temporal*, and 8 questions expect answers of the type *Description, Named Entity, Temporal*, which means that they also require a year.

With respect to the *Intention Count* category, the dataset exclusively contains *Single Intention* questions, as multiple intentions were not considered. For the *Question Goal* category, due to insufficient variability in the publication data, only the class *Information Lookup* applies. Regarding *Answer Credibility*, all questions meet the *Objective* criterion.

The data has been extracted from the [dataset_analysis.ipynb](./qa_datasets/dataset_analysis.ipynb) notebook.

### Dataset Creation
To generate the dataset, we used the semi-automatic cluster and subgraph-based generation methods from the SQA system. 
For each template question there is one instance question in the dataset, resulting in the final dataset consisting of 170 competency questions in total.

The generation process is documented in the [qa_dataset_generation.ipynb](./generation/qa_dataset_generation.ipynb) notebook. 

After the generation of the question-answer pairs, we manually reviewed each question-answer pair for the quality of the question and the consistency of the generated answer. 
For this process, we employed a LLM that provides feedback which we then manually reviewed to consider whether the feedback should be incorporated. 
The LLM based validation is documented in the [/validation/](./validation/) folder.