
# Project Description
The overarching question this project seeks to answer is whether 4chan, despite its controversial reputation, can serve as a potent forum for free and open discussion, offering early insights into significant societal, political, and global phenomena. To assess this claim, the following project description outlines the machine learning classification pipeline that attempts to categorize text for coherent dialogue. It does this through three distinct steps that employ various machine learning concepts through both foundational and novel approaches.

Before geting into the Project structure and File directories, it's important to qualify various concepts and definitions used within this project. 

Project Definitions:
- **Dialog**: Dialog: The term dialog is used to describe coherent dataframe text. It is a conversation that is logical, consistent, and meaningful. It is not a monologue or a series of unrelated statements. It can contain profanity as long as it is used like any other word would. Belligerent and incoherent profanity is not considered dialog and will not be labeled as dialog.

 Examples of rows of qualifying dialog:![Dialog example 1](plots\dialogdata_mainexample_readme.png)
 Examples of rows of qualifying non-dialog: ![Dialog example 2](plots\nondialogdata_mainexample_readme.png)

- **Dialog mapping**:
The images below represent two comments that have been mapped to eachother using the thread id's found in raw data. This illistrates the dialog that this project is aiming to identify and select for.

Example 1: ![Dialog example 1](plots\dialog_2.png)
Example 2: ![Dialog example 2](plots\dialog_5.png)


Methods and Approaches:
- **Custom Attention Filtering**: This approach leverages BERT's attention mechanisms to extract a five-number summary from the attention values of text documents (where a "document" refers to a single dataframe row, representing the tokenized text of that row). Using KMeans clustering and PCA, the code tokenizes and embeds text with BERT, then processes the attention values to identify key statistics for each attention head. This is followed by clustering the documents and visualizing them in a two-dimensional PCA plot. The function `document_embedding_Five_Number_Summary` extracts the five-number summary from attention values and utilizes a recursive function `flatten` to handle nested lists. The resulting clusters are labeled and can be analyzed to understand their dialog or non-dialog nature, allowing for insights into document categorization.

- **UST (Unsupervised and Semi-Supervised Training) with Self-Training Process**: Incorporates a specialized self-training process, leveraging both unsupervised and semi-supervised learning methods. The iterative UST approach leverages labeled, unlabeled, and validation data. Through a teacher-student paradigm, the student model learns from pseudo-labels generated from high-confidence predictions, while the teacher model guides the learning process. The model continually refines its understanding and enhances its adaptability, responding to changes in the underlying data characteristics.

- **FnWC (Frequency number Word Count number) Method**: This method is applied at various stages of the pipeline. In step 1, FnWC is used for similarity matching, identifying patterns based on frequency and word count, aligning them with corresponding data clusters. As the data changes, the method dynamically adjusts, enabling efficient clustering and adaptation to evolving data trends. It plays a vital role in pattern recognition and clustering within the dataset.

- **Time-Stratified Random Sampling**: A specific sampling technique used to ensure the quality and robustness of the data, enabling the model to better understand the nuances and underlying patterns of the text.

## Data Directory (`data/`)
- Relevant data files such as training data, baselines, or log files.
	- `datasets/`
        - `baselines/`
        - `TRS/`
    - `samples/`
    - `supporting/`
    - `tests/`

## Models Directory (`models/`)
- Trained models and related files.

## Notebooks (`notebooks/`)
- Jupyter notebooks for exploratory analysis, sampling, thresholds and performance evaluations. 
    - `SPT.ipynb`
    - `performance_evaluations.ipynb`
    - `classification_pipeline.ipynb`

## Plots (`plots/`)
- EDA and performance analysis plots.

## Source Files (`srcs/`)
- Source code files.
    - `classifier/`
        - `classifier.py`
        - `load_classifier.py`
        - `UCA_trainer.py`
    - `attention_filtering.py`
    - `freq_logging.py`
    - `labeler.py`

## Utils (`utils/`)
- Directory containing utility scripts and helper functions.
    - `contraction_mapping.json`
    - `fnPlots.py`
    - `fnProcessing.py`
    - `fnSampling.py`
    - `fnTesting.py`
    - `fnUtils.py`

## Configuration and Setup
- `config.ini`: Configuration file containing various settings and parameters.
- `Dockerfile`: Dockerfile to build and run the project in a containerized environment.
- `requirements.txt`: File listing the Python package dependencies required for the project.

## Data Directory (data)
The data directory includes the following subdirectories with the associated description:
- `datasets/`: Directory containing the datasets used for training and testing.
    - `baselines/`: Directory containing the original dataset `dialog_dataset.py` and baseline datasets used in performance evaluations.
    - `TSRS/`: Directory containing `Time Stratified Random Samples` that is the data used during development that was sampled from the main data source

## Models Directory (models)
Includes best performing model and related files.

## Notebooks (notebooks)
The notebooks in this directory provide visualizations and results analyasis. The main notebook is `SPT.ipynb` (Samples, Proportions, and Thresholds), which contains just that, the process used to obtain appropriate samples, proportions, and thresholds used throughout. The `performance_evaluations.ipynb` notebook provides the final performance against various baselines.

## Source files (srcs)
Contains the main source code files for the project. I break down each of the main files below, dsicusing the methods used and intended outcome. 

- `labeler.py` contains the core logic for detecting and labeling dialog in the dataset. It introduces the `DialogDetector` class, which includes a variety of methods to identify dialog, such as pattern recognition, cosine similarity computations (against training data and `FnWCn logs`), profanity checks, and length thresholds.
- `FnWCn.py` (Frequency number Word Count number) contains the code for the frequency logging process that deals with the logging of high-frequency rows, working based on dynamically set frequency and word count thresholds. It either updates existing files or creates new ones, depending on each specific frequency threshold and word count combination.
- The `attention_filtering` script is designed to cluster and analyze text documents based on attention values. It utilizes BERT's attention mechanisms, applies KMeans clustering, finds the five-number summary of attention values, and optionally visualizes the clustering in a scatter plot using PCA. It provides insights into the data, such as the junk ratio within each cluster, and enables exploratory data analysis for understanding the structure and dialog distribution within the dataset.
- `classifer.py` contains two main components:
    1. **Model Architecture (via the `create_model` function)**:
    - Constructs a sequential neural network with TensorFlow.
    - Comprises dense layers with specific activations, regularization, and optional batch normalization, and dropout.
    - Ends with a sigmoid activation for binary classification.
    2. **Model Training (via the `train_model` function)**:
    - Uses the Adam optimizer, complemented with an exponential decay learning rate.
    - Integrates callbacks for learning rate adjustment, overfitting prevention, and best model saving.
    - The final model is saved in the `saved_models` directory.
- `UST_trainer.py` contains the core logic for unsupervised and semi-supervised training through self-training methods:
    1. **Self-Training Process (`self_train` function)**:
        - Leverages labeled, unlabeled, and validation data to iteratively refine model training.
        - Utilizes a teacher-student paradigm, where the student model learns from pseudo-labels of unlabeled data, and the teacher guides the process.
        - Incorporates dropout during inference for robustness in pseudo-labeling.
        - Handles uncertainty filtering and dynamic updating of training data with selected samples from unlabeled data.
        - The process iteratively resets weights, shuffles data, trains the student model, evaluates uncertainty, updates labels, and transfers knowledge from student to teacher.
        - Final results include the trained teacher model, evaluation metrics, and histories of training.
    2. **Model Reset (`reset_weights` function)**:
        - Allows for the resetting of model weights, providing fresh starts in successive iterations.
    3. **Dropout During Inference (`DropoutInference` class)**:
        - A specialized model class that includes dropout during the inference phase, facilitating uncertainty analysis in pseudo-labeling.
## Utils (utils)
The files in this directory are commonly found in ML projects. However, a distinctive feature is the `contraction_mapping.json` file, which houses a mapping of contractions to their expanded forms. This serves as a reference file for punctuated text, allowing for its transformation without significant loss of meaning. Additionally, the `fnSampling.py` file is noteworthy as it contains time-stratified random sampling functions, essential for providing high-quality and robust data.
## Configuration and Setup (config.ini)
Details on how to use `Config.ini`, `Dockerfile`, and `Requirements.txt` for configuring and setting up the project environment.

## Use Cases and Future Work
Refer to the notebook results_analysis.ipynb for examples of applications.

Future work involves finalizing the narrative tracking feature and using generative models to create dynamic training datasets. 
The example below illistrates the function text_generator.generate() completing the text generation.
 ![Dialog example 1](plots\genpreview1.png)

## References
- Bird, S., Klein, E., & Loper, E. (2021). Natural Language Processing with Python. In *Natural Language Processing with Python* (3rd ed., ch. 9). O'Reilly Media. Retrieved from https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/ch09.html
- Bird, S., Klein, E., & Loper, E. (2021). Natural Language Processing with Python. In *Natural Language Processing with Python* (3rd ed., ch. 9). O'Reilly Media. Retrieved from https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/ch09.html#:-:text=Implementing%20a%20Naive,reasons%0Afor%20this%3A
- Brownlee, J. (n.d.). How to Develop a Word Embedding Model for Predicting Movie Review Sentiment. *Machine Learning Mastery*. Retrieved from https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
- Almeida,Tiago and Hidalgo,Jos. (2012). SMS dialog Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
- Gunasekar, S., Zhang, Y., Aneja, J., Mendes, C.C., Giorno, A.D., Gopi, S., Javaheripi, M., Kauffmann, P.C., Rosa, G.D., Saarikivi, O., Salim, A., Shah, S., Behl, H.S., Wang, X., Bubeck, S., Eldan, R., Kalai, A.T., Lee, Y.T., & Li, Y. (2023). Textbooks Are All You Need. ArXiv, abs/2306.11644.
- Longpre, S., Yauney, G., Reif, E., Lee, K., Roberts, A., Zoph, B., Zhou, D., Wei, J., Robinson, K., Mimno, D.M., & Ippolito, D. (2023). A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity. ArXiv, abs/2305.13169.
- Jung, J., West, P., Jiang, L., Brahman, F., Lu, X., Fisher, J., Sorensen, T., & Choi, Y. (2023). Impossible Distillation: from Low-Quality Model to High-Quality Dataset & Model for Summarization and Paraphrasing. ArXiv, abs/2305.16635.
- Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. NIPS.