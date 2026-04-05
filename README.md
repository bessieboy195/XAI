Explainable AI code demonstration:

structure:
project.py → Runs the base model and prints predictions
lime_project.py → Generates LIME explanations and saves HTML outputs
project_shap.py → Generates SHAP explanations and saves HTML outputs
anchors.py → Generates Anchor explanations and saves results to text files


requirements: 
Install the required Python libraries - pip install transformers datasets lime shap alibi spacy matplotlib scikit-learn
download a SpaCy model for anchors-python -m spacy download en_core_web_sm


how to run:
1. base model - python project.py
2. lime - python lime_project.py
3. shap - python project_shap.py
4. anchors - python anchors.py


dataset:
the project uses the bias in bios dataset from huggingface (LabHC/bias_in_bios) 
only a small subset of the dataset was used.


model:
A pre-trained transformer model from HuggingFace is used- distilbert-base-uncased-finetuned-sst-2-english  


output:
1. lime- HTML explanations + plots
2. shap- HTML explanations
3. anchors- text files with anchors, precision and coverage


important notes:
- anchors may be very slow due to sampling
- results may vary for lime due to randomness
- shap provides stable explanations
- the dataset consists of neutral biographies, the sentiment predictions are not meaningful. the focus is on analyzing which words influence predictions
- a small subset of the dataset is used due to computational limitations
