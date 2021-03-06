# module-classifier

This is a Python application for a module (topic/category) classifier for The Syllabus.

## Module Classification

### Usage


To predict the module(s) for a text, use the `ModuleClassifier.predict_text()`
 method:
```
from module_classifier.classification.module_classifier import ModuleClassifier

classifier = ModuleClassifier() # use the default model on disk

# Or download the model from S3 and use it (uses local copy if already present):
classifier = ModuleClassifier.from_s3()

# Alternatively, specify a custom model file:
classifier = ModuleClassifier(model_path=model_file_path)

classifier.predict_text("This text is about automation and AI", k=3)

[Prediction(module=Module(section=6, module=8), prob=1.0000072),
 Prediction(module=Module(section=6, module=2), prob=1.12474345e-05),
 Prediction(module=Module(section=6, module=9), prob=1.099022e-05)]
```

The method returns a list of `Prediction` objects of length `k`.
Each of them compises comprises a `Module` object and the respective model 
confidence.
 
The function parameter `k` determines the number of results.
If `k` is set to 1 (default), only the most probable module is returned.

The `predict_row()` method expects a row from a CSV file in the form of a 
dictionary as input.
It extracts the text fields and returns the same output format.

## Main Edition Classifier

The Main Edition Classifier is a binary classifier that estimates whether an item is suitable for the main edition.
It returns a single prediction, either `True` or `False`, along with a probability score.

### Usage

Usage is essentially the same as for the module classifier above:

```
from module_classifier.classification.binary_classifier import MainEditionClassifier

c = MainEditionClassifier.from_s3()

classifier.predict_text("This text is about automation and AI")
```

The classifier also implements the same additional methods, such as `predict_row()` and `predict_texts()`.
## Explanation

### Command Line Tool

The built-in script `explain` generates a simplified visualization of the words that contributed to the classifier's decision.
Run it from the command line for instance like this:

```
explain -i <input.txt> -o <output.html> -k 3
```

The <input.txt> file is a file containing a single text to be explained.
The <output.html> file is the output file to which the output is written in HTML format.
The `-k` parameter determines the number of labels to be explained (defaults to 1, ie. explain only the highest scoring label).

Optionally, you can specify a custom model file with the `--model-file` parameter:

```
explain -i <input.txt> -o <output.html> --model-file <my_model_file>
```

Run `explain --help` for a full list of parameters.

### Python Interface

The module `module_classifier.explaination.explainer.Explainer` provides a Python class for explainations.
It is initialized with a `Classifier` object:

```
from module_classifier.classification import Classifier
from module_classifier.explain import Explainer

classifier = Classifier()
explainer = Explainer(classifier)

explainer.explain("text to explain", k=3)
```

The `explain()` method returns an `Explanation` object.
See the [Lime documentation](https://lime-ml.readthedocs.io/en/latest/lime.html#lime.explanation.Explanation) for available methods.




## Train a New Model

### Command Line Tool

To train a new model, either use the `train_module_classifier` script like this:

```shell
train_module_classifier --input data.csv --output model.bin --text-fields abstract description --class-field module_id_for_all
```

The `--text-field` arguments define the columns from the input CSV file that contain the texts to use for classification.
The `--class-field` argument defines the column in the input CSV file that contains the module ID.

Run `train_module_classifier --help` to get more detailed instructions.

### Python

```python
from module_classifier.training import Trainer

input_csv = "/path/to/training_data.csv"
model_file = "/path/to/model"
text_fields=("item_title","authors","publication_name","abstract_description")

trainer = Trainer()
trainer.train_model(input_csv, target_file=model_file, text_fields=text_fields)
```
The output should look like this:
```
Read 4M words
Number of words:  157148
Number of labels: 86
Progress: 100.0% words/sec/thread:  243635 lr:  0.000000 avg.loss:  1.353899 ETA:   0h 0m 0s
```

Mind that the `text_fields` should correspond to what is defined in the input CSV file.
The contents of these fields are extracted to generate the training data.  

The `class_field` argument specifies the column which contains the assigned module labels for each 
row in the input CSV file.
It defaults to `module_id_for_all`. 

The training time varies greatly, depending on the input data size, and the number and speed of CPUs.
It can take between a few minutes and hours.
 
Once done, the model can be used in the `module_classifier_api`:

```python
from module_classifier.classification import Classifier

c = Classifier(model_file)  # model_file generated above
``` 

### Training effects

The training can be done in order to improve the model with added or updated training data.
However, this needs to be handled with care.
Re-training the model with all new training data might look tempting, 
but can result in a model that overfits to the new training data. 
