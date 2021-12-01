# module-classifier

This is a Python module for a module classifier for The Syllabus.

## Classification

### Usage
To predict the module(s) for a text, use the `Classifier.predict_text()`
 method:
```
from module_classifier.classification import Classifier

classifier = Classifier() # use the default model
# Alternatively, specify a custom model file:
classifier = Classifier(model_path=model_file_path)

classifier.predict_text("This text is about automation and AI", k=3)

[(Module(S=6, module=8), 0.9990239143371582),
 (Module(S=6, module=2), 0.0004524348769336939),
 (Module(S=6, module=9), 0.0002751499123405665)]
```

The method returns a list of tuples where each tuple comprises the module and the respective model 
confidence.
 
The function parameter `k` determines the number of results.
If `k` is set to 1 (default), only the most probable module is returned.

The `predict_row()` method expects a row from a CSV file in the form of a 
dictionary as input.
It extracts the text fields and returns the same output format.

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