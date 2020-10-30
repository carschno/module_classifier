# module_trainer

This is a Python module for training a model to be used in the module classifier for The Syllabus.

## Usage

To train a new model:

```python
from module_trainer.training import Trainer

input_csv = "/path/to/training_data.csv"
model_file = "/path/to/model"
text_fields=("item_title","authors","publication_name","abstract_description")

trainer = Trainer()
trainer.train_model(input_csv, target_file=model_file, text_fields=text_fields)
Read 4M words
Number of words:  157148
Number of labels: 86
Progress: 100.0% words/sec/thread:  243635 lr:  0.000000 avg.loss:  1.353899 ETA:   0h 0m 0s
```

Mind that the `text_fields` have to correspond to what is defined in the input CSV file.
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

## Training effects

The training can be done in order to improve the model with added or updated training data.
However, this needs to be handled with care.
Re-training the model with all new training data might look tempting, 
but can result in a model that overfits to the new training data. 