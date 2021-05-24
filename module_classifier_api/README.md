# module-classifier-api

This is a Python module for a module classifier for The Syllabus.


## Usage
To predict the module(s) for a text, use the `Classifier.predict_text()`
 method:
```
from module_classifier.classification import Classifier

classifier = Classifier() # use the default model
# Alternatively, specify a custom model file:
classifier = Classifier(model_path=model_file_path)

classifier.predict_text("This text is about automation and AI", k=3)

[('S6.M8', 0.9990239143371582),
 ('S6.M2', 0.0004524348769336939),
 ('S6.M9', 0.0002751499123405665)]
```

The method returns a list of tuples where each tuple comprises the module and the respective model 
confidence.
 
The function parameter `k` determines the number of results.
If `k` is set to 1 (default), only the most probable module is returned.

The `predict_row()` method expects a row from a CSV file in the form of a 
dictionary as input.
It extracts the text fields and returns the same output format.

