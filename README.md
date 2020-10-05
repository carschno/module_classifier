# module-classifier-api

This is a Python module for a module classifier for The Syllabus.


## Usage

To predict the module(s) for a text, use the `Classifier.predict_text()`
 method:
```
from module_classifier.classification import Classifier

classifier = Classifier()
classifier.predict_text("This is a text", k=3)

[('S3.M2', 0.5078761577606201), ('S3.M4', 0.456290066242218), ('S2.M3', 0.016183091327548027)]
```

The method returns a list of tuples where each tuple comprises the module and the respective model 
confidence.
 
The function parameter `k` determines the number of results.
If `k` is set to 1 (default), only the most probably module is output.
