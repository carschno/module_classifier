import argparse
import csv

from module_classifier.classification import Classifier
from module_classifier.settings import TEXT_FIELDS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a CSV file.')
    parser.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("r"),
        help="Input CSV file.",
        required=True,
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Number of predictions to output per row.",
    )
    parser.add_argument(
        "--output", "-o", type=argparse.FileType("w"), help="Output CSV file."
    )

    args = parser.parse_args()

    reader = csv.DictReader(args.input)
    headers = list(next(reader).keys())

    predicted_label_headers = [f"predicted_{i}" for i in range(args.k)]
    predicted_confidence_headers = [f"confidence_{i}" for i in range(args.k)]

    writer = csv.DictWriter(
        args.output,
        fieldnames=headers
        + predicted_label_headers
        + predicted_confidence_headers,
    )

    classifier = Classifier()

    writer.writeheader()
    for row in reader:
        output_row = row.copy()
        predictions = classifier.predict_row(
            row, k=args.k, columns=TEXT_FIELDS
        )
        for ((label, conf), label_header, conf_header,) in zip(
            predictions, predicted_label_headers, predicted_confidence_headers
        ):
            output_row[label_header] = label
            output_row[conf_header] = conf

        writer.writerow(output_row)
