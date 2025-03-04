# MOAST: Mechanism of Action Similarity Tool

## Overview

MOAST (Mechanism of Action Similarity Tool) is a project, inspired by the popular sequence alignment tool BLAST, that I put together as part of my dissertation. The aim of the tool is to provide a framework in order to take phenotypic screening data from a large annotated reference screen and construct a database against which you can compare smaller screening runs of the same assay to obtain hypothesis for  Mechanism of Action using pairwise similarities of the smaller screening run to the reference screening data, much like BLAST.

## Evaluation Story

Check out the MOAST chapter from my [dissertation](https://escholarship.org/uc/item/572197kr)) or the preprint excerpt (biorxive.) for details on rational and validation I performed on the method using data from High-Content Image-based screens.

## Development Notebooks

Check out "KDEdraws\_fakeBLAST\_BayesianClassifier.ipynb" notebook for development of the KDE integration approach initially conceived and implemented as part of the installable package source.

The notebook "MOAST\_pycaretMLtesting.ipynb" contains the development of the pivot to a formal Machine-Learning implementation for the same task that I evaluated to perform better.

## Contribution

Contributions are welcome! If you find a bug or have feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, reach out via GitHub Issues or contact **alohith**.
