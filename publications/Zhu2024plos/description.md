# Code Description

This is the training code used in the paper entitled "Learning to integrate parts for whole through correlated neural variability".

Please follows the tutorial to run the code in this fold.

1. Clone the repository to your local drive.
2. Copy the demo files in this fold (*.py and *.yaml) to the root directory.
3. Create two directories, **./checkpoint/** (for saving trained model results) and **./data/** (downloading the Caltech-UCSD Birds-200-2011 dataset into it).
4. Run the following command to call the script named `mnist.py` with the config file specified through the option:


   ```
   python main.py --config=TASK_SPECIFIC.yaml
   ```

## Citation

[Zhu, Zhichao, et al. "Learning to integrate parts for whole through correlated neural variability." PLOS Computational Biology 20.9 (2024): e1012401.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012401)

```
@article{Zhu2024plos,
    author = {Zhu, Zhichao and Qi, Yang and Lu, Wenlian and Feng, Jianfeng},
    journal = {PLoS Computational Biology},
    title = {Learning to integrate parts for whole through correlated neural variability},
    year = {2024},
    month = {09},
    volume = {20},
    pages = {1-25},
    number = {9},
}
```

