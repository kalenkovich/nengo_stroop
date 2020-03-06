The code is run in a dedicated [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) environment called 'nengo_stroop'.
To recreate it, run the code below:

```cmd
conda env update -f nengo_stroop.yml -n nengo_stroop --prune
conda activate phonemes
```