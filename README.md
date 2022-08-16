# TF predictions

## Models

### Scbasset modified

### TFBanformer

## Analysis

### Region and TF representation

### In silico mutagenesis

For the deepexplainer, you will need to modify one line in order to be able to to choose which TF the analyse

Go to `shap/explainers/_deep/deep_pytorch.py`

```python
elif output_rank_order.isnumeric():
    _, model_output_ranks = torch.sort(model_output_values, descending=True)
    model_output_ranks[0] = int(output_rank_order)
```