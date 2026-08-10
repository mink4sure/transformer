# Transformer
A toy example to get a feeling for the transformer architecture.

## Components
 - Some form of scraper/converter
    - to get data and stores in a pd dataframe
 - Custom Torch dataset
    - should create sequences of the desired lenght
 - Transformer model
 - Training 

### The Dataset
Inspired by some dataset found on `huggingface.co`, I think the dataset can be a Pandas Dataframe with some relevant fields.
 - `text`
 - `content_type`
 - ...
 On a second thought... I just will just use a dataset from `huggingface.co`.


