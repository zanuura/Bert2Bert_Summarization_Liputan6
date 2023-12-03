# Bert2Bert Liputan6

This is Bert2Bert [EncoderDecoderModel](https://huggingface.co/docs/transformers/model_doc/encoder-decoder) train on [Liputan6](https://huggingface.co/datasets/id_liputan6) Dataset Canonical, 
this model was base on this [Documentation](https://huggingface.co/docs/transformers/model_doc/bert-generation) 
and this [notebook](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb#scrollTo=w67vkz3KP9eZ) 

# How to Use?

## Install the package
Colab: 
```shell

!pip install torch
!pip install transformers[torch]
!pip install evaluate
!pip install datasets

```
Cmd:
```shell

pip install torch
pip install transformers[torch]
pip install evaluate
pip install datasets

```

## Install the Model
```shell

git clone https://github.com/zanuura/Bert2Bert_Summarization_Liputan6

```
## Import Package 
```python

from transformers import EncoderDecoderModel, AutoTokenizer, pipeline
import datasets

```

## Load Model and Tokenizer
```python

model = EncoderDecoderModel.from_pretrained("Bert2Bert_Summarization_Liputan6/model/") # insert the path
tokenizer = AutoTokenizer.from_pretrained("Bert2Bert_Summarization_Liputan6/model/") # you also can change the tokenizer from bert-base-uncased

```

## Test the Model
```python
## this is test with Liputan6 Test Dataset

## Load rouge for validation

rouge = datasets.load_metric("rouge")

def generate_summary(batch):

  inputs = tokenizer(batch['clean_article'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
  input_ids = inputs.input_ids.to("cuda")
  attention_mask = inputs.attention_mask.to("cuda")

  outputs = model.generate(input_ids, attention_mask=attention_mask)
  outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

  batch['pred'] = outputs_str

  return batch

results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["clean_article"])

pred_str = results['pred']
label_str = results['clean_summary']

rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

print(rouge_output)

```

References:
- [EncoderDecoderModel](https://huggingface.co/docs/transformers/model_doc/encoder-decoder)
- [BertGeneration](https://huggingface.co/docs/transformers/model_doc/bert-generation)
- [NoteBook Bert2Bert CNN Daily](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb#scrollTo=w67vkz3KP9eZ)
- Datasets Liputan6 [info](https://huggingface.co/datasets/id_liputan6)
- [Datasets](https://drive.google.com/file/d/1ixaIO24XBZX-BFVyHIk0FG0kI2W3lACD/view)

Hope you enjoyit 😎.
