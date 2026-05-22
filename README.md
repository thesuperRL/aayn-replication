# Transformer Implementation - "Attention is All You Need" Replication

An excercise in creating a PyTorch replication of the Transformer model from the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Ryan Li.

## Dataset

The model trains on the Multi30k dataset (English to German translation), which is a smaller alternative to WMT datasets.

- Training samples: ~29,000
- Validation samples: ~1,000
- Test samples: ~1,000

## Implementation
All code in [`layers`](layers) is written by Ryan.

Cursor AI was used in implementing data processing and the train file itself because that is not the goal of this project/excercise.
