import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

from modules.build import LANGUAGE_REGISTRY


@LANGUAGE_REGISTRY.register()
class BERTLanguageEncoder(nn.Module):
    def __init__(self, cfg, weights="bert-base-uncased", hidden_size=768,
                 num_hidden_layers=4, num_attention_heads=12, type_vocab_size=2):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            weights, do_lower_case=True
        )
        self.bert_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            type_vocab_size=type_vocab_size
        )
        self.model = BertModel.from_pretrained(
            weights, config=self.bert_config
        )

    def forward(self, txt_ids, txt_masks, **kwargs):
        return self.model(txt_ids, txt_masks).last_hidden_state
