from typing import List
import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaModel, AutoTokenizer


class QuestionEncoderBERT(nn.Module):
    def __init__(self, args, tokenizer):
        super(QuestionEncoderBERT, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        # load pre-trained BERT model
        config = AutoConfig.from_pretrained(args.bert_model)
        self.BERTmodel = RobertaModel.from_pretrained(args.bert_model, config=config)

    def forward(self, input_ids, q_attn_mask):
        # feed question to BERT model
        outputs = self.BERTmodel(input_ids, attention_mask=q_attn_mask)
        # get word embeddings
        word_embeddings = outputs.last_hidden_state

        return word_embeddings
    
    def encode_phrases(self, phrases, return_first_token_only=True):
        tokenized = self.tokenizer(
            phrases,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True  # ensures consistent shape
        ).to(device=next(self.parameters()).device)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            if return_first_token_only:
                return outputs[:, 0:1, :]  # shape: [batch_size, 1, hidden_dim]
            else:
                return outputs  # shape: [batch_size, seq_len, hidden_dim]
            
    def encode_options(self, options: List[str], return_first_token_only=True):
        
        encoded_options = {}
        for option in options:
            tokenized = self.tokenizer(
                option,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True  # ensures consistent shape
            ).to(device=next(self.parameters()).device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            with torch.no_grad():
                outputs = self.forward(input_ids, attention_mask)
                if return_first_token_only:
                    encoded_options[option] = outputs[:, 0:1, :]
                    #return outputs[:, 0:1, :]  # shape: [batch_size, 1, hidden_dim]
                else:
                    encoded_options[option] = outputs
                    #return outputs  # shape: [batch_size, seq_len, hidden_dim]
        return encoded_options

