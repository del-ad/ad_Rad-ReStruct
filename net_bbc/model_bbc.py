from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import random
import time
from typing import Dict, List, Optional

import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate_fn_map, collate
from torchvision import transforms
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler

from data_utils.data_radrestruct import RadReStructEval, RadReStructPrecomputed, RadReStructPrecomputedEval, get_targets_for_split
from evaluation.evaluator_radrestruct import AutoregressiveEvaluator
from evaluation.predict_autoregressive_VQA_radrestruct import predict_BBC, predict_autoregressive_VQA
from net.image_encoding import ImageEncoderEfficientNet
from net.question_encoding import QuestionEncoderBERT
#KB
from knowledge_base.knowledge_base_loader import KnowledgeBase, KnowledgeBasePostProcessor
from knowledge_base.constants import Constants, Mode
## BBC variants
#from net_bbc.variant_bbc_noncomposite_sequence import forward, training_epoch_end, training_step, validation_epoch_end, validation_step
#from net_bbc.variant_bbc_noncomposite_OVERFIT import forward, training_epoch_end, training_step, validation_epoch_end, validation_step
#from net_bbc.variant_bbc_noncomposite_OVERFIT_nopatch import forward, training_epoch_end, training_step, validation_epoch_end, validation_step
#from net_bbc.variant_bbc_noncomposite_CLSfusion_overfit_nopatch import forward, training_epoch_end, training_step, validation_epoch_end, validation_step
#from net_bbc.variant_bbc_simpleffn_noncomposite_overfit import forward, training_epoch_end, training_step, validation_epoch_end, validation_step
from net_bbc.variant_bbc_simpleffn_noncomposite import forward, training_epoch_end, training_step, validation_epoch_end, validation_step
#
from transformers import AutoTokenizer
from transformers import get_scheduler
from memory_profiler import profile


CONSTANTS = Constants(Mode.CLUSTER)

# handle info dicts in collate_fn
def collate_dict_fn(batch, *, collate_fn_map):
    return batch


def custom_collate(batch):
    default_collate_fn_map.update({dict: collate_dict_fn})
    return collate(batch, collate_fn_map=default_collate_fn_map)

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

def print_mem(tag):
    print(f"{timestamp()}[{tag}] RSS: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

# implementation of 2D positional encoding from https://github.com/gaopengcuhk/Stable-Pix2Seq
def create_pos_encoding(args):
    temperature = 10000
    hidden_size = args.hidden_size
    max_position_embeddings = args.max_position_embeddings  # args.num_image_tokens + 3 + args.num_question_tokens
    num_pos_feats = hidden_size // 4
    img_mask = torch.ones((1, args.img_feat_size, args.img_feat_size))  # bs, img_tokens
    y_embed = img_mask.cumsum(1, dtype=torch.float32)
    x_embed = img_mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_img = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    pos_img = pos_img.flatten(2)

    # extend pos_img with zeros to match the size of pos_seq
    pos_img = torch.cat(
        (pos_img, torch.zeros((pos_img.shape[0], pos_img.shape[1], max_position_embeddings - pos_img.shape[2]), device=pos_img.device)), dim=2)
    # switch last two dimensions
    pos_img = pos_img.permute(0, 2, 1)

    return pos_img

# Original 4 token version
class MyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # no special image positional embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # when using patch image features
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings,
        #                                         config.hidden_size // 2)  # other half is reserved for spatial image embeddings
        if args.progressive:
            ## added knowledge token
            #self.token_type_embeddings = nn.Embedding(5, config.hidden_size)  # image = 0, history Q - 2, history A - 3, current question - 1, knowledge - 4
            self.token_type_embeddings = nn.Embedding(8, config.hidden_size)  # image = 0, history Q - 2, history A - 3, current question - 1, knowledge - 4
            
        else:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.img_pos_embeddings = create_pos_encoding(self.args)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,

    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # ## inspecting token type embeddings
        # e0 = self.token_type_embeddings.weight[0:1]
        # e1 = self.token_type_embeddings.weight[1:2]
        # e2 = self.token_type_embeddings.weight[2:3]
        # e3 = self.token_type_embeddings.weight[3:4]
        # e4 = self.token_type_embeddings.weight[4:5]
        # e5 = self.token_type_embeddings.weight[5:6]
        # e6 = self.token_type_embeddings.weight[6:7]
        # e7 = self.token_type_embeddings.weight[7:8]
        # e8 = self.token_type_embeddings.weight[8:9]



        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            ## no special image positional embeddings ---- to be used when no patch image features are used in the bbc fusion sequence
            # position_embeddings = torch.cat((self.img_pos_embeddings.to(position_embeddings.device), position_embeddings),
            #                                 dim=-1)  # add image position embeddings and sequence position embeddings
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MyBertModel(BertModel):
    """

    Overwrite BERTModel in order to adapt the positional embeddings for images
    """

    def __init__(self, config, add_pooling_layer=True, args=None):
        super().__init__(config)
        self.config = config

        self.embeddings = MyBertEmbeddings(config, args=args)
        self.encoder = BertEncoder(config)
        self.cls_fusion_token = nn.Parameter(torch.randn(1, 1, args.hidden_size))
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class ModelBBC(nn.Module):
    def __init__(self, args):
        super(ModelBBC, self).__init__()
        self.args = args

        self.image_encoder = ImageEncoderEfficientNet(args)
        #self.frozen_image_encoder = FrozenImageEncoderEfficientNet(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)
        self.question_encoder = QuestionEncoderBERT(args, self.tokenizer)
        
        if args.use_precomputed:
            with open(f'/home/guests/adrian_delchev/code/ad_Rad-ReStruct/precomputed/text_options_embeddings.pkl', 'rb') as f:
                self.precomputed_text_options = pickle.load(f)

        #self.knowledge_base = KnowledgeBase(args.kb_dir)
        self.knowledge_base_processor = KnowledgeBasePostProcessor(text_encoder=self.question_encoder, image_encoder=self.image_encoder)

        
        # self.fusion_config = BertConfig(vocab_size=1, hidden_size=args.hidden_size, num_hidden_layers=args.n_layers,
        #                                 num_attention_heads=args.heads, intermediate_size=args.hidden_size * 4,
        #                                 max_position_embeddings=args.max_position_embeddings)

        # self.bbc_fusion = MyBertModel(config=self.fusion_config, args=args)
        # self.bbc_fusion.train()

        self.bbc_simple_ffn = nn.Sequential(
            nn.Linear(args.hidden_size * 2, 768),
            nn.ReLU(),
            nn.Dropout(args.classifier_dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(args.classifier_dropout),
            nn.Linear(64, 1),
        )

        if "vqarad" in args.data_dir:
            self.bbc_classifier = nn.Linear(args.hidden_size, args.num_classes)
        else:
            if not args.use_simple_classifier:
                self.bbc_classifier = nn.Sequential(
                    ## comment out when training / doing non overfit stuff
                    ### Original - comment out next line during overfitting
                    nn.Dropout(args.classifier_dropout),
                    nn.Linear(args.hidden_size, 256),
                    nn.ReLU(),
                    # nn.BatchNorm1d(256),
                    ### output is a single output
                    nn.Linear(256, 1))
                self.bbc_classifier.train()

    

    def forward(self, img_and_global, input_ids, q_attn_mask, attn_mask, token_type_ids_q=None, batch_metadata=None, mode='train'):
        return forward(model=self, img_and_global=img_and_global, \
                        input_ids=input_ids, q_attn_mask=q_attn_mask, \
                        attn_mask=attn_mask, token_type_ids_q=token_type_ids_q, \
                        batch_metadata=batch_metadata, mode=mode)
        
    def get_forward_origin(self):
        # Access the _origin_file attribute of the imported function
        return forward._origin_file
        
        
    def get_global_embeddings_in_batches(self, encoder, images, max_batch_size, device):
        """
        encoder: your image_encoder
        images: tensor of shape (total_images, 3, 456, 456)
        max_batch_size: int, e.g., 64
        device: torch.device
        """
        all_embeddings = []
        n_images = images.shape[0]
        with torch.no_grad():
            for start in range(0, n_images, max_batch_size):
                end = min(start + max_batch_size, n_images)
                batch = images[start:end].to(device)
                emb = encoder.get_global_embeddings(batch)
                all_embeddings.append(emb.cpu())  # Move to CPU to save GPU memory if needed
                del batch, emb
        # Concatenate results back together along the first dimension
        return torch.cat(all_embeddings, dim=0)

    def produce_option_embeddings_batched(self, kb_examples: List[Dict], image_encoder):

        used_device = next(image_encoder.parameters()).device
        image_batches = []
        slice_metadata = []
        current_index = 0

        was_training = image_encoder.training
        image_encoder.eval()
        try:
            for batch_idx, single_batch in enumerate(kb_examples):
                for option, images in single_batch.items():
                    if images:  # Skip options with no images
                        flattened_images = []
                        sizes = []
                        is_batched_list = []
                        for img in images:
                            if img.dim() == 3 and img.shape[0] == 3:
                                flattened_images.append(img)
                                sizes.append(1)
                                is_batched_list.append(False)
                            elif img.dim() == 4 and img.shape[1] == 3:
                                n = img.shape[0]
                                for i in range(n):
                                    flattened_images.append(img[i])
                                sizes.append(n)
                                is_batched_list.append(True)
                            else:
                                raise ValueError(f"Unexpected image tensor shape: {img.shape}")
                        if flattened_images:
                            tensor = torch.stack(flattened_images)  # shape: (N_i, 3, 456, 456)
                            image_batches.append(tensor)
                            start = current_index
                            end = start + tensor.shape[0]
                            # Save both batch_idx, option, and the sizes list to group for pooling later
                            slice_metadata.append((batch_idx, option, start, end, sizes, is_batched_list))
                            current_index = end

            pooled_embeddings = [{} for _ in range(len(kb_examples))]
            if len(image_batches) > 0:
                batched_images_cpu = torch.cat(image_batches, dim=0)
                batched_images_gpu = batched_images_cpu.to(used_device)
                # single forward pass - fails for large batches
                # with torch.no_grad():
                #     global_embeddings = image_encoder.get_global_embeddings(batched_images_gpu)
                #     del batched_images_gpu, batched_images_cpu
                with torch.no_grad():
                    global_embeddings = self.get_global_embeddings_in_batches(image_encoder, batched_images_gpu, \
                                                                              max_batch_size=64, device=used_device)
                del batched_images_cpu, batched_images_gpu

                ###
                for batch_idx, option, start, end, sizes, is_batched_list in slice_metadata:
                    embeddings = global_embeddings[start:end]  # shape: (N_i, D)
                    pooled_list = []
                    idx = 0
                    for sz, is_batched in zip(sizes, is_batched_list):
                        group = embeddings[idx:idx+sz]  # group of embeddings for each input tensor
                        if is_batched:  # Came from a (n, 3, 456, 456) input
                            # group is (n, D) - L1 questions
                            # Either keep as is, or if you want shape (n, 1, D):
                            pooled_list.append(group)  # (n,1,D)
                        else:
                            # group is (1, D) - L2 and L3 questions
                            #pooled, _ = torch.max(group, dim=0, keepdim=True)  # (1, D)
                            pooled_list.append(group)
                        idx += sz
                    if len(pooled_list) > 1:
                        s = torch.cat(pooled_list, dim=0)
                        option_embeddings, _ = torch.max(s, dim=0, keepdim=True)
                    else:
                        option_embeddings = torch.cat(pooled_list, dim=0)
                    # Concatenate all (n, 1, D) and (1, D) tensors as you wish, or keep as list:
                    pooled_embeddings[batch_idx][option] = option_embeddings
                    del embeddings, pooled_list, option_embeddings
                image_batches.clear()
                slice_metadata.clear()
                del global_embeddings, image_batches, slice_metadata

            # Add any missing knowledge
            for batch_idx, single_batch in enumerate(kb_examples):
                for option in single_batch:
                    if option not in pooled_embeddings[batch_idx]:
                        pooled_embeddings[batch_idx][option] = image_encoder.missing_knowledge_embedding

        finally:
            if was_training:
                image_encoder.train()

        return pooled_embeddings
    
    
    def encode_batch_options_precomputed(self, batch_metadata):
        text_options_embeddings = [{} for _ in batch_metadata]
        for batch_index, item in enumerate(batch_metadata):
            for option in item['options']:
                text_options_embeddings[batch_index][option] = self.precomputed_text_options[option]


        return text_options_embeddings, self.precomputed_text_options['colon_token'], self.precomputed_text_options['sep_token'], self.precomputed_text_options['pad_token']
    
    def encode_batch_sep_pad_precomputed(self, batch_metadata):
        text_options_embeddings = [{} for _ in batch_metadata]
        for batch_index, item in enumerate(batch_metadata):
            for option in item['options']:
                text_options_embeddings[batch_index][option] = self.precomputed_text_options[option]


        return self.precomputed_text_options['sep_token'], self.precomputed_text_options['pad_token']
    
    def produce_option_embeddings_precomputed(self, kb_examples: List[Dict]):

        pooled_embeddings = [{} for _ in range(len(kb_examples))]
        for batch_idx, dict in enumerate(kb_examples):
            for option, tensors in dict.items():
                pooled_embeddings[batch_idx][option] = random.choice(tensors)   
        
        return pooled_embeddings
    
    def produce_option_embeddings(self, kb_examples: List[Dict], batch_metadata, global_image_embedding):
        kb_batch_embeddings = []
        kb_embeddings = {}
        used_device = next(self.image_encoder.parameters()).device
        
        was_training = self.image_encoder.training
        self.image_encoder.eval()
        try:
            with torch.no_grad():
                for idx, single_batch in enumerate(batch_metadata):
                    kb_embeddings = {}

                    ### positive options is available:
                    if 'positive_option' in single_batch:
                        for option in single_batch['options']:
                            if option in single_batch['positive_option']:
                                kb_embeddings[option] = global_image_embedding[idx:idx+1,]
                            else:
                                ## not a positive option = is there a kb example
                                if len(kb_examples[idx][option])>0:
                                    kb_embeddings[option] = kb_examples[idx][option]
                                ## otherwise a missing knowledge embedding
                                else:
                                    kb_embeddings[option] = self.image_encoder.missing_knowledge_embedding.to(device=used_device)
                        kb_batch_embeddings.append(kb_embeddings)
                    ### no positive options = entirely from kb
                    else:
                        for option in single_batch['options']:
                            if len(kb_examples[idx][option])>0:
                                kb_embeddings[option] = kb_examples[idx][option]
                                ## otherwise a missing knowledge embedding
                            else:
                                kb_embeddings[option] = self.image_encoder.missing_knowledge_embedding.to(device=used_device)
                        kb_batch_embeddings.append(kb_embeddings)    




                    # ### positive options is not available:
                    # for option in single_batch['options']:
                    #     if 'positive_options' in single_batch:
                        
                    #     else:

                    #     if kb_examples[option]:
                    #         # kb_embeddings[option] = self.image_encoder.encode_and_max_pool(
                    #         #     torch.stack(examples).to(device=used_device))
                    #         kb_embeddings[option] = kb_examples[option]
                    #     #### CASE WHEN NO KNOWLEDGE IS AVAILABLE FROM THE KB
                    #     else:
                    #         #kb_embeddings[option] = torch.zeros(1, 1, 768).to(device=used_device)
                    #         kb_embeddings[option] = self.image_encoder.missing_knowledge_embedding.to(device=used_device)
                    # kb_batch_embeddings.append(kb_embeddings)
        finally:
            if was_training:
                self.image_encoder.train()
                     
                    
        return kb_batch_embeddings
       

   

class ModelWrapperBBC(pl.LightningModule):
    def __init__(self, args, train_df=None, val_df=None):
        super(ModelWrapperBBC, self).__init__()
        self.args = args
        self.model = ModelBBC(args)
        self.train_df = train_df
        self.val_df = val_df

        if 'radrestruct' in args.data_dir:
            self.test_data_train = get_targets_for_split('train', limit_data=30)
            self.test_data_val = get_targets_for_split('val', limit_data=None)
            self.train_ar_evaluator_vqa = AutoregressiveEvaluator()
            self.val_ar_evaluator_vqa = AutoregressiveEvaluator()

            # handle info dicts in collate_fn
            def collate_dict_fn(batch, *, collate_fn_map):
                return batch
            
            def custom_collate(batch):
                default_collate_fn_map.update({dict: collate_dict_fn})
                return collate(batch, collate_fn_map=default_collate_fn_map)

            img_tfm = self.model.image_encoder.img_tfm
            norm_tfm = self.model.image_encoder.norm_tfm
            test_tfm = transforms.Compose([img_tfm, norm_tfm]) if norm_tfm is not None else img_tfm
            
            if args.use_precomputed:
                self.ar_valdataset = RadReStructPrecomputed(tfm=test_tfm, mode='val', args=args, precompute=True)
                self.ar_val_loader_vqa = DataLoader(self.ar_valdataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                                    collate_fn=custom_collate)
                self.ar_traindataset = RadReStructPrecomputed(tfm=test_tfm, mode='train', args=args, precompute=True)
                self.ar_train_loader_vqa = DataLoader(self.ar_traindataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                                    collate_fn=custom_collate)
            else:
                self.ar_valdataset = RadReStructEval(tfm=test_tfm, mode='val', args=args, limit_data=None)
                self.ar_val_loader_vqa = DataLoader(self.ar_valdataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                                    collate_fn=custom_collate)
                self.ar_traindataset = RadReStructEval(tfm=test_tfm, mode='train', args=args, limit_data=30)
                self.ar_train_loader_vqa = DataLoader(self.ar_traindataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                                    collate_fn=custom_collate)

            pos_weights_path = 'data/radrestruct/all_pos_weights.json'
            with open(pos_weights_path, 'r') as f:
                self.pos_weight = torch.tensor(json.load(f), dtype=torch.float32)

            ### Original - use for training/comment out for overfitting
            #self.loss_fn = BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")
            self.loss_fn = BCEWithLogitsLoss(reduction="none")

        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.train_preds = []
        self.val_preds = []
        self.val_infos = []
        self.val_soft_scores = []
        self.val_gen_preds = []
        self.train_gen_preds = []
        self.train_targets = []
        self.train_soft_scores = []
        self.train_infos = []
        self.train_gen_labels = []
        self.val_targets = []
        self.val_gen_labels = []
        self.train_answer_types = []
        self.val_answer_types = []

        if "radrestruct" in args.data_dir:
            with open('data/radrestruct/answer_options.json', 'r') as f:
                self.answer_options = json.load(f)

    def get_masked_loss(self, loss, mask, targets, path_pos_weight=None):

        if path_pos_weight is not None:
            # for pos predictions multiply loss with path_pos_weight, for negatives with 1 (no change)
            loss = (targets * loss * path_pos_weight) + ((1 - targets) * loss)

        # get mean loss when only considering non-masked options
        masked_loss = loss.masked_select(mask).mean()

        return masked_loss

    def forward(self, img_and_global, input_ids, q_attn_mask, attn_mask, token_type_ids_q, batch_metadata, mode='train'):
        return self.model(img_and_global, input_ids, q_attn_mask, attn_mask, token_type_ids_q, batch_metadata, mode=mode)

    # ### Original
    # def training_step(self, batch, batch_idx, dataset="vqarad"):
    #     if "vqarad" in self.args.data_dir:
    #         img, question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
    #     else:
    #         img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch

    #     question_token = question_token.squeeze(1)
    #     attn_mask = attn_mask.squeeze(1)
    #     q_attention_mask = q_attention_mask.squeeze(1)
    #     ## metadata such as path for the given batch - returned from getitem as info
    #     batch_info = batch[6][0]

    #     out, _ = self(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, batch_info, mode='train')

    #     logits = out
    #     if "vqarad" in self.args.data_dir:
    #         pred = logits.softmax(1).argmax(1).detach()
    #     else:  # multi-label classification
    #         pred = (logits.sigmoid().detach() > 0.5).detach().long()
    #         self.train_soft_scores.append(logits.sigmoid().detach())

    #     self.train_preds.append(pred)
    #     self.train_targets.append(target)

    #     if "vqarad" in self.args.data_dir:
    #         self.train_answer_types.append(answer_type)
    #     else:
    #         self.train_infos.append(info)

    #     loss = self.loss_fn(logits, target)
    #     loss = self.get_masked_loss(loss, mask, target, None)  # only use loss of occuring classes

    #     self.log('Loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    #     return loss


    ### NEW
    def training_step(self, batch, batch_idx, dataset="vqarad"):
        return training_step(self, batch, batch_idx, dataset)

    # NEW
    def validation_step(self, batch, batch_idx):
        return validation_step(self, batch, batch_idx)

    ### Original
    def configure_optimizers(self):
        # real training numbers
        #total_steps = ((165002 * 3) / self.args.batch_size) * self.args.epochs
        #overfit
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer
    
    # from scratch bbc fusion
    # def configure_optimizers(self):
    #     # real training numbers
    #     #total_steps = ((165002 * 3) / self.args.batch_size) * self.args.epochs
    #     #overfit
        
    #     lr = 5e-4
    #     weight_decay = 0.01
    #     batch_size = 32
    #     max_epochs = 5
    #     warmup_ratio = 0.1
    #     scheduler_type = "linear"  # or "cosine"
        
    #     total_training_steps  = ((8 * 3) / self.args.batch_size) * self.args.epochs
    #     warmup_steps = int(warmup_ratio * total_training_steps)
        
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=weight_decay)
    #     scheduler = get_scheduler(
    #         name=scheduler_type,  # 'linear', 'cosine', etc.
    #         optimizer=optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=total_training_steps,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",  # important! because scheduler.step() is called every step
    #             "frequency": 1,
    #         },
    #     }
    
    ## Steady decrease in LR
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
    #     scheduler = LinearLR(
    #         optimizer,
    #         start_factor=1.0,
    #         end_factor=0.02,
    #         total_iters=28  # set to number of epochs you want the transition over
    #     )
    #     return [optimizer], [{
    #         "scheduler": scheduler,
    #         "interval": "epoch",      # Step the scheduler every epoch
    #         "frequency": 1,
    #         "monitor": None,          # Not needed for LinearLR
    #         "name": "linear_lr"
    #     }]
    
    # ### New
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
    #     #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    #     scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6),
    #         'monitor': 'loss',  # or 'val_loss' if you're using validation
    #         'interval': 'epoch',
    #         'frequency': 1
    #     }
    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': scheduler,
    #         'monitor': 'loss',  # optional, used for ReduceLROnPlateau
    #     }


        # Implementation for PyTorch Lightning < 2.0
    def training_epoch_end(self, outputs):
        training_epoch_end(self,outputs)

        # Optionally return something, but usually unnecessary
        # return {'accuracy': accuracy}


    # def on_train_epoch_end(self):

    #     e0 = self.model.fusion.embeddings.token_type_embeddings.weight[0:1]
    #     e1 = self.model.fusion.embeddings.token_type_embeddings.weight[1:2]
    #     e2 = self.model.fusion.embeddings.token_type_embeddings.weight[2:3]
    #     e3 = self.model.fusion.embeddings.token_type_embeddings.weight[3:4]
    #     e4 = self.model.fusion.embeddings.token_type_embeddings.weight[4:5]
    #     e5 = self.model.fusion.embeddings.token_type_embeddings.weight[5:6]
    #     e6 = self.model.fusion.embeddings.token_type_embeddings.weight[6:7]
    #     e7 = self.model.fusion.embeddings.token_type_embeddings.weight[7:8]
    #     e8 = self.model.fusion.embeddings.token_type_embeddings.weight[8:9]
    #     print("done")
    #     return super().on_train_epoch_end()
    
    # def on_save_checkpoint(self, checkpoint):
    #     print(self.model.fusion.embeddings.token_type_embeddings.weight.shape)
    #     print(checkpoint['state_dict']['model.fusion.embeddings.token_type_embeddings.weight'].shape) 
    

    def validation_epoch_end(self, outputs):
        validation_epoch_end(self, outputs)



