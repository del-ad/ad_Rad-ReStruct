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
#
from transformers import AutoTokenizer
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
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size // 2)  # other half is reserved for spatial image embeddings
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

        ## inspecting token type embeddings
        e0 = self.token_type_embeddings.weight[0:1]
        e1 = self.token_type_embeddings.weight[1:2]
        e2 = self.token_type_embeddings.weight[2:3]
        e3 = self.token_type_embeddings.weight[3:4]
        e4 = self.token_type_embeddings.weight[4:5]
        e5 = self.token_type_embeddings.weight[5:6]
        e6 = self.token_type_embeddings.weight[6:7]
        e7 = self.token_type_embeddings.weight[7:8]
        e8 = self.token_type_embeddings.weight[8:9]



        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = torch.cat((self.img_pos_embeddings.to(position_embeddings.device), position_embeddings),
                                            dim=-1)  # add image position embeddings and sequence position embeddings
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

        
        self.fusion_config = BertConfig(vocab_size=1, hidden_size=args.hidden_size, num_hidden_layers=args.n_layers,
                                        num_attention_heads=args.heads, intermediate_size=args.hidden_size * 4,
                                        max_position_embeddings=args.max_position_embeddings)

        # self.alignment_image = nn.Sequential(
        #     nn.Linear(args.hidden_size, args.hidden_size),
        #     nn.ReLU()
        # )

        # self.alignment_text = nn.Sequential(
        #     nn.Linear(args.hidden_size, args.hidden_size),
        #     nn.ReLU()
        # )
        self.fusion = MyBertModel(config=self.fusion_config, args=args)

        if "vqarad" in args.data_dir:
            self.classifier = nn.Linear(args.hidden_size, args.num_classes)
        else:
            self.classifier = nn.Sequential(
                ## comment out when training / doing non overfit stuff
                ### Original - comment out next line during overfitting
                nn.Dropout(args.classifier_dropout),
                nn.Linear(args.hidden_size, 256),
                nn.ReLU(),
                # nn.BatchNorm1d(256),
                ### output is a single output
                nn.Linear(256, 1))

    ###################### CANDIDATE ######################## WORKING FORWARD PASS
    def forward(self, img_and_global, input_ids, q_attn_mask, attn_mask, token_type_ids_q=None, batch_metadata=None, mode='train'):
        
        
        if self.args.use_precomputed:
            image_features, global_embedding = img_and_global
            text_features = self.question_encoder(input_ids, q_attn_mask)
            cls_tokens = text_features[:, 0:1]
            
            kb_examples = self.knowledge_base.get_images_for_paths(batch_metadata)
            image_options_embedding_dict = self.produce_option_embeddings_precomputed(kb_examples=kb_examples)
            
            # options_embedding_text, col_embedding, seperator_embedding, pad_embedding = self.encode_batch_options_precomputed(batch_metadata)
            #pad_embedding = self.knowledge_base_processor._get_pad_embedding()
            seperator_embedding, pad_embedding = self.encode_batch_sep_pad_precomputed(batch_metadata)
            
            # knowledge_sequence, knowledge_ttid = self.generate_knowledge_sequence_manyyes_precomputed(global_img_embeddings=global_embedding, options_embeddings_image=image_options_embedding_dict,
            #                                                         options_embedding_text=options_embedding_text,
            #                                                         col_embedding=col_embedding,
            #                                                         separator_embedding=seperator_embedding,
            #                                                         batch_metadata=batch_metadata,
            #                                                         use_noise='no')

            batch_labels = self.generate_labels_bbc(global_img_embeddings=global_embedding, options_embeddings_image=image_options_embedding_dict,
                                                                    separator_embedding=seperator_embedding,
                                                                    batch_metadata=batch_metadata)                                                                                
            #print("hi")
            
        else:
            ## global image embedding serves as positive example in sanity check
            image_features, global_embedding = self.image_encoder(img_and_global, mode=mode)
            #image_features = image_features.detach()
            #global_image_embedding = global_image_embedding.detach()
            text_features = self.question_encoder(input_ids, q_attn_mask)
            cls_tokens = text_features[:, 0:1]

            # a list of dicts, where each list element represents a batch element, and each dict has options and paths
            kb_examples = self.knowledge_base.get_images_for_paths(batch_metadata)


            ##### need to change the representation of the MISSING_KNOWLEDGE embedding
            image_options_embedding_dict = self.produce_option_embeddings_batched(kb_examples=kb_examples, image_encoder=self.image_encoder)
            
            # ### convert to SANITY EMBEDDINGS used for sanitycheck 
            # for idx, batch in enumerate(batch_metadata):
            #     positive_options = set(batch['positive_option'])
            #     for option,tensor in image_options_embedding_dict[idx].items():
            #         if option in positive_options:
            #             image_options_embedding_dict[idx][option] = global_image_embedding[idx:idx+1]
            #             del tensor

            batch_labels = self.generate_labels_bbc(global_img_embeddings=global_embedding, options_embeddings_image=image_options_embedding_dict,
                                                                    separator_embedding=seperator_embedding,
                                                                    batch_metadata=batch_metadata)    

            
            options_embedding_text, col_embedding, seperator_embedding = self.knowledge_base_processor.encode_batch_options_batched(batch_metadata)
            pad_embedding = self.knowledge_base_processor._get_pad_embedding()
            
            # knowledge_sequence, knowledge_ttid = self.generate_knowledge_sequence_manyyes_precomputed(global_img_embeddings=global_embedding, options_embeddings_image=image_options_embedding_dict,
            #                                                         options_embedding_text=options_embedding_text,
            #                                                         col_embedding=col_embedding,
            #                                                         separator_embedding=seperator_embedding,
            #                                                         batch_metadata=batch_metadata,
            #                                                         use_noise='no')
            
            
            
                    
        ### CHECK THAT THE FOLLOWING IS LEGIT!
        h, ttids, attn_masks, gt_labels = self.generate_full_batch_sequence_bbc(cls_tokens, image_features, \
                                                global_embedding, image_options_embedding_dict, \
                                                text_features, seperator_embedding, \
                                                pad_embedding, token_type_ids_q, \
                                                batch_metadata, batch_labels, \
                                                attn_mask)

        # g_list = []
        # for lst in gt_labels:
        #     g_list.extend(lst)
        
        # g_list = g_list

        if self.args.progressive:
            assert token_type_ids_q is not None
            #token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            
            #ttid, attention_mask_modified = self.generate_tokentypeid_attnmask_forbatch_bbc(h, token_type_ids_q, image_features, attn_mask)
            
            
        else:
            token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            token_type_ids[:, 0] = 1
            token_type_ids[:, 1:image_features.size(1) + 1] = 0
            token_type_ids[:, image_features.size(1) + 1:] = 1


        #aligned = self.alignment(h)

        out = self.fusion(inputs_embeds=h, attention_mask=attn_masks, token_type_ids=ttids, output_attentions=True)
        h = out['last_hidden_state']
        attentions = out['attentions'][0]
        logits = self.classifier(h.mean(dim=1))
        
        ### clearing stuff kb_examples - list of dicts
        for kb_exam in kb_examples:
            for key, values in kb_exam.items():
                values.clear()
            kb_exam.clear()
        kb_examples.clear()
        
        
        for _ in image_options_embedding_dict:
            _.clear()
        image_options_embedding_dict.clear()
        
        ## uncomment if using options embed sanity
        # for _ in options_embedding_img_sanity:
        #     _.clear()
        # options_embedding_img_sanity.clear()
        
        # for _ in options_embedding_text:
        #     _.clear()
        # options_embedding_text.clear()

        del kb_examples, image_options_embedding_dict #, options_embedding_img_sanity
        

        if not torch.isfinite(logits).all():
            print(f'{timestamp()}Fusion model produced nan/inf in ints output')

        return logits, gt_labels, attentions
    

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
    
    ### TRASH - delete
    def combine_knowledge(prediction_tensors):
        # prediction_tensor is of shape (batch_size, n_tokens, 1, 768)
        # grab the tensor corresponding to the batch

        num_batches = prediction_tensors.size(0) # the first dimension is the batch dimesnion
        for batch_idx in range(num_batches):
            batch_tensor = prediction_tensors[batch_idx: batch_idx+1] # slices the tensor so that (num_tokebs, 1, 768) tensor is returned
            num_tokens_in_batch_tensor = batch_tensor.size(0)
    
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
       
    
    def produce_option_sanity_embeddings_fast_fix(self, kb_examples, global_embeddings, batch_metadata, verify=False):
        # Pre-allocate result list
        sanity_embeddings = [{} for _ in batch_metadata]
        detached_global_embeddings = global_embeddings.clone().detach()

        for idx, batch in enumerate(batch_metadata):
            pos_set = set(batch['positive_option'])  # Set lookup is faster
            #global_emb = detached_global_embeddings[idx:idx+1].clone().detach()  # Slice once per sample
            global_emb = detached_global_embeddings[idx:idx+1].clone() # no detach, already done

            ##dry run fix
            #kb_dict = kb_examples[idx]
            for option in batch['options']:
                if option in pos_set:
                    sanity_embeddings[idx][option] = global_emb
                else:
                    sanity_embeddings[idx][option] = kb_examples[idx][option]
            #kb_dict.clear()

        if verify:
            for idx, batch in enumerate(batch_metadata):
                pos_set = set(batch['positive_option'])
                for option in batch['options']:
                    sanity = sanity_embeddings[idx][option]
                    original = kb_examples[idx][option]
                    global_emb = detached_global_embeddings[idx:idx+1]
                    if option in pos_set:
                        assert not torch.equal(sanity, original), \
                            f"Positive option '{option}' should not match original tensor."
                        assert torch.equal(sanity, global_emb), \
                            f"Positive option '{option}' does not match global embedding."
                    else:
                        assert torch.equal(sanity, original), \
                            f"Negative option '{option}' should remain unchanged."

        return sanity_embeddings
    
    def generate_knowledge_sequence_fix(self, options_embeddings_image, options_embedding_text, col_embedding, separator_embedding, batch_metadata, use_noise='no'):
        used_device = next(self.image_encoder.parameters()).device
        
        assert len(options_embeddings_image) == len(options_embedding_text)
        
        # Move static embeddings to the correct device once
        col_embedding = col_embedding.to(device=used_device)
        separator_embedding = separator_embedding.to(device=used_device)
        
        ### move tensors to GPU
        for batch in range(len(options_embeddings_image)):
            img_dict = options_embeddings_image[batch]
            txt_dict = options_embedding_text[batch]
            for key in img_dict.keys():
                img_dict[key] = img_dict[key].to(device=used_device)
                txt_dict[key] = txt_dict[key].to(device=used_device)

        batch_knowledge_sequence = []
        batch_knowledge_ttid = []

        for batch_index in range(len(options_embedding_text)):
            img_options_embedding = options_embeddings_image[batch_index]
            text_option_embedding = options_embedding_text[batch_index]
            
            assert img_options_embedding.keys() == text_option_embedding.keys(), \
                "Mismatched keys between image and text embeddings."


            total_seq_len = 0
            for option in batch_metadata[batch_index]['options']:
                img_emb = img_options_embedding[option]
                text_emb = text_option_embedding[option]       # (num_tokens x 1 x 768)


                ### replace embeddings with 0 / noise
                if use_noise == 'yes':
                    img_emb = torch.zeros(img_emb.size(0), 1, 768).to(device=used_device)
                    text_emb =  torch.zeros(text_emb.size(0), 1, 768).to(device=used_device)
                
                
                total_seq_len += text_emb.size(0)
                total_seq_len += col_embedding.size(0)      # usually 1
                total_seq_len += img_emb.size(0)
                total_seq_len += separator_embedding.size(0) # usually 1
                

            ttid_tensor = torch.empty(total_seq_len, dtype=torch.long, device=used_device)

            pieces = []
            cursor = 0
            
            for option in batch_metadata[batch_index]['options']:
                text_emb = text_option_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True
                img_emb  = img_options_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True

                if use_noise == 'yes':
                    text_emb = torch.zeros_like(text_emb).detach()
                    img_emb  = torch.zeros_like(img_emb).detach()


                # ─(a) TEXT EMBEDDING──────────────────────────────────────────────────────────────
                pieces.append(text_emb)
                n_text = text_emb.size(0)
                # All text tokens share token‐type ID “5” (for example). Adjust if needed.
                ttid_tensor[cursor : cursor + n_text] = 5
                cursor += n_text
                # ─(b) COLUMN EMBEDDING────────────────────────────────────────────────────────────
                pieces.append(col_embedding)
                n_col = col_embedding.size(0)  # usually 1
                # We give col_embedding the SAME ttid as text (5), or pick your convention.
                ttid_tensor[cursor : cursor + n_col] = 5
                cursor += n_col
                # ─(c) IMAGE EMBEDDING─────────────────────────────────────────────────────────────
                pieces.append(img_emb)
                n_img = img_emb.size(0)
                # All image tokens share token‐type ID “6,” for instance.
                ttid_tensor[cursor : cursor + n_img] = 6
                cursor += n_img
                # ─(d) SEPARATOR EMBEDDING────────────────────────────────────────────────────────
                pieces.append(separator_embedding)
                n_sep = separator_embedding.size(0)  # usually 1
                ttid_tensor[cursor : cursor + n_sep] = 6
                cursor += n_sep
                
            assert cursor == total_seq_len, f"Cursor mismatch: {cursor} vs {total_seq_len}"
            
            

            # ─── (4C) Concatenate all pieces into one long tensor ─────────────────────────────────
            # Because `text_emb` and `img_emb` were not detached, they carry requires_grad=True,
            # so the resulting `batch_tensor` automatically becomes part of the autograd graph.
            batch_tensor = torch.cat(pieces, dim=0)  # shape: (total_seq_len × 1 × 768)
            # At this point: batch_tensor.requires_grad == True
            # The `.grad_fn` on batch_tensor will point back into each text_emb and img_emb.

            # ─── (4D) Append to the outputs ────────────────────────────────────────────────────────
            

            
            batch_knowledge_sequence.append(batch_tensor)
            batch_knowledge_ttid.append(ttid_tensor)
            
            pieces.clear()
            del pieces

        return batch_knowledge_sequence, batch_knowledge_ttid
    
    def generate_knowledge_sequence_vanilla_precomputed(self, options_embeddings_image, options_embedding_text, col_embedding, separator_embedding, batch_metadata, use_noise='no'):
        used_device = next(self.image_encoder.parameters()).device
        
        assert len(options_embeddings_image) == len(options_embedding_text)
        #sep_token_embedding = self.tokenizer.encode(self.tokenizer.sep_token_id)
        # Move static embeddings to the correct device once
        col_embedding = col_embedding.to(device=used_device)
        separator_embedding = separator_embedding.to(device=used_device)
        #global_embeddings_image = global_embeddings_image.to(used_device=used_device)
        
        ### move tensors to GPU
        for batch in range(len(options_embeddings_image)):
            img_dict = options_embeddings_image[batch]
            txt_dict = options_embedding_text[batch]
            for key in img_dict.keys():
                img_dict[key] = img_dict[key].to(device=used_device)
                txt_dict[key] = txt_dict[key].to(device=used_device)

        batch_knowledge_sequence = []
        batch_knowledge_ttid = []

        for batch_index in range(len(options_embedding_text)):
            img_options_embedding = options_embeddings_image[batch_index]
            text_option_embedding = options_embedding_text[batch_index]
            
            assert img_options_embedding.keys() == text_option_embedding.keys(), \
                "Mismatched keys between image and text embeddings."


            total_seq_len = 0
            for option in batch_metadata[batch_index]['options']:
                img_emb = img_options_embedding[option]
                text_emb = text_option_embedding[option]       # (num_tokens x 1 x 768)


                ### replace embeddings with 0 / noise
                if use_noise == 'yes':
                    img_emb = torch.zeros(img_emb.size(0), 1, 768).to(device=used_device)
                    text_emb =  torch.zeros(text_emb.size(0), 1, 768).to(device=used_device)
                
                
                total_seq_len += text_emb.size(0)
                total_seq_len += col_embedding.size(0)      # usually 1
                total_seq_len += 1 if img_emb.size(0) > 1 else img_emb.size(0) ### for l1_yes questions, we'll max later, so still only 1 token
                total_seq_len += separator_embedding.size(0) # usually 1
                

            ttid_tensor = torch.empty(total_seq_len, dtype=torch.long, device=used_device)

            pieces = []
            cursor = 0
            
            for option in batch_metadata[batch_index]['options']:
                img_emb  = img_options_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True
                text_emb = text_option_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True

                ### in vanilla mode we have yes:1x1768 img embedding
                if img_emb.size(0) > 1:
                    img_emb, _ = torch.max(img_emb, dim=0, keepdim=True)

                if use_noise == 'yes':
                    text_emb = torch.zeros_like(text_emb).detach()
                    img_emb  = torch.zeros_like(img_emb).detach()


                # ─(a) TEXT EMBEDDING──────────────────────────────────────────────────────────────
                pieces.append(text_emb)
                n_text = text_emb.size(0)
                # All text tokens share token‐type ID “5” (for example). Adjust if needed.
                ttid_tensor[cursor : cursor + n_text] = 5
                cursor += n_text
                # ─(b) COLUMN EMBEDDING────────────────────────────────────────────────────────────
                pieces.append(col_embedding)
                n_col = col_embedding.size(0)  # usually 1
                # We give col_embedding the SAME ttid as text (5), or pick your convention.
                ttid_tensor[cursor : cursor + n_col] = 5
                cursor += n_col
                # ─(c) IMAGE EMBEDDING─────────────────────────────────────────────────────────────
                pieces.append(img_emb)
                n_img = img_emb.size(0)
                # All image tokens share token‐type ID “6,” for instance.
                ttid_tensor[cursor : cursor + n_img] = 6
                cursor += n_img
                # ─(d) SEPARATOR EMBEDDING────────────────────────────────────────────────────────
                pieces.append(separator_embedding)
                n_sep = separator_embedding.size(0)  # usually 1
                ttid_tensor[cursor : cursor + n_sep] = 6
                cursor += n_sep
                
            assert cursor == total_seq_len, f"Cursor mismatch: {cursor} vs {total_seq_len}"
            
            

            # ─── (4C) Concatenate all pieces into one long tensor ─────────────────────────────────
            # Because `text_emb` and `img_emb` were not detached, they carry requires_grad=True,
            # so the resulting `batch_tensor` automatically becomes part of the autograd graph.
            batch_tensor = torch.cat(pieces, dim=0)  # shape: (total_seq_len × 1 × 768)
            # At this point: batch_tensor.requires_grad == True
            # The `.grad_fn` on batch_tensor will point back into each text_emb and img_emb.

            # ─── (4D) Append to the outputs ────────────────────────────────────────────────────────
            

            
            batch_knowledge_sequence.append(batch_tensor)
            batch_knowledge_ttid.append(ttid_tensor)
            
            pieces.clear()
            del pieces

        return batch_knowledge_sequence, batch_knowledge_ttid
    
    
    def generate_knowledge_sequence_bigyes_precomputed(self, options_embeddings_image, options_embedding_text, col_embedding, separator_embedding, batch_metadata, use_noise='no'):
        used_device = next(self.image_encoder.parameters()).device
        
        assert len(options_embeddings_image) == len(options_embedding_text)
        #sep_token_embedding = self.tokenizer.encode(self.tokenizer.sep_token_id)
        # Move static embeddings to the correct device once
        col_embedding = col_embedding.to(device=used_device)
        separator_embedding = separator_embedding.to(device=used_device)
        #global_embeddings_image = global_embeddings_image.to(used_device=used_device)
        
        ### move tensors to GPU
        for batch in range(len(options_embeddings_image)):
            img_dict = options_embeddings_image[batch]
            txt_dict = options_embedding_text[batch]
            for key in img_dict.keys():
                img_dict[key] = img_dict[key].to(device=used_device)
                txt_dict[key] = txt_dict[key].to(device=used_device)

        batch_knowledge_sequence = []
        batch_knowledge_ttid = []

        for batch_index in range(len(options_embedding_text)):
            img_options_embedding = options_embeddings_image[batch_index]
            text_option_embedding = options_embedding_text[batch_index]
            
            assert img_options_embedding.keys() == text_option_embedding.keys(), \
                "Mismatched keys between image and text embeddings."


            total_seq_len = 0
            for option in batch_metadata[batch_index]['options']:
                img_emb = img_options_embedding[option]
                text_emb = text_option_embedding[option]       # (num_tokens x 1 x 768)


                ### replace embeddings with 0 / noise
                if use_noise == 'yes':
                    img_emb = torch.zeros(img_emb.size(0), 1, 768).to(device=used_device)
                    text_emb =  torch.zeros(text_emb.size(0), 1, 768).to(device=used_device)
                
                # vanilla
                # total_seq_len += text_emb.size(0)
                # total_seq_len += col_embedding.size(0)      # usually 1
                # total_seq_len += 1 if img_emb.size(0) > 1 else img_emb.size(0) ### for l1_yes questions, we'll max later, so still only 1 token
                # total_seq_len += separator_embedding.size(0) # usually 1
                
                # vanilla
                total_seq_len += text_emb.size(0)
                total_seq_len += col_embedding.size(0)      # usually 1
                total_seq_len += img_emb.size(0)  ### for l1_yes questions, we'll max later, so still only 1 token
                total_seq_len += separator_embedding.size(0) # usually 1
                

            ttid_tensor = torch.empty(total_seq_len, dtype=torch.long, device=used_device)

            pieces = []
            cursor = 0
            
            for option in batch_metadata[batch_index]['options']:
                img_emb  = img_options_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True
                text_emb = text_option_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True

                ### in vanilla mode we have yes:1x1768 img embedding
                # if img_emb.size(0) > 1:
                #     img_emb, _ = torch.max(img_emb, dim=0, keepdim=True)

                if use_noise == 'yes':
                    text_emb = torch.zeros_like(text_emb).detach()
                    img_emb  = torch.zeros_like(img_emb).detach()


                # ─(a) TEXT EMBEDDING──────────────────────────────────────────────────────────────
                pieces.append(text_emb)
                n_text = text_emb.size(0)
                # All text tokens share token‐type ID “5” (for example). Adjust if needed.
                ttid_tensor[cursor : cursor + n_text] = 5
                cursor += n_text
                # ─(b) COLUMN EMBEDDING────────────────────────────────────────────────────────────
                pieces.append(col_embedding)
                n_col = col_embedding.size(0)  # usually 1
                # We give col_embedding the SAME ttid as text (5), or pick your convention.
                ttid_tensor[cursor : cursor + n_col] = 5
                cursor += n_col
                # ─(c) IMAGE EMBEDDING─────────────────────────────────────────────────────────────
                if option == 'yes' and img_emb.size(0) > 1:
                    for gl_embed_idx in range(img_emb.size(0)):
                        pieces.append(img_emb[gl_embed_idx : gl_embed_idx + 1])
                    n_img = img_emb.size(0)
                    # All image tokens share token‐type ID “6,” for instance.
                    ttid_tensor[cursor : cursor + n_img] = 6
                    cursor += n_img
                else:
                    pieces.append(img_emb)
                    n_img = img_emb.size(0)
                    # All image tokens share token‐type ID “6,” for instance.
                    ttid_tensor[cursor : cursor + n_img] = 6
                    cursor += n_img
                # ─(d) SEPARATOR EMBEDDING────────────────────────────────────────────────────────
                pieces.append(separator_embedding)
                n_sep = separator_embedding.size(0)  # usually 1
                ttid_tensor[cursor : cursor + n_sep] = 6
                cursor += n_sep
                
            assert cursor == total_seq_len, f"Cursor mismatch: {cursor} vs {total_seq_len}"
            
            

            # ─── (4C) Concatenate all pieces into one long tensor ─────────────────────────────────
            # Because `text_emb` and `img_emb` were not detached, they carry requires_grad=True,
            # so the resulting `batch_tensor` automatically becomes part of the autograd graph.
            batch_tensor = torch.cat(pieces, dim=0)  # shape: (total_seq_len × 1 × 768)
            # At this point: batch_tensor.requires_grad == True
            # The `.grad_fn` on batch_tensor will point back into each text_emb and img_emb.

            # ─── (4D) Append to the outputs ────────────────────────────────────────────────────────
            

            
            batch_knowledge_sequence.append(batch_tensor)
            batch_knowledge_ttid.append(ttid_tensor)
            
            pieces.clear()
            del pieces

        return batch_knowledge_sequence, batch_knowledge_ttid
    
    def generate_labels_bbc(self, global_img_embeddings, options_embeddings_image, separator_embedding, batch_metadata):
        used_device = next(self.image_encoder.parameters()).device
        batch_labels = []
        #sep_token_embedding = self.tokenizer.encode(self.tokenizer.sep_token_id)
        # Move static embeddings to the correct device once
        for batch_index in range(len(options_embeddings_image)):
            batch_labels.append([])
            img_options_embedding = options_embeddings_image[batch_index]
            
            for option in batch_metadata[batch_index]['options']:
                img_emb = img_options_embedding[option]
                
                # many yes - L1 yes question - produces 
                if option == 'yes' and img_emb.size(0) > 1:
                    for global_embed_option in range(img_emb.size(0)):
                        if option in batch_metadata[batch_index]['positive_option']:
                            batch_labels[batch_index].append(1)
                        else:
                            batch_labels[batch_index].append(0)
                ## non L1 uestion
                else:
                    if option in batch_metadata[batch_index]['positive_option']:
                        batch_labels[batch_index].append(1)
                    else:
                        batch_labels[batch_index].append(0)
        return batch_labels
    
    def generate_full_batch_sequence_bbc(self, cls_tokens, image_features, \
                                        global_embeddings, image_options_g_embeddings, \
                                        text_features, seperator_embedding, \
                                        pad_embedding, token_type_ids_q, \
                                        batch_metadata, batch_labels, \
                                        attn_mask):
        """
        This version pre-allocates a single tensor of shape
        (batch_size, max_position_embeddings, 768)
        and then fills in each row [i, :] by copying the
        sub-segments (cls_tok, img_feat, know_feat, txt_feat)
        directly into that buffer, up to max_position_embeddings.
        """
        ## by the end I want to have a sequence for each sample in the batch
        ## need to go through each batch element -> from it create N batch samples -> for each batch sample create its sequence
        ## what are the common and repeated parts, what is unique
        ### 

        batch_size = cls_tokens.size(0)
        hidden_dim_size = cls_tokens.size(-1)  # typically 768
        sequence_length = self.args.max_position_embeddings

        # 1) Create the output buffer all at once. By default, it's zero-initialized.
        #    We match the dtype/device of cls_tokens for safety.
        device = cls_tokens.device
        dtype = cls_tokens.dtype
        h = cls_tokens.new_zeros((sum(len(batch) for batch in batch_labels), sequence_length, hidden_dim_size), dtype=dtype, device=device)
        ttids = token_type_ids_q.new_zeros((sum(len(batch) for batch in batch_labels), sequence_length), dtype=token_type_ids_q.dtype, device=device)
        attn_masks_modified = attn_mask.new_zeros((sum(len(batch) for batch in batch_labels), sequence_length), dtype=attn_mask.dtype, device=device)

        gt_for_batch = []

        ## go through each batch
        h_write_pos = 0
        for batch_idx in range(batch_size):
            gt_for_batch.append([])

            ### things that should be common for each new batch element
            cls_tokens_for_batch = cls_tokens[batch_idx:batch_idx+1]
            image_tokens_for_batch = image_features[batch_idx:batch_idx+1]
            #num_qa_tokens_for_batch = sum(token_type_ids_q[batch_idx]).item()
            #text_tokens_for_batch = text_features[batch_size : batch_size+1, 1:num_qa_tokens]
            global_image_embedding = global_embeddings[batch_idx: batch_idx+1]

            kb_example_embeddings = image_options_g_embeddings[batch_idx] # dict containing each option for the batch
            ttids_for_batch = token_type_ids_q[batch_idx]
            attn_mask_for_batch = attn_mask[batch_idx]


            for option in batch_metadata[batch_idx]['options']:
                # L1 questio
                
                if option == 'yes' and kb_example_embeddings[option].size(0) > 1:
                    # global embeddings for each l1 yes option
                    for l1_option in range(kb_example_embeddings[option].size(0)):

                        write_pos = 0
                        ### CLS token
                        h[h_write_pos, write_pos : write_pos+1, :] = cls_tokens_for_batch
                        h_inspect = h[h_write_pos]
                        ### CLS token - attn mask
                        ttids[h_write_pos, write_pos : write_pos + 1] = 1
                        attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1

                        write_pos += 1


                        ### Path Image Features
                        img_feat_len = image_tokens_for_batch.size(1)
                        # How many image tokens can we still fit?
                        remain = sequence_length - write_pos
                        copy_len = min(img_feat_len, remain)
                        if copy_len > 0:
                            # We slice both sides:
                            #   - source: img_feat_i[:, :copy_len, :] → (1, copy_len, D)
                            #   - destination: h[i, write_pos : write_pos+copy_len, :]
                            h[h_write_pos, write_pos : write_pos+copy_len, :] = image_tokens_for_batch[:, :copy_len, :]
                            h_inspect = h[h_write_pos]
                            ### Patch Image Features - ttid + attn_mask
                            ttids[h_write_pos, write_pos : write_pos+copy_len] = 4
                            attn_masks_modified[h_write_pos, write_pos : write_pos+copy_len] = 1

                            write_pos += copy_len


                        # SEP embedding
                        h[h_write_pos, write_pos : write_pos+1, :] = seperator_embedding
                        h_inspect = h[h_write_pos]
                        ### SEP embedding - ttid + attn_mask
                        ttids[h_write_pos, write_pos : write_pos + 1] = 4
                        attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1                        
                        
                        write_pos += 1



                        ### Global input image embedding
                        h[h_write_pos, write_pos : write_pos+1, :] = global_image_embedding
                        h_inspect = h[h_write_pos]
                        ### Global input image embedding - ttid + attn_mask
                        ttids[h_write_pos, write_pos : write_pos + 1] = 6
                        attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1

                        write_pos += 1



                        # SEP embedding
                        h[h_write_pos, write_pos : write_pos+1, :] = seperator_embedding
                        h_inspect = h[h_write_pos]
                        ### Global input image embedding - ttid + attn_mask
                        ttids[h_write_pos, write_pos : write_pos + 1] = 6
                        attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1
                        
                        write_pos += 1



                        ## Knowledge image embedding
                        knowledge_global_embedding = kb_example_embeddings[option][l1_option: l1_option+1]
                        know_len = knowledge_global_embedding.size(1)
                        remain = sequence_length - write_pos
                        copy_len = min(know_len, remain)
                        if copy_len > 0:
                            h[h_write_pos, write_pos : write_pos+copy_len, :] = knowledge_global_embedding
                            h_inspect = h[h_write_pos]
                            ### Knowledge image embedding - ttid + attn_mask
                            ttids[h_write_pos, write_pos : write_pos+copy_len] = 6
                            attn_masks_modified[h_write_pos, write_pos : write_pos+copy_len] = 1
                            
                            write_pos += copy_len

                        


                        ### Question features
                        # txt_feat_i = text_features[i : i+1, 1:num_qa_tokens]  # (1, T-1, D)
                        # #inspct_txt_feat = text_features[i]
                        # #inspct_txt_feat2 = text_features[i : i + 1, 1:num_qa_tokens]
                        # #inspct_txt_feat3 = text_features[i : i + 1, 1:num_qa_tokens, 0:5]
                        # txt_len = txt_feat_i.size(1)
                        # remain = sequence_length - write_pos
                        # copy_len = min(txt_len, remain)
                        # if copy_len > 0:
                        #     h[i, write_pos : write_pos+copy_len, :] = txt_feat_i[:, :copy_len, :]
                        #     #h_inspect = h[i]
                        #     write_pos += copy_len
                        
                        ### pad the rest of the sequence
                        remain = sequence_length - write_pos
                        if remain > 0:
                            h[h_write_pos, write_pos : write_pos+remain, :] = pad_embedding
                            h_inspect = h[h_write_pos]
                            ### Pad - ttid + attn_mask
                            ttids[h_write_pos, write_pos : write_pos+remain] = 0
                            attn_masks_modified[h_write_pos, write_pos : write_pos+remain] = 0
                            
                            write_pos += remain



                        h_write_pos += 1

                        if option in batch_metadata[batch_idx]['positive_option']:
                            gt_for_batch[batch_idx].append(1)
                        else:
                            gt_for_batch[batch_idx].append(0)


      
                # L2 + L2 questions
                else:
                    write_pos = 0
                    ### CLS token
                    h[h_write_pos, write_pos : write_pos+1, :] = cls_tokens_for_batch
                    h_inspect = h[h_write_pos]
                    ### CLS token - attn mask
                    ttids[h_write_pos, write_pos : write_pos + 1] = 1
                    attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1
                    
                    write_pos += 1

                    ### Path Image Features
                    img_feat_len = image_tokens_for_batch.size(1)
                    # How many image tokens can we still fit?
                    remain = sequence_length - write_pos
                    copy_len = min(img_feat_len, remain)
                    if copy_len > 0:
                        # We slice both sides:
                        #   - source: img_feat_i[:, :copy_len, :] → (1, copy_len, D)
                        #   - destination: h[i, write_pos : write_pos+copy_len, :]
                        h[h_write_pos, write_pos : write_pos+copy_len, :] = image_tokens_for_batch[:, :copy_len, :]
                        h_inspect = h[h_write_pos]
                        ### Patch Image Features - ttid + attn_mask
                        ttids[h_write_pos, write_pos : write_pos+copy_len] = 4
                        attn_masks_modified[h_write_pos, write_pos : write_pos+copy_len] = 1
                        
                        write_pos += copy_len

                    # SEP embedding
                    h[h_write_pos, write_pos : write_pos+1, :] = seperator_embedding
                    h_inspect = h[h_write_pos]
                    ### SEP embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos : write_pos + 1] = 4
                    attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1                          
                    
                    write_pos += 1

                    ### Global input image embedding
                    h[h_write_pos, write_pos : write_pos+1, :] = global_image_embedding
                    h_inspect = h[h_write_pos]
                    ### Global input image embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos : write_pos + 1] = 6
                    attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1                    
                    
                    write_pos += 1

                    # SEP embedding
                    h[h_write_pos, write_pos : write_pos+1, :] = seperator_embedding
                    h_inspect = h[h_write_pos]
                    ### SEP embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos : write_pos + 1] = 6
                    attn_masks_modified[h_write_pos, write_pos : write_pos + 1] = 1  
                    write_pos += 1

                    ## Knowledge image embedding
                    knowledge_global_embedding = kb_example_embeddings[option]
                    know_len = knowledge_global_embedding.size(1)
                    remain = sequence_length - write_pos
                    copy_len = min(know_len, remain)
                    if copy_len > 0:
                        h[h_write_pos, write_pos : write_pos+copy_len, :] = knowledge_global_embedding
                        h_inspect = h[h_write_pos]
                        ### Knowledge image embedding - ttid + attn_mask
                        ttids[h_write_pos, write_pos : write_pos+copy_len] = 6
                        attn_masks_modified[h_write_pos, write_pos : write_pos+copy_len] = 1                        
                        
                        write_pos += copy_len

                    
                    ### Question features
                    # txt_feat_i = text_features[i : i+1, 1:num_qa_tokens]  # (1, T-1, D)
                    # #inspct_txt_feat = text_features[i]
                    # #inspct_txt_feat2 = text_features[i : i + 1, 1:num_qa_tokens]
                    # #inspct_txt_feat3 = text_features[i : i + 1, 1:num_qa_tokens, 0:5]
                    # txt_len = txt_feat_i.size(1)
                    # remain = sequence_length - write_pos
                    # copy_len = min(txt_len, remain)
                    # if copy_len > 0:
                    #     h[i, write_pos : write_pos+copy_len, :] = txt_feat_i[:, :copy_len, :]
                    #     #h_inspect = h[i]
                    #     write_pos += copy_len
                    
                    ### pad the rest of the sequence
                    remain = sequence_length - write_pos
                    if remain > 0:
                        h[h_write_pos, write_pos : write_pos+remain, :] = pad_embedding
                        h_inspect = h[h_write_pos]
                        ### Pad - ttid + attn_mask
                        ttids[h_write_pos, write_pos : write_pos+remain] = 0
                        attn_masks_modified[h_write_pos, write_pos : write_pos+remain] = 0

                        write_pos += remain

                    h_write_pos += 1

                    if option in batch_metadata[batch_idx]['positive_option']:
                        gt_for_batch[batch_idx].append(1)
                    else:
                        gt_for_batch[batch_idx].append(0)



        # gt_labels is a list of lists or similar iterable
        # Example: gt_labels = [[0, 1], [2, 3, 4], [5]]
        # or gt_labels = torch.tensor([[0, 1], [2, 3, 4], [5]]) if already a tensor
        gt_for_batch = torch.cat([torch.tensor(lst, dtype=torch.float32) if not isinstance(lst, torch.Tensor) else lst.float() for lst in gt_for_batch]).unsqueeze(1)
        return h, ttids, attn_masks_modified, gt_for_batch
        
    
    def generate_tokentypeid_attnmask_forbatch_bbc(
    self,
    sequence,            # [batch_size, seq_len]
    token_type_ids_q,    # [batch_size, seq_len]  (we’ll slice out of this)
    image_features,      # [batch_size, num_image_feats, ...]  (we only need num_image_feats)
    attn_mask,           # [batch_size, seq_len], dtype long with 0/1
):
        """
        Returns:
        token_type_ids:        [batch_size, seq_len] with the desired marks
        modified_attention_mask: [batch_size, seq_len] (0/1) extended by positive-KB length
        """

        device = sequence.device
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        num_image_feats = image_features.size(1)   # e.g. 196

        # ---------------------------------------------------
        # 1) PRE-ALLOCATE or RE-USE buffers instead of torch.zeros(...) every time.
        #    Here we store them as attributes so future calls will re-use the same memory:
        if not hasattr(self, "_ttids_buffer") or self._ttids_buffer.size() != (batch_size, seq_len):
            # If it doesn’t exist yet, allocate it once:
            self._ttids_buffer = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )
            self._mask_buffer = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )

        token_type_ids = self._ttids_buffer
        modified_attention_mask = self._mask_buffer

        # Zero them out in-place (no new allocation):
        token_type_ids.fill_(0)
        modified_attention_mask.fill_(0)

        # ---------------------------------------------------
        # 2) Convert knowledge_ttid (list of Python lists or numpy arrays) into
        #    a list of LongTensors on the correct device, ONCE.
        #    If they’re already tensors on the right device, skip this step.
        knowledge_ttid_tensors = []
        for idx in range(batch_size):
            t = knowledge_ttid[idx]
            if isinstance(t, torch.Tensor):
                # Make sure it’s long & on the right device
                knowledge_ttid_tensors.append(t.long().to(device))
            else:
                # If it’s a Python list or numpy array, convert once:
                knowledge_ttid_tensors.append(torch.LongTensor(t).to(device))

        # ---------------------------------------------------
        # 3) Fill in the “CLS token” (index 0) and “image features” (indices 1..num_image_feats)
        #    in *one* vectorized step across all examples:
        #
        #   token_type_ids[:, 0] = 1       # CLS
        #   token_type_ids[:, 1 : 1 + num_image_feats] = 4  # image-features
        #
        #   These two lines allocate no new memory; they simply write into our pre-allocated buffer.
        token_type_ids[:, 0] = 1 # CLS token
        token_type_ids[:, 1 : 1 + num_image_feats] = 4 # image patch features
        token_type_ids[:, 1 + num_image_feats : 1 + 1 + num_image_feats] = 4 ######## sep token
        token_type_ids[:, 1 + 1 + num_image_feats :  1 + 1 + 1 + num_image_feats] = 6 ## global image embedding
        token_type_ids[:, 1 + 1 + 1 + num_image_feats :  1 + 1 + 1 + 1 + num_image_feats] = 6 ## sep token

        # [CLS][IMAGE_PATCH_FEATURES][SEP][IMAGEGLOBAL_EMBEDDING][SEP][KNOWLEDGE-includes sep at end][Q/A]

        # ---------------------------------------------------
        # 4) Handle variable-length “knowledge” slices.
        #    We must do a short Python loop, because each example i might have a different K_i length.
        #
        #    For each example idx:
        #       start_i = 1 + num_image_feats
        #       end_i   = start_i + K_i
        #       token_type_ids[idx, start_i : end_i] = knowledge_ttid_tensors[idx]
        #       Then text_start_i = end_i, and we copy from token_type_ids_q[idx, 1 : 1 + (seq_len - end_i)]
        #       into token_type_ids[idx, end_i : ]
        #
        # We also build a small Python list of knowledge_lengths, so we can do the new attention-mask vectorization later.
        knowledge_lengths = [ks.size(0) for ks in knowledge_sequence]
        #knowledge_lengths = [k_len + 3 for k_len in knowledge_lengths] # 3 extra tokens (sep, global_embed, sep)
        for idx in range(batch_size):
            Ki = knowledge_lengths[idx]
            start_i = 4 + num_image_feats ############# 3 (sep, global_embed, sep) because of sep token added at 1024
            end_i = start_i + Ki

            # 4.1) Assign the “positive KB example” token-type IDs from the pre-converted tensor:
            token_type_ids[idx, start_i : end_i] = knowledge_ttid_tensors[idx]

            # 4.2) Compute how many leftover “text positions” remain in this sequence:
            #      We assume `self.args.max_position_embeddings` equals seq_len here.
            #      So `remaining_i = seq_len - end_i`.
            remaining_i = seq_len - end_i #### +1 becuase of sep
            if remaining_i > 0:
                # Copy from token_type_ids_q[idx, 1:1 + remaining_i] into token_type_ids[idx, end_i:]
                token_type_ids[idx, end_i : end_i + remaining_i] = token_type_ids_q[
                    idx, 1 : 1 + remaining_i
                ]

        # ---------------------------------------------------
        # 5) Build modified_attention_mask in a vectorized way.
        #    Original code did, for each idx:
        #        attn_mask_items = (attn_mask[idx] == 1).sum().item()
        #        modified_attention_mask[idx, 0 : attn_mask_items + Ki] = 1
        #
        #    Instead, we:
        #      (a) compute attn_mask_items_per_example = attn_mask.sum(dim=1) -> [batch_size]
        #      (b) compute thresholds = attn_mask_items_per_example + knowledge_lengths
        #          (convert knowledge_lengths list → a tensor on the same device)
        #      (c) build a row-vector `positions = torch.arange(seq_len, device=device)`
        #      (d) compare `positions.unsqueeze(0) < thresholds.unsqueeze(1)` → a [batch_size, seq_len] boolean mask
        #
        #    That single boolean mask covers all rows in one shot:
        with torch.no_grad():
            # (a) how many of the original tokens are “active” (==1) in attn_mask, per example:
            attn_counts = (attn_mask == 1).sum(dim=1)          # shape: [batch_size]

            # (b) add corresponding knowledge_lengths:
            #    convert knowledge_lengths list into a LongTensor on `device`
            knowledge_lengths = [k_len + 3 for k_len in knowledge_lengths] ################### + 3 because of the 3 extra tokens added to the sequence - sep , global embed, sep
            kl_tensor = torch.LongTensor(knowledge_lengths).to(device)
            thresholds = attn_counts + kl_tensor            # shape: [batch_size]

            # (c) build positions = [0, 1, 2, ..., seq_len-1] want shape [1, seq_len]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]

            # (d) each row i: mark 1 for positions < thresholds[i]
            #     This comparison returns a boolean Tensor [batch_size, seq_len]
            mask_bool = positions < thresholds.unsqueeze(1)  # [batch_size, seq_len]

            # Write it into our preallocated `modified_attention_mask` (long dtype 0/1):
            modified_attention_mask.copy_(mask_bool.long())
            
        knowledge_ttid_tensors.clear()

        return token_type_ids, modified_attention_mask
    

    def generate_tokentypeid_attnmask_forbatch_fix(
    self,
    sequence,            # [batch_size, seq_len]
    token_type_ids_q,    # [batch_size, seq_len]  (we’ll slice out of this)
    image_features,      # [batch_size, num_image_feats, ...]  (we only need num_image_feats)
    knowledge_sequence,  # list of length batch_size; each entry is a Tensor [K_i, ...]
    attn_mask,           # [batch_size, seq_len], dtype long with 0/1
    knowledge_ttid       # list of length batch_size; each is list or 1D array of length K_i
):
        """
        Returns:
        token_type_ids:        [batch_size, seq_len] with the desired marks
        modified_attention_mask: [batch_size, seq_len] (0/1) extended by positive-KB length
        """

        device = sequence.device
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        num_image_feats = image_features.size(1)   # e.g. 196

        # ---------------------------------------------------
        # 1) PRE-ALLOCATE or RE-USE buffers instead of torch.zeros(...) every time.
        #    Here we store them as attributes so future calls will re-use the same memory:
        if not hasattr(self, "_ttids_buffer") or self._ttids_buffer.size() != (batch_size, seq_len):
            # If it doesn’t exist yet, allocate it once:
            self._ttids_buffer = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )
            self._mask_buffer = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )

        token_type_ids = self._ttids_buffer
        modified_attention_mask = self._mask_buffer

        # Zero them out in-place (no new allocation):
        token_type_ids.fill_(0)
        modified_attention_mask.fill_(0)

        # ---------------------------------------------------
        # 2) Convert knowledge_ttid (list of Python lists or numpy arrays) into
        #    a list of LongTensors on the correct device, ONCE.
        #    If they’re already tensors on the right device, skip this step.
        knowledge_ttid_tensors = []
        for idx in range(batch_size):
            t = knowledge_ttid[idx]
            if isinstance(t, torch.Tensor):
                # Make sure it’s long & on the right device
                knowledge_ttid_tensors.append(t.long().to(device))
            else:
                # If it’s a Python list or numpy array, convert once:
                knowledge_ttid_tensors.append(torch.LongTensor(t).to(device))

        # ---------------------------------------------------
        # 3) Fill in the “CLS token” (index 0) and “image features” (indices 1..num_image_feats)
        #    in *one* vectorized step across all examples:
        #
        #   token_type_ids[:, 0] = 1       # CLS
        #   token_type_ids[:, 1 : 1 + num_image_feats] = 4  # image-features
        #
        #   These two lines allocate no new memory; they simply write into our pre-allocated buffer.
        token_type_ids[:, 0] = 1 # CLS token
        token_type_ids[:, 1 : 1 + num_image_feats] = 4 # image patch features
        token_type_ids[:, 1 + num_image_feats : 1 + 1 + num_image_feats] = 4 ######## sep token
        token_type_ids[:, 1 + 1 + num_image_feats :  1 + 1 + 1 + num_image_feats] = 6 ## global image embedding
        token_type_ids[:, 1 + 1 + 1 + num_image_feats :  1 + 1 + 1 + 1 + num_image_feats] = 6 ## sep token

        # [CLS][IMAGE_PATCH_FEATURES][SEP][IMAGEGLOBAL_EMBEDDING][SEP][KNOWLEDGE-includes sep at end][Q/A]

        # ---------------------------------------------------
        # 4) Handle variable-length “knowledge” slices.
        #    We must do a short Python loop, because each example i might have a different K_i length.
        #
        #    For each example idx:
        #       start_i = 1 + num_image_feats
        #       end_i   = start_i + K_i
        #       token_type_ids[idx, start_i : end_i] = knowledge_ttid_tensors[idx]
        #       Then text_start_i = end_i, and we copy from token_type_ids_q[idx, 1 : 1 + (seq_len - end_i)]
        #       into token_type_ids[idx, end_i : ]
        #
        # We also build a small Python list of knowledge_lengths, so we can do the new attention-mask vectorization later.
        knowledge_lengths = [ks.size(0) for ks in knowledge_sequence]
        #knowledge_lengths = [k_len + 3 for k_len in knowledge_lengths] # 3 extra tokens (sep, global_embed, sep)
        for idx in range(batch_size):
            Ki = knowledge_lengths[idx]
            start_i = 4 + num_image_feats ############# 3 (sep, global_embed, sep) because of sep token added at 1024
            end_i = start_i + Ki

            # 4.1) Assign the “positive KB example” token-type IDs from the pre-converted tensor:
            token_type_ids[idx, start_i : end_i] = knowledge_ttid_tensors[idx]

            # 4.2) Compute how many leftover “text positions” remain in this sequence:
            #      We assume `self.args.max_position_embeddings` equals seq_len here.
            #      So `remaining_i = seq_len - end_i`.
            remaining_i = seq_len - end_i #### +1 becuase of sep
            if remaining_i > 0:
                # Copy from token_type_ids_q[idx, 1:1 + remaining_i] into token_type_ids[idx, end_i:]
                token_type_ids[idx, end_i : end_i + remaining_i] = token_type_ids_q[
                    idx, 1 : 1 + remaining_i
                ]

        # ---------------------------------------------------
        # 5) Build modified_attention_mask in a vectorized way.
        #    Original code did, for each idx:
        #        attn_mask_items = (attn_mask[idx] == 1).sum().item()
        #        modified_attention_mask[idx, 0 : attn_mask_items + Ki] = 1
        #
        #    Instead, we:
        #      (a) compute attn_mask_items_per_example = attn_mask.sum(dim=1) -> [batch_size]
        #      (b) compute thresholds = attn_mask_items_per_example + knowledge_lengths
        #          (convert knowledge_lengths list → a tensor on the same device)
        #      (c) build a row-vector `positions = torch.arange(seq_len, device=device)`
        #      (d) compare `positions.unsqueeze(0) < thresholds.unsqueeze(1)` → a [batch_size, seq_len] boolean mask
        #
        #    That single boolean mask covers all rows in one shot:
        with torch.no_grad():
            # (a) how many of the original tokens are “active” (==1) in attn_mask, per example:
            attn_counts = (attn_mask == 1).sum(dim=1)          # shape: [batch_size]

            # (b) add corresponding knowledge_lengths:
            #    convert knowledge_lengths list into a LongTensor on `device`
            knowledge_lengths = [k_len + 3 for k_len in knowledge_lengths] ################### + 3 because of the 3 extra tokens added to the sequence - sep , global embed, sep
            kl_tensor = torch.LongTensor(knowledge_lengths).to(device)
            thresholds = attn_counts + kl_tensor            # shape: [batch_size]

            # (c) build positions = [0, 1, 2, ..., seq_len-1] want shape [1, seq_len]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]

            # (d) each row i: mark 1 for positions < thresholds[i]
            #     This comparison returns a boolean Tensor [batch_size, seq_len]
            mask_bool = positions < thresholds.unsqueeze(1)  # [batch_size, seq_len]

            # Write it into our preallocated `modified_attention_mask` (long dtype 0/1):
            modified_attention_mask.copy_(mask_bool.long())
            
        knowledge_ttid_tensors.clear()

        return token_type_ids, modified_attention_mask
    



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
        if self.args.use_precomputed:
            if "vqarad" in self.args.data_dir:
                (img, global_embed), question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
            else:
                (img, global_embed), question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch
        else:
            if "vqarad" in self.args.data_dir:
                img, question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
            else:
                img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch

        question_token = question_token.squeeze(1)
        attn_mask = attn_mask.squeeze(1)
        q_attention_mask = q_attention_mask.squeeze(1)

        if self.args.use_precomputed:
            out, gt_vectors, attentions = self((img, global_embed), question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')
        else:
            out, gt_vectors, attentions  = self(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')
            
        #print_mem(f"after {batch_idx}")
        
        
        logits = out
        gt_vectors = gt_vectors.to(device=logits.device).detach()
        target = gt_vectors

        if "vqarad" in self.args.data_dir:
            pred = logits.softmax(1).argmax(1).detach()
        else:  
            pred = (logits.sigmoid().detach() > 0.5).detach().long()

            self.train_soft_scores.append(logits.sigmoid().detach())

        self.train_preds.append(pred)
        self.train_targets.append(target)

        if "vqarad" in self.args.data_dir:
            self.train_answer_types.append(answer_type)
        else:
            self.train_infos.append(info)

        loss = self.loss_fn(logits, target) # N x 1 per class loss
        mean_loss = loss.mean() # a single scalar
        #don't need for bbc
        #loss = self.get_masked_loss(loss, mask, target, None)  # only use loss of occuring classes        

        self.log('Loss/train', mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return mean_loss

    ## NEW
    # def validation_step(self, batch, batch_idx):
    #     if self.args.use_precomputed:
    #         if "vqarad" in self.args.data_dir:
    #             (img, global_embedding), question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
    #         else:
    #             (img, global_embedding), question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch
    #     else:
    #         if "vqarad" in self.args.data_dir:
    #             img, question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
    #         else:
    #             img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch            

    #     question_token = question_token.squeeze(1)
    #     attn_mask = attn_mask.squeeze(1)
    #     q_attention_mask = q_attention_mask.squeeze(1)
    #     batch_info = batch[6]
        
    #     if self.args.use_precomputed:
    #         out, gt_vectors, attentions = self((img, global_embedding), question_token, q_attention_mask, attn_mask, token_type_ids_q, batch_info, mode='val')
    #     else:
    #         out, gt_vectors, attentions = self(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, batch_info, mode='val')

    #     logits = out
    #     gt_vectors = gt_vectors.to(device=logits.device).detach()
    #     target = gt_vectors

    #     if "vqarad" in self.args.data_dir:
    #         pred = logits.softmax(1).argmax(1).detach()
    #         self.val_soft_scores.append(logits.softmax(1).detach())
    #     else:  # multi-label classification
    #         pred = (logits.sigmoid().detach() > 0.5).detach().long()
    #         self.val_soft_scores.append(logits.sigmoid().detach())

    #     self.val_preds.append(pred)
    #     self.val_targets.append(target)
    #     if "vqarad" in self.args.data_dir:
    #         self.val_answer_types.append(answer_type)
    #     else:
    #         self.val_infos.append(info)

    #     if "vqarad" in self.args.data_dir:
    #         val_loss = self.loss_fn(logits[target != -1], target[target != -1])
    #     else:
    #         val_loss = self.loss_fn(logits, target)
    #         # only use loss of occuring classes --- not relevant for bbc
    #         # if "radrestruct" in self.args.data_dir:
    #         #     val_loss = self.get_masked_loss(val_loss, mask, target, None)
    #     mean_val_loss = val_loss.mean()
    #     self.log('Loss/val', mean_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #     return mean_val_loss

    ### Original
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer
    
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
    
    ### New
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
    #     #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    #     scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6),
    #         'monitor': 'Loss/train',  # or 'val_loss' if you're using validation
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
        # Concatenate predictions and targets from all batches
        preds = torch.cat(self.train_preds, dim=0)
        targets = torch.cat(self.train_targets, dim=0)
        
        # For binary classification, targets may be float, so ensure type match
        # If multiclass, ensure argmax etc. (already done in training_step)
        # If targets are one-hot or probabilities, you may need to convert

        if "vqarad" in self.args.data_dir:
            # Multiclass classification
            correct = (preds == targets).sum().item()
            total = targets.numel()
            accuracy = correct / total
            self.log('Accuracy/train', accuracy, on_epoch=True, prog_bar=True, logger=True)
            # Add more metrics here if needed (F1, etc.)
        else:
            # Binary classification (multi-label)
            # Often want to calculate metrics per label or overall
            # Example: simple accuracy (all labels correct)
            correct = (preds == targets).float().sum()
            total = torch.numel(targets)
            accuracy = correct / total
            self.log('Accuracy/train', accuracy, on_epoch=True, prog_bar=True, logger=True)
            # You might want to use sklearn.metrics for F1, precision, recall, etc.
            # with torch -> numpy conversion if needed

        # Optionally: log more detailed metrics (F1, confusion matrix, etc.)

        # Clear out your lists so they don't grow across epochs!
        self.train_preds = []
        self.train_targets = []
        if hasattr(self, "train_answer_types"):
            self.train_answer_types = []
        if hasattr(self, "train_infos"):
            self.train_infos = []
        if hasattr(self, "train_soft_scores"):
            self.train_soft_scores = []

        self.train_preds = []
        self.train_targets = []
        self.train_soft_scores = []
        self.train_infos = []
        self.train_answer_types = []
        self.train_gen_labels = []
        self.train_gen_preds = []

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
        # Concatenate all predictions and targets for the epoch
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        if "vqarad" in self.args.data_dir:
            correct = (preds == targets).sum().item()
            total = targets.numel()
            accuracy = correct / total
            self.log('Accuracy/val', accuracy, on_epoch=True, prog_bar=True, logger=True)
        else:
            correct = (preds == targets).float().sum()
            total = torch.numel(targets)
            accuracy = correct / total
            self.log('Accuracy/val', accuracy, on_epoch=True, prog_bar=True, logger=True)

        # Optionally compute F1 score for both cases
        try:
            from sklearn.metrics import f1_score
            f1 = f1_score(targets.cpu().numpy().reshape(-1), preds.cpu().numpy().reshape(-1), average='macro')
            self.log('F1/val', f1, on_epoch=True, prog_bar=True, logger=True)
        except ImportError:
            pass  # sklearn not available, just skip

        # Reset lists for next epoch
        self.val_preds = []
        self.val_targets = []
        if hasattr(self, "val_soft_scores"):
            self.val_soft_scores = []
        if hasattr(self, "val_answer_types"):
            self.val_answer_types = []
        if hasattr(self, "val_infos"):
            self.val_infos = []

        self.val_preds = []
        self.val_targets = []
        self.val_soft_scores = []
        self.val_answer_types = []
        self.val_gen_labels = []
        self.val_gen_preds = []
        self.val_infos = []



