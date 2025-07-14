

from datetime import datetime
import random
from typing import Dict, List
import torch

from knowledge_base.constants import Constants, Mode

CONSTANTS = Constants(Mode.CLUSTER)

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"


### training the bbc using NO composite imates for L1 questions, instead - matching to each individual l2 image possible
### frozen image encoder, text encoder, using BCE, precomputed KB + input images and g embedding of input images


# python -m net_bbc.train_bbc \
#     --run_name "variant_bbc_noncomposite_sequence" \
#     --data_dir /home/guests/adrian_delchev/code/Rad-ReStruct/data/radrestruct \
#     --classifier_dropout 0.2 \
#     --num_workers 0 \
#     --lr 1e-5 \
#     --epochs 30 \
#     --batch_size 4 \
#     --acc_grad_batches 4 \
#     --progressive \
#     --mixed_precision \
#     --kb_dir /home/guests/adrian_delchev/code/ad_Rad-ReStruct/knowledge_base/tiny_KB.json \
#     --model_dir /home/guests/adrian_delchev/code/ad_Rad-ReStruct-Original/checkpoints_radrestruct/train_rrs_wdb_b4/epoch=20-F1/val=0.32.ckpt \
#     --use_precomputed \
#     --freeze_image_encoder \
#     --freeze_question_encoder 

def forward(model, img_and_global, input_ids, q_attn_mask, attn_mask, token_type_ids_q=None, batch_metadata=None, mode='train'):
    
    if model.args.use_precomputed:
        image_features, global_embedding = img_and_global
        text_features = model.question_encoder(input_ids, q_attn_mask)
        cls_tokens = text_features[:, 0:1]
        
        kb_examples = model.knowledge_base.get_images_for_paths(batch_metadata)
        image_options_embedding_dict = produce_option_embeddings_precomputed(kb_examples=kb_examples)
        
        # options_embedding_text, col_embedding, seperator_embedding, pad_embedding = self.encode_batch_options_precomputed(batch_metadata)
        #pad_embedding = self.knowledge_base_processor._get_pad_embedding()
        seperator_embedding, pad_embedding = encode_batch_sep_pad_precomputed(model,batch_metadata)
        
        # knowledge_sequence, knowledge_ttid = self.generate_knowledge_sequence_manyyes_precomputed(global_img_embeddings=global_embedding, options_embeddings_image=image_options_embedding_dict,
        #                                                         options_embedding_text=options_embedding_text,
        #                                                         col_embedding=col_embedding,
        #                                                         separator_embedding=seperator_embedding,
        #                                                         batch_metadata=batch_metadata,
        #                                                         use_noise='no')
                                                                             
        
    else:
        ## global image embedding serves as positive example in sanity check
        image_features, global_embedding = model.image_encoder(img_and_global, mode=mode)
        #image_features = image_features.detach()
        #global_image_embedding = global_image_embedding.detach()
        text_features = model.question_encoder(input_ids, q_attn_mask)
        cls_tokens = text_features[:, 0:1]

        # a list of dicts, where each list element represents a batch element, and each dict has options and paths
        kb_examples = model.knowledge_base.get_images_for_paths(batch_metadata)


        ##### need to change the representation of the MISSING_KNOWLEDGE embedding
        image_options_embedding_dict = model.produce_option_embeddings_batched(kb_examples=kb_examples, image_encoder=model.image_encoder)
        
        # ### convert to SANITY EMBEDDINGS used for sanitycheck 
        # for idx, batch in enumerate(batch_metadata):
        #     positive_options = set(batch['positive_option'])
        #     for option,tensor in image_options_embedding_dict[idx].items():
        #         if option in positive_options:
        #             image_options_embedding_dict[idx][option] = global_image_embedding[idx:idx+1]
        #             del tensor

        # batch_labels = model.generate_labels_bbc(global_img_embeddings=global_embedding, options_embeddings_image=image_options_embedding_dict,
        #                                                         separator_embedding=seperator_embedding,
        #                                                         batch_metadata=batch_metadata)    

        
        #options_embedding_text, col_embedding, seperator_embedding = model.knowledge_base_processor.encode_batch_options_batched(batch_metadata)
        seperator_embedding = model.knowledge_base_processor._get_sep_embedding()
        pad_embedding = model.knowledge_base_processor._get_pad_embedding()
        
        # knowledge_sequence, knowledge_ttid = self.generate_knowledge_sequence_manyyes_precomputed(global_img_embeddings=global_embedding, options_embeddings_image=image_options_embedding_dict,
        #                                                         options_embedding_text=options_embedding_text,
        #                                                         col_embedding=col_embedding,
        #                                                         separator_embedding=seperator_embedding,
        #                                                         batch_metadata=batch_metadata,
        #                                                         use_noise='no')
        
        
                     
    ### CHECK THAT THE FOLLOWING IS LEGIT!
    input_tokens, ttids, attn_masks, gt_labels, batch_options = generate_full_batch_sequence_bbc_noncomposite(model, cls_tokens, image_features, \
                                            global_embedding, image_options_embedding_dict, \
                                            text_features, seperator_embedding, \
                                            pad_embedding, token_type_ids_q, \
                                            batch_metadata, attn_mask, \
                                            model.knowledge_base.rrs_as_kb_train if mode=='train' else model.knowledge_base.rrs_as_kb_val)


    if model.args.progressive:
        assert token_type_ids_q is not None
        #token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
        
        #ttid, attention_mask_modified = self.generate_tokentypeid_attnmask_forbatch_bbc(h, token_type_ids_q, image_features, attn_mask)
        
        
    else:
        token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
        token_type_ids[:, 0] = 1
        token_type_ids[:, 1:image_features.size(1) + 1] = 0
        token_type_ids[:, image_features.size(1) + 1:] = 1


    #aligned = self.alignment(h)

    out = model.bbc_fusion(inputs_embeds=input_tokens, attention_mask=attn_masks, token_type_ids=ttids, output_attentions=True)
    h = out['last_hidden_state']
    attentions = out['attentions'][0]
    ### uses the mean of the sequence
    #logits = model.bbc_classifier(h.mean(dim=1))
    ### USES the [CLS] token of the fusion model instead of mean
    logits = model.bbc_classifier(h[:, 0, :])
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


def training_step(model, batch, batch_idx, dataset="vqarad"):
    if model.args.use_precomputed:
        if "vqarad" in model.args.data_dir:
            (img, global_embed), question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
        else:
            (img, global_embed), question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch
    else:
        if "vqarad" in model.args.data_dir:
            img, question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
        else:
            img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch

    question_token = question_token.squeeze(1)
    attn_mask = attn_mask.squeeze(1)
    q_attention_mask = q_attention_mask.squeeze(1)

    if model.args.use_precomputed:
        out, gt_vectors, attentions = model((img, global_embed), question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')
    else:
        out, gt_vectors, attentions  = model(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')
        
    #print_mem(f"after {batch_idx}")
    
    
    logits = out
    gt_vectors = gt_vectors.to(device=logits.device).detach()
    target = gt_vectors

    if "vqarad" in model.args.data_dir:
        pred = logits.softmax(1).argmax(1).detach()
    else:  
        pred = (logits.sigmoid().detach() > 0.5).detach().long()

        model.train_soft_scores.append(logits.sigmoid().detach())

    model.train_preds.append(pred)
    model.train_targets.append(target)

    if "vqarad" in model.args.data_dir:
        model.train_answer_types.append(answer_type)
    else:
        model.train_infos.append(info)

    loss = model.loss_fn(logits, target) # N x 1 per class loss
    mean_loss = loss.mean() # a single scalar
    #don't need for bbc
    #loss = self.get_masked_loss(loss, mask, target, None)  # only use loss of occuring classes        

    model.log('Loss/train', mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    current_lr = model.optimizers().param_groups[0]['lr']
    model.log('LR', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return mean_loss

def validation_step(model, batch, batch_idx):
    return 1

def training_epoch_end(model, outputs):
    # Concatenate predictions and targets from all batches
    preds = torch.cat(model.train_preds, dim=0)
    targets = torch.cat(model.train_targets, dim=0)
    
    # For binary classification, targets may be float, so ensure type match
    # If multiclass, ensure argmax etc. (already done in training_step)
    # If targets are one-hot or probabilities, you may need to convert

    if "vqarad" in model.args.data_dir:
        # Multiclass classification
        correct = (preds == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total
        model.log('Accuracy/train', accuracy, on_epoch=True, prog_bar=True, logger=True)
        # Add more metrics here if needed (F1, etc.)
    else:
        # Binary classification (multi-label)
        # Often want to calculate metrics per label or overall
        # Example: simple accuracy (all labels correct)
        correct = (preds == targets).float().sum()
        total = torch.numel(targets)
        accuracy = correct / total
        model.log('Accuracy/train', accuracy, on_epoch=True, prog_bar=True, logger=True)
        # You might want to use sklearn.metrics for F1, precision, recall, etc.
        # with torch -> numpy conversion if needed

    # Optionally: log more detailed metrics (F1, confusion matrix, etc.)

    # Clear out your lists so they don't grow across epochs!
    model.train_preds = []
    model.train_targets = []
    if hasattr(model, "train_answer_types"):
        model.train_answer_types = []
    if hasattr(model, "train_infos"):
        model.train_infos = []
    if hasattr(model, "train_soft_scores"):
        model.train_soft_scores = []

    model.train_preds = []
    model.train_targets = []
    model.train_soft_scores = []
    model.train_infos = []
    model.train_answer_types = []
    model.train_gen_labels = []
    model.train_gen_preds = []

def validation_epoch_end(model, outputs):
    model.log('F1/val', 0, on_epoch=True, prog_bar=True, logger=True)

## Helpers
def produce_option_embeddings_precomputed(kb_examples: List[Dict]):

    pooled_embeddings = [{} for _ in range(len(kb_examples))]
    for batch_idx, dict in enumerate(kb_examples):
        for option, tensors in dict.items():
            pooled_embeddings[batch_idx][option] = random.choice(tensors)   
    
    return pooled_embeddings

def encode_batch_sep_pad_precomputed(model, batch_metadata):
    text_options_embeddings = [{} for _ in batch_metadata]
    for batch_index, item in enumerate(batch_metadata):
        for option in item['options']:
            text_options_embeddings[batch_index][option] = model.precomputed_text_options[option]


    return model.precomputed_text_options['sep_token'], model.precomputed_text_options['pad_token']

def generate_labels_bbc(model, global_img_embeddings, options_embeddings_image, separator_embedding, batch_metadata):
    used_device = next(model.image_encoder.parameters()).device
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

def generate_full_batch_sequence_bbc(model, cls_tokens, image_features, \
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
    sequence_length = model.args.max_position_embeddings

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

def generate_full_batch_sequence_bbc_noncomposite(model, cls_tokens, image_features, \
                                    global_embeddings, image_options_g_embeddings, \
                                    text_features, seperator_embedding, \
                                    pad_embedding, token_type_ids_q, \
                                    batch_metadata, attn_mask, \
                                    rrs_as_kb):
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
    sequence_length = model.args.max_position_embeddings
    total_new_samples = 0
    # learnable CLS token for the fusion model
    cls_fusion_token = model.bbc_fusion.cls_fusion_token.to(device=cls_tokens.device)
    
    for batch_idx, batch in enumerate(batch_metadata):
        samples_per_option = 0
        for batch_option in batch['options']:
            samples_per_option += image_options_g_embeddings[batch_idx][batch_option].size(0)
        total_new_samples += samples_per_option
    num_options_in_batch = total_new_samples

    # 1) Create the output buffer all at once. By default, it's zero-initialized.
    #    We match the dtype/device of cls_tokens for safety.
    device = cls_tokens.device
    dtype = cls_tokens.dtype

    h = cls_tokens.new_zeros((num_options_in_batch, sequence_length, hidden_dim_size), dtype=dtype, device=device)
    ttids = token_type_ids_q.new_zeros((num_options_in_batch, sequence_length), dtype=token_type_ids_q.dtype, device=device)
    attn_masks_modified = attn_mask.new_zeros((num_options_in_batch, sequence_length), dtype=attn_mask.dtype, device=device)

    gt_for_batch = []
    batch_options = []

    ## go through each batch
    h_write_pos = 0
    for batch_idx in range(batch_size):
        h_write_pos = 0
        gt_for_batch.append([])
        batch_options.append([])

        ### things that should be common for each new batch element
        cls_tokens_for_batch = cls_tokens[batch_idx:batch_idx + 1]
        image_tokens_for_batch = image_features[batch_idx:batch_idx + 1]
        # num_qa_tokens_for_batch = sum(token_type_ids_q[batch_idx]).item()
        # text_tokens_for_batch = text_features[batch_size : batch_size+1, 1:num_qa_tokens]
        global_image_embedding = global_embeddings[batch_idx: batch_idx + 1]

        kb_example_embeddings = image_options_g_embeddings[batch_idx]  # dict containing each option for the batch
        ttids_for_batch = token_type_ids_q[batch_idx]
        attn_mask_for_batch = attn_mask[batch_idx]

        for option in batch_metadata[batch_idx]['options']:
            # L1 questio

            if option == 'yes' and kb_example_embeddings[option].size(0) > 1:
                # global embeddings for each l1 yes option
                for l1_option in range(kb_example_embeddings[option].size(0)):

                    write_pos = 0
                    
                    ### CLS FUSION token (new learnable param)
                    h[h_write_pos, write_pos: write_pos + 1, :] = cls_fusion_token
                    # h_inspect = h[h_write_pos]
                    ### CLS token - attn mask
                    ttids[h_write_pos, write_pos: write_pos + 1] = 1
                    attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                    write_pos += 1
                    
                    
                    # SEP embedding
                    h[h_write_pos, write_pos: write_pos + 1, :] = seperator_embedding
                    h_inspect = h[h_write_pos]
                    ### SEP embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos: write_pos + 1] = 1
                    attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                    write_pos += 1
                    
                         
                    ### CLS - text token
                    h[h_write_pos, write_pos: write_pos + 1, :] = cls_tokens_for_batch
                    # h_inspect = h[h_write_pos]
                    ### CLS token - attn mask
                    ttids[h_write_pos, write_pos: write_pos + 1] = 7
                    attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                    write_pos += 1

                    # ### Patch Image Features
                    # img_feat_len = image_tokens_for_batch.size(1)
                    # # How many image tokens can we still fit?
                    # remain = sequence_length - write_pos
                    # copy_len = min(img_feat_len, remain)
                    # if copy_len > 0:
                    #     # We slice both sides:
                    #     #   - source: img_feat_i[:, :copy_len, :] → (1, copy_len, D)
                    #     #   - destination: h[i, write_pos : write_pos+copy_len, :]
                    #     h[h_write_pos, write_pos: write_pos + copy_len, :] = image_tokens_for_batch[:, :copy_len, :]
                    #     # h_inspect = h[h_write_pos]
                    #     ### Patch Image Features - ttid + attn_mask
                    #     ttids[h_write_pos, write_pos: write_pos + copy_len] = 4
                    #     attn_masks_modified[h_write_pos, write_pos: write_pos + copy_len] = 1

                    #     write_pos += copy_len

                    # SEP embedding
                    h[h_write_pos, write_pos: write_pos + 1, :] = seperator_embedding
                    h_inspect = h[h_write_pos]
                    ### SEP embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos: write_pos + 1] = 7
                    attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                    write_pos += 1

                    ### Global input image embedding
                    h[h_write_pos, write_pos: write_pos + 1, :] = global_image_embedding
                    h_inspect = h[h_write_pos]
                    ### Global input image embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos: write_pos + 1] = 6
                    attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                    write_pos += 1

                    # SEP embedding
                    h[h_write_pos, write_pos: write_pos + 1, :] = seperator_embedding
                    h_inspect = h[h_write_pos]
                    ### Global input image embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos: write_pos + 1] = 6
                    attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                    write_pos += 1

                    ## Knowledge image embedding
                    knowledge_global_embedding = kb_example_embeddings[option][l1_option: l1_option + 1]
                    know_len = knowledge_global_embedding.size(1)
                    remain = sequence_length - write_pos
                    copy_len = min(know_len, remain)
                    if copy_len > 0:
                        h[h_write_pos, write_pos: write_pos + copy_len, :] = knowledge_global_embedding
                        # h_inspect = h[h_write_pos]
                        ### Knowledge image embedding - ttid + attn_mask
                        ttids[h_write_pos, write_pos: write_pos + copy_len] = 6
                        attn_masks_modified[h_write_pos, write_pos: write_pos + copy_len] = 1

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
                        h[h_write_pos, write_pos: write_pos + remain, :] = pad_embedding
                        # h_inspect = h[h_write_pos]
                        ### Pad - ttid + attn_mask
                        ttids[h_write_pos, write_pos: write_pos + remain] = 0
                        attn_masks_modified[h_write_pos, write_pos: write_pos + remain] = 0

                        write_pos += remain

                    h_write_pos += 1

                    batch_options[batch_idx].append(option)
                    if option in batch_metadata[batch_idx]['positive_option']:
                        current_path = batch_metadata[batch_idx]['path']
                        current_path = current_path.replace('body_regions', 'body regions')
                        ## need to change path for body region/body regions
                        l2_paths = CONSTANTS.L1_TO_L2_MAPPING_DICT[f"{current_path}_{option}"]
                        rrs_for_path = rrs_as_kb.get(l2_paths[l1_option], [])
                        img_name = batch_metadata[batch_idx]['img_name']
                        if img_name in rrs_for_path:
                            gt_for_batch[batch_idx].append(1)
                        else:
                            gt_for_batch[batch_idx].append(0)
                    else:
                        gt_for_batch[batch_idx].append(0)



            # L2 + L2 questions
            else:
                
                write_pos = 0
                
                ### CLS FUSION token (new learnable param)
                h[h_write_pos, write_pos: write_pos + 1, :] = cls_fusion_token
                # h_inspect = h[h_write_pos]
                ### CLS token - attn mask
                ttids[h_write_pos, write_pos: write_pos + 1] = 1
                attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                write_pos += 1
                
                # SEP embedding
                h[h_write_pos, write_pos: write_pos + 1, :] = seperator_embedding
                h_inspect = h[h_write_pos]
                ### SEP embedding - ttid + attn_mask
                ttids[h_write_pos, write_pos: write_pos + 1] = 1
                attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                write_pos += 1
                ### CLS token
                h[h_write_pos, write_pos: write_pos + 1, :] = cls_tokens_for_batch
                h_inspect = h[h_write_pos]
                ### CLS token - attn mask
                ttids[h_write_pos, write_pos: write_pos + 1] = 7
                attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                write_pos += 1

                # ### Path Image Features
                # img_feat_len = image_tokens_for_batch.size(1)
                # # How many image tokens can we still fit?
                # remain = sequence_length - write_pos
                # copy_len = min(img_feat_len, remain)
                # if copy_len > 0:
                #     # We slice both sides:
                #     #   - source: img_feat_i[:, :copy_len, :] → (1, copy_len, D)
                #     #   - destination: h[i, write_pos : write_pos+copy_len, :]
                #     h[h_write_pos, write_pos: write_pos + copy_len, :] = image_tokens_for_batch[:, :copy_len, :]
                #     h_inspect = h[h_write_pos]
                #     ### Patch Image Features - ttid + attn_mask
                #     ttids[h_write_pos, write_pos: write_pos + copy_len] = 4
                #     attn_masks_modified[h_write_pos, write_pos: write_pos + copy_len] = 1

                #     write_pos += copy_len

                # SEP embedding
                h[h_write_pos, write_pos: write_pos + 1, :] = seperator_embedding
                h_inspect = h[h_write_pos]
                ### SEP embedding - ttid + attn_mask
                ttids[h_write_pos, write_pos: write_pos + 1] = 7
                attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                write_pos += 1

                ### Global input image embedding
                h[h_write_pos, write_pos: write_pos + 1, :] = global_image_embedding
                h_inspect = h[h_write_pos]
                ### Global input image embedding - ttid + attn_mask
                ttids[h_write_pos, write_pos: write_pos + 1] = 6
                attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1

                write_pos += 1

                # SEP embedding
                h[h_write_pos, write_pos: write_pos + 1, :] = seperator_embedding
                h_inspect = h[h_write_pos]
                ### SEP embedding - ttid + attn_mask
                ttids[h_write_pos, write_pos: write_pos + 1] = 6
                attn_masks_modified[h_write_pos, write_pos: write_pos + 1] = 1
                write_pos += 1

                ## Knowledge image embedding
                knowledge_global_embedding = kb_example_embeddings[option]
                know_len = knowledge_global_embedding.size(1)
                remain = sequence_length - write_pos
                copy_len = min(know_len, remain)
                if copy_len > 0:
                    h[h_write_pos, write_pos: write_pos + copy_len, :] = knowledge_global_embedding
                    h_inspect = h[h_write_pos]
                    ### Knowledge image embedding - ttid + attn_mask
                    ttids[h_write_pos, write_pos: write_pos + copy_len] = 6
                    attn_masks_modified[h_write_pos, write_pos: write_pos + copy_len] = 1

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
                    h[h_write_pos, write_pos: write_pos + remain, :] = pad_embedding
                    h_inspect = h[h_write_pos]
                    ### Pad - ttid + attn_mask
                    ttids[h_write_pos, write_pos: write_pos + remain] = 0
                    attn_masks_modified[h_write_pos, write_pos: write_pos + remain] = 0

                    write_pos += remain

                h_write_pos += 1

                batch_options[batch_idx].append(option)
                if option in batch_metadata[batch_idx]['positive_option']:
                    gt_for_batch[batch_idx].append(1)
                else:
                    gt_for_batch[batch_idx].append(0)

    # gt_labels is a list of lists or similar iterable
    # Example: gt_labels = [[0, 1], [2, 3, 4], [5]]
    # or gt_labels = torch.tensor([[0, 1], [2, 3, 4], [5]]) if already a tensor
    gt_for_batch = torch.cat(
        [torch.tensor(lst, dtype=torch.float32) if not isinstance(lst, torch.Tensor) else lst.float() for lst in
         gt_for_batch]).unsqueeze(1)
    return h, ttids, attn_masks_modified, gt_for_batch, batch_options


forward._origin_file = __file__