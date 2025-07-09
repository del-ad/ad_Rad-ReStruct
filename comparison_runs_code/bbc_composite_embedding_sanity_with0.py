from collections import defaultdict
from datetime import datetime
import random
from typing import Dict, List
import torch

from knowledge_base.constants import Constants, Mode

CONSTANTS = Constants(Mode.CLUSTER)

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

##### This run is identical to bbc_composite_embedding_sanity_check.py - with the difference that as
##### negative KB examples a 0 tensor is used instead of the KB embedding for that example
### The goal for this run is to be used with a precomputed KB only - to be run with a frozen img + text encoder
### KB = precomputed
### img_encoder = frozen
### text_encoder = frozen
### BBC Model ###
### BBC - trained together with RRS Fusion model
### BBC sequence: [CLS][IMG_PATCH_FEATURES][GLOBAL_IMAGE_EMBEDDING][EXAMPLE_IMAGE_EMBEDDING]
### for L1 questions - the example consists of a composite image computed from the possible L2 options
### Fusion Model ###
### the Knowledge sequence for the fusion model is as follows:
### [TEXT_OPTION_EMBEDDIG][:][IMG_OPTION_GLOBAL_EMBEDDING][SEP] --- for the correct option 
### IMG_OPTION_GLOBAL_EMBEDDING is replaced with the INPUT_IMAGE_GLOBAL_EMBEDDING - the sanity check part
### Next steps: ###
### Both models are trained together but they have no other interaction, BBC is not used to influence RRS fusion
### essentially the goal is to measure performace of the sanity check when training both RRS fusion and BBC fusion models together
### next step would be to disable the sanity check so that the global embedding is set to only examples that the BBC predicted it
### should be set for
def forward(model, img_and_global, input_ids, q_attn_mask, attn_mask, token_type_ids_q=None, batch_metadata=None, mode='train'):
        
        
        if model.args.use_precomputed:
            image_features, global_embedding = img_and_global
            text_features = model.question_encoder(input_ids, q_attn_mask)
            cls_tokens = text_features[:, 0:1]
            
            kb_examples = model.knowledge_base.get_images_for_paths(batch_metadata)
            image_options_embedding_dict = produce_option_embeddings_precomputed(kb_examples=kb_examples)
            
            options_embedding_text, col_embedding, seperator_embedding, pad_embedding = encode_batch_options_precomputed(model, batch_metadata)
            #pad_embedding = self.knowledge_base_processor._get_pad_embedding()

            if model.args.use_kb_adapter:
                input_tokens, ttids, attn_masks, gt_labels, batch_options = generate_full_batch_sequence_bbc_composite(model.args, cls_tokens, image_features, \
                                            global_embedding, image_options_embedding_dict, \
                                            text_features, seperator_embedding, \
                                            pad_embedding, token_type_ids_q, \
                                            batch_metadata, attn_mask)
                
                out = model.bbc(inputs_embeds=input_tokens, attention_mask=attn_masks, token_type_ids=ttids, output_attentions=True)
                h_log = out['last_hidden_state']
                logits = model.bbc_classifier(h_log.mean(dim=1))
                preds_bbc = logits.sigmoid().detach()
                score_boosts, labels_to_preds = get_score_boosts(preds_bbc, gt_labels, batch_options)
            
            ### bbc bias
            # knowledge_sequence, knowledge_ttid = self.generate_knowledge_sequence_manyyes_precomputed_bbcbias(options_embeddings_image=image_options_embedding_dict,
            #                                                         options_embedding_text=options_embedding_text,
            #                                                         g_embedding_image=global_embedding,
            #                                                         col_embedding=col_embedding,
            #                                                         separator_embedding=seperator_embedding,
            #                                                         batch_metadata=batch_metadata,
            #                                                         labels_preds=labels_to_preds,
            #                                                         use_noise='no')

            # ### convert to SANITY EMBEDDINGS used for sanitycheck
            ### positive example is the global image embedding for the input image
            ### negative example/s is the 0 embedding 
            for idx, batch in enumerate(batch_metadata):
                positive_options = set(batch['positive_option'])
                for option,tensor in image_options_embedding_dict[idx].items():
                    if option in positive_options:
                        image_options_embedding_dict[idx][option] = global_embedding[idx:idx+1]
                        del tensor
                    else:
                        image_options_embedding_dict[idx][option] = model.image_encoder.zero_embedding
                        del tensor


            score_boosts = []
            knowledge_sequence, knowledge_ttid = generate_knowledge_sequence_vanilla_precomputed(model, options_embeddings_image=image_options_embedding_dict,
                                                                    options_embedding_text=options_embedding_text,
                                                                    col_embedding=col_embedding,
                                                                    separator_embedding=seperator_embedding,
                                                                    batch_metadata=batch_metadata,
                                                                    use_noise='no')
            
            #print("hi")
            
        else:
            pass
     
        ### CHECK THAT THE FOLLOWING IS LEGIT!
        h = generate_full_batch_sequence_fix(model, cls_tokens, image_features, global_embedding, knowledge_sequence, text_features, seperator_embedding, pad_embedding, token_type_ids_q)


        if model.args.progressive:
            assert token_type_ids_q is not None
            #token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            
            ttid, attention_mask_modified = generate_tokentypeid_attnmask_forbatch_fix(model, h, token_type_ids_q, image_features, knowledge_sequence, attn_mask, knowledge_ttid)
            
            
        else:
            token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            token_type_ids[:, 0] = 1
            token_type_ids[:, 1:image_features.size(1) + 1] = 0
            token_type_ids[:, image_features.size(1) + 1:] = 1


        #aligned = self.alignment(h)

        out = model.fusion(inputs_embeds=h, attention_mask=attention_mask_modified, token_type_ids=ttid, output_attentions=True)
        h = out['last_hidden_state']
        attentions = out['attentions'][0]
        logits = model.classifier(h.mean(dim=1))
        
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
        
        for _ in options_embedding_text:
            _.clear()
        options_embedding_text.clear()
        knowledge_sequence.clear()
        knowledge_ttid.clear()
        del attention_mask_modified, ttid, kb_examples, image_options_embedding_dict #, options_embedding_img_sanity
        

        if not torch.isfinite(logits).all():
            print(f'{timestamp()}Fusion model produced nan/inf in ints output')

        if model.args.use_kb_adapter:
            return logits, attentions, preds_bbc, gt_labels
        else:
            return logits, attentions, score_boosts

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
        out, _, bbc_preds, bbc_gt = model((img, global_embed), question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')
    else:
        out, _, score_boosts = model(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')
        

    logits = out
    #score_boosts = score_boosts.to(device=logits.device)
    if "vqarad" in model.args.data_dir:
        pred = logits.softmax(1).argmax(1).detach()
    else:  # multi-label classification
        ### Sanity check
        # pred = (logits.sigmoid().detach() > 0.5).detach().long()
        # self.train_soft_scores.append(logits.sigmoid().detach())
        pred = (logits.sigmoid().detach() > 0.5).detach().long()
        #boosted_preds = ((logits + score_boosts).sigmoid().detach() > 0.5).detach().long()
        
        ### Logging monitoring metrics - comment out during full training
        # preddd = logits.sigmoid().detach()
        # predictions_per_option = {option: preddd[0, CONSTANTS.ANSWER_OPTIONS_OPTION_STR_TO_CODE_INT[option]].detach().item() for option in all_options}
        model.train_soft_scores.append(logits.sigmoid().detach())

    #self.train_preds.append(boosted_preds)
    model.train_preds.append(pred)
    model.train_targets.append(target)

    if "vqarad" in model.args.data_dir:
        model.train_answer_types.append(answer_type)
    else:
        model.train_infos.append(info)

    loss = model.loss_fn(logits, target)
    loss = model.get_masked_loss(loss, mask, target, None)  # only use loss of occuring classes     
    ### For joint training of bbc + normal model
    loss_bbc = model.loss_fn_bbc(bbc_preds, bbc_gt.to(device=bbc_preds.device)).mean()

    loss = loss + loss_bbc   

    model.log('Loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return loss

def validation_step(model, batch, batch_idx):
    if model.args.use_precomputed:
        if "vqarad" in model.args.data_dir:
            (img, global_embedding), question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
        else:
            (img, global_embedding), question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch
    else:
        if "vqarad" in model.args.data_dir:
            img, question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
        else:
            img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch            

    question_token = question_token.squeeze(1)
    attn_mask = attn_mask.squeeze(1)
    q_attention_mask = q_attention_mask.squeeze(1)
    batch_info = batch[6]
    
    if model.args.use_precomputed:
        out, _, bbc_pred, bbc_gt = model((img, global_embedding), question_token, q_attention_mask, attn_mask, token_type_ids_q, batch_info, mode='val')
    else:
        out, _, score_boosts = model(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, batch_info, mode='val')

    logits = out
    #score_boosts = score_boosts.to(device=logits.device)
    if "vqarad" in model.args.data_dir:
        pred = logits.softmax(1).argmax(1).detach()
        model.val_soft_scores.append(logits.softmax(1).detach())
    else:  # multi-label classification
        pred = (logits.sigmoid().detach() > 0.5).detach().long()
        #boosted_preds = ((logits + score_boosts).sigmoid().detach() > 0.5).detach().long()
        model.val_soft_scores.append(logits.sigmoid().detach())

    #self.val_preds.append(boosted_preds)
    model.val_preds.append(pred)
    model.val_targets.append(target)
    if "vqarad" in model.args.data_dir:
        model.val_answer_types.append(answer_type)
    else:
        model.val_infos.append(info)

    if "vqarad" in model.args.data_dir:
        val_loss = model.loss_fn(logits[target != -1], target[target != -1])
    else:
        val_loss = model.loss_fn(logits, target)
        bbc_val_loss = model.loss_fn_bbc(bbc_pred, bbc_gt.to(device=bbc_pred.device)).mean()
        # only use loss of occuring classes
        if "radrestruct" in model.args.data_dir:
            val_loss = model.get_masked_loss(val_loss, mask, target, None)
        val_loss = val_loss + bbc_val_loss
    model.log('Loss/val', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    return val_loss



### HELPERS
def generate_full_batch_sequence_bbc_composite(args, cls_tokens, image_features, \
                                    global_embeddings, image_options_g_embeddings, \
                                    text_features, seperator_embedding, \
                                    pad_embedding, token_type_ids_q, \
                                    batch_metadata, attn_mask):
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
    sequence_length = args.max_position_embeddings

    # 1) Create the output buffer all at once. By default, it's zero-initialized.
    #    We match the dtype/device of cls_tokens for safety.
    device = cls_tokens.device
    dtype = cls_tokens.dtype
    num_options_in_batch = sum(len(batch['options']) for batch in batch_metadata)
    h = cls_tokens.new_zeros((num_options_in_batch, sequence_length, hidden_dim_size), dtype=dtype, device=device)
    ttids = token_type_ids_q.new_zeros((num_options_in_batch, sequence_length), dtype=token_type_ids_q.dtype, device=device)
    attn_masks_modified = attn_mask.new_zeros((num_options_in_batch, sequence_length), dtype=attn_mask.dtype, device=device)

    gt_for_batch = []
    batch_options = []

    ## go through each batch
    h_write_pos = 0
    for batch_idx in range(batch_size):
        gt_for_batch.append([])
        batch_options.append([])

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
            
            ### why both yes and size0 - size should only ever be > 1 for L1 questions, right ?
            ### it's a way to differentiate between L1 and L2 questions ?
            if option == 'yes' and kb_example_embeddings[option].size(0) > 1:
                # global embeddings for each l1 yes option
                composite_g_img_embedding, _ = torch.max(kb_example_embeddings[option], dim=0, keepdim=True)
                kb_example_embeddings[option] = composite_g_img_embedding
            write_pos = 0
            ### CLS token
            h[h_write_pos, write_pos : write_pos+1, :] = cls_tokens_for_batch
            #h_inspect = h[h_write_pos]
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
                #h_inspect = h[h_write_pos]
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
            knowledge_global_embedding = kb_example_embeddings[option]
            know_len = knowledge_global_embedding.size(1)
            remain = sequence_length - write_pos
            copy_len = min(know_len, remain)
            if copy_len > 0:
                h[h_write_pos, write_pos : write_pos+copy_len, :] = knowledge_global_embedding
                #h_inspect = h[h_write_pos]
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
                #h_inspect = h[h_write_pos]
                ### Pad - ttid + attn_mask
                ttids[h_write_pos, write_pos : write_pos+remain] = 0
                attn_masks_modified[h_write_pos, write_pos : write_pos+remain] = 0
                
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
    gt_for_batch = torch.cat([torch.tensor(lst, dtype=torch.float32) if not isinstance(lst, torch.Tensor) else lst.float() for lst in gt_for_batch]).unsqueeze(1)
    return h, ttids, attn_masks_modified, gt_for_batch, batch_options


def generate_labels_bbc(self, global_img_embeddings, options_embeddings_image, separator_embedding, batch_metadata):
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

def get_total_samples(self, global_img_embeddings, options_embeddings_image, separator_embedding, batch_metadata):
    total_samples = 0
    for batch_index in range(len(options_embeddings_image)):
        img_options_embedding = options_embeddings_image[batch_index]
        
        for option in batch_metadata[batch_index]['options']:
            img_emb = img_options_embedding[option]
            
            # many yes - L1 yes question - produces 
            if option == 'yes' and img_emb.size(0) > 1:
                total_samples += img_emb.size(0)
            ## non L1 uestion
            else:
                total_samples += 1
    return total_samples

def produce_option_embeddings_precomputed(kb_examples: List[Dict]):

    pooled_embeddings = [{} for _ in range(len(kb_examples))]
    for batch_idx, dict in enumerate(kb_examples):
        for option, tensors in dict.items():
            pooled_embeddings[batch_idx][option] = random.choice(tensors)   
    
    return pooled_embeddings

def encode_batch_options_precomputed(model, batch_metadata):
    text_options_embeddings = [{} for _ in batch_metadata]
    for batch_index, item in enumerate(batch_metadata):
        for option in item['options']:
            text_options_embeddings[batch_index][option] = model.precomputed_text_options[option]


    return text_options_embeddings, model.precomputed_text_options['colon_token'], model.precomputed_text_options['sep_token'], model.precomputed_text_options['pad_token']

def get_score_boosts(predictions, gt_labels, batch_options):
    score_boosts = torch.zeros([len(batch_options), 96])
    sample_index = 0
    batch_predictions = []
    SIMILARITY_THRESHOLD = 0.5

    # Define which labels are considered L1/L2 (mutually exclusive)
    L1_L2_LABELS = {'yes', 'no'}

    for batch_idx, batch in enumerate(batch_options):
        batch_size = len(batch)
        # bps = predictions[sample_index : sample_index + batch_size].shape
        # batch_preds = predictions[sample_index : sample_index + batch_size].squeeze()
        # bps = batch_preds.shape
        bps_initial = predictions[sample_index : sample_index + batch_size].shape
        batch_preds = predictions[sample_index : sample_index + batch_size].reshape(-1) # Flattens to 1D
        bps_final = batch_preds.shape # This will now be (N,) where N is the number of elements
        # Group predictions by label
        label_to_preds = defaultdict(list)
        batch_predictions.append({})
        for i, label in enumerate(batch):
            label_to_preds[label].append(batch_preds[i].item())
            batch_predictions[batch_idx][label] = False

        labels_in_batch = set(label_to_preds.keys())
        is_L1_L2 = labels_in_batch.issubset(L1_L2_LABELS)

        if is_L1_L2:
            # Mutually exclusive: only boost the highest-scoring label (if any)
            best_label = None
            best_score = -float('inf')
            for label, preds in label_to_preds.items():
                max_pred = max(preds)
                if max_pred >= SIMILARITY_THRESHOLD and max_pred > best_score:
                    best_score = max_pred
                    best_label = label
            if best_label:
                score_boosts[batch_idx][CONSTANTS.ANSWER_OPTIONS_OPTION_STR_TO_CODE_INT[best_label]] = 0.4
                batch_predictions[batch_idx][best_label] = True

        else:
            # L3: allow multiple boosts
            for label, preds in label_to_preds.items():
                if any(pred >= SIMILARITY_THRESHOLD for pred in preds):
                    score_boosts[batch_idx][CONSTANTS.ANSWER_OPTIONS_OPTION_STR_TO_CODE_INT[label]] = 0.4
                    batch_predictions[batch_idx][label] = True

        sample_index += batch_size

    return score_boosts, batch_predictions

def generate_knowledge_sequence_vanilla_precomputed(model, options_embeddings_image, options_embedding_text, col_embedding, separator_embedding, batch_metadata, use_noise='no'):
    used_device = next(model.image_encoder.parameters()).device
    
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

def generate_full_batch_sequence_fix(model, cls_tokens, image_features, \
                                        global_embeddings, knowledge_sequence, \
                                        text_features, seperator_embedding, \
                                        pad_embedding, token_type_ids_q):
        """
        This version pre-allocates a single tensor of shape
        (batch_size, max_position_embeddings, 768)
        and then fills in each row [i, :] by copying the
        sub-segments (cls_tok, img_feat, know_feat, txt_feat)
        directly into that buffer, up to max_position_embeddings.
        """

        batch_size = cls_tokens.size(0)
        hidden_dim_size = cls_tokens.size(-1)  # typically 768
        sequence_length = model.args.max_position_embeddings

        # 1) Create the output buffer all at once. By default, it's zero-initialized.
        #    We match the dtype/device of cls_tokens for safety.
        device = cls_tokens.device
        dtype = cls_tokens.dtype
        h = cls_tokens.new_zeros((batch_size, sequence_length, hidden_dim_size), dtype=dtype, device=device)
        # Now `h` is (B, K, 768), filled with zeros (or whatever default).

        # 2) For each sample i, copy in segments one by one:
        for i in range(batch_size):
            # We keep a running offset into the “position” dimension of h[i].
            write_pos = 0

            #inspect_ttidq = token_type_ids_q[i]
            num_qa_tokens = sum(token_type_ids_q[i]).item()

            # a) Copy the CLS token (shape (1,1,D)) → h[i, 0:1, :]
            #    cls_tokens[i] has shape (1, 1, D) if cls_tokens is (B,1,D).
            h[i, write_pos : write_pos+1, :] = cls_tokens[i]
            #h_inspect = h[i]
            write_pos += 1

            # b) Copy image_features[i], but truncate if it would exceed K.
            #    img_feat_i: shape (img_len, D) or (1, img_len, D)? Let’s assume (1, img_len, D).
            img_feat_i = image_features[i : i+1]  # shape (1, img_len, D)
            img_len = img_feat_i.size(1)
            # How many image tokens can we still fit?
            remain = sequence_length - write_pos
            copy_len = min(img_len, remain)
            if copy_len > 0:
                # We slice both sides:
                #   - source: img_feat_i[:, :copy_len, :] → (1, copy_len, D)
                #   - destination: h[i, write_pos : write_pos+copy_len, :]
                h[i, write_pos : write_pos+copy_len, :] = img_feat_i[:, :copy_len, :]
                #h_inspect = h[i]
                write_pos += copy_len

            # c) Copy knowledge_sequence[i].permute(1, 0, 2). 
            #    Suppose knowledge_sequence[i] has shape (N_i, 1, D).
            #    Then after permute(1,0,2) it becomes (1, N_i, D).
            #    Let’s call that know_i:

            # d) Copy the SEP token token (shape (1,1,D)) → h[i, 0:1, :]
            #    cls_tokens[i] has shape (1, 1, D) if cls_tokens is (B,1,D).
            h[i, write_pos : write_pos+1, :] = seperator_embedding
            #h_inspect = h[i]
            write_pos += 1

            #### UNCOMMENT TO INCLUDE THE GLOBAL IMAGE EMBEDDING OF THE INPUT IMAGE IN THE SEQUENCE
            ## c) Copy the GLOBAL IMAGE EMBEDDING token token (shape (1,1,D)) → h[i, 0:1, :]
            ##   cls_tokens[i] has shape (1, 1, D) if cls_tokens is (B,1,D).
            h[i, write_pos : write_pos+1, :] = global_embeddings[i]
            #h_inspect = h[i]
            write_pos += 1

            # d) Copy the SEP token token (shape (1,1,D)) → h[i, 0:1, :]
            #    cls_tokens[i] has shape (1, 1, D) if cls_tokens is (B,1,D).
            h[i, write_pos : write_pos+1, :] = seperator_embedding
            #h_inspect = h[i]
            write_pos += 1

            ### only if there is knowledge sequence 
            if knowledge_sequence[i].ndim > 1:
                know_i = knowledge_sequence[i].permute(1, 0, 2)  # (1, N_i, D)
                know_len = know_i.size(1)
                remain = sequence_length - write_pos
                copy_len = min(know_len, remain)
                if copy_len > 0:
                    h[i, write_pos : write_pos+copy_len, :] = know_i[:, :copy_len, :]
                    #h_inspect = h[i]
                    write_pos += copy_len

            # d) Copy the text features (skip the CLS in text_features). 
            #    text_features[i] is (T, D) if text_features is (B,T,D).
            #    We want text_features[i, 1:] → shape (T-1, D).
            txt_feat_i = text_features[i : i+1, 1:num_qa_tokens]  # (1, T-1, D)
            #inspct_txt_feat = text_features[i]
            #inspct_txt_feat2 = text_features[i : i + 1, 1:num_qa_tokens]
            #inspct_txt_feat3 = text_features[i : i + 1, 1:num_qa_tokens, 0:5]
            txt_len = txt_feat_i.size(1)
            remain = sequence_length - write_pos
            copy_len = min(txt_len, remain)
            if copy_len > 0:
                h[i, write_pos : write_pos+copy_len, :] = txt_feat_i[:, :copy_len, :]
                #h_inspect = h[i]
                write_pos += copy_len
            
            ### pad the rest of the sequence
            remain = sequence_length - write_pos
            if remain > 0:
                h[i, write_pos : write_pos+remain, :] = pad_embedding
                #h_inspect = h[i]
                write_pos += remain


            # If write_pos < K, the tail of h[i] remains as all zeros (which you can treat
            # as padding if that makes sense for your model). If your model needs a special
            # pad token embedding instead of zero, you could fill that here instead.

        # 3) Return h, which is (B, K, 768).
        return h

def generate_tokentypeid_attnmask_forbatch_fix(
    model,
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
        if not hasattr(model, "_ttids_buffer") or model._ttids_buffer.size() != (batch_size, seq_len):
            # If it doesn’t exist yet, allocate it once:
            model._ttids_buffer = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )
            model._mask_buffer = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )

        token_type_ids = model._ttids_buffer
        modified_attention_mask = model._mask_buffer

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

forward._origin_file = __file__