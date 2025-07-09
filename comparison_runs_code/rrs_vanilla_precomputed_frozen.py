from collections import defaultdict
from datetime import datetime
import random
from typing import Dict, List
import torch

from knowledge_base.constants import Constants, Mode

CONSTANTS = Constants(Mode.CLUSTER)

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

### Vanilla RRS run with no knowledge base and precomputed images - some of the RRS + KB model components are present but not used
### KB = precomputed
### img_encoder = frozen
### text_encoder = frozen
### BBC Model - not initialized due to 
### Fusion Model - the only one being trained ###
### Next steps: ###
# python -m train.train_radrestruct \
#     --run_name "rrs_vanilla_precomputedimages_frozen" \
#     --data_dir /home/guests/adrian_delchev/code/Rad-ReStruct/data/radrestruct \
#     --lr 1e-5 \
#     --classifier_dropout 0.2 \
#     --epochs 30 \
#     --batch_size 4 \
#     --acc_grad_batches 4 \
#     --progressive \
#     --mixed_precision \
#     --use_precomputed \
#     --freeze_image_encoder \
#     --freeze_question_encoder


def forward(model, img_and_global, input_ids, q_attn_mask, attn_mask, token_type_ids_q=None, batch_metadata=None, mode='train'):
        
        if model.args.use_precomputed:
            image_features, global_embedding = img_and_global
             
        else:
            pass
        
        text_features = model.question_encoder(input_ids, q_attn_mask)
        cls_tokens = text_features[:, 0:1]
     
        ### vanilla rrs
        h = torch.cat((cls_tokens, image_features, text_features[:, 1:]), dim=1)[:, :model.args.max_position_embeddings]


        if model.args.progressive:
            assert token_type_ids_q is not None
            token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            token_type_ids[:, 0] = 1  # cls token
            token_type_ids[:, 1:image_features.size(1) + 1] = 0  # image features
            token_type_ids[:, image_features.size(1) + 1:] = token_type_ids_q[:, 1:model.args.max_position_embeddings - (
                image_features.size(1))]  # drop CLS token_type_id as added before already and cut unnecessary padding
        else:
            token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            token_type_ids[:, 0] = 1
            token_type_ids[:, 1:image_features.size(1) + 1] = 0
            token_type_ids[:, image_features.size(1) + 1:] = 1


        #aligned = self.alignment(h)

        out = model.fusion(inputs_embeds=h, attention_mask=attn_mask, token_type_ids=token_type_ids, output_attentions=True)
        h = out['last_hidden_state']
        attentions = out['attentions'][0]
        logits = model.classifier(h.mean(dim=1))
        
        
        if not torch.isfinite(logits).all():
            print(f'{timestamp()}Fusion model produced nan/inf in ints output')

        if model.args.use_precomputed:
            return logits, attentions, [], [] # bbc preds, bbc_gt
        else:
            return logits, attentions, [] # score boosts
        
# def forward(self, img, input_ids, q_attn_mask, attn_mask, token_type_ids_q=None, mode='train', precomputed=False):
        
#     if model.args.use_precomputed:
#         # convert from precomputed numpy to tensor
#         image_features =  img
#     else:
#         image_features, _ = self.image_encoder(img, mode=mode)
#     text_features = self.question_encoder(input_ids, q_attn_mask)
#     cls_tokens = text_features[:, 0:1]

#     h = torch.cat((cls_tokens, image_features, text_features[:, 1:]), dim=1)[:, :self.args.max_position_embeddings]
#     if self.args.progressive:
#         assert token_type_ids_q is not None
#         token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
#         token_type_ids[:, 0] = 1  # cls token
#         token_type_ids[:, 1:image_features.size(1) + 1] = 0  # image features
#         token_type_ids[:, image_features.size(1) + 1:] = token_type_ids_q[:, 1:self.args.max_position_embeddings - (
#             image_features.size(1))]  # drop CLS token_type_id as added before already and cut unnecessary padding
#     else:
#         token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
#         token_type_ids[:, 0] = 1
#         token_type_ids[:, 1:image_features.size(1) + 1] = 0
#         token_type_ids[:, image_features.size(1) + 1:] = 1

#     out = self.fusion(inputs_embeds=h, attention_mask=attn_mask, token_type_ids=token_type_ids, output_attentions=True)
#     h = out['last_hidden_state']
#     attentions = out['attentions'][0]
#     logits = self.classifier(h.mean(dim=1))

#     return logits, attentions

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
    #loss_bbc = model.loss_fn_bbc(bbc_preds, bbc_gt.to(device=bbc_preds.device)).mean()

    #loss = loss + loss_bbc   

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
        #bbc_val_loss = model.loss_fn_bbc(bbc_pred, bbc_gt.to(device=bbc_pred.device)).mean()
        # only use loss of occuring classes
        if "radrestruct" in model.args.data_dir:
            val_loss = model.get_masked_loss(val_loss, mask, target, None)
        #val_loss = val_loss + bbc_val_loss
    model.log('Loss/val', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    return val_loss


forward._origin_file = __file__