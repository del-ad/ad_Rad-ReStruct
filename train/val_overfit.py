import argparse
import json
import os
import warnings
import wandb

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger

from data_utils.data_radrestruct import RadReStruct, RadReStructCOMBINEDEval
from net.model import ModelWrapper

warnings.simplefilter("ignore", UserWarning)


# handle info dicts in collate_fn
def collate_dict_fn(batch, *, collate_fn_map):
    return batch

def custom_collate(batch):
    default_collate_fn_map.update({dict: collate_dict_fn})
    return collate(batch, collate_fn_map=default_collate_fn_map)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Finetune on RadReStruct")

    parser.add_argument('--run_name', type=str, required=False, default="debug", help="run name for wandb")
    parser.add_argument('--data_dir', type=str, required=False, default="data/radrestruct", help="path for data")
    parser.add_argument('--model_dir', type=str, required=False, default="", help="path to load weights")
    parser.add_argument('--save_dir', type=str, required=False, default="checkpoints_radrestruct", help="path to save weights")
    parser.add_argument('--question_type', type=str, required=False, default=None, help="choose specific category if you want")
    parser.add_argument('--use_pretrained', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="use mixed precision or not")
    parser.add_argument('--bert_model', type=str, required=False, default="zzxslp/RadBERT-RoBERTa-4m", help="pretrained question encoder weights")

    parser.add_argument('--progressive', action='store_true', default=False, help="use progressive answering of questions")
    parser.add_argument('--match_instances', action='store_true', default=False, help="do optimal instance matching")
    parser.add_argument('--aug_history', action='store_true', default=False, help="do history augmentation")

    parser.add_argument('--seed', type=int, required=False, default=42, help="set seed for reproducibility")
    parser.add_argument('--num_workers', type=int, required=False, default=12, help="number of workers")
    parser.add_argument('--epochs', type=int, required=False, default=100, help="num epochs to train")
    parser.add_argument('--classifier_dropout', type=float, required=False, default=0.0, help="how often should image be dropped")

    parser.add_argument('--max_position_embeddings', type=int, required=False, default=12, help="max length of sequence")
    parser.add_argument('--max_answer_len', type=int, required=False, default=29, help="padding length for free-text answers")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="batch size")
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate'")

    parser.add_argument('--hidden_dropout_prob', type=float, required=False, default=0.3, help="hidden dropout probability")

    parser.add_argument('--img_feat_size', type=int, required=False, default=14, help="dimension of last pooling layer of img encoder")
    parser.add_argument('--num_question_tokens', type=int, required=False, default=20, help="number of tokens for question")
    parser.add_argument('--hidden_size', type=int, required=False, default=768, help="hidden size")
    parser.add_argument('--vocab_size', type=int, required=False, default=30522, help="vocab size")
    parser.add_argument('--type_vocab_size', type=int, required=False, default=2, help="type vocab size")
    parser.add_argument('--heads', type=int, required=False, default=16, help="heads")
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="num of fusion layers")
    parser.add_argument('--acc_grad_batches', type=int, required=False, default=None, help="how many batches to accumulate gradients")
    ## KB
    parser.add_argument('--kb_dir', type=str, required=False, default=None, help="the path to the knowledge base index file")

    args = parser.parse_args()

    # same as vqarad progressive
    args.num_image_tokens = args.img_feat_size ** 2
    args.max_position_embeddings = 458
    args.hidden_size_img_enc = args.hidden_size
    args.num_question_tokens = 458 - 3 - args.num_image_tokens

    # create directory for saving params
    if not os.path.exists(f'{args.save_dir}/{args.run_name}'):
        os.makedirs(f'{args.save_dir}/{args.run_name}')
    with open(os.path.join(args.save_dir, f'{args.run_name}/commandline_args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    pl.seed_everything(args.seed, workers=True)

    args.num_classes = 96

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ModelWrapper(args)

    # use torchinfo to see model architecture and trainable parameters
    from torchinfo import summary

    summary(model)
    
    def collate_dict_fn(batch, *, collate_fn_map):
        return batch
    
    def custom_collate(batch):
        default_collate_fn_map.update({dict: collate_dict_fn})
        return collate(batch, collate_fn_map=default_collate_fn_map)

    if args.use_pretrained:
        print(f"Loading model from checkpoint: {args.model_dir}")
        model = ModelWrapper.load_from_checkpoint(args.model_dir, args=args)
        model.eval()
        model.freeze()  # Optional: disables gradients


        img_tfm = model.model.image_encoder.img_tfm
        norm_tfm = model.model.image_encoder.norm_tfm
        resize_size = model.model.image_encoder.resize_size
    
        aug_tfm = transforms.Compose([transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                                  # Cutout(),
                                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                  transforms.RandomResizedCrop(resize_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                                  transforms.RandomRotation(10)])

        train_tfm = transforms.Compose([img_tfm, aug_tfm, norm_tfm]) if norm_tfm is not None else transforms.Compose([img_tfm, aug_tfm])
        test_tfm = transforms.Compose([img_tfm, norm_tfm]) if norm_tfm is not None else img_tfm

        kb_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        *img_tfm.transforms])
    
        ## apply the normalization transforms from efficient net
        kb_transforms = transforms.Compose([kb_transforms,
                                        norm_tfm])
    
        model.model.knowledge_base.train_transform = kb_transforms ## USING TEST SINCE NO AUGMENTATION ? 
        model.model.knowledge_base.test_transform = kb_transforms


        # Prepare the test dataset and loader
        testdataset = RadReStructCOMBINEDEval(tfm=test_tfm, mode='train', args=args, type='binary')   
        testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

        # Load one batch (or just the first sample)
        #sample_batch = next(iter(testloader))

        # Move data to the appropriate device
        #img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = sample_batch

        preds = []
        for batch in testloader:
             # Move data to the appropriate device
            img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch
            # Inference (no grad, eval mode)
            with torch.no_grad():
                output,_ = model(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')
                
                #out, _ = self(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, info, mode='train')

            logits = output
            predictions = logits.sigmoid().detach()
            preds.append(predictions)
            print("\n=== Inference Output ===")
            print(predictions)
        print(f"Difference between predictions: {(preds[0]-preds[1]).abs().max()}")

    img_tfm = model.model.image_encoder.img_tfm
    norm_tfm = model.model.image_encoder.norm_tfm
    resize_size = model.model.image_encoder.resize_size
    
    aug_tfm = transforms.Compose([transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                                  # Cutout(),
                                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                  transforms.RandomResizedCrop(resize_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                                  transforms.RandomRotation(10)])

    train_tfm = transforms.Compose([img_tfm, aug_tfm, norm_tfm]) if norm_tfm is not None else transforms.Compose([img_tfm, aug_tfm])
    test_tfm = transforms.Compose([img_tfm, norm_tfm]) if norm_tfm is not None else img_tfm

    ## SET THE APPROPRIATE TRANSFORMS FOR THE KNOWLEDGE BASE
    ## make images fake 3 channels by copying the existing channel 3 times
    ## add the image transforms from the efficient net transforms
    kb_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        *img_tfm.transforms])
    
    ## apply the normalization transforms from efficient net
    kb_transforms = transforms.Compose([kb_transforms,
                                        norm_tfm])
    
    model.model.knowledge_base.train_transform = kb_transforms ## USING TEST SINCE NO AUGMENTATION ? 
    model.model.knowledge_base.test_transform = kb_transforms

    traindataset = RadReStruct(tfm=train_tfm, mode='train', args=args)
    #valdataset = RadReStruct(tfm=test_tfm, mode='val', args=args)

    #handle info dicts in collate_fn
    def collate_dict_fn(batch, *, collate_fn_map):
        return batch
    
    def custom_collate(batch):
        default_collate_fn_map.update({dict: collate_dict_fn})
        return collate(batch, collate_fn_map=default_collate_fn_map)

    trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate, pin_memory=True)
    #valloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate, pin_memory=True)

