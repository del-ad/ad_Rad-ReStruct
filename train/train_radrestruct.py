import argparse
from collections import OrderedDict
from datetime import datetime
import gc
import json
import os
import pickle
import warnings
import wandb

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger

from data_utils.data_radrestruct import RadReStruct, RadReStructCOMBINED, RadReStructPrecomputed, RadReStructReversed, RadReStructCOMBINEDEval
from knowledge_base.knowledge_base_loader import KnowledgeBase,CachedKnowledgeBase, PrecomputedKnowledgeBase
from net.model import ModelWrapper

import tracemalloc, linecache
import objgraph

warnings.simplefilter("ignore", UserWarning)

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
# handle info dicts in collate_fn
def collate_dict_fn(batch, *, collate_fn_map):
    return batch

def custom_collate(batch):
    default_collate_fn_map.update({dict: collate_dict_fn})
    return collate(batch, collate_fn_map=default_collate_fn_map)

import pytorch_lightning as pl
import gc, ctypes, ctypes.util
from typing import Optional

def trim_cpu_cache():
    """Release freed CPU heap pages back to the OS."""
    gc.collect()
    try:                             # PyTorch ≥ 2.2
        import torch
        torch.malloc_trim()
    except (ImportError, AttributeError):
        # earlier versions – call glibc directly
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.malloc_trim(0)

class MallocTrim(pl.Callback):
    def __init__(self, every_steps: int = 500):
        self.every_steps = every_steps

    def on_train_batch_end(self, trainer, *_):
        if trainer.global_step and trainer.global_step % self.every_steps == 0:
            trim_cpu_cache()

if __name__ == '__main__':
    tracemalloc.start()
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
    parser.add_argument('--initialize_kb', action='store_true', default=False, help="is the KB actually being loaded into memory")
    parser.add_argument('--kb_dir', type=str, required=False, default=None, help="the path to the knowledge base index file")
    parser.add_argument('--use_kb_adapter', action='store_true', default=False, help="use the bbc kb adapter")
    parser.add_argument('--pretrained_kb_adapter', action='store_true', default=False, help="use the bbc kb adapter")
    parser.add_argument('--kb_adapter_dir', type=str, required=False, default=None, help="the path to the knowledge base bbc adapter model")
    ## Freezing parts of the model
    parser.add_argument('--freeze_image_encoder', action='store_true', default=False, help="freeze the image encoder so its weights don't get updated")
    parser.add_argument('--freeze_question_encoder', action='store_true', default=False, help="freeze the question encoder so its weights don't get updated")
    ## Precomputed
    parser.add_argument('--use_precomputed', action='store_true', default=False, help="use precomputed KB, image features, global embeddings and text features")

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
    print(f"{timestamp()} ==================================================================")
    print(f"{timestamp()} Using the forward() defined in: {model.model.get_forward_origin()}")
    # move the missing knowledge embedding from the image encoder to the gpu
    model.model.image_encoder.missing_knowledge_embedding = model.model.image_encoder.missing_knowledge_embedding.to(device=device)


    if args.use_pretrained:
        print(f"{timestamp()}Loading model from checkpoint: {args.model_dir}")
        
        checkpoint = torch.load(args.model_dir, map_location=torch.device('cpu'))
        full_state_dict = checkpoint['state_dict']
        
        image_encoder_state_dict = OrderedDict()
        for k, v in full_state_dict.items():
            if k.startswith('model.image_encoder.'):
                # Remove the 'image_encoder.' prefix
                new_key = k.replace('model.image_encoder.', '')
                image_encoder_state_dict[new_key] = v
        
        print(f"\n {timestamp()}Attempting to load state_dict into image_encoder...")
        missing_keys, unexpected_keys = model.model.image_encoder.load_state_dict(image_encoder_state_dict)
        
        ### Old - loading full model weights
        #checkpoint = torch.load(args.model_dir, map_location=torch.device('cpu'))
        #missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'])
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0

    ### Load pre-trained bbc and freeze it
    # if args.use_kb_adapter:
    #     print(f"{timestamp()}Loading BBC model from checkpoint: {args.kb_adapter_dir}")

    #     checkpoint = torch.load(args.kb_adapter_dir, map_location=torch.device('cpu'))
    #     full_state_dict = checkpoint['state_dict']

    #     fusion_bbc_state_dict = OrderedDict()
    #     bbc_classifier_state_dict = OrderedDict()
    #     for k, v in full_state_dict.items():
    #         if k.startswith('model.fusion.'):
    #             # Remove the 'image_encoder.' prefix
    #             new_key = k.replace('model.fusion.', '')
    #             fusion_bbc_state_dict[new_key] = v

    #         if k.startswith('model.classifier.'):
    #             # Remove the 'image_encoder.' prefix
    #             new_key = k.replace('model.classifier.', '')
    #             bbc_classifier_state_dict[new_key] = v

    #     print(f"\n {timestamp()}Attempting to load state_dict into bbc fusion...")
    #     missing_keys, unexpected_keys = model.model.bbc.load_state_dict(fusion_bbc_state_dict)

    #     print(f"\n {timestamp()}Attempting to load state_dict into bbc classifier...")
    #     missing_keys, unexpected_keys = model.model.bbc_classifier.load_state_dict(bbc_classifier_state_dict)

    #     assert len(missing_keys) == 0
    #     assert len(unexpected_keys) == 0

    #     for param in model.model.bbc.parameters():
    #         param.requires_grad = False
    #     print(f"{timestamp()}The BBC fusion has been frozen")

    #     for param in model.model.bbc_classifier.parameters():
    #         param.requires_grad = False
    #     print(f"{timestamp()}The BBC classifier has been frozen")
    if args.use_kb_adapter:
        if args.pretrained_kb_adapter:
            print(f"{timestamp()}Loading BBC model from checkpoint: {args.kb_adapter_dir}")

            checkpoint = torch.load(args.kb_adapter_dir, map_location=torch.device('cpu'))
            full_state_dict = checkpoint['state_dict']

            bbc_classifier_state_dict = OrderedDict()
            for k, v in full_state_dict.items():

                if k.startswith('model.bbc_simple_ffn.'):
                    # Remove the 'image_encoder.' prefix
                    new_key = k.replace('model.bbc_simple_ffn.', '')
                    bbc_classifier_state_dict[new_key] = v


            print(f"\n {timestamp()}Attempting to load state_dict into simple bbc bbc classifier...")
            missing_keys, unexpected_keys = model.model.bbc_simple_ffn.load_state_dict(bbc_classifier_state_dict)

            assert len(missing_keys) == 0
            assert len(unexpected_keys) == 0

            for param in model.model.bbc_simple_ffn.parameters():
                param.requires_grad = False
            print(f"{timestamp()}The BBC classifier has been frozen")
        
    if args.freeze_image_encoder:
        for param in model.model.image_encoder.parameters():
            param.requires_grad = False
        print(f"{timestamp()}The image encoder has been frozen")
    
    if args.freeze_question_encoder:
        for param in model.model.question_encoder.parameters():
            param.requires_grad = False
        print(f"{timestamp()}The question encoder has been frozen")
        
        
    # use torchinfo to see model architecture and trainable parameters
    from torchinfo import summary
    summary(model)

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
    
    if args.use_precomputed:
        if args.initialize_kb:
            model.model.knowledge_base = PrecomputedKnowledgeBase(args.kb_dir, kb_transforms, img_transform=img_tfm,
                                                                norm_transform=norm_tfm, precomputed_path='/home/guests/adrian_delchev/code/ad_Rad-ReStruct/precomputed/kb_5samples_3composite.pkl')
            print(f"{timestamp()} Using PrecomputedKnowledgeBase: {model.model.knowledge_base.precomputed_path}")
        traindataset = RadReStructPrecomputed(tfm=train_tfm, mode='train', args=args, precompute=args.use_precomputed)
        valdataset = RadReStructPrecomputed(tfm=test_tfm, mode='val', args=args, precompute=args.use_precomputed)
        
    else:
        if args.initialize_kb:
            model.model.knowledge_base = CachedKnowledgeBase(args.kb_dir, model.model.image_encoder, kb_transforms, img_transform=img_tfm, norm_transform=norm_tfm)
        traindataset = RadReStruct(tfm=train_tfm, mode='train', args=args)
        valdataset = RadReStruct(tfm=test_tfm, mode='val', args=args)
        print(f"{timestamp()} Using CachedKnowledgeBase")
    
    ### set KB transforms
    if args.initialize_kb:
        model.model.knowledge_base.train_transform = kb_transforms ## USING TEST SINCE NO AUGMENTATION ? 
        model.model.knowledge_base.test_transform = kb_transforms

    ### Original
    # traindataset = RadReStruct(tfm=train_tfm, mode='train', args=args)
    # valdataset = RadReStruct(tfm=test_tfm, mode='val', args=args)

    ### New (overfitting)
    # traindataset = RadReStruct(tfm=train_tfm, mode='train', args=args)
    # valdataset = RadReStruct(tfm=test_tfm, mode='val', args=args)

    #handle info dicts in collate_fn
    def collate_dict_fn(batch, *, collate_fn_map):
        return batch

    def custom_collate(batch):
        default_collate_fn_map.update({dict: collate_dict_fn})
        return collate(batch, collate_fn_map=default_collate_fn_map)

    trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate, pin_memory=True)
    valloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate, pin_memory=True)

    #logger = pl.loggers.TensorBoardLogger('runs_radrestruct', name=args.run_name, version=0)
    logger = WandbLogger(project='train_radrestruct', name=args.run_name, config=args)
    ### Original
    checkpoint_callback = ModelCheckpoint(monitor='F1/val', dirpath=os.path.join(args.save_dir, args.run_name), filename='{epoch}-{F1/val:.2f}',
                                          mode='max', every_n_epochs=1, save_last=True)
    ### New - Overfitting tests
    #checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.save_dir, args.run_name), filename='last-epoch-{epoch}', save_last=True)

    # trainer = Trainer(
    #     accelerator="gpu" if torch.cuda.is_available() else None,
    #     devices=1 if torch.cuda.is_available() else None,
    #     max_epochs=args.epochs,
    #     precision=16 if args.mixed_precision and torch.cuda.is_available() else 32,
    #     num_sanity_val_steps=0,
    #     accumulate_grad_batches=args.acc_grad_batches,
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    #     benchmark=False,
    #     deterministic=True
    # )

    ### New - Overfitting
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.epochs,
        precision=16 if args.mixed_precision and torch.cuda.is_available() else 32,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.acc_grad_batches,
        logger=logger,
        callbacks=[checkpoint_callback,MallocTrim(every_steps=1500)],
        #callbacks=[checkpoint_callback,MallocTrim(every_steps=200)],
        benchmark=False,
        deterministic=True
    )
    # print("Before training:")
    # objgraph.show_most_common_types(limit=20)
    # before = objgraph.by_type('Tensor')
    # before = objgraph.by_type('Tensor')
    
    
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    
    # print("After some training:")
    # objgraph.show_most_common_types(limit=20)
    
    # after = objgraph.by_type('Tensor')
    # print(f"Tensors before: {len(before)}, after: {len(after)}")
    # objgraph.show_backrefs(
    # objgraph.by_type('Tensor')[:20],  # Pick 3 sample Tensors
    # max_depth=4,                      # How deep to search the reference chain
    # filename='/home/guests/adrian_delchev/tensor_leak.dot'        # This will output a PNG file
    # )

    # if args.use_pretrained:
    #     trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader, ckpt_path=args.model_dir)
    # else:
    #     trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

