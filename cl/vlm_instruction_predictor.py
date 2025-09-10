"""
Vision-Language Model for Instruction Prediction
输入: goal文本 + instr(high level)文本 + 第一张图片
输出: instr(high level)文本

使用 LLaMA 作为文本编码器，CLIP 作为图像编码器
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image

# Import existing components
from models.model.seq2seq import Module as BaseModule
from models.nn.llama_encoder import LlamaTextEncoder, LlamaEncoderConfig
from models.nn.clip import _CLIPViTL14_336
import models.nn.vnn as vnn

try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False


class VLMInstructionPredictor(nn.Module):
    """
    Vision-Language Model for predicting high-level instructions
    """
    
    def __init__(self, args, vocab):
        super().__init__()
        
        self.args = args
        self.vocab = vocab # word <<pad>> <<seg>> <<goal>>  action:<<stop>>
        self.device = torch.device('cuda' if args.gpu else 'cpu')
        
        # Initialize text encoder (LLaMA)
        self._init_text_encoder()
        self.tokenizer = self.text_encoder.tokenizer
        
        # Initialize image encoder (CLIP)
        self._init_image_encoder()
        
        # Fusion and prediction layers
        self._init_fusion_layers()
        
        # Output projection to instruction vocabulary
        self._init_output_layers()
        
    def _init_text_encoder(self):
        """Initialize LLaMA text encoder"""
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers not available. Please install transformers package.")
            
        llama_model_name = getattr(self.args, 'llama_model_name', 'initial_model/llama')
        llama_dtype = getattr(self.args, 'llama_dtype', 'float16')
        llama_max_length = int(getattr(self.args, 'llama_max_length', 512))
        
        dtype = {
            'float16': torch.float16, 
            'bfloat16': torch.bfloat16, 
            'float32': torch.float32
        }.get(str(llama_dtype).lower(), torch.float16)
        
        config = LlamaEncoderConfig(
            model_name_or_path=llama_model_name,
            device='cuda' if self.args.gpu else 'cpu',
            dtype=dtype,
            max_length=llama_max_length
        )
        
        self.text_encoder = LlamaTextEncoder(config)
        self.text_hidden_size = getattr(self.text_encoder.model.config, 'hidden_size', 4096)
        
    def _init_image_encoder(self):
        """Initialize CLIP image encoder"""
        if not _HAS_CLIP:
            raise ImportError("CLIP not available. Please install openai-clip.")
            
        # Use existing CLIP wrapper
        self.image_encoder = _CLIPViTL14_336(self.args, eval=True, share_memory=False)
        self.image_hidden_size = self.image_encoder.output_channels  # 768 for ViT-L/14
        
    def _init_fusion_layers(self):
        """Initialize fusion layers for combining text and image features"""
        # Project text and image features to common dimension
        self.fusion_dim = getattr(self.args, 'fusion_dim', 512)
        
        self.text_projection = nn.Linear(self.text_hidden_size, self.fusion_dim)
        self.image_projection = nn.Linear(self.image_hidden_size, self.fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=getattr(self.args, 'num_attention_heads', 8),
            dropout=getattr(self.args, 'attention_dropout', 0.1),
            batch_first=True
        )
        
        # Fusion transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=getattr(self.args, 'num_attention_heads', 8),
            dim_feedforward=self.fusion_dim * 4,
            dropout=getattr(self.args, 'fusion_dropout', 0.1),
            batch_first=True
        )
        
        self.fusion_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=getattr(self.args, 'num_fusion_layers', 2)
        )
        
    def _init_output_layers(self):
        """Initialize output layers for instruction generation"""
        # Get instruction vocabulary size
        self.instr_vocab_size = len(self.vocab.get('word', {}))
        
        # Instruction decoder (LSTM-based)
        self.decoder_hidden_size = getattr(self.args, 'decoder_hidden_size', 512)
        
        # Embedding layer for instruction tokens
        self.instr_embedding = nn.Embedding(self.instr_vocab_size, self.decoder_hidden_size)
        
        # LSTM decoder
        # self.decoder_lstm = nn.LSTM(
        #     input_size=self.decoder_hidden_size + self.fusion_dim,  # instruction embedding + context
        #     hidden_size=self.decoder_hidden_size,
        #     num_layers=getattr(self.args, 'num_decoder_layers', 2),
        #     dropout=getattr(self.args, 'decoder_dropout', 0.1),
        #     batch_first=True
        # )
        
        # Output projection
        self.output_projection = nn.Linear(self.decoder_hidden_size, self.instr_vocab_size)
        
        # Attention mechanism for focusing on fused features
        self.decoder_attention = nn.MultiheadAttention(
            embed_dim=self.decoder_hidden_size,
            num_heads=getattr(self.args, 'num_attention_heads', 8),
            dropout=getattr(self.args, 'attention_dropout', 0.1),
            batch_first=True
        )
        
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode goal and instruction texts using LLaMA
        
        Args:
            texts: List of descriptions
            
        Returns:
            Encoded text features [B, T, H]
        """

        text_features = self.text_encoder.encode(texts) # [B, T, H]

        dtype = self.text_projection.weight.dtype
        text_features = text_features.to(dtype)
        # Project to fusion dimension
        text_features = self.text_projection(text_features)  # [B, T, fusion_dim]
        return text_features
    
    def encode_image(self, images) -> torch.Tensor:
        """
        Encode images using CLIP
        
        Args:
            images: Batch of images [B, 3, 336, 336] or list of PIL Images
            
        Returns:
            Encoded image features [B, H]
        """
        if isinstance(images, list):
            # Handle PIL Images
            image_features = self.image_encoder.encode_image(images)
        else:
            # Handle tensor input
            image_features = self.image_encoder.extract(images)
        
        dtype = self.image_projection.weight.dtype
        image_features = image_features.to(dtype)
        # Project to fusion dimension
        image_features = self.image_projection(image_features)  # [B, fusion_dim]
        
        return image_features
    
    def fuse_modalities(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse text and image features using cross-modal attention
        
        Args:
            text_features: [B, T_text, fusion_dim]
            image_features: [B, fusion_dim]
            
        Returns:
            Fused features [B, T_fused, fusion_dim]
        """
        batch_size = text_features.size(0)
        
        # Expand image features to match text sequence dimension
        image_features = image_features.unsqueeze(1)  # [B, 1, fusion_dim]
        
        # Concatenate text and image features
        fused_features = torch.cat([text_features, image_features], dim=1)  # [B, T_text+1, fusion_dim]
        
        # Apply cross-modal attention and fusion transformer
        fused_features = self.fusion_transformer(fused_features)
        
        return fused_features

    def decode_instruction(self, fused_features: torch.Tensor, target_instrs: Optional[List[str]] = None) -> torch.Tensor:
        """
        Decode instruction sequence from fused features
        
        Args:
            fused_features: [B, T_fused, fusion_dim]
            target_instrs: [B, T_instr] target instruction texts
            
        Returns:
            Instruction logits [B, T_instr, vocab_size]
        """
        batch_size = fused_features.size(0)
        
        if target_instrs is not None:
            tokenized_instr = self.tokenizer(
                target_instrs,
                padding=True,
                truncation=True,
                max_length=getattr(self.args, 'max_decode_length', 100),
                return_tensors="pt",
            ).input_ids.to(self.device) # [B, T_instr]
            decoder_input_ids = tokenized_instr.input_ids  # [B, T_instr]
            decoder_attention_mask = tokenized_instr.attention_mask  # [B, T_instr]
            labels = decoder_input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

            decoder_output, _ = self.decoder_attention(decoder_input_ids)  # [B, T_instr, hidden]

            # Project to vocabulary
            logits = self.output_projection(decoder_output)  # [B, T_instr, vocab_size]
            
        else:
            # Inference mode: autoregressive generation
            max_len = getattr(self.args, 'max_decode_length', 50)
            
            # Initialize with start token
            start_token = self.vocab['word'].word2index('<start>', train=False) if '<start>' in self.vocab['word'].index2word else 0
            current_token = torch.full((batch_size, 1), start_token, device=self.device)
            
            logits_list = []
            hidden = None
            
            for t in range(max_len):
                # Embed current token
                instr_embed = self.instr_embedding(current_token)  # [B, 1, hidden]
                
                # Get context from fused features
                context = fused_features.mean(dim=1, keepdim=True)  # [B, 1, fusion_dim]
                
                # Combine embedding with context
                decoder_input = torch.cat([instr_embed, context], dim=-1)  # [B, 1, hidden+fusion_dim]
                
                # LSTM step
                decoder_output, hidden = self.decoder_attention(decoder_input, hidden)  # [B, 1, hidden]
                
                # Project to vocabulary
                step_logits = self.output_projection(decoder_output)  # [B, 1, vocab_size]
                logits_list.append(step_logits)
                
                # Get next token (greedy decoding)
                current_token = step_logits.argmax(dim=-1)  # [B, 1]
                
                # Check for stop condition
                stop_token = self.vocab['word'].word2index('<stop>', train=False) if '<stop>' in self.vocab['word'].index2word else 1
                if (current_token == stop_token).all():
                    break
            
            logits = torch.cat(logits_list, dim=1)  # [B, T_generated, vocab_size]
        
        return logits
    
    def forward(self, 
                goal_texts: List[str], 
                images, 
                target_instrs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            goal_texts: List of goal descriptions
            images: Batch of images
            target_instrs: Target instruction
            
        Returns:
            Dictionary containing logits and other outputs
        """
        # Encode modalities
        text_features = self.encode_text(goal_texts) #[B, T_text, H] [16,11,4096]
        image_features = self.encode_image(images)
        
        # Fuse modalities
        fused_features = self.fuse_modalities(text_features, image_features)
        
        # Decode instruction
        logits = self.decode_instruction(fused_features, target_instrs)
        
        return {
            'logits': logits,
            'text_features': text_features,
            'image_features': image_features,
            'fused_features': fused_features
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for instruction prediction
        """
        logits = outputs['logits']  # [B, T, vocab_size]
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # [B*T, vocab_size]
        targets_flat = targets.view(-1)  # [B*T]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)  # Assuming 0 is padding token
        
        return loss
    
    def generate_instruction(self, 
                           goal_text: str, 
                           image) -> List[str]:
        """
        Generate instruction sequence for a single input
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward([goal_text], [image])
            logits = outputs['logits']  # [1, T, vocab_size]
            
            # Convert logits to tokens
            predicted_tokens = logits.argmax(dim=-1).squeeze(0)  # [T]
            
            # Convert tokens to words
            words = []
            for token in predicted_tokens:
                word = self.vocab['word'].index2word.get(token.item(), '<unk>')
                if word in ['<stop>', '<pad>']:
                    break
                words.append(word)
            
            return words


class Module(nn.Module):
    """
    Wrapper class to integrate with existing training framework
    """
    
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab

        # Initialize VLM model
        self.vlm_model = VLMInstructionPredictor(args, vocab)
        
        # Move to device
        if args.gpu:
            self.vlm_model = self.vlm_model.to(torch.device('cuda'))
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _get_image_path(self,image_name, task_type,split,root):
        """Get full image path"""
        # print(self.args.data)
        # print(split)
        # print(task_id)
        # print(image_name)
        # print(os.path.join(self.args.data, 'images', split, task_id,image_name))
        # data/json_feat_2.1.0/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T20190907_174127_043461
        root = root.replace('data/json_feat_2.1.0/',"")
        image_name = image_name.replace('.png','.jpg')
        # data/full/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T20190907_174127_043461/raw_images/1.png
        return os.path.join('data', 'full', split, root,'raw_images',image_name)

    def featurize(self, batch):
        """
        Extract features from batch data
        
        Expected batch format:
        [
            (traj_data, swapColor),
            ...
        ]
        """
        feat = {}
        
        goal_texts = []
        target_instrs = []
        images = []

        #only the first ann
        #for each traj , 1 goal, multiple instr, multiple images. each image corresponds to one instr
        for traj_data, swapColor in batch:
            goal_ann = traj_data['turk_annotations']['anns'][0]['task_desc'] # eg. Pick up the alarm clock and turn on the lamp.
            
            # Extract high-level instruction,
            high_descs = traj_data['turk_annotations']['anns'][0]['high_descs']
            image_infos = traj_data['images']
            id = 0

            #find the high level instruction corresponding to the first image
            for image_info in image_infos:
                if image_info['high_idx'] == id:
                    target_instrs.append(high_descs[id])
                    images.append(Image.open(self._get_image_path(image_name=image_info['image_name'],task_type=traj_data['task_type'],split = traj_data['split'],root=traj_data['root'])).convert('RGB'))
                    goal_texts.append(goal_ann)
                    
                    id += 1

        feat['goal_texts'] = goal_texts
        feat['images'] = images
        feat['target_instrs'] = target_instrs
        
        return feat
    
    def forward(self, feat, max_decode=300):
        """Forward pass"""
        return self.vlm_model(
            feat['goal_texts'],
            feat['images'],
            feat['target_instrs']
        )
    
    def compute_loss(self, out, batch, feat):
        """Compute loss"""
        targets = feat['target_instrs']
        if self.args.gpu:
            targets = targets.to(torch.device('cuda'))
        
        loss = self.vlm_model.compute_loss(out, targets)
        return {'total_loss': loss}
    
    # def extract_preds(self, out, batch, feat, clean_special_tokens=True):
    #     """Extract predictions for evaluation"""
    #     logits = out['logits']
    #     predicted_tokens = logits.argmax(dim=-1)  # [B, T]
        
    #     preds = []
    #     for i, tokens in enumerate(predicted_tokens):
    #         words = []
    #         for token in tokens:
    #             word = self.vocab['word'].index2word.get(token.item(), '<unk>')
    #             if word in ['<stop>', '<pad>'] and clean_special_tokens:
    #                 break
    #             words.append(word)
    #         preds.append(' '.join(words))
        
    #     return {'predicted_instructions': preds}
    
    @classmethod
    def load(cls, fsave):
        """Load model from checkpoint"""
        save = torch.load(fsave, map_location='cpu')
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if 'optim' in save:
            optimizer.load_state_dict(save['optim'])
            
        return model, optimizer
