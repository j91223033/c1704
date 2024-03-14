import logging
import os.path
from typing import List
import json
import csv

import torch
from header import *
import torch.nn.functional as F
from .modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
# from diffusers import StableDiffusionPipeline
from .common.utils import *
from .visual_encoder import VisualEncoder

def create_mapping(input_dim, output_dim, dropout_rate=0.1):
    return nn.Sequential(nn.Linear(input_dim, output_dim//4),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim//4),
                        nn.Linear(output_dim//4, output_dim//2),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim//2),
                        nn.Linear(output_dim//2, output_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim),
                        )


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens: List[List[int]] = None, encounters: int = 1):
        super().__init__()
        self.stop_tokens = stop_tokens  # List of lists of token IDs
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Flatten the 2D list into a 1D list
        input_ids_list = [token for sublist in input_ids.tolist() for token in sublist]

        for stop_seq in self.stop_tokens:
            for i in range(len(input_ids_list) - len(stop_seq) + 1):
                if input_ids_list[i:i+len(stop_seq)] == stop_seq:
                    return True
        return False

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    #multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        #if any(mm_keyword in name for mm_keyword in multimodal_keywords):
        #    continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

def to_one_hot(class_ids_tensor, num_classes=525):
    # Expand the dimensions of class_ids_tensor to (B, N, 1)
    expanded_tensor = torch.tensor(class_ids_tensor).unsqueeze(-1)

    # Create a tensor of zeros with the desired shape (B, N, 525)
    one_hot_tensor = torch.zeros(class_ids_tensor.size(0), class_ids_tensor.size(1), num_classes)

    # Use scatter_ to set the appropriate indices to 1
    one_hot_tensor.scatter_(-1, expanded_tensor, 1)
    
    return torch.tensor(one_hot_tensor)

class LLMAOModel(nn.Module):
    """LoRA for LLaMa model"""
    def __init__(self, **args):
        super(LLMAOModel, self).__init__()
        self.args = args
        self.dataset=args['dataset']
        self.max_length = args['max_length']
        self.device = torch.cuda.current_device()
        self.stage = args['stage']
        self.max_anchor = 12
        self.max_distractor = 5
        print('args max_length', args['max_length'])

        imagebind_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'imagebind_ckpt',
                                           self.args['imagebind_version'])
        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        # TODO
        self.visual_hidden_size = 768
        self.visual_encoder = VisualEncoder()
        checkpoint = torch.load("ckpt/visual_ckpt/best_model.pth")
        # Assuming 'checkpoint' is your loaded checkpoint and 'model' is your current model instance
        obj_feature_mapping_state_dict = {k.replace('obj_feature_mapping.', ''): v 
                                  for k, v in checkpoint['model'].items() 
                                  if k.startswith('obj_feature_mapping')}
        self.visual_encoder.obj_feature_mapping.load_state_dict(obj_feature_mapping_state_dict)

        box_feature_mapping_state_dict = {k.replace('box_feature_mapping.', ''): v 
                                  for k, v in checkpoint['model'].items() 
                                  if k.startswith('box_feature_mapping')}
        self.visual_encoder.box_feature_mapping.load_state_dict(box_feature_mapping_state_dict)
        
        object_encoder_state_dict = {k.replace('object_encoder.', ''): v 
                                  for k, v in checkpoint['model'].items() 
                                  if k.startswith('object_encoder')}
        self.visual_encoder.object_encoder.load_state_dict(object_encoder_state_dict)
        
        post_obj_enc_state_dict = {k.replace('post_obj_enc.', ''): v 
                                  for k, v in checkpoint['model'].items() 
                                  if k.startswith('post_obj_enc')}
        self.visual_encoder.post_obj_enc.load_state_dict(post_obj_enc_state_dict)
        post_object_clf_state_dict = {k.replace('post_object_clf.', ''): v 
                                  for k, v in checkpoint['model'].items() 
                                  if k.startswith('post_object_clf')}            
        self.visual_encoder.post_object_clf.load_state_dict(post_object_clf_state_dict)
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
            
        print('Visual encoder initialized.')

        self.vicuna_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'vicuna_ckpt', self.args['vicuna_version'])
        print(f'Initializing language decoder from {self.vicuna_ckpt_path} ...')

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_ckpt_path)
        if self.stage == 1:
            print("Freezing the LLaMa ...")
            for param in self.llama_model.parameters():
                param.requires_grad = False
            self.llama_model.eval()
        else: #self.args.get('freeze_lm'):
            print("Applying LoRA ...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                bias=self.args['lora_bias'],
                target_modules=find_all_linear_names(self.llama_model)
            )

            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        print('Language decoder initialized.')
        # use the new trained tokenizer
        tokenizer_path = self.vicuna_ckpt_path
        print(f'Initializing tokenizer from {self.vicuna_ckpt_path} ...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.vicuna_ckpt_path)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        # self.llama_tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self._add_pc_token()
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        print('Tokenizer initialized.')
        self.special_token_proj = nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size)
        
        
        self.llama_proj = create_mapping(self.visual_hidden_size, self.llama_model.config.hidden_size)
        if self.stage != 1:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
        self.spatial_enc = create_mapping(2, self.llama_model.config.hidden_size)
        self.spatial_self_attn = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=self.llama_model.config.hidden_size, 
                                                nhead=8, dim_feedforward=self.llama_model.config.hidden_size, activation="gelu"), num_layers=2)
        self.input_embeddings = self.llama_model.get_input_embeddings()
        self.mse_loss = torch.nn.MSELoss()
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)

    def _add_pc_token(self):
        # Add an image token for loss masking (and visualization) purposes.
        self.llama_tokenizer.add_tokens(["<PC>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["</PC>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Anchor>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["</Anchor>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Position>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Target>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Distractor>"])  # add special image token to tokenizer

    def special_tokens_embed(self, batch_size):
        PC_start = self.llama_tokenizer('<PC>', return_tensors="pt", add_special_tokens=False).to(self.device)
        PC_start = self.llama_model.model.model.embed_tokens(PC_start.input_ids).expand(batch_size, -1, -1)
        assert PC_start.shape[1] == 1, PC_start.shape[1]
        PC_end = self.llama_tokenizer('</PC>', return_tensors="pt", add_special_tokens=False).to(self.device)
        PC_end = self.llama_model.model.model.embed_tokens(PC_end.input_ids).expand(batch_size, -1, -1)
        Anchor_start = self.llama_tokenizer('<Anchor>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Anchor_start = self.llama_model.model.model.embed_tokens(Anchor_start.input_ids).expand(batch_size, -1, -1)
        
        Anchor_end = self.llama_tokenizer('</Anchor>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Anchor_end = self.llama_model.model.model.embed_tokens(Anchor_end.input_ids).expand(batch_size, -1, -1)
        Position = self.llama_tokenizer('<Position>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Position = self.llama_model.model.model.embed_tokens(Position.input_ids).expand(batch_size, -1, -1)
        
        Target = self.llama_tokenizer('<Target>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Target = self.llama_model.model.model.embed_tokens(Target.input_ids).expand(batch_size, -1, -1)
        
        Distractor = self.llama_tokenizer('<Distractor>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Distractor = self.llama_model.model.model.embed_tokens(Distractor.input_ids).expand(batch_size, -1, -1)
        
        PC_start = self.special_token_proj(PC_start)
        PC_end = self.special_token_proj(PC_end)
        Anchor_start = self.special_token_proj(Anchor_start)
        Anchor_end = self.special_token_proj(Anchor_end)
        Position = self.special_token_proj(Position)
        Target = self.special_token_proj(Target)
        Distractor = self.special_token_proj(Distractor)
        return PC_start, PC_end, Anchor_start, Anchor_end, Position, Target, Distractor
            
    def prompt_wrap(self, pc_embeds, ids_after_prompt, target_ids, attention_mask, mm_mask=None, padding=True):
        '''
            bos: <bos>
            p_before: ### Human:
            pc_embeds: pc_embeds
            p_after: caption(if any) \n ### Assistant:
        '''
        ids_after_prompt = ids_after_prompt.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2>
        batch_size = ids_after_prompt.shape[0]
        bos = torch.ones([batch_size, 1], dtype=ids_after_prompt.dtype, device=ids_after_prompt.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
        
        # Llama has a bug handling \n###, the tokenizer will parse it as \n + # + ##        
        p_before_tokens = self.llama_tokenizer('Human:', return_tensors="pt", add_special_tokens=False).to(self.device)
        start_token_tensor = torch.tensor([835], device=p_before_tokens.input_ids.device).unsqueeze(0)  # Shape: [1, 1]
        p_before_tokens = torch.cat([start_token_tensor, p_before_tokens.input_ids], dim=1) # "### Human:"
        
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(ids_after_prompt).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
            
        # the input contains point cloud
        if pc_embeds is not None:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, pc_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (1+s1+max_pc_embeds+s2) x embed_dim
            empty_targets = (
                torch.ones([batch_size, 1 + p_before_embeds.size()[1] + pc_embeds.size()[1]], dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1 + 1 + max_anchor)            
            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device) # b, 5            
            atts_prefix = torch.cat([atts_prefix, mm_mask], dim=1)
        else: # only text as input
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (1+s1+s2) x embed_dim
            empty_targets = (
                torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1)
            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device)  # bsz x (1 + s1)
            
        targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + 1 + max_anchor + s2)            
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
        if padding:
            assert inputs_embeds.size()[1] == targets.size()[1]      
            assert attention_mask.size() == targets.size()
        return inputs_embeds, targets, attention_mask
    
    @torch.no_grad()
    def get_pairwise_distance(self, x):
        #torch.set_printoptions(profile="full")
        B, N, _ = x.shape
        relative_positions = x[:, None] - x[:, :, None]
        
        # Obtain the xy distances
        xy_distances = relative_positions[..., :2].norm(dim=-1, keepdim=True) + 1e-5
        # Append the distances to the relative_positions tensor
        relative_positions = torch.cat([relative_positions[..., 2].unsqueeze(-1), xy_distances], dim=-1)      
        relative_positions[..., 0] = scale_to_unit_range(relative_positions[..., 0])
        relative_positions[..., 1] = scale_to_01(relative_positions[..., 1])  
        return relative_positions.to(self.device)

    @torch.no_grad()
    def get_label_embed(self, label, color):
        #print(label)
        B = len(label)
        N = len(label[0])
        pad_token_id = self.llama_tokenizer.pad_token_id
        flat_labels = [lbl if lbl else pad_token_id for sublist in label for lbl in sublist]
        tokenized_flat = [self.llama_tokenizer(str(lbl), add_special_tokens=False).input_ids if lbl != pad_token_id else [pad_token_id] for lbl in flat_labels]
        max_token_num = max(len(tokens) for tokens in tokenized_flat)
        
        tokenized_inputs = [tokens + [pad_token_id] * (max_token_num - len(tokens)) for tokens in tokenized_flat]
        tokenized_inputs = torch.tensor(tokenized_inputs).to(self.device).reshape(B, N, max_token_num)
        if self.stage == 1:
            label_embeds = self.llama_model.model.embed_tokens(tokenized_inputs)
        else:
            label_embeds = self.llama_model.model.model.embed_tokens(tokenized_inputs)
        
        # Averaging the embeddings, ignoring pad tokens
        pad_mask = tokenized_inputs != pad_token_id
        pad_mask = pad_mask.unsqueeze(-1).expand_as(label_embeds)
        label_embeds = label_embeds * pad_mask  # Zero out pad token embeddings
        label_embeds = label_embeds.sum(dim=-2) / (pad_mask.sum(dim=-2) + 1e-8)
        
        color = [tensor for sublist in color for tensor in sublist]
        color = torch.stack(color).view(B, N, 3)
        return label_embeds + self.color_enc(color.to(self.device)) # (B, 1+max_anchor, 4096)
    
    def get_mm_embeds(self, objects_pc, instance_label, box_info, target_id, distractor_ids, anchor_ids, object_ids, color):
        """
        objects_pc: (B, N, P, 6)
        boxes: B, N, 7
        """
        B, N, _ = box_info.shape
        PC_start, PC_end, Anchor_start, Anchor_end, Position, Target, Distractor = self.special_tokens_embed(batch_size=1)
        xyz = box_info[...,:3].to(dtype=next(self.spatial_enc.parameters()).dtype) # (B,N,3)
        relative_positions = self.get_pairwise_distance(xyz) # (B, N, N, 4)
        output_embeds = []
        output_mask = []
        yep = torch.ones([1, 1], dtype=torch.long).to(self.device)
        nope = torch.ones([1, 1], dtype=torch.long).to(self.device)
        objects_embeds, CLASS_LOGITS = self.visual_encoder(objects_pc, box_info)
        for b in range(B):
            # Preparing target
            target_idx = (object_ids[b] == target_id[b]).nonzero(as_tuple=True)[0].item()
            target_obj = {
                "pc": objects_pc[b, target_idx], # (P, 6)
                "color": color[b, target_idx],
                "label": instance_label[b][target_idx],
                "box": box_info[b, target_idx], # (7)
                "idx": target_idx,
                "map": relative_positions[b][target_idx]
            }
            target_embed = objects_embeds[b][target_obj['idx']].unsqueeze(0).unsqueeze(0)
            target_embed = self.llama_proj(target_embed)
            target_class = CLASS_LOGITS[b, target_idx].argmax()
            mask = CLASS_LOGITS[b].argmax(dim=-1) == target_class
            same_class_indices = mask.nonzero().squeeze(-1)

            assert target_embed.shape[0]==1 and target_embed.shape[1]==1, target_embed.shape
            # preparing distractor
            
            distractor_objs = []
            if self.training and self.dataset=='sr3d':
                for distractor_id in distractor_ids[b]:
                    distractor_id_tensor = torch.tensor(distractor_id, device=object_ids.device, dtype=object_ids.dtype)
                    distractor_idx = (object_ids[b] == distractor_id_tensor).nonzero(as_tuple=True)[0].item()
                    distractor_objs.append({
                        "pc": objects_pc[b, distractor_idx], # (P, 6)
                        "color": color[b, distractor_idx],
                        "label": instance_label[b][distractor_idx],
                        "box": box_info[b, distractor_idx], # (7)
                        "idx": distractor_idx,
                        "map": relative_positions[b][distractor_idx]
                        })
                    assert instance_label[b][distractor_idx] == instance_label[b][target_idx], (instance_label[b][distractor_idx], instance_label[b][target_idx])
                target_cate_idx = [target_obj["idx"]]+[distractor_obj['idx'] for distractor_obj in distractor_objs]
            else:
                for distractor_idx in same_class_indices:
                    distractor_objs.append({
                        "pc": objects_pc[b, distractor_idx], # (P, 6)
                        "color": color[b, distractor_idx],
                        "label": instance_label[b][distractor_idx],
                        "box": box_info[b, distractor_idx], # (7)
                        "idx": distractor_idx,
                        "map": relative_positions[b][distractor_idx]
                        })
                    #assert instance_label[b][distractor_idx] == instance_label[b][target_idx], (instance_label[b][distractor_idx], instance_label[b][target_idx])
                target_cate_idx = [target_obj["idx"]]+[distractor_obj['idx'].item() for distractor_obj in distractor_objs]
            target_cate_maps = [target_obj["map"]]+[distractor_obj['map'] for distractor_obj in distractor_objs]
            # preparing anchor
            anchor_objs = []
            for anchor_id in anchor_ids[b]:
                # Find the index of each anchor_id in object_ids
                anchor_id_tensor = torch.tensor(anchor_id, device=object_ids.device, dtype=object_ids.dtype)
                anchor_idx = (object_ids[b] == anchor_id_tensor).nonzero(as_tuple=True)[0].item()
                anchor_objs.append({
                    "pc": objects_pc[b, anchor_idx], # (P, 6)
                    "color": color[b, anchor_idx],
                    "label": instance_label[b][anchor_idx],
                    "box": box_info[b, anchor_idx], # (7)
                    "idx": anchor_idx,
                    "map": relative_positions[b][anchor_idx]
                    })
            valid_indices = [i for i in range(N) if object_ids[b, i] != -1 and 
                                                    object_ids[b, i] not in anchor_ids[b] and                                                    
                                                    instance_label[b][i] not in ['wall', 'floor', 'ceiling', instance_label[b][target_idx]] and
                                                    ambiguous_anchor(target_cate_maps, i)]
            num_anchors = self.max_anchor-len(anchor_objs) 
            padd_num = 0
            if num_anchors < len(valid_indices):
                selected_anchors = random.sample(valid_indices, num_anchors)
                for idx in selected_anchors:
                    anchor_objs.append({
                        "pc": objects_pc[b, idx], # (P, 6)
                        "color": color[b, idx],
                        "label": instance_label[b][idx],
                        "box": box_info[b, idx], # (7)
                        "idx": idx
                    })
            else:
                for idx in valid_indices:
                    anchor_objs.append({
                        "pc": objects_pc[b, idx], # (P, 6)
                        "color": color[b, idx],
                        "label": instance_label[b][idx],
                        "box": box_info[b, idx], # (7)
                        "idx": idx
                    })
                padd_num = self.max_anchor-len(anchor_objs)
            
            anchor_objs = [anchor_objs[i] for i in torch.randperm(len(anchor_objs))]
            
            
            # Anchors
            anchor_embeds = torch.stack([objects_embeds[b][anchor_obj['idx']] for anchor_obj in anchor_objs]).unsqueeze(0)
            anchor_embeds = self.llama_proj(anchor_embeds)
            assert anchor_embeds.shape==(1, len(anchor_objs), self.llama_model.config.hidden_size), anchor_embeds.shape
            anchor_with_token_embeds = []
            anchor_masks = []
            for j in range(len(anchor_objs)):
                # Get the spatial map of this potential anchor
                anchor_sp_map = relative_positions[b][anchor_objs[j]['idx']] # (N, 8)
                # Extract the spatial features of target & distractors form anchor's spatial map
                anchor_sps = anchor_sp_map[target_cate_idx].unsqueeze(0) # (1, 1+len(distractor_objs), 8)+
                assert not torch.isnan(anchor_sps).any(), "anchor_sps contains NaN"
                #anchor_sps[..., 0] = scale_to_unit_range(anchor_sps[..., 0])
                
                assert not torch.isnan(anchor_sps).any(), "anchor_sps contains NaN"
                #anchor_sps[..., 1] = scale_to_01(anchor_sps[..., 1])
                assert not torch.isnan(anchor_sps).any(), "anchor_sps contains NaN"
                assert anchor_sps.shape == (1, 1+len(distractor_objs), 2), anchor_sps.shape
                assert not torch.isnan(anchor_sps).any(), ("anchor_sps contains NaN", anchor_sps, next(self.spatial_enc.parameters()).dtype)
                
                # MLP mapping 8 --> D
                anchor_sps_embed = self.spatial_enc(anchor_sps) # (1, 1+len(distractor_objs), D)
                anchor_sps_embed = torch.cat([anchor_sps_embed[:,:1],torch.max(anchor_sps_embed, dim=1, keepdim=True)[0]], dim=1)
                assert anchor_sps_embed.shape[:2] == (1, 2), anchor_sps_embed.shape
                assert not torch.isnan(anchor_sps_embed).any(), ("anchor_sps_embed contains NaN", anchor_sps)
                assert not torch.isnan(anchor_embeds[:,j:j+1]).any(), "anchor_embeds[:,j:j+1] contains NaN"                
                anchor_with_token_embed = torch.cat([Anchor_start, anchor_embeds[:,j:j+1], Position, anchor_sps_embed, Anchor_end], dim=1) # (1, 1+1+1+1+1, D)
                assert anchor_with_token_embed.shape[:2] == (1, 1+1+1+2+1), anchor_with_token_embed.shape
                assert not torch.isnan(anchor_with_token_embed).any(), "anchor_with_token_embed contains NaN"
                # mask out the padded distractors
                anchor_masks.append(torch.cat([yep]*6, dim=1)) # (1, 4+max_anchor+1)
                assert anchor_masks[0].shape==(1, 6)
                anchor_with_token_embeds.append(anchor_with_token_embed)
            anchor_with_token_embeds = torch.cat(anchor_with_token_embeds, dim=1) # (1, len(anchor_objs)*(4+max_distractor+1), D)
            assert anchor_with_token_embeds.shape == (1, len(anchor_objs)*(6), self.llama_model.config.hidden_size)            
            anchor_masks = torch.cat(anchor_masks, dim=1)
            assert anchor_masks.shape==(1,len(anchor_objs)*(6)), anchor_masks.shape
            
            anchor_with_token_embeds = F.pad(anchor_with_token_embeds, (0,0,0,(self.max_anchor-len(anchor_objs))*(6),0,0), 'constant', 0) # (1, max_anchor*(4+max_distractor+1), D)
            assert not torch.isnan(anchor_with_token_embeds).any(), "anchor_with_token_embeds contains NaN"
            assert anchor_with_token_embeds.shape == (1, self.max_anchor*(6), self.llama_model.config.hidden_size)
            anchor_masks = F.pad(anchor_masks, (0, (self.max_anchor-len(anchor_objs))*(6)), "constant", 0)
            assert anchor_masks.shape==(1,self.max_anchor*(6)), anchor_masks.shape
                        
            # single batch of the visual input embedding
            single_embeds = torch.cat([PC_start, Target, target_embed, anchor_with_token_embeds]+[PC_end], dim=1)
            assert single_embeds.shape[:2]== (1, 1+1+1+self.max_anchor*(6)+1), single_embeds.shape
            output_embeds.append(single_embeds)
            single_mask = torch.cat([yep, yep, yep, anchor_masks, yep], dim=1)
            assert single_mask.shape == (1, 1+1+1+self.max_anchor*(6)+1), single_mask.shape
            output_mask.append(single_mask)
        output_embeds = torch.cat(output_embeds, dim=0)
        assert not torch.isnan(output_embeds).any(), "output_embeds contains NaN"
        assert output_embeds.shape == (B, 1+1+1+self.max_anchor*(6)+1, self.llama_model.config.hidden_size), output_embeds.shape
        output_mask = torch.cat(output_mask, dim=0)
        assert output_mask.shape == (B, output_embeds.shape[1]), output_mask.shape
        assert not torch.isnan(output_mask).any(), "output_mask contains NaN"
        return output_embeds, output_mask
        
    def get_object_label_loss(self, pc_embeds, label_embeds, mse_w=0.5, cos_w=1):
        mse_loss_value = self.mse_loss(pc_embeds, label_embeds)
        pc_embeds_normalized = F.normalize(pc_embeds, dim=-1, eps=1e-8)
        label_embeds_normalized = F.normalize(label_embeds, dim=-1, eps=1e-8)
        cos_loss_value = 1 - self.cos_loss(pc_embeds_normalized, label_embeds_normalized)
        assert not torch.isnan(cos_loss_value).any(), "cos_loss_value contains NaN"
        print("mse_loss: ", mse_loss_value)
        print("cos_loss: ", cos_loss_value.mean())
        return mse_w*mse_loss_value + cos_w*cos_loss_value.mean()
       
    def _training_stage_1(self, inputs_from_2_ds):
        inputs = inputs_from_2_ds["dataset1"]
        input2 = inputs_from_2_ds["dataset2"]
        pc_embeds, _ = self.visual_encoder(inputs['objects'], inputs['box_info'])
        pc_embeds = self.llama_proj(pc_embeds)  # B x (1+max_anchors_n) x llama_size 
        label_embeds = self.get_label_embed(inputs['instance_label']).detach()
        object_label_loss = self.get_object_label_loss(pc_embeds, label_embeds)
        
        return object_label_loss, 0
    
    def _training_stage_2(self, inputs_from_2_ds):
        inputs = inputs_from_2_ds["dataset1"]
        if self.dataset=="sr3d":
            anchor_ids = inputs['anchor_ids']
            distractor_ids = inputs['distractor_ids']
        else:
            anchor_ids =  select_random_anchors(inputs['object_ids'], inputs['instance_label'], self.max_anchor)
            distractor_ids = None
        mm_embeds, mm_mask = self.get_mm_embeds(inputs['objects'], inputs['instance_label'], inputs['box_info'], inputs['target_id'], distractor_ids, anchor_ids, inputs['object_ids'], inputs['objects_color'])
        
        input_ids_after_prompt, target_ids, attention_mask = process_batch_stage_2(tokenizer=self.llama_tokenizer,
                                                                      batch_of_captions=inputs['utterance'],
                                                                      max_tgt_len=self.max_length,
                                                                      prompt=self.args['prompt']) # 'generate a caption'        
        # print(input_ids)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(mm_embeds, input_ids_after_prompt, target_ids, attention_mask, mm_mask)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )
        loss = outputs.loss
        assert not torch.isnan(loss).any(), "loss contains NaN"
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        return loss, gen_acc
    
    def forward(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = None

        if self.stage == 1:
            loss, gen_acc = self._training_stage_1(inputs)
        elif self.stage == 2:
            loss, gen_acc = self._training_stage_2(inputs)
        else:
            raise NotImplementedError(f"stage {self.stage} is not implemented, now it only support [1, 2, 3]")

        return loss, gen_acc, mse_loss

    def get_random_ids(self, object_ids, instance_labels):
        """
        object_ids: (B, N)
        instance_label: list(B, N)
        """
        # 1. randomly select a object in the room as target, but this object need to have at least one distractor
        target_id, target_index, target_instance_label = select_random_target_with_label_constraint(object_ids, instance_labels)
        # 2. if the need_distractor is true then choose 1~5 distractor which belongs to the same category
        distractor_ids = gather_same_instance_indices(object_ids, target_index, instance_labels)            
        # 3. randomly choose 1~3 anchor, and i think at least the category should be different, also if need_distractor ==True anchor should be one
        anchor_ids = select_random_anchors(object_ids, instance_labels, self.max_anchor)        
        return target_id, anchor_ids, distractor_ids, target_instance_label
   
    def evaluate(self, inputs, output_file):
        target_id, anchor_ids, distractor_ids, target_instance_label = self.get_random_ids(inputs['object_ids'], inputs['instance_label'])
        mm_embeds, mm_mask = self.get_mm_embeds(inputs['objects'], inputs['instance_label'], inputs['box_info'], target_id, distractor_ids, anchor_ids, inputs['object_ids'], inputs['objects_color'])
        input_ids_after_prompt, target_ids, attention_mask = process_batch_stage_2(tokenizer=self.llama_tokenizer,
                                                                      batch_of_captions=inputs['utterance'],
                                                                      max_tgt_len=self.max_length,
                                                                      prompt=self.args['prompt'],
                                                                      padding=False) # 'generate a caption'        
        # print(input_ids)
        inputs_embeds, _, attention_mask = self.prompt_wrap(mm_embeds, input_ids_after_prompt, target_ids, attention_mask, mm_mask, padding=False)
        stops_id = [[13, 2277, 29937], [835]]
        stopping_criteria_instance = StoppingCriteriaSub(stop_tokens=stops_id, encounters=1)
        stopping_criteria = StoppingCriteriaList([stopping_criteria_instance])
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=512,
            top_p=0.1,
            temperature=0.1,
            # repeat_pen,
            do_sample=False,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True
        )
        caption = self.llama_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        
        print(caption)
        print(outputs.sequences)
        append_to_csv(inputs['scan_id'], target_id, anchor_ids, distractor_ids, caption, target_instance_label, file_path=output_file)

def process_and_append_to_csv(target_ids, anchor_ids, distractor_ids, utterances, stimulus_id, csv_file_path):
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Iterate over each item in the batch
        for i in range(len(utterances)):
            target_id = target_ids[i].item()
            anchors = [aid.item() for aid in anchor_ids[i] if aid.numel() == 1 and aid.item() != -1]
            distractors = [did.item() for did in distractor_ids[i] if did.numel() == 1 and did.item() != -1]
            utterance = utterances[i].replace('\n', '').replace('###', '')

            # Write to CSV
            writer.writerow([target_id, anchors, distractors, utterance, stimulus_id[i]])