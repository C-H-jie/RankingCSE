import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
        

def RCL_loss(cos_sim,N_cos,labels=None):
    

    # RCL第一部分,正例和负例的loss
    diagonal_matrix = torch.diag(cos_sim)
      
    result = cos_sim - diagonal_matrix

    lpair_components = result


    # #对角线取极小值，排除对角线正例-正例的影响 
    # ner_diag = torch.diag(result)
    # ner_diag = torch.where(ner_diag <= 0, -ner_diag, ner_diag)
    # ner_diag.fill_(-1e+12)
    # ner_diag.fill_(0) 

    # print(ner_diag)
    # lpair_components.diagonal().copy_(ner_diag)


    # lpair_components = torch.where(lpair_components <= 0, torch.tensor(-1e+12), lpair_components)

    lpair_components = torch.cat((torch.zeros(1).to(lpair_components.device), lpair_components.view(-1)), dim=0)

    loss = torch.logsumexp(lpair_components,dim=-1) + 0

    return loss



class CustomSortingLoss(nn.Module):
    def __init__(self, hinge_margin=0.5):
        super(CustomSortingLoss, self).__init__()
        self.hinge_margin = hinge_margin


    def logexp(self,x1,x2):
        
        lpair_components = x2 - x1

        lpair_components = torch.where(lpair_components <= self.hinge_margin, torch.tensor(-1e+12), lpair_components)

        lpair_components = torch.cat((torch.zeros(1).to(lpair_components.device), lpair_components.view(-1)), dim=0)
        # loss = torch.logsumexp(lpair_components,dim=-1) + 0
        return lpair_components
        

    def logthirrank(self,p1,p2,p3):
        loss = 0
        loss += torch.logsumexp(self.logexp(p1,p2),dim=-1) + 0
        loss += torch.logsumexp(self.logexp(p1,p3),dim=-1) + 0
        loss += torch.logsumexp(self.logexp(p2,p3),dim=-1) + 0
        return loss/3

    def logfourrank(self,p0,p1,p2,p3):
        loss = 0
        loss += torch.logsumexp(self.logexp(p0,p1),dim=-1) + 0
        loss += torch.logsumexp(self.logexp(p0,p2),dim=-1) + 0
        loss += torch.logsumexp(self.logexp(p0,p3),dim=-1) + 0
        loss += torch.logsumexp(self.logexp(p1,p2),dim=-1) + 0
        loss += torch.logsumexp(self.logexp(p1,p3),dim=-1) + 0
        loss += torch.logsumexp(self.logexp(p2,p3),dim=-1) + 0
        return loss/6

        pass
    def thirrank(self,p1,p2,p3):
        loss = 0
        loss += torch.mean(torch.clamp(self.hinge_margin + p2 - p1, min=0.0))

        loss += torch.mean(torch.clamp(self.hinge_margin + p3 - p1, min=0.0))

        loss += torch.mean(torch.clamp(self.hinge_margin + p3 - p2, min=0.0))

        return loss/3




    def fourrank(self,p0,p1,p2,p3):

        loss = 0
        loss += torch.mean(torch.clamp(self.hinge_margin + p1 - p0, min=0.0))
        loss += torch.mean(torch.clamp(self.hinge_margin + p2 - p0, min=0.0))
        loss += torch.mean(torch.clamp(self.hinge_margin + p3 - p0, min=0.0))

        loss += torch.mean(torch.clamp(self.hinge_margin + p2 - p1, min=0.0))
        loss += torch.mean(torch.clamp(self.hinge_margin + p3 - p1, min=0.0))

        loss += torch.mean(torch.clamp(self.hinge_margin + p3 - p2, min=0.0))

        return loss/6


    def forward(self,p0,p1,p2,p3,p4,p5,p6,p7):

        method = 6

        # p1 p2 p6 p4 logexp
        if method == 1:
            # max_p4_per_row,max_indices = p4.max(dim=1)

            loss1 = 0
            # p1>p2>p4
            loss1 += self.logfourrank(p1,p0,p2,p4)
            # p1>p3>p5
            loss1 += self.logfourrank(p1,p0,p3,p5)
            # p1>p2>p6
            loss1 += self.logfourrank(p1,p0,p2,p6)
            # p1>p3>p6
            loss1 += self.logfourrank(p1,p0,p3,p6)
            
            return loss1 / 4

        elif method == 2:

            loss1 = 0
            # p1>p2>p4
            loss1 += self.logfourrank(p0,p1,p2,p4)
            # p1>p3>p5
            loss1 += self.logfourrank(p0,p1,p3,p5)
            # p1>p2>p6
            loss1 += self.logfourrank(p0,p1,p2,p6)
            # p1>p3>p6
            loss1 += self.logfourrank(p0,p1,p3,p6)
            
            return loss1 / 4

        # p1>p2>p4
        # p1>p3>p5
        elif method == 3:

            loss1 = 0
            loss1 += self.logthirrank(p1,p2,p4)
            loss1 += self.logthirrank(p1,p3,p5)
            loss1 += self.logthirrank(p1,p2,p6)
            loss1 += self.logthirrank(p1,p3,p6)

            return loss1 / 4
        
        # p1 p2 p6 p4
        elif method == 4:

            loss1 = 0
            # p0 > p1
            # loss1 += torch.mean(torch.clamp(self.hinge_margin + p1 - p0, min=0.0)) 
            # loss1 += torch.mean(torch.clamp(self.hinge_margin + p4 - p0, min=0.0)) 
            # loss1 += torch.mean(torch.clamp(self.hinge_margin + p2 - p0, min=0.0)) 

            # p1>p2
            loss1 += torch.mean(torch.clamp(self.hinge_margin + p2 - p1, min=0.0)) 
            # p1>p4
            loss1 += torch.mean(torch.clamp(self.hinge_margin + p4 - p1, min=0.0)) 
            # p1>p3
            loss1 += torch.mean(torch.clamp(self.hinge_margin + p3 - p1, min=0.0)) 
            # p1>p5
            loss1 += torch.mean(torch.clamp(self.hinge_margin + p5 - p1, min=0.0)) 

            # p2>p4
            loss1 += torch.mean(torch.clamp(p4 - p2, min=0.0)) 
            # p3>p5
            loss1 += torch.mean(torch.clamp(p5 - p3, min=0.0)) 


            return loss1 / 6
        
        elif method == 5:
            loss1 = 0
            # p1>p2>p4
            loss1 += self.thirrank(p1,p4,p2)
            # p1>p3>p5
            loss1 += self.thirrank(p1,p5,p3)
            # p1>p2>p6
            loss1 += self.thirrank(p1,p6,p2)
            # p1>p3>p6
            loss1 += self.thirrank(p1,p6,p3)
            
            return loss1 / 4
        

        # 论文最好 25/03/12
        elif method == 6:
            loss1 = 0
            # p1>p2>p4
            loss1 += self.thirrank(p1,p2,p4)
            # p1>p3>p5
            loss1 += self.thirrank(p1,p3,p5)
            # p1>p2>p6
            loss1 += self.thirrank(p1,p2,p6)
            # p1>p3>p6
            loss1 += self.thirrank(p1,p3,p6)
            
            return loss1 / 4

        elif method == 7:
            # 欧氏距离
            loss1 = 0
            # p4>p2>p1
            loss1 += self.thirrank(p4,p2,p1)
            # p5>p3>p1
            loss1 += self.thirrank(p5,p3,p1)
            # p6>p2>p1
            loss1 += self.thirrank(p6,p2,p1)
            # p6>p3>p1
            loss1 += self.thirrank(p6,p3,p1)
            
            return loss1 / 4

        elif method == 8:
            
            loss1 = 0
            # p1>p2>p4
            loss1 += self.fourrank(p0,p1,p2,p4)
            # p1>p3>p5
            loss1 += self.fourrank(p0,p1,p3,p5)
            # p1>p2>p6
            loss1 += self.fourrank(p0,p1,p2,p6)
            # p1>p3>p6
            loss1 += self.fourrank(p0,p1,p3,p6)
            
            return loss1 / 4
        

        elif method == 9:
            
            loss1 = 0
            # p1>p2>p4
            loss1 += self.fourrank(p1,p0,p2,p4)
            # p1>p3>p5
            loss1 += self.fourrank(p1,p0,p3,p5)
            # p1>p2>p6
            loss1 += self.fourrank(p1,p0,p2,p6)
            # p1>p3>p6
            loss1 += self.fourrank(p1,p0,p3,p6)
            
            return loss1 / 4
            

        elif method == 10:
            
            loss1 = 0
            # p1>p2>p4
            loss1 += self.fourrank(p1,p2,p0,p4)
            # p1>p3>p5
            loss1 += self.fourrank(p1,p3,p0,p5)
            # p1>p2>p6
            loss1 += self.fourrank(p1,p2,p0,p6)
            # p1>p3>p6
            loss1 += self.fourrank(p1,p3,p0,p6)
            
            return loss1 / 4
        pass



def RCL_lossV2(cls,z0,z1,z2,z3,lable=None):

    p0 = None
    if z0 is not None:
        p0 = cls.sim(z0,z1)

    p1 = cls.sim(z1,z2)
    p2 = cls.sim(z1,z3)
    p3 = cls.sim(z2,z3)

    p4 = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))
    ner_diag = torch.diag(p4)
    ner_diag.fill_(0) 
    p4.diagonal().copy_(ner_diag)
    p4 = p4.mean(dim=1, keepdim=True).view(-1)
    # p4 = torch.max(p4, dim=1)[0].view(-1)



    p5 =  cls.sim(z2.unsqueeze(1), z2.unsqueeze(0))
    ner_diag = torch.diag(p5)
    ner_diag.fill_(0) 
    p5.diagonal().copy_(ner_diag)
    p5 = p5.mean(dim=1, keepdim=True).view(-1)
    # p5 = torch.max(p5,dim=1)[0].view(-1)

    
    p6 =  cls.sim(z3.unsqueeze(1), z1.unsqueeze(0))
    ner_diag = torch.diag(p6)
    ner_diag.fill_(0) 
    p6.diagonal().copy_(ner_diag)
    p6 = p6.mean(dim=1, keepdim=True).view(-1)
    # p6 = torch.max(p6,dim=1)[0].view(-1)


    p7 =  cls.sim(z3.unsqueeze(1), z2.unsqueeze(0))
    ner_diag = torch.diag(p7)
    ner_diag.fill_(0) 
    p7.diagonal().copy_(ner_diag)
    p7 = p7.mean(dim=1, keepdim=True).view(-1)
    


    

    loss_func = CustomSortingLoss()
    loss = loss_func(p0,p1,p2,p3,p4,p5,p6,p7)

    return loss
    # pass

def RCL_lossV3(cls,z0,z1,z2,z3,lable=None):
    p0 = None
    if z0 is not None:
        p0 = cls.sim2(z0,z1)

    p1 = cls.sim2(z1,z2)
    p2 = cls.sim2(z1,z3)
    p3 = cls.sim2(z2,z3)

    p4 = cls.sim2(z1.unsqueeze(1), z1.unsqueeze(0))
    ner_diag = torch.diag(p4)
    ner_diag.fill_(0) 
    p4.diagonal().copy_(ner_diag)

    # p4 = p4.mean(dim=1, keepdim=True).view(-1)
    # p4 = torch.max(p4, dim=1)[0].view(-1)
    # print(p4)



    p5 =  cls.sim2(z2.unsqueeze(1), z2.unsqueeze(0))
    ner_diag = torch.diag(p5)
    ner_diag.fill_(0) 
    p5.diagonal().copy_(ner_diag)

    # p5 = p5.mean(dim=1, keepdim=True).view(-1)
    # p5 = torch.max(p5, dim=1)[0].view(-1)


    
    p6 =  cls.sim2(z3.unsqueeze(1), z1.unsqueeze(0))
    ner_diag = torch.diag(p6)
    ner_diag.fill_(0) 
    p6.diagonal().copy_(ner_diag)
    # p6 = p6.mean(dim=1, keepdim=True).view(-1)
    # p6 = torch.max(p6, dim=1)[0].view(-1)

    p7 =  cls.sim2(z3.unsqueeze(1), z2.unsqueeze(0))
    ner_diag = torch.diag(p7)
    ner_diag.fill_(0) 
    p7.diagonal().copy_(ner_diag)
    p7 = p7.mean(dim=1, keepdim=True).view(-1)
    


    

    loss_func = CustomSortingLoss()
    loss = loss_func(p0,p1,p2,p3,p4,p5,p6,p7)

    return loss







def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim_ex = torch.cat([cos_sim, z1_z3_cos], 1)

        # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = 0.0
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights


    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    loss1 = loss_fct(cos_sim, labels)
    
    z0 = None
    z1_z2_cos = cls.sim(z1, z2)
    RCL_loss=RCL_lossV2(cls,z0,z1,z2,z3,labels)

    loss = loss1 + RCL_loss * cls.model_args.rcl_lamal

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim_ex,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=True)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
