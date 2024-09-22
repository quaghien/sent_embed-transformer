import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import copy
import numpy as np
import random
from safetensors.torch import load_file

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import MT5Config, AutoTokenizer
from transformers.models.mt5.modeling_mt5 import MT5Stack
from transformers import MT5PreTrainedModel, MT5EncoderModel
import time

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def get_random_state():
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }

def set_random_state(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])

class EmbedFusion(nn.Module):
    def __init__(self, fusion_type='attention'):  #fusion_type= 'attention' / 'average'
        super(EmbedFusion, self).__init__()
        self.fusion_type = fusion_type

        # Lưu trữ trạng thái ngẫu nhiên hiện tại
        random_state = get_random_state()

        # Đặt seed ngẫu nhiên mới
        set_seed(42)

        # Khởi tạo lớp Linear với trọng số cố định
        self.dense = nn.Linear(1536, 768)

        # Khôi phục trạng thái ngẫu nhiên ban đầu
        set_random_state(random_state)

    def forward(self, sent_embed, word_embed):
        # Ensure input dimensions are as expected
        word_embeddings = word_embed
        sentence_embedding = sent_embed

        if self.fusion_type == 'attention':
            # Prepare K and V
            K_transposed = word_embeddings.transpose(1, 2)  # K: [1, 768, L]
            V = word_embeddings  # V: [1, L, 768]

            # Prepare Q
            Q = sentence_embedding

            # Compute attention scores: (Q * K^T) / sqrt(d)
            attention_scores = torch.bmm(Q, K_transposed) / torch.sqrt(torch.tensor(sentence_embedding.size(-1), dtype=torch.float32))

            # Compute softmax over the last dimension
            attention_weights = F.softmax(attention_scores, dim=-1)

            weighted_values = attention_weights.transpose(1, 2) * V

            # Expand sentence_embedding to match the dimensions for concatenation
            # Adjust e to match the sentence_embedding dimensions
            target_length = word_embeddings.size(1)
            expanded_sentence_embedding = sentence_embedding.repeat(1, target_length, 1)

            # Concatenate sentence embedding with weighted word embeddings
            concat_output = torch.cat((expanded_sentence_embedding, weighted_values), dim=2)

        elif self.fusion_type == 'average':
            # Calculate the average of word embeddings
            e = torch.mean(word_embeddings, dim=1, keepdim=True)

            # Adjust e to match the sentence_embedding dimensions
            target_length = word_embeddings.size(1)

            # Concatenate sentence embedding with expanded average word embeddings
            concat_output = torch.cat((sentence_embedding, e), dim=2)
            concat_output = concat_output.expand(-1, target_length, -1)

        fused_output = self.dense(concat_output)
        return fused_output

class MT5DecoderModel(MT5PreTrainedModel):
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        def forward(
            self,
            input_ids = None,
            attention_mask = None,
            decoder_input_ids = None,
            decoder_attention_mask = None,
            head_mask = None,
            decoder_head_mask = None,
            cross_attn_head_mask = None,
            encoder_outputs = None,
            past_key_values = None,
            inputs_embeds = None,
            decoder_inputs_embeds = None,
            labels = None,
            use_cache = False,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = True,
            hidden_states = None,
        ):
            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]

            lm_logits = self.lm_head(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # move labels to correct device to enable PP
                labels = labels.to(lm_logits.device)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
                return ((loss,) + output) if loss is not None else output

            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

class SMT5Model(nn.Module):
    def __init__(self):
        super(SMT5Model, self).__init__()
        self.embed_fusion = EmbedFusion(fusion_type='attention')
        self.sent_model = MT5EncoderModel.from_pretrained('wanhin/msim-mt5-luat-atien')
        self.encoder_model = MT5EncoderModel.from_pretrained('google/mt5-base')
        self.decoder_model = MT5DecoderModel.from_pretrained('google/mt5-base')
        self.softmax = nn.Softmax(dim=-1)
        self.config = MT5Config.from_pretrained('google/mt5-base')
        
        self.remove_shared()
        
        self.freeze_sent_model()

    def remove_shared(self):
        """
        Xóa thuộc tính 'shared' của một mô hình nếu nó tồn tại.

        Args:
            model: Mô hình cần xóa thuộc tính 'shared'.
        """
        if hasattr(self.encoder_model, 'shared'):
            del self.encoder_model.shared
        if hasattr(self.decoder_model, 'shared'):
            del self.decoder_model.shared
        if hasattr(self.sent_model, 'shared'):
            del self.sent_model.shared

    def freeze_sent_model(self):
        """
        Đóng băng tất cả các tham số của mô hình mst5.
        """

        # Đóng băng các tham số
        for param in self.sent_model.parameters():
            param.requires_grad = False

        # Kiểm tra xem tất cả các tham số đã được đóng băng chưa
        for param in self.sent_model.parameters():
            assert param.requires_grad == False, "Có tham số chưa được đóng băng"


    def forward(self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # start_time = time.time()
        # with torch.no_grad():
        #     sent_output = self.sent_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        # print(f"Sent model forward pass time: {time.time() - start_time:.4f} seconds")
        
        # start_time = time.time()
        # sent_embed = sent_output.last_hidden_state.mean(dim=1, keepdim=True)  # Average pooling
        # print(f"Sent embed calculation time: {time.time() - start_time:.4f} seconds")
        
        # start_time = time.time()
        # encoder_outputs = self.encoder_model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        # print(f"Encoder model forward pass time: {time.time() - start_time:.4f} seconds")
        
        # start_time = time.time()
        # word_embed = encoder_outputs.last_hidden_state
        # print(f"Word embed extraction time: {time.time() - start_time:.4f} seconds")
        
        # start_time = time.time()
        # embed_fusion = self.embed_fusion(sent_embed, word_embed)
        # print(f"Embed fusion time: {time.time() - start_time:.4f} seconds")
        
        # start_time = time.time()
        # decoder_outputs = self.decoder_model.decoder(inputs_embeds=embed_fusion, return_dict=return_dict)
        # print(f"Decoder model forward pass time: {time.time() - start_time:.4f} seconds")
        
        # start_time = time.time()
        # sequence_output = decoder_outputs.last_hidden_state
        # print(f"Sequence output extraction time: {time.time() - start_time:.4f} seconds")
        
        # start_time = time.time()
        # lm_logits = self.decoder_model.lm_head(sequence_output)
        # print(f"Logits calculation time: {time.time() - start_time:.4f} seconds")

        # loss = None
        # if labels is not None:
        #     start_time = time.time()
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     labels = labels.to(lm_logits.device)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     print(f"Loss calculation time: {time.time() - start_time:.4f} seconds")
        
        
        with torch.no_grad():
            sent_output = self.sent_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
            sent_embed = sent_output.last_hidden_state.mean(dim=1, keepdim=True)  # Average pooling

        encoder_outputs = self.encoder_model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        word_embed = encoder_outputs.last_hidden_state

        embed_fusion = self.embed_fusion(sent_embed, word_embed)

        decoder_outputs = self.decoder_model.decoder(inputs_embeds = embed_fusion, return_dict=return_dict)
        sequence_output = decoder_outputs.last_hidden_state

        lm_logits = self.decoder_model.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(self, sentences=None, max_length=512, top_k=3, early_stopping=True):       #max_length_train=512

        tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
        inputs = tokenizer(sentences,padding="max_length",truncation=True,max_length=max_length, return_tensors="pt")

        sent_output = self.sent_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True)
        sent_embed = sent_output.last_hidden_state.mean(dim=1, keepdim=True)  # Average pooling

        encoder_outputs = self.encoder_model.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True)
        word_embed = encoder_outputs.last_hidden_state

        embed_fusion = self.embed_fusion(sent_embed, word_embed)

        decoder_outputs = self.decoder_model.decoder(inputs_embeds = embed_fusion, return_dict=True)
        sequence_output = decoder_outputs.last_hidden_state

        lm_logits = self.decoder_model.lm_head(sequence_output)

        # Apply softmax to get probabilities
        probs = self.softmax(lm_logits[:, -1, :])

        # Sampling from the probability distribution
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            next_token = torch.multinomial(top_k_probs, 1)
            next_token = torch.gather(top_k_indices, -1, next_token)
        else:
            next_token = torch.multinomial(probs, 1)

        # Check for early stopping condition
        if early_stopping and (next_token == tokenizer.eos_token_id).any():
            return tokenizer.decode(next_token.squeeze().tolist(), skip_special_tokens=True)

        return tokenizer.decode(next_token.squeeze().tolist(), skip_special_tokens=True)

    def load_weights(self, safetensors_path):
        """
        Tải state_dict từ tệp .safetensors và nạp vào mô hình.

        Args:
            custom_model: Mô hình để nạp trọng số.
            safetensors_path (str): Đường dẫn đến tệp .safetensors chứa state_dict.
        """
        # Tải state_dict từ tệp .safetensors
        state_dict = load_file(safetensors_path)
        
        # Nạp state_dict vào mô hình
        self.load_state_dict(state_dict)
