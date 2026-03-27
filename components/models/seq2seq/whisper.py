import math
from typing import Any, Generator

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from components.models.base_model import BaseModel
from components.metrics.base_metric import BaseMetric
from components.tokenizers.base_tokenizer import BaseTokenizer
from components.modules.transformer import TransformerEncoder, TransformerDecoder, TransformerLMHead
from components.modules.embedding import PosNTokEmbedding, PositionalEmbedding
from components.modules.whisper import AudioEncoder


class Whisper(BaseModel):
	''' Whisper-style audio-to-text model with CNN audio encoder + transformer seq2seq.

	Properties:
		1. criterion: nn.CrossEntropyLoss  loss function.
		2. tokenizer: BaseTokenizer | None  tokenizer for BLEU reports and translation.
		3. mel_pos_embedding: PositionalEmbedding  positional embeddings for mel features.
		4. transcript_embeddings: PosNTokEmbedding  token + position embeddings for transcripts.
		5. audio_encoder: AudioEncoder  CNN encoder for mel spectrograms.
		6. encoder: TransformerEncoder  transformer encoder stack.
		7. decoder: TransformerDecoder  transformer decoder stack.
		8. lm_head: TransformerLMHead  projection to vocabulary logits.
	'''

	def __init__(self,
				 n_cnn_layers: int,
				 enc_max_len: int,
				 dec_max_len: int,
				 vocab_size: int,
				 n_blocks: int,
				 n_heads: int,
				 emb_dim: int,
				 dropout: float,
				 bias: bool = False,
				 weight_tying: bool = False,
				 use_grad_ckpt: bool = False,
				 pad_index: int = 3,
				 attention_type: str = 'vanilla',
				 output_attention: bool = False,
				 tokenizer: BaseTokenizer | None = None,
				 optimizer: dict[str, Any] = {},
				 metrics: dict[str, list[BaseMetric]] | None = None):
		''' Initialize the Whisper model.

		Args:
			1. n_cnn_layers: int  number of CNN layers in the audio encoder.
			2. enc_max_len: int  maximum mel spectrogram length.
			3. dec_max_len: int  maximum transcript length.
			4. vocab_size: int  vocabulary size V.
			5. n_blocks: int  number of transformer blocks.
			6. n_heads: int  number of attention heads H.
			7. emb_dim: int  embedding dimension D.
			8. dropout: float  dropout rate.
			9. bias: bool  attention projection bias.
			10. weight_tying: bool  tie decoder embedding to LM head weights.
			11. use_grad_ckpt: bool  gradient checkpointing.
			12. pad_index: int  padding token index.
			13. attention_type: str  attention variant.
			14. output_attention: bool  hook attention weights.
			15. tokenizer: BaseTokenizer | None  for BLEU reports.
			16. optimizer: dict[str, Any]  optimizer config with 'cls' key and kwargs.
			17. metrics: dict[str, list[BaseMetric]] | None  stage-keyed metrics.
		'''
		super().__init__(optimizer=optimizer, metrics=metrics)
		self.save_hyperparameters(ignore=['tokenizer', 'metrics'])
		self.tokenizer = tokenizer
		self.enc_max_len = enc_max_len
		self.dec_max_len = dec_max_len
		self.criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

		# model components
		self.mel_pos_embedding = PositionalEmbedding(emb_dim, enc_max_len)
		self.transcript_embeddings = PosNTokEmbedding(vocab_size, emb_dim, dec_max_len)
		self.audio_encoder = AudioEncoder(n_cnn_layers, 3, emb_dim)
		self.encoder = TransformerEncoder(n_blocks, n_heads, emb_dim, dropout, bias,
										  use_grad_ckpt, attention_type, output_attention)
		self.decoder = TransformerDecoder(n_blocks, n_heads, emb_dim, dropout, bias,
										  use_grad_ckpt, attention_type, output_attention)
		self.lm_head = TransformerLMHead(emb_dim, vocab_size)

		# weight tying
		if weight_tying:
			self.transcript_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight

		# attention hooking
		if output_attention:
			self.attention_weights: dict[str, Tensor] = {}
			self._hook_attention_layers()

	def _hook_attention_layers(self) -> None:
		''' Register forward hooks on attention modules to capture weights. '''
		def hook_fn(module, input, output):
			self.attention_weights[module.tag] = output[1]
		for module in self.modules():
			if hasattr(module, 'tag') and hasattr(module, 'output_attention') and module.output_attention:
				module.register_forward_hook(hook_fn)

	# --- Forward methods ---

	def encoder_forward(self, src: Tensor, src_tok_mask: Tensor) -> Tensor:
		''' Forward pass through CNN + positional embedding + encoder.

		Args:
			1. src: Tensor  [float32, (B, T, M)] mel spectrogram.
			2. src_tok_mask: Tensor  [bool, (B, T//2)] source mask (post-CNN length).
		Returns:
			enc: Tensor  [float32, (B, T//2, D)] encoder output.
		'''
		features = self.audio_encoder(src)  # (B, T//2, D)
		mel_pos = self.mel_pos_embedding(features.size(1))  # (T//2, D)
		return self.encoder(mel_pos + features, src_tok_mask)

	def incremental_forward(self, enc: Tensor, tgt: Tensor,
							src_tok_mask: Tensor, tgt_tok_mask: Tensor) -> Tensor:
		''' Forward pass through decoder + LM head.

		Args:
			1. enc: Tensor  [float32, (B, Ts, D)] encoder output.
			2. tgt: Tensor  [int64, (B, Tt)] target token ids.
			3. src_tok_mask: Tensor  [bool, (B, Ts)] source mask.
			4. tgt_tok_mask: Tensor  [bool, (B, Tt)] target mask.
		Returns:
			logits: Tensor  [float32, (B, Tt, V)] vocabulary logits.
		'''
		dec = self.decoder(enc, self.transcript_embeddings(tgt), src_tok_mask, tgt_tok_mask)
		return self.lm_head(dec)

	def forward(self, src: Tensor, tgt: Tensor,
				src_tok_mask: Tensor, tgt_tok_mask: Tensor) -> Tensor:
		''' Full forward: CNN encode → transformer encode → decode → logits.

		Args:
			1. src: Tensor  [float32, (B, T, M)] mel spectrogram.
			2. tgt: Tensor  [int64, (B, Tt)] target token ids.
			3. src_tok_mask: Tensor  [bool, (B, T//2)] source mask (post-CNN).
			4. tgt_tok_mask: Tensor  [bool, (B, Tt)] target mask.
		Returns:
			logits: Tensor  [float32, (B, Tt, V)] vocabulary logits.
		'''
		features = self.audio_encoder(src)
		mel_pos = self.mel_pos_embedding(features.size(1))
		enc = self.encoder(mel_pos + features, src_tok_mask)
		dec = self.decoder(enc, self.transcript_embeddings(tgt), src_tok_mask, tgt_tok_mask)
		return self.lm_head(dec)

	# --- BaseModel contract ---

	def produce(self, batch: dict[str, Any], requested: set[str]) -> dict[str, Any]:
		''' Produce requested outputs from a batch.

		Args:
			1. batch: dict[str, Any]  keys: x_src, x_tgt, x_src_mask, x_tgt_mask, y_tgt.
			2. requested: set[str]  subset of supports().
		Returns:
			outputs: dict[str, Any]  requested outputs.
		'''
		y_pred = self(batch['x_src'], batch['x_tgt'], batch['x_src_mask'], batch['x_tgt_mask'])
		results: dict[str, Any] = {}
		if 'loss' in requested:
			B, T = batch['y_tgt'].shape
			results['loss'] = self.criterion(y_pred.view(B * T, -1), batch['y_tgt'].reshape(B * T))
		if 'y_pred' in requested:
			results['y_pred'] = y_pred
		if 'y_true' in requested:
			results['y_true'] = batch['y_tgt']
		return results

	def supports(self) -> set[str]:
		''' Return the set of all output keys this model can produce.

		Returns:
			keys: set[str]  {'loss', 'y_pred', 'y_true'}.
		'''
		return {'loss', 'y_pred', 'y_true'}

	# --- BLEU reporting (model-specific) ---

	def on_validation_epoch_end(self) -> None:
		''' Override to add BLEU report after validation. '''
		super().on_validation_epoch_end()
		if self.global_step > 0 and self.tokenizer is not None and hasattr(self, '_val_dataloader'):
			samples_length = 32
			samples = self._val_dataloader.dataset.df.head(samples_length).mel.tolist()
			references = self._val_dataloader.dataset.df.head(samples_length).transcript.tolist()
			print('Generating BLEU report...')
			self._generate_bleu_report(samples, references)

	@torch.inference_mode()
	def _generate_bleu_report(self, samples: list, references: list[str]) -> None:
		''' Generate and log BLEU scores for sample transcriptions. '''
		from nltk.translate.bleu_score import sentence_bleu

		greedy_scores: list[float] = []
		bs_scores: list[float] = []
		greedy_gens: list[str] = []
		bs_gens: list[str] = []
		self.eval()

		for sample, reference in zip(samples, references):
			src = torch.tensor(sample.T, dtype=torch.float, device=self.device)
			ref_tokens = self.tokenizer.tokenize(reference)
			greedy = list(self.translate_with_sampling(src, 1, 2, sampling='argmax', max_new_tokens=128))
			greedy_gens.append(self.tokenizer.detokenize(greedy))
			greedy_scores.append(self._bleu(greedy, ref_tokens))
			bs = [int(b[0]) for b in self.translate_with_beams(src, 1, 2, beam_width=16, max_new_tokens=128)]
			bs_gens.append(self.tokenizer.detokenize(bs))
			bs_scores.append(self._bleu(bs, ref_tokens))

		g_avg = sum(greedy_scores) / len(greedy_scores)
		b_avg = sum(bs_scores) / len(bs_scores)
		self.log_dict({'bleu_greedy': g_avg, 'bleu_bs16': b_avg, 'bleu_ave': (g_avg + b_avg) / 2})
		for ref, g, b in zip(references, greedy_gens, bs_gens):
			print(f'--- Step {self.global_step}')
			print(f'Ref:    {ref}')
			print(f'Greedy: {g}')
			print(f'Beam16: {b}\n')

	def _bleu(self, pred: list[int], ref: list[int]) -> float:
		''' Compute BLEU-4 via detokenize/retokenize normalization. '''
		p = self.tokenizer.tokenize(self.tokenizer.detokenize(pred), add_special_tokens=False)
		r = self.tokenizer.tokenize(self.tokenizer.detokenize(ref), add_special_tokens=False)
		from nltk.translate.bleu_score import sentence_bleu
		return sentence_bleu([r], p)

	# --- Translation methods ---

	@torch.inference_mode()
	def translate_with_sampling(self, src: Tensor, bos_idx: int, eos_idx: int,
								sampling: str = 'multinomial', temperature: float = 1.0,
								max_new_tokens: int = 1000) -> Generator[int, None, None]:
		''' Translate audio to text token-by-token via sampling.

		Args:
			1. src: Tensor  [float32, (T, M)] mel spectrogram (unbatched).
			2. bos_idx: int  BOS token id.
			3. eos_idx: int  EOS token id.
			4. sampling: str  'multinomial' or 'argmax'.
			5. temperature: float  softmax temperature.
			6. max_new_tokens: int  max tokens to generate.
		Yields:
			token: int  next generated token id.
		'''
		assert sampling in ['multinomial', 'argmax']
		self.eval()
		src = src.to(self.device).unsqueeze(0)  # (1, T, M)
		src_mask = torch.ones((1, math.ceil(src.size(1) / 2)), dtype=torch.bool, device=self.device)
		enc = self.encoder_forward(src[:, -self.enc_max_len:], src_mask)
		tgt = torch.tensor([bos_idx], dtype=torch.long, device=self.device).unsqueeze(0)

		for _ in range(max_new_tokens):
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			logits = self.incremental_forward(enc, tgt[:, -self.dec_max_len:], src_mask, tgt_mask)
			logits = logits[:, -1, :] / temperature
			probs = F.softmax(logits, dim=-1)
			idx_next = (torch.multinomial(probs, 1) if sampling == 'multinomial'
						else torch.argmax(probs, dim=-1, keepdim=True))
			tgt = torch.cat((tgt, idx_next), dim=1)
			token = idx_next.item()
			yield token
			if token == eos_idx:
				break

	@torch.inference_mode()
	def translate_with_beams(self, src: Tensor, bos_idx: int, eos_idx: int,
							 beam_width: int = 16, max_new_tokens: int = 1000) -> Generator[np.ndarray, None, None]:
		''' Translate audio to text using beam search.

		Args:
			1. src: Tensor  [float32, (T, M)] mel spectrogram (unbatched).
			2. bos_idx: int  BOS token id.
			3. eos_idx: int  EOS token id.
			4. beam_width: int  number of beams.
			5. max_new_tokens: int  max tokens to generate.
		Yields:
			tokens: np.ndarray  [int64, (beam_width,)] next token for each beam.
		'''
		self.eval()
		src = src.to(self.device).unsqueeze(0)  # (1, T, M)
		src_mask = torch.ones((1, math.ceil(src.size(1) / 2)), dtype=torch.bool, device=self.device)
		enc = self.encoder_forward(src[:, -self.enc_max_len:], src_mask)
		tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=self.device)
		tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
		tgt_probs = torch.ones(beam_width, dtype=torch.float, device=self.device)
		eos_reached = torch.zeros(beam_width, dtype=torch.bool, device=self.device)

		logits = self.incremental_forward(enc, tgt, src_mask, tgt_mask)
		logits = logits[0, 0, :]
		_, topk_idx = torch.topk(logits, k=beam_width, dim=-1)
		eos_reached = eos_reached | (topk_idx == eos_idx)
		yield topk_idx.cpu().numpy()

		enc = enc.repeat(beam_width, 1, 1)
		tgt = torch.concat((tgt.repeat(beam_width, 1), topk_idx.unsqueeze(1)), dim=1)

		for _ in range(max_new_tokens - 1):
			if torch.all(eos_reached):
				break
			tgt = tgt[:, -self.dec_max_len:]
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			logits = self.incremental_forward(enc, tgt, src_mask, tgt_mask)
			logits = logits[:, -1, :]
			next_probs = F.softmax(logits, dim=-1)
			joint_probs = tgt_probs.unsqueeze(1) * next_probs
			topk_probs, topk_flat = torch.topk(joint_probs.flatten(), k=beam_width, dim=-1)
			topk_idx = torch.tensor(
				np.stack(np.unravel_index(topk_flat.cpu().numpy(), joint_probs.shape)),
				device=self.device).T
			eos_reached = eos_reached | (topk_idx[:, 1] == eos_idx)
			tgt = torch.concat((tgt, torch.zeros((beam_width, 1), dtype=torch.long, device=self.device)), dim=1)
			for (b, idx) in topk_idx:
				tgt[b, -1] = idx
				tgt_probs[b] = topk_probs[b]
			yield tgt[:, -1].cpu().numpy()
			if torch.all(tgt == eos_idx):
				break
