import torch
import torch.functional as F
import pytorch_lightning as pl
from model import TransformerModel, TransformerConfig
from config import TrainingConfig

class TransformerModelLN(pl.LightningModule):

    def __init__(self, model_config: TransformerConfig, training_config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters(ignore='compile')
        self.model_config = model_config
        self.learning_rate = training_config.learning_rate
        self.batch_size = training_config.batch_size
        if training_config.compile_model:
            self.model = torch.compile(TransformerModel(model_config))
        else:
            self.model = TransformerModel(model_config)
    
    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        B, T = y.shape
        y_pred = self(x)
        loss = F.cross_entropy(y_pred.view(B * T, -1), y.view(B * T))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B, T = y.shape
        y_pred = self(x)
        loss = F.cross_entropy(y_pred.view(B * T, -1), y.view(B * T))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        B, T = y.shape
        y_pred = self(x)
        loss = F.cross_entropy(y_pred.view(B * T, -1), y.view(B * T))
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def load_inference_model_from_compiled_checkpoint(path, config: TransformerConfig):
        ''' Load a checkpoint containing a compiled model into inference mode. '''
        state_dict = torch.load(path)['state_dict']
        state_dict = { k.replace('._orig_mod', ''):v for k, v in state_dict.items() }
        model = TransformerModel(config)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def generate(self, prompt: str, tokenizer, max_new_tokens: int):
        ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        gen_ids = self._generate(ids, max_new_tokens).squeeze(0).cpu().numpy()
        return tokenizer.decode([int(i) for i in gen_ids])

    @torch.inference_mode()
    def _generate(self, x, max_new_tokens: int):
        # put self into eval mode
        self.eval()
        # x is (B, T) input in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(x[:, -self.seq_len:])
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, idx_next), dim=1) # (B, T)
        # return self to train mode
        self.train()
        return x