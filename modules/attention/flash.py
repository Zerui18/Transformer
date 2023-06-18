from .stock import MultiHeadCrossAttention as MCA, MultiHeadSelfAttention as MSA

class MultiHeadSelfAttention(MSA):
    
    def get_attention_args(self):
        return {
            'enable_math': False,
            'enable_flash': True,
            'enable_mem_efficient': False
        }

class MultiHeadCrossAttention(MCA):

    def get_attention_args(self):
        return {
            'enable_math': False,
            'enable_flash': True,
            'enable_mem_efficient': False
        }