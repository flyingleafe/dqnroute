from .common import *


class PPOActor(SaveableModel):
    def __init__(
            self,
            layers: List[int],
            activation: str = 'relu',
            embedding_dim: int = None,
            embedding_shift: bool = True,
            scope='',
            optimizer='adam'
    ):

        super(PPOActor, self).__init__()

        self.uses_embedding = embedding_dim is not None
        self.embedding_shift = embedding_shift

        # Set input_dim and output_dim size
        if self.uses_embedding:
            if self.embedding_shift:
                input_dim = embedding_dim  # Input = (dst_emb - addr_emb)
                output_dim = embedding_dim  # Output = (delta_step_emb)
            else:
                input_dim = 2 * embedding_dim  # Input = (addr_emb, dst_emb)
                output_dim = embedding_dim  # Output = (predicted_next_emb)
        else:
            input_dim = 2  # Input = (addr_idx, dst_idx)
            output_dim = 1  # Output = (next_addr_idx)

        self.actor = FFNetwork(input_dim, output_dim, layers, activation)
        self.optimizer = get_optimizer(optimizer)(self.actor.parameters())
        self._scope = scope if len(scope) > 0 else None
        self._label = None

    def forward(self, addr, dst):
        if self.uses_embedding:
            addr_ = atleast_dim(addr, 2)
            dst_ = atleast_dim(dst, 2)

            if self.embedding_shift:
                delta_ = dst_ - addr_
                input_tensors = torch.FloatTensor(delta_)
            else:
                input_tensors = [addr_, dst_]
                input_tensors = torch.cat(input_tensors, dim=1)
        else:
            raise NotImplementedError()

        outputs = self.actor(input_tensors)

        if self.uses_embedding and self.embedding_shift:
            outputs += addr_

        return outputs

    def init_xavier(self):
        self.actor.apply(xavier_init)

    def change_label(self, new_label):
        self._label = new_label


class PPOCritic(SaveableModel):
    def __init__(
            self,
            layers: List[int],
            activation: str = 'relu',
            embedding_dim: int = None,
            scope='',
            optimizer='adam'
    ):

        super(PPOCritic, self).__init__()

        self.uses_embedding = embedding_dim is not None

        input_dim = None if not self.uses_embedding else 2 * embedding_dim
        output_dim = 1

        self.critic = FFNetwork(input_dim, output_dim, layers, activation)
        self.optimizer = get_optimizer(optimizer)(self.critic.parameters())
        self._scope = scope if len(scope) > 0 else None

    def forward(self, addr, dst):
        if self.uses_embedding:
            addr_ = atleast_dim(addr, 2)
            dst_ = atleast_dim(dst, 2)

            input_tensors = [addr_, dst_]
        else:
            raise NotImplementedError()

        inputs = torch.cat(input_tensors, dim=1)
        outputs = self.critic(inputs)

        return outputs

    def init_xavier(self):
        self.critic.apply(xavier_init)