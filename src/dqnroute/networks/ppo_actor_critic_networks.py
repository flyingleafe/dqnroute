from .common import *


class PPOActor(SaveableModel):
    def __init__(
            self,
            layers: List[int],
            activation: str = 'relu',
            embedding_dim: int = None,
            scope='',
            optimizer='adam'
    ):

        super(PPOActor, self).__init__()

        self.uses_embedding = embedding_dim is not None

        input_dim = 2 if not self.uses_embedding else 2 * embedding_dim
        output_dim = None if not self.uses_embedding else embedding_dim

        self.actor = FFNetwork(input_dim, output_dim, layers, activation)
        self.optimizer = get_optimizer(optimizer)(self.actor.parameters())
        self._scope = scope if len(scope) > 0 else None

    def forward(self, addr, dst):
        if self.uses_embedding:
            addr_ = atleast_dim(addr, 2)
            dst_ = atleast_dim(dst, 2)

            input_tensors = [addr_, dst_]
        else:
            raise NotImplementedError()

        inputs = torch.cat(input_tensors, dim=1)
        outputs = self.actor(inputs)

        return outputs

    def init_xavier(self):
        self.actor.apply(xavier_init)


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