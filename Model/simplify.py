from torch import nn, Tensor


class Simple(nn.Module):
    def __init__(self) -> None:
        super(Simple, self).__init__()


        #self.conv_block3 = conv2d_approx(64, 3, (9, 9), (1, 1), (4, 4), True, config.sample_ratio)
        self.conv_up = nn.Sequential(
            nn.Conv2d(3, 16, (9, 9), (1, 1), (4, 4)),
            nn.PReLU()
        )

        self.conv_down1 = nn.Sequential(
            nn.Conv2d(16, 8, (10, 10), (2, 2), (4, 4)),
            nn.PReLU()
        )

        self.conv_down2 = nn.Sequential(
            nn.Conv2d(8, 3, (10, 10), (2, 2), (4, 4)),
            nn.PReLU()
        )

        # Initialize neural network weights.
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        return self.conv_down2(self.conv_down1(self.conv_up(x)))


    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)