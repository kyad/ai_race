import torch
import torch.nn as nn

class ViT2(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        if type(image_size) is int:
            assert image_size % patch_size == 0
            num_patches = (image_size // patch_size) ** 2
        elif len(image_size) == 2:
            assert image_size[0] % patch_size == 0
            assert image_size[1] % patch_size == 0
            num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        else:
            raise NotImplementedError()
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=depth,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size
        _, c, h, w = img.shape
        x = img.view(-1, h // p * w // p, p * p * c)

        x = self.patch_to_embedding(x)  # b * n * dim
        b, n, _ = x.shape

        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)  # b * (n + 1) * dim

        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        x = torch.transpose(x, 0, 1)  # (n + 1) * b * dim
        x = self.transformer_encoder(x)  # (n + 1) * b * dim
        x = torch.transpose(x, 0, 1)  # b * (n + 1) * dim

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # b * dim

        x = self.to_latent(x)
        return self.mlp_head(x)

def pred_test():
    v = ViT2(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
    )
    img = torch.randn(1, 3, 256, 256)
    preds = v(img)
    print(preds)

def pred_rect_test():
    v = ViT2(
        image_size=(240, 320),
        patch_size=40,
        num_classes=3,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
    )
    img = torch.randn(1, 3, 240, 320)
    preds = v(img)
    print(preds)

def mnist_test():
    import time
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision

    torch.manual_seed(42)

    DOWNLOAD_PATH = '/data/mnist'
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 1000

    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                           transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                          transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    def train_epoch(model, optimizer, data_loader, loss_history):
        total_samples = len(data_loader.dataset)
        model.train()

        for i, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                      ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                      '{:6.4f}'.format(loss.item()))
                loss_history.append(loss.item())

    def evaluate(model, data_loader, loss_history):
        model.eval()

        total_samples = len(data_loader.dataset)
        correct_samples = 0
        total_loss = 0

        with torch.no_grad():
            for data, target in data_loader:
                output = F.log_softmax(model(data), dim=1)
                loss = F.nll_loss(output, target, reduction='sum')
                _, pred = torch.max(output, dim=1)

                total_loss += loss.item()
                correct_samples += pred.eq(target).sum()

        avg_loss = total_loss / total_samples
        loss_history.append(avg_loss)
        print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
              '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
              '{:5}'.format(total_samples) + ' (' +
              '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    N_EPOCHS = 25

    start_time = time.time()
    model = ViT2(image_size=28, patch_size=7, num_classes=10, channels=1,
                 dim=64, depth=6, heads=8, mlp_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        evaluate(model, test_loader, test_loss_history)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if __name__ == '__main__':
    pred_test()
    pred_rect_test()
    mnist_test()
