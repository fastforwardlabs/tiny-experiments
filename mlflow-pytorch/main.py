import torch

# generate some fake data

batch_size = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, output_dim)


# define a model

model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim)
)

mse = torch.nn.MSELoss()


# define an optimizer

lr = 1e-4
opt = torch.optim.SGD(model.parameters(), lr=lr)


# run a training loop

for epoch in range(100):
    preds = model(x)
    loss = mse(preds, y)
    
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
