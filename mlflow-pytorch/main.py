import torch
import mlflow
import mlflow.pytorch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden', type=int, default=100)
args = parser.parse_args()

# generate some fake data

batch_size = 64
input_dim = 1000
hidden_dim = args.hidden
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
l1 = torch.nn.L1Loss()

# define an optimizer

opt = torch.optim.SGD(model.parameters(), lr=args.lr)


# run a training loop

with mlflow.start_run():

    mlflow.log_params({
        'lr': args.lr,
        'hidden': args.hidden
    })

    for epoch in range(100):
        preds = model(x)
        loss = mse(preds, y)
        l1_loss = l1(preds, y)        
 
        mlflow.log_metrics({
            'loss': loss.item(),
            'squared loss': loss.item()**2,
            'l1 loss': l1_loss.item()
        }, step=epoch)
        
        opt.zero_grad()
        loss.backward()
        opt.step() 
    
    mlflow.pytorch.log_model(model ,'model')
