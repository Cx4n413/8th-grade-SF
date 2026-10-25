import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from tqdm import tqdm
import multiprocessing as mp
import sys

class PrecomputedDataset(Dataset):
    def __init__(self, pt_file_path): 
        self.vectors = torch.load(pt_file_path).float()
        self.min_val = self.vectors.min()
        self.max_val = self.vectors.max()
        self.range = self.max_val - self.min_val if self.max_val > self.min_val else 1.0
    def __len__(self): return self.vectors.shape[0]
    def __getitem__(self, idx): return self.vectors[idx]
    def get_input_dim(self): return self.vectors.shape[1]

class Encoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(Encoder, self).__init__()
        self.shared_layers = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(True), nn.Linear(256, 128), nn.ReLU(True))
        self.fc_mu = nn.Linear(128, bottleneck_dim)
        self.fc_log_var = nn.Linear(128, bottleneck_dim)
    def forward(self, x): 
        hidden = self.shared_layers(x)
        return self.fc_mu(hidden), self.fc_log_var(hidden)

class Decoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.Sequential(nn.Linear(bottleneck_dim, 256), nn.ReLU(True), nn.Linear(256, input_dim))
    def forward(self, z): 
        return self.decoder_layers(z)

class ConceptEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(ConceptEncoder, self).__init__()
        self.encoder = Encoder(input_dim, bottleneck_dim)
        self.decoder = Decoder(input_dim, bottleneck_dim)
    def reparameterize(self, mu, log_var): 
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon
    def forward(self, x): 
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def main():
    parser = argparse.ArgumentParser(description="Train a Beta-VAE Concept Encoder.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--precomputed_dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='concept_encoder_runs')
    parser.add_argument('--run_name', type=str, default=f'run_{time.strftime("%Y%m%d-%H%M%S")}')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.001, help="")
    parser.add_argument('--warmup_epochs', type=int, default=40, help="")
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"Run Name: {args.run_name}")

    full_dataset = PrecomputedDataset(args.precomputed_dataset)
    print(f"scaled using Min: {full_dataset.min_val:.4f}, Max: {full_dataset.max_val:.4f}")
    
    train_len = int(len(full_dataset) * 0.9)
    val_len = len(full_dataset) - train_len
    train_data, val_data = random_split(full_dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    
    input_dim = full_dataset.get_input_dim()
    print(f"Inferred input dimension: {input_dim}")
    
    concept_encoder = ConceptEncoder(input_dim=input_dim).to(device)
    optimizer = optim.Adam(concept_encoder.parameters(), lr=args.lr)
    
    loss_fn = nn.MSELoss(reduction='mean')
    warmup_steps = args.warmup_epochs * len(train_loader)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        concept_encoder.train()
        train_loss = 0
        
        current_beta = 0.0 if epoch < args.warmup_epochs else args.beta
        phase = 1 if current_beta == 0.0 else 2
        
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs} [Phase {phase}, Beta={current_beta:.4f}]", leave=False)

        for i, thought_vectors in train_loop:
            inputs = thought_vectors.to(device)
            optimizer.zero_grad()
            
            reconstructed, mu, log_var = concept_encoder(inputs)
            
            reconstruction_loss = loss_fn(reconstructed, inputs)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss = kl_divergence / args.batch_size
            
            total_loss = reconstruction_loss + (current_beta * kl_loss)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_loop.set_postfix(recon_loss=reconstruction_loss.item(), kl_loss=kl_loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        concept_encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for thought_vectors in val_loader:
                inputs = thought_vectors.to(device)
                reconstructed, _, _ = concept_encoder(inputs)
                val_loss += loss_fn(reconstructed, inputs).item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss (Recon): {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(concept_encoder.state_dict(), os.path.join(run_dir, 'best_concept_encoder.pth'))
            print(f"New best model saved (Val Loss: {best_val_loss:.6f})")
            

    final_save_path = os.path.join(run_dir, 'last.pth')
    torch.save(concept_encoder.state_dict(), final_save_path)
    print(f"saved final model state to: {final_save_path}")
    # -------------------------------------
    
    writer.close()
    print("\ndone")

if __name__ == "__main__":
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        mp.set_start_method('spawn', force=True)
    main()