import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import os
import multiprocessing as mp

from transformers import AutoModel
class Encoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(Encoder, self).__init__(); self.shared_layers = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(True), nn.Linear(256, 128), nn.ReLU(True))
        self.fc_mu = nn.Linear(128, bottleneck_dim); self.fc_log_var = nn.Linear(128, bottleneck_dim)
    def forward(self, x): hidden = self.shared_layers(x); return self.fc_mu(hidden), self.fc_log_var(hidden)

class Decoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(Decoder, self).__init__(); self.decoder_layers = nn.Sequential(nn.Linear(bottleneck_dim, 256), nn.ReLU(True), nn.Linear(256, input_dim))
    def forward(self, z): return self.decoder_layers(z)

class ConceptEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(ConceptEncoder, self).__init__(); self.encoder = Encoder(input_dim, bottleneck_dim); self.decoder = Decoder(input_dim, bottleneck_dim)
    def reparameterize(self, mu, log_var): std = torch.exp(0.5 * log_var); epsilon = torch.randn_like(std); return mu + std * epsilon
    def forward(self, x): mu, log_var = self.encoder(x); z = self.reparameterize(mu, log_var); return self.decoder(z), mu, log_var

def fen_to_tensor(fen: str):
    board = chess.Board(fen); P = 19; tensor = np.zeros((8, 8, P), dtype=np.float32)
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8); plane = piece_map[piece.symbol()]; tensor[7 - rank, file, plane] = 1.0
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.ep_square is not None: rank, file = divmod(board.ep_square, 8); tensor[7 - rank, file, 13] = 1.0
    tensor[:, :, 14] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0; tensor[:, :, 15] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 16] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0; tensor[:, :, 17] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
    return tensor

class Hook:
    def __init__(self): self.value = None
    def __call__(self, module, input, output): self.value = output.detach()

class FENDataset(Dataset):
    def __init__(self, fen_file_path):
        with open(fen_file_path, 'r') as f: self.fens = [line.strip() for line in f if line.strip()]
    def __len__(self): return len(self.fens)
    def __getitem__(self, idx): return self.fens[idx]

def collate_fn_chessbot(batch_fens: list):
    tensors = [fen_to_tensor(fen) for fen in batch_fens]
    return batch_fens, np.stack(tensors)[:, np.newaxis, :, :, :]

def calculate_proxies_batch(board_batch: list) -> list:
    proxies_list = []; material_map = {'p':-100,'n':-300,'b':-320,'r':-500,'q':-900, 'P':100,'N':300,'B':320,'R':500,'Q':900}
    for board in board_batch:
        proxies = {}; proxies['dim_material'] = sum(material_map.get(p.symbol(), 0) for p in board.piece_map().values()); proxies['dim_mobility'] = board.legal_moves.count()
        king_square = board.king(board.turn); proxies['dim_king_freedom'] = len([m for m in board.legal_moves if m.from_square == king_square]) if king_square else 0
        pawn_weakness = 0
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            for pawn_square in pawns:
                pawn_file = chess.square_file(pawn_square)
                if len(board.pieces(chess.PAWN, color) & chess.BB_FILES[pawn_file]) > 1: pawn_weakness += 1
                adj_files_mask = 0;
                if pawn_file > 0: adj_files_mask |= chess.BB_FILES[pawn_file - 1]
                if pawn_file < 7: adj_files_mask |= chess.BB_FILES[pawn_file + 1]
                if adj_files_mask and not (pawns & adj_files_mask): pawn_weakness += 1
        proxies['dim_pawn_weakness'] = pawn_weakness; proxies['dim_game_phase'] = max(0.0, (len(board.piece_map()) - 4) / 28.0); proxies_list.append(proxies)
    return proxies_list

def main():
    parser = argparse.ArgumentParser(description="Professional Inference for the Unified Audit.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fen_file', type=str, required=True); parser.add_argument('--concept_encoder_ckpt', type=str, required=True)
    parser.add_argument('--output_h5', type=str, default="audit_data.h5"); parser.add_argument('--num_workers', type=int, default=8); parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"--- Unified Audit Inference (Rich Vector Mode) ---"); print(f"Using device: {device}")
    
    powercore_model = AutoModel.from_pretrained("Maxlegrec/ChessBot", trust_remote_code=True).to(device).eval()
    concept_encoder = ConceptEncoder(input_dim=1536).to(device).eval()
    concept_encoder.load_state_dict(torch.load(args.concept_encoder_ckpt, map_location=device))
    thought_vector_hook = Hook(); hook_handle = powercore_model.layers[-1].register_forward_hook(thought_vector_hook)
    dataset = FENDataset(args.fen_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_chessbot, pin_memory=True)
    all_data, is_first_write = [], True
    column_names = ['y_pc'] + [f'factor_{i+1}' for i in range(8)] + list(calculate_proxies_batch([chess.Board()])[0].keys())  
    for fen_batch, tensor_batch in tqdm(dataloader, desc="Processing batches"):
        with torch.no_grad():
            inputs = torch.from_numpy(tensor_batch).to(device)
            powercore_model(inputs_embeds=inputs)
            raw_thought_tensor = thought_vector_hook.value
            mean_vec = raw_thought_tensor.mean(dim=1)
            std_vec = raw_thought_tensor.std(dim=1)
            max_vec = torch.max(raw_thought_tensor, dim=1).values
            captured_thought_vector = torch.cat((mean_vec, std_vec, max_vec), dim=1)
            
            mu, log_var = concept_encoder.encoder(captured_thought_vector)
            X_factors_batch = mu.cpu().numpy()
            
            y_pc_batch = captured_thought_vector.mean(dim=1).cpu().numpy()
        
        board_batch = [chess.Board(f) for f in fen_batch]; proxies_batch = calculate_proxies_batch(board_batch)
        for i in range(len(fen_batch)):
            all_data.append([y_pc_batch[i]] + X_factors_batch[i].tolist() + list(proxies_batch[i].values()))

        if len(all_data) >= 20000:
            df = pd.DataFrame(all_data, columns=column_names); mode = 'w' if is_first_write else 'a'
            with pd.HDFStore(args.output_h5, mode=mode) as store: store.append('audit_data', df, index=False)
            all_data, is_first_write = [], False
    
    if all_data:
        df = pd.DataFrame(all_data, columns=column_names); mode = 'w' if is_first_write else 'a'
        with pd.HDFStore(args.output_h5, mode=mode) as store: store.append('audit_data', df, index=False)
    
    hook_handle.remove()
    print(f"\ndone!-"); print(f"Total positions processed: {len(dataset)}"); print(f"saved to: {args.output_h5}")

if __name__ == "__main__":
    if sys.platform.startswith("linux") or sys.platform == "darwin": mp.set_start_method('spawn', force=True)
    main()