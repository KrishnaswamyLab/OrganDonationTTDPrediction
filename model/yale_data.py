from dataset import YaleDataset
import torch

if __name__ == "__main__":
    static_file_path = '../output_data/preprocessed/static_target.csv'
    dynamic_file_path = '../output_data/preprocessed/dynamic.csv'
    dataset = YaleDataset(static_file_path, dynamic_file_path)
    torch.save(dataset, "../dataset/yale_imputed.pt")