import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from transformers import RobertaTokenizer, RobertaModel # Used in RobertaTextDataset

# --- Re-define RobertaTextDataset (Must be identical to your training script's version) ---
class RobertaTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_length=128, device='cpu'):
        self.device = device
        self.data = pd.read_csv(csv_file)
        self.tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/huggingface/roberta-large')
        self.model = RobertaModel.from_pretrained('/root/autodl-tmp/huggingface/roberta-large').to(device)
        self.model.eval() 
        
        self.max_length = max_length
        self.proj_concat = nn.Linear(3072, 1024).to(device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1).to(device)
        self.norm_single = nn.LayerNorm(1024).to(device)    

        self.title_embedding_list = []
        self.summaried_embedding_list = []
        self.synopsis_embedding_list = []
        self.title_tf_list = []
        self.summaried_tf_list = []
        self.synopsis_tf_list = []
        self.multiembedding_list = [] 

        self.label_columns = [
            "Drama", "Thriller", "Comedy", "Action", "Adventure", "Crime", "Romance", "Mystery", "Sci-Fi", "Fantasy",
            "Horror", "Dark Comedy", "Family", "Period Drama", "Biography", "Animation", "Romantic Comedy", "Tragedy",
            "Psychological Thriller", "Psychological Drama", "Supernatural Horror", "Slapstick", "War", "History",
            "Coming-of-Age", "Superhero", "Docudrama"
        ]
        
        self._preprocess_all()

    def _get_embedding(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze(0) 

    def _transform_and_norm(self, embedding_list):
        batch_tensor = torch.stack(embedding_list).unsqueeze(1).to(self.device) 
        transformed = self.transformer(batch_tensor).squeeze(1) 
        return [self.norm_single(tensor) for tensor in transformed]

    def _preprocess_all(self):
        print(f"开始编码文本和生成所有中间嵌入（共 {len(self.data)} 条数据）...")

        temp_title_raw = []
        temp_summaries_raw = []
        temp_synopsis_raw = []

        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Generating raw embeddings"):
            temp_title_raw.append(self._get_embedding(str(row['title'])))
            temp_summaries_raw.append(self._get_embedding(str(row['summaries'])))
            temp_synopsis_raw.append(self._get_embedding(str(row['synopsis'])))

        self.title_embedding_list = temp_title_raw
        self.summaried_embedding_list = temp_summaries_raw
        self.synopsis_embedding_list = temp_synopsis_raw
        
        del self.tokenizer
        del self.model
        torch.cuda.empty_cache()

        print("Transformer 处理原始嵌入...")
        self.title_tf_list = self._transform_and_norm(self.title_embedding_list)
        self.summaried_tf_list = self._transform_and_norm(self.summaried_embedding_list)
        self.synopsis_tf_list = self._transform_and_norm(self.synopsis_embedding_list)

        print("生成最终多模态 embedding...")
        for t, s, y in tqdm(zip(self.title_tf_list, self.summaried_tf_list, self.synopsis_tf_list), 
                            total=len(self.title_tf_list), desc="Generating multi-modal embedding"):
            concat = torch.cat([t, s, y], dim=0) 
            reduced = self.proj_concat(concat) 
            self.multiembedding_list.append(self.norm_single(reduced))

        self.labels_tensor = torch.tensor(self.data[self.label_columns].values, dtype=torch.float32).to(self.device)
        print("预处理完成.")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return {
            'title_embedding': self.title_embedding_list[idx].detach(),
            'summaried_embedding': self.summaried_embedding_list[idx].detach(),
            'synopsis_embedding': self.synopsis_embedding_list[idx].detach(),
            'title_tf': self.title_tf_list[idx].detach(),
            'summaried_tf': self.summaried_tf_list[idx].detach(),
            'synopsis_tf': self.synopsis_tf_list[idx].detach(),
            'multi_embedding': self.multiembedding_list[idx].detach(),
            'label': self.labels_tensor[idx] 
        }

# --- Re-define SevenVectorClassifier (Must be identical to your training script's version) ---
class SevenVectorClassifier(nn.Module):
    def __init__(self, input_embedding_dim, output_dim, num_input_vectors=7):
        super().__init__()
        self.input_embedding_dim = input_embedding_dim 
        self.output_dim = output_dim 
        self.num_input_vectors = num_input_vectors 

        self.classifier = nn.Linear(self.num_input_vectors * self.input_embedding_dim, self.output_dim)

    def forward(self, 
                title_embedding, summaried_embedding, synopsis_embedding, 
                title_tf, summaried_tf, synopsis_tf, 
                multi_embedding, 
                labels=None):
        
        combined_features = torch.cat([
            title_embedding, 
            summaried_embedding, 
            synopsis_embedding, 
            title_tf, 
            summaried_tf, 
            synopsis_tf, 
            multi_embedding
        ], dim=1) 

        logits = self.classifier(combined_features) 

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float()) 

        return {'loss': loss, 'logits': logits}


# --- Test Function ---
def test_model(
    model,
    test_dataloader,
    device,
    output_dir="./seven_vector_classifier_output" # Ensure this matches your training script's output directory
):
    model.to(device)
    model.eval() # Set model to evaluation mode

    print("Starting evaluation on the test set...")
    all_preds = []
    all_labels = []
    test_progress_bar = tqdm(test_dataloader, desc="Testing")

    with torch.no_grad(): # Disable gradient calculations during testing
        for batch in test_progress_bar:
            # Extract all 7 input embeddings and labels
            title_embedding = batch['title_embedding'].to(device)
            summaried_embedding = batch['summaried_embedding'].to(device)
            synopsis_embedding = batch['synopsis_embedding'].to(device)
            title_tf = batch['title_tf'].to(device)
            summaried_tf = batch['summaried_tf'].to(device)
            synopsis_tf = batch['synopsis_tf'].to(device)
            multi_embedding = batch['multi_embedding'].to(device)
            labels = batch['label'].to(device) # Note: 'label' here, matches your __getitem__

            # Pass all 7 embeddings to the model's forward method
            outputs = model(
                title_embedding=title_embedding, 
                summaried_embedding=summaried_embedding, 
                synopsis_embedding=synopsis_embedding, 
                title_tf=title_tf, 
                summaried_tf=summaried_tf, 
                synopsis_tf=synopsis_tf, 
                multi_embedding=multi_embedding, 
                labels=labels # Labels are optional for inference, but useful for calculating loss during testing
            )
            
            # Get predictions and convert to binary
            preds = (torch.sigmoid(outputs['logits']) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate and print evaluation metrics
    overall_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    hamming = hamming_loss(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    print("\n--- Test Results ---")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")

# --- Main Execution Part ---
if __name__ == "__main__":
    # Configuration (Ensure these match your training setup)
    TEST_CSV_FILE = '/root/autodl-tmp/MovieLabeling/datasets/test_data.csv'
    MODEL_WEIGHTS_PATH = '../weights/best_model.pt' # Path to your trained model weights
    MAX_LENGTH = 128 
    BATCH_SIZE = 32 # Can be larger for testing as no backprop

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for necessary files
    if not os.path.exists(TEST_CSV_FILE):
        print(f"Error: Test CSV file not found at {TEST_CSV_FILE}")
        exit()

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        print("Please ensure your training script ran successfully and saved 'best_model.pt' in the specified output directory.")
        exit()

    # Instantiate the test dataset and dataloader
    print("Loading and pre-processing test data with RobertaTextDataset...")
    test_dataset = RobertaTextDataset(csv_file=TEST_CSV_FILE, max_length=MAX_LENGTH, device=device)
    # num_workers should be 0 because all data is pre-processed and loaded into memory
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize the model and load its weights
    # input_embedding_dim is 1024 (from RoBERTa's hidden_size)
    # output_dim is 27 (number of labels)
    model = SevenVectorClassifier(
        input_embedding_dim=1024, 
        output_dim=len(test_dataset.label_columns),
        num_input_vectors=7 
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    print(f"Model weights loaded from {MODEL_WEIGHTS_PATH}")

    # Run the testing process
    test_model(
        model,
        test_dataloader,
        device
    )

    print("Test script finished.")