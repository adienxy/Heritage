import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import ViTModel, RobertaTokenizer, RobertaModel
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import pearsonr
import logging
from imblearn.over_sampling import RandomOverSampler
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UNESCO_CRITERIA = [
    "to represent a masterpiece of human creative genius",
    "to exhibit an important interchange of human values, over a span of time or within a cultural area of the world, on developments in architecture or technology, monumental arts, town-planning or landscape design",
    "to bear a unique or at least exceptional testimony to a cultural tradition or to a civilization which is living or which has disappeared",
    "to be an outstanding example of a type of building, architectural or technological ensemble or landscape which illustrates (a) significant stage(s) in human history",
    "to be an outstanding example of a traditional human settlement, land-use, or sea-use which is representative of a culture (or cultures), or human interaction with the environment especially when it has become vulnerable under the impact of irreversible change",
    "to be directly or tangibly associated with events or living traditions, with ideas, or with beliefs, with artistic and literary works of outstanding universal significance",
    "to contain superlative natural phenomena or areas of exceptional natural beauty and aesthetic importance",
    "to be outstanding examples representing major stages of earth's history, including the record of life, significant on-going geological processes in the development of landforms, or significant geomorphic or physiographic features",
    "to be outstanding examples representing significant on-going ecological and biological processes in the evolution and development of terrestrial, fresh water, coastal and marine ecosystems and communities of plants and animals",
    "to contain the most important and significant natural habitats for in-situ conservation of biological diversity, including those containing threatened species of outstanding universal value from the point of view of science or conservation"
]

class MultiImageHeritageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=5):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'), encoding="MacRoman")
        self.scores = pd.read_csv(os.path.join(root_dir, 'scores.csv'))
        self.criteria = pd.read_csv(os.path.join(root_dir, 'criteria.csv'))
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained('./robertabase')
        self.max_images = max_images
        
        self.data = pd.merge(self.metadata, self.scores[['id', 'score']], on='id', how='left')
        self.data = pd.merge(self.data, self.criteria, on='id', how='left')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        folder_name = f"{row['unique_number_x']}_"
        folder_path = os.path.join(self.root_dir, 'datas', folder_name)
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif'))]
        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")
        
        selected_images = random.sample(image_files, min(len(image_files), self.max_images))
        images = []
        for image_file in selected_images:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        while len(images) < self.max_images:
            images.append(torch.zeros_like(images[0]))
        
        images = torch.stack(images)
        
        description_path = os.path.join(folder_path, 'description.txt')
        try:
            with open(description_path, 'r', encoding='utf-8') as f:
                description = f.read()
        except FileNotFoundError:
            description = ""
            logger.warning(f"Description file not found for {folder_name}")
        
        tokens = self.tokenizer(description, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        
        score = row['score']
        score = torch.tensor(score, dtype=torch.float) if pd.notna(score) else torch.tensor(0.0, dtype=torch.float)
        
        criteria_columns = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
        criteria = row[criteria_columns].values
        criteria = torch.tensor([float(c) if pd.notna(c) else 0.0 for c in criteria], dtype=torch.float)
        
        score_mask = torch.tensor(1.0, dtype=torch.float) if pd.notna(row['score']) else torch.tensor(0.0, dtype=torch.float)
        
        return images, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), score, criteria, score_mask

    def calculate_sample_weights(self):
        criteria_counts = self.criteria.sum()
        class_weights = 1.0 / criteria_counts
        class_weights = class_weights.values
        
        sample_weights = np.zeros(len(self.data))
        for i, row in self.criteria.iterrows():
            sample_weights[i] = np.sum(row * class_weights)
            
        return sample_weights

    def get_oversampled_indices(self):
        ros = RandomOverSampler(random_state=42)
        indices = np.arange(len(self.data)).reshape(-1, 1)
        oversampled_indices, _ = ros.fit_resample(indices, self.criteria.values)
        return oversampled_indices.flatten()

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        
    def forward(self, query, key, value, attn_mask=None):
        return self.attention(query, key, value, attn_mask)[0]

class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.image_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, image_features, text_features):
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)
        
        v2t_attn = self.attention(
            text_proj,
            image_proj,
            image_proj
        )[0]
        
        t2v_attn = self.attention(
            image_proj,
            text_proj,
            text_proj
        )[0]

        fused = self.norm1(v2t_attn + t2v_attn)
        output = self.norm2(fused + self.ffn(fused))
        
        return output

class MutualInformationMaximizer(nn.Module):
    def __init__(self, dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, image_features, text_features):
        f_i = self.projection(image_features)
        f_t = self.projection(text_features)
        
        sim_matrix = torch.matmul(f_i, f_t.transpose(-2, -1)) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.transpose(-2, -1), labels)
        
        return (loss_i2t + loss_t2i) / 2

class EnhancedMultiImageHeritageEncoder(nn.Module):
    def __init__(self, output_dim=256, max_images=5):
        super().__init__()
        self.image_encoder = ViTModel.from_pretrained('./vit-base-patch16-224-in21k')
        self.text_encoder = RobertaModel.from_pretrained('./robertabase')
        self.tokenizer = RobertaTokenizer.from_pretrained('./robertabase')
        
        self.hidden_dim = 768
        
        self.multi_scale_heads = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, num_heads=h, batch_first=True)
            for h in [4, 8, 12]
        ])
        
        self.scale_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        
        self.domain_embeddings = nn.Parameter(torch.randn(10, self.hidden_dim))
        self.semantic_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            bidirectional=False,
            batch_first=True
        )
        
        self.text_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        self.modality_fusion = CrossModalFusion(self.hidden_dim)
        self.mutual_information = MutualInformationMaximizer(self.hidden_dim)
        
        self.score_encoder = nn.Linear(1, 128)
        self.criteria_encoder = nn.Embedding(10, self.hidden_dim)
        
        fusion_input_dim = self.hidden_dim + self.hidden_dim + 128  # fused_features + criteria_embeds + score_embed
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim)
        )
        
        self.criterion_predictor = nn.Linear(output_dim, 10)
        self.score_predictor = nn.Linear(output_dim, 1)
        
        self.initialize_criteria_embeddings()
    
    def initialize_criteria_embeddings(self):
        with torch.no_grad():
            for i, criterion in enumerate(UNESCO_CRITERIA):
                tokens = self.tokenizer(criterion, return_tensors='pt', padding=True, truncation=True)
                outputs = self.text_encoder(**tokens)
                embedding = outputs.last_hidden_state.mean(dim=1)
                self.criteria_encoder.weight.data[i] = embedding.squeeze()

    def forward(self, image_features, text_ids, text_mask, score, criteria):
        batch_size, num_images, channels, height, width = image_features.shape
        
        image_features = image_features.view(-1, channels, height, width)
        base_image_embed = self.image_encoder(image_features).last_hidden_state
        _, seq_len, hidden_dim = base_image_embed.shape
        base_image_embed = base_image_embed.view(batch_size, num_images * seq_len, hidden_dim)
        
        multi_scale_features = []
        for attention_head in self.multi_scale_heads:
            scale_feature, _ = attention_head(base_image_embed, base_image_embed, base_image_embed)
            multi_scale_features.append(scale_feature)
        
        scale_weights = self.scale_attention(base_image_embed.mean(dim=1))
        scale_weights = scale_weights.unsqueeze(1).unsqueeze(2)
        
        stacked_features = torch.stack(multi_scale_features, dim=-1)
        weighted_features = (stacked_features * scale_weights).sum(dim=-1)
        
        weighted_features = weighted_features.view(batch_size, num_images, seq_len, hidden_dim)
        image_features = weighted_features.mean(dim=2)
        
        text_embed = self.text_encoder(input_ids=text_ids, attention_mask=text_mask).last_hidden_state
        

        domain_context = torch.matmul(criteria, self.domain_embeddings)
        h0 = domain_context.unsqueeze(0)
        
        enhanced_text, _ = self.semantic_gru(text_embed, h0)
        
        attention_weights = self.text_attention(enhanced_text).squeeze(-1)
        attention_mask = text_mask.float()
        attention_weights = torch.softmax(
            attention_weights * attention_mask + (1 - attention_mask) * -1e9, 
            dim=1
        )
        text_embed = torch.sum(enhanced_text * attention_weights.unsqueeze(-1), dim=1)
        
        image_proj = self.image_projection(image_features.mean(dim=1))
        text_proj = self.text_projection(text_embed)
        
        image_proj = image_proj.unsqueeze(1)
        text_proj = text_proj.unsqueeze(1)
        
        fused_features = self.modality_fusion(image_proj, text_proj)
        
        mi_loss = self.mutual_information(image_proj.squeeze(1), text_proj.squeeze(1))
        
        score_embed = self.score_encoder(score.unsqueeze(1))
        criteria_embeds = self.criteria_encoder(torch.arange(10).to(criteria.device))
        criteria_embeds = self.text_projection(criteria_embeds).mean(dim=0)
        
        combined = torch.cat([
            fused_features.squeeze(1),
            criteria_embeds.expand(batch_size, -1),
            score_embed
        ], dim=-1)
        
        output = self.fusion(combined)
        criterion_pred = self.criterion_predictor(output)
        score_pred = self.score_predictor(output)
        
        return output, criterion_pred, score_pred, mi_loss

class EnhancedSemiSupervisedLoss(nn.Module):
    def __init__(self, lambda_unsupervised=0.1, lambda_consistency=0.05, lambda_contrastive=0.01):
        super().__init__()
        self.lambda_unsupervised = lambda_unsupervised
        self.lambda_consistency = lambda_consistency
        self.lambda_contrastive = lambda_contrastive

    def forward(self, model_output, targets, score_mask):
        output, criterion_pred, score_pred, mi_loss = model_output
        image, text_ids, text_mask, score, criteria, _ = targets

        criterion_loss = F.binary_cross_entropy_with_logits(criterion_pred, criteria)
        score_loss = F.mse_loss(score_pred.squeeze(), score) * score_mask.float()

        consistency_loss = F.mse_loss(score_pred.squeeze(), score) * (1 - score_mask.float())

        supervised_loss = criterion_loss.mean() + score_loss.mean()
        
        total_loss = supervised_loss + \
                    self.lambda_unsupervised * mi_loss + \
                    self.lambda_consistency * consistency_loss.mean()

        return total_loss

def train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, num_epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar:
        batch = [b.to(device) for b in batch]
        optimizer.zero_grad()

        model_output = model(*batch[:-1])
        loss = criterion(model_output, batch, batch[-1])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1} - Average Train Loss: {avg_loss:.4f}")
    
    return avg_loss

def evaluate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []
    all_true_scores = []
    
    progress_bar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            batch = [b.to(device) for b in batch]
            model_output = model(*batch[:-1])
            loss = criterion(model_output, batch, batch[-1])
            
            _, criterion_pred, score_pred, _ = model_output
            preds = torch.sigmoid(criterion_pred)
            
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(batch[4].cpu())
            all_scores.append(score_pred.cpu())
            all_true_scores.append(batch[3].cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_true_scores = torch.cat(all_true_scores, dim=0).numpy()
    
    avg_loss = total_loss / len(val_loader)
    
    metrics = calculate_metrics(all_preds, all_labels, all_scores, all_true_scores)
    criterion_stats, error_analysis = analyze_prediction_patterns(all_preds, all_labels)

    print_detailed_metrics(metrics, criterion_stats, error_analysis)
    
    return avg_loss, metrics

def calculate_metrics(all_preds, all_labels, all_scores, all_true_scores):
    metrics = {}
    
    pred_labels = (all_preds > 0.5).astype(int)
    
    sample_accuracy = np.mean(np.all(pred_labels == all_labels, axis=1))
    metrics['sample_accuracy'] = sample_accuracy
    
    label_accuracy = np.mean(pred_labels == all_labels)
    metrics['label_accuracy'] = label_accuracy
    
    individual_accuracies = np.mean(pred_labels == all_labels, axis=0)
    for i, acc in enumerate(individual_accuracies):
        metrics[f'criterion_{i+1}_accuracy'] = acc
    
    hamming_accuracy = 1 - np.mean(np.abs(pred_labels - all_labels))
    metrics['hamming_accuracy'] = hamming_accuracy
    
    metrics['subset_accuracy'] = sample_accuracy
    
    correct_labels_per_sample = np.sum(pred_labels == all_labels, axis=1)
    metrics['avg_correct_labels'] = np.mean(correct_labels_per_sample)
    metrics['median_correct_labels'] = np.median(correct_labels_per_sample)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        pred_labels,
        average='weighted',
        zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['mAP'] = average_precision_score(all_labels, all_preds, average='weighted')
    
    try:
        lb = LabelBinarizer()
        lb.fit(all_labels)
        all_labels_bin = lb.transform(all_labels)
        if all_labels_bin.shape[1] == 1:
            all_labels_bin = np.hstack((1 - all_labels_bin, all_labels_bin))
        metrics['auc_roc'] = roc_auc_score(all_labels_bin, all_preds, average='weighted')
    except ValueError as e:
        logger.warning(f"ROC AUC calculation failed: {e}")
        metrics['auc_roc'] = 0
    
    metrics['mae'] = np.mean(np.abs(all_true_scores - all_scores))
    metrics['rmse'] = np.sqrt(np.mean((all_true_scores - all_scores)**2))
    
    if len(all_true_scores) > 1:
        metrics['pcc'], _ = pearsonr(all_true_scores.flatten(), all_scores.flatten())
    else:
        metrics['pcc'] = 0
    
    return metrics

def analyze_prediction_patterns(all_preds, all_labels):
    pred_labels = (all_preds > 0.5).astype(int)
    
    criterion_stats = {}
    for i in range(all_labels.shape[1]):
        stats = {
            'true_positives': np.sum((pred_labels[:, i] == 1) & (all_labels[:, i] == 1)),
            'false_positives': np.sum((pred_labels[:, i] == 1) & (all_labels[:, i] == 0)),
            'false_negatives': np.sum((pred_labels[:, i] == 0) & (all_labels[:, i] == 1)),
            'true_negatives': np.sum((pred_labels[:, i] == 0) & (all_labels[:, i] == 0))
        }
        criterion_stats[f'criterion_{i+1}'] = stats
    
    error_analysis = {
        'over_prediction': np.mean(np.sum(pred_labels, axis=1) > np.sum(all_labels, axis=1)),
        'under_prediction': np.mean(np.sum(pred_labels, axis=1) < np.sum(all_labels, axis=1)),
        'avg_predicted_criteria': np.mean(np.sum(pred_labels, axis=1)),
        'avg_true_criteria': np.mean(np.sum(all_labels, axis=1))
    }
    
    return criterion_stats, error_analysis

def print_detailed_metrics(metrics, criterion_stats, error_analysis):
    print("\n=== Detailed Evaluation Results ===")
    
    print("\n1. Overall Accuracy Metrics:")
    print(f"Sample-level Accuracy: {metrics['sample_accuracy']:.4f}")
    print(f"Label-level Accuracy: {metrics['label_accuracy']:.4f}")
    print(f"Hamming Accuracy: {metrics['hamming_accuracy']:.4f}")
    
    print("\n2. Individual Criterion Performance:")
    for i in range(len(UNESCO_CRITERIA)):
        criterion_acc = metrics[f'criterion_{i+1}_accuracy']
        stats = criterion_stats[f'criterion_{i+1}']
        print(f"\nCriterion {i+1}:")
        print(f"  Accuracy: {criterion_acc:.4f}")
        print(f"  True Positives: {stats['true_positives']}")
        print(f"  False Positives: {stats['false_positives']}")
        print(f"  False Negatives: {stats['false_negatives']}")
    
    print("\n3. Error Analysis:")
    print(f"Over-prediction Rate: {error_analysis['over_prediction']:.4f}")
    print(f"Under-prediction Rate: {error_analysis['under_prediction']:.4f}")
    print(f"Average Predicted Criteria per Sample: {error_analysis['avg_predicted_criteria']:.2f}")
    print(f"Average True Criteria per Sample: {error_analysis['avg_true_criteria']:.2f}")
    
    print("\n4. Other Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MultiImageHeritageDataset('./final_dataset/', transform=transform, max_images=5)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    sample_weights = dataset.calculate_sample_weights()
    train_sampler = WeightedRandomSampler(sample_weights[train_indices], len(train_indices))
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    model = EnhancedMultiImageHeritageEncoder(max_images=5).to(device)
    criterion = EnhancedSemiSupervisedLoss(
        lambda_unsupervised=0.1,
        lambda_consistency=0.05,
        lambda_contrastive=0.01
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': [],
        'learning_rate': []
    }
    
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, num_epochs)
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion)
        
        scheduler.step()

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        for key, value in val_metrics.items():
            if key not in history:
                history[key] = []
            history[key].append(float(value))
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info("Validation Metrics:")
        for key, value in val_metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_enhanced_heritage_model.pth')
            with open('best_enhanced_heritage_model_history.json', 'w') as f:
                json.dump(history, f, indent=4)
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            with open(f'training_history_epoch_{epoch+1}.json', 'w') as f:
                json.dump(history, f, indent=4)

    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    with open('final_training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    main()