import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json
from torchvision import transforms
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from main import (
    EnhancedMultiImageHeritageEncoder,
    MultiImageHeritageDataset,
    UNESCO_CRITERIA
)

class HeritageModelAnalyzer:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = EnhancedMultiImageHeritageEncoder(max_images=5).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.color_theme = {
            'primary': '#20407D',
            'secondary': '#CC444B',
            'tertiary': '#2A7F62',
            'background': '#FFFFFF',
            'text': '#333333',
            'grid': '#E5E5E5'
        }
        
        self.set_plotting_style()
        self.attention_cmap = self.create_attention_colormap()

    def set_plotting_style(self):
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.0,
            'axes.edgecolor': self.color_theme['text']
        })
    
    def create_attention_colormap(self):
        colors = ['#FFFFFF', '#FFE5E5', '#FFB3B3', '#FF8080', '#FF4D4D', '#FF0000']
        return LinearSegmentedColormap.from_list('attention', colors)

    def get_attention_maps(self, batch):
        images, text_ids, text_mask, score, criteria, score_mask = batch
        
        with torch.no_grad():
            image_features = images.to(self.device)
            base_image_embed = self.model.image_encoder(
                image_features.view(-1, 3, 224, 224)
            ).last_hidden_state
            
            attention_maps = []
            for head in self.model.multi_scale_heads:
                _, attn_weights = head(
                    base_image_embed,
                    base_image_embed,
                    base_image_embed,
                    need_weights=True
                )
                attention_maps.append(attn_weights.cpu().numpy())
            
            text_embed = self.model.text_encoder(
                input_ids=text_ids.to(self.device),
                attention_mask=text_mask.to(self.device)
            ).last_hidden_state
            
            text_attention = self.model.text_attention(text_embed).squeeze(-1)
            
            return {
                'visual_attention': attention_maps,
                'text_attention': text_attention.cpu().numpy(),
                'scale_weights': self.model.scale_attention(
                    base_image_embed.mean(dim=1)
                ).cpu().numpy(),
                'score_mask': score_mask.cpu().numpy()
            }

    def visualize_attention_analysis(self, image, attention_weights, save_path=None):
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, height_ratios=[3, 1])
        
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        
        attention = cv2.resize(attention_weights, (image.shape[1], image.shape[0]))
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('(a) Original Image', pad=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(attention, cmap=self.attention_cmap)
        ax2.set_title('(b) Attention Heatmap', pad=10)
        ax2.axis('off')
        
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Intensity', fontsize=9)
        
        ax3 = fig.add_subplot(gs[0, 2])
        heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.uint8(255 * image), 0.7, heatmap, 0.3, 0)
        ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax3.set_title('(c) Attention Overlay', pad=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, :])
        sns.histplot(data=attention.flatten(), bins=50, color=self.color_theme['primary'], 
                    ax=ax4, stat='density', alpha=0.7)
        ax4.set_title('(d) Attention Distribution', pad=10)
        ax4.set_xlabel('Attention Value')
        ax4.set_ylabel('Density')
        
        stats_text = (f'Mean: {attention.mean():.3f}\n'
                     f'Std: {attention.std():.3f}\n'
                     f'Max: {attention.max():.3f}')
        ax4.text(0.95, 0.95, stats_text,
                transform=ax4.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_text_analysis(self, text, attention_weights, save_path=None, top_k=15):
        if isinstance(text, torch.Tensor):
            text = self.model.tokenizer.decode(text, skip_special_tokens=True)
        
        words = text.split()
        attention = attention_weights[:len(words)]
        
        word_importance = list(zip(words, attention))
        word_importance.sort(key=lambda x: x[1], reverse=True)
        top_words = word_importance[:top_k]
        
        plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 1, height_ratios=[1.5, 1])
        
        ax1 = plt.subplot(gs[0])
        
        top_30_indices = sorted(
            range(len(attention)), 
            key=lambda i: attention[i], 
            reverse=True
        )[:30]
        
        selected_words = [words[i] for i in top_30_indices]
        selected_attention = [attention[i] for i in top_30_indices]
        
        im = ax1.imshow(
            np.array(selected_attention).reshape(1, -1),
            cmap='YlOrRd',
            aspect='auto'
        )
        
        ax1.set_xticks(range(len(selected_words)))
        ax1.set_xticklabels(
            selected_words,
            rotation=45,
            ha='right',
            fontsize=8
        )
        
        ax1.set_yticks([])
        
        cbar = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.01)
        cbar.set_label('Attention Weight', fontsize=9)
        
        ax1.set_title('(a) Text Attention Heatmap (Top 30 Words)', pad=10)
        
        ax2 = plt.subplot(gs[1])
        
        bars = ax2.bar(
            range(len(top_words)),
            [w[1] for w in top_words],
            color=self.color_theme['primary'],
            alpha=0.7
        )
        
        ax2.set_xticks(range(len(top_words)))
        ax2.set_xticklabels(
            [w[0] for w in top_words],
            rotation=45,
            ha='right',
            fontsize=9
        )
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        ax2.set_title(f'(b) Top {top_k} Important Words', pad=10)
        ax2.set_ylabel('Attention Weight')
        
        stats_text = (
            f'Mean Attention: {np.mean(attention):.3f}\n'
            f'Max Attention: {np.max(attention):.3f}\n'
            f'Word Count: {len(words)}'
        )
        
        ax2.text(
            0.95, 0.95,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8,
                edgecolor=self.color_theme['primary'],
                pad=0.5
            ),
            fontsize=8
        )
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def visualize_criteria_analysis(self, importance_scores, save_path=None):
        criteria_names = [f'Criterion {i+1}' for i in range(len(importance_scores))]
        
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(criteria_names, importance_scores, color=self.color_theme['primary'])
        ax1.set_xticklabels(criteria_names, rotation=45, ha='right')
        ax1.set_title('(a) Criteria Importance Scores', pad=10)
        ax1.set_ylabel('Importance Score')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom',
                    fontsize=8)
        
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        angles = np.linspace(0, 2*np.pi, len(importance_scores), endpoint=False)
        values = np.concatenate((importance_scores, [importance_scores[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax2.plot(angles, values, 'o-', linewidth=2, color=self.color_theme['primary'])
        ax2.fill(angles, values, alpha=0.25, color=self.color_theme['primary'])
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(criteria_names, size=8)
        ax2.set_title('(b) Criteria Importance Radar', pad=10)
        
        ax3 = fig.add_subplot(gs[1, :])
        sns.kdeplot(data=importance_scores, color=self.color_theme['primary'], ax=ax3)
        ax3.set_title('(c) Importance Score Distribution', pad=10)
        ax3.set_xlabel('Importance Score')
        ax3.set_ylabel('Density')
        
        stats_text = (f'Mean: {np.mean(importance_scores):.3f}\n'
                     f'Std: {np.std(importance_scores):.3f}\n'
                     f'Max: {np.max(importance_scores):.3f}')
        ax3.text(0.95, 0.95, stats_text,
                transform=ax3.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def analyze_sample(self, sample_data, save_dir='analysis_results'):
        os.makedirs(save_dir, exist_ok=True)
        
        attention_data = self.get_attention_maps(sample_data)
        
        images = sample_data[0]
        processed_images = []
        for i in range(images.size(1)):
            if torch.sum(images[0, i]) != 0:
                image_attention = np.mean([attn[0, i] for attn in attention_data['visual_attention']], axis=0)
                self.visualize_attention_analysis(
                    images[0, i],
                    image_attention,
                    save_path=os.path.join(save_dir, f'fig1_attention_image_{i}.png')
                )
                processed_images.append(i)
        
        text = sample_data[1][0]
        text_decoded = self.model.tokenizer.decode(text, skip_special_tokens=True)
        text_attention = attention_data['text_attention'][0]
        
        self.visualize_text_analysis(
            text_decoded,
            text_attention,
            save_path=os.path.join(save_dir, 'fig2_text_attention.png'),
            top_k=15
        )
        
        with torch.no_grad():
            model_inputs = [x.to(self.device) for x in sample_data[:5]]
            output, criterion_pred, score_pred, _ = self.model(*model_inputs)
            
            importance_scores = torch.sigmoid(criterion_pred)[0].cpu().numpy()
            
            self.visualize_criteria_analysis(
                importance_scores,
                save_path=os.path.join(save_dir, 'fig3_criteria_analysis.png')
            )
            
            analysis_results = {
                'predicted_score': float(score_pred[0].item()),
                'true_score': float(sample_data[3][0].item()),
                'score_mask': float(sample_data[5][0].item()),  # 添加score_mask
                'processed_images': processed_images,
                'criteria_importance': {
                    f'criterion_{i+1}': {
                        'description': UNESCO_CRITERIA[i],
                        'importance_score': float(importance_scores[i])
                    }
                    for i in range(len(UNESCO_CRITERIA))
                },
                'text_analysis': {
                    'text': text_decoded,
                    'attention_weights': text_attention.tolist()
                },
                'attention_statistics': {
                    'scale_weights': attention_data['scale_weights'][0].tolist(),
                    'mean_text_attention': float(np.mean(text_attention)),
                    'max_text_attention': float(np.max(text_attention))
                }
            }
            
            self.generate_analysis_report(analysis_results, save_dir)
            
            with open(os.path.join(save_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=4, ensure_ascii=False)
            
            return analysis_results

    def generate_analysis_report(self, results, save_dir):
        report_path = os.path.join(save_dir, 'analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Heritage Site Analysis Report\n\n")
            
            f.write("## 1. Overall Score Analysis\n")
            f.write(f"- Predicted Heritage Score: {results['predicted_score']:.3f}\n")
            f.write(f"- True Heritage Score: {results['true_score']:.3f}\n")
            f.write(f"- Score Difference: {abs(results['predicted_score'] - results['true_score']):.3f}\n")
            if results['score_mask'] > 0:
                f.write("- Score is verified (has ground truth)\n")
            else:
                f.write("- Score is predicted (no ground truth available)\n")
            f.write("\n")
            
            f.write("## 2. UNESCO Criteria Analysis\n")
            
            criteria_scores = [
                (k, v['importance_score'], v['description'])
                for k, v in results['criteria_importance'].items()
            ]
            criteria_scores.sort(key=lambda x: x[1], reverse=True)
            
            f.write("### Top 3 Most Important Criteria:\n")
            for criterion, score, desc in criteria_scores[:3]:
                f.write(f"#### {criterion}: {score:.3f}\n")
                f.write(f"- Description: {desc}\n")
                f.write(f"- Importance Score: {score:.3f}\n\n")
            
            f.write("## 3. Text Content Analysis\n")
            f.write("### Site Description:\n")
            f.write(f"{results['text_analysis']['text']}\n\n")
            
            words = results['text_analysis']['text'].split()
            word_weights = results['text_analysis']['attention_weights']
            word_importance = list(zip(words, word_weights[:len(words)]))
            word_importance.sort(key=lambda x: x[1], reverse=True)
            
            f.write("### Key Phrases (by attention weight):\n")
            for word, weight in word_importance[:5]:
                f.write(f"- {word}: {weight:.3f}\n")
            f.write("\n")
            
            f.write("## 4. Visual Analysis\n")
            f.write(f"- Number of processed images: {len(results['processed_images'])}\n")
            f.write("- Scale attention weights:\n")
            for i, weight in enumerate(results['attention_statistics']['scale_weights']):
                f.write(f"  - Scale {i+1}: {weight:.3f}\n")
            f.write("\n")
            
            f.write("## 5. Visualization Guide\n")
            f.write("The following visualizations have been generated:\n\n")
            
            f.write("1. **Attention Analysis** (fig1_attention_image_*.png):\n")
            f.write("   - Original image\n")
            f.write("   - Attention heatmap showing model focus areas\n")
            f.write("   - Overlay visualization\n")
            f.write("   - Attention distribution analysis\n\n")
            
            f.write("2. **Text Attention Analysis** (fig2_text_attention.png):\n")
            f.write("   - Word-level attention weights\n")
            f.write("   - Key phrase identification\n")
            f.write("   - Attention distribution across text\n\n")
            
            f.write("3. **Criteria Analysis** (fig3_criteria_analysis.png):\n")
            f.write("   - UNESCO criteria importance scores\n")
            f.write("   - Comparative visualization of criteria weights\n")
            f.write("   - Distribution of importance scores\n")

def main():
    model_path = './pthfile/best_enhanced_heritage_model.pth'
    data_root = './final_dataset/'
    
    analyzer = HeritageModelAnalyzer(model_path)
    
    dataset = MultiImageHeritageDataset(
        data_root,
        transform=analyzer.transform,
        max_images=5
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    print("\nStarting analysis of heritage sites...")
    for i, batch in enumerate(tqdm(dataloader, desc="Analyzing samples")):
        print(f"\nAnalyzing sample {i+1}")
        results = analyzer.analyze_sample(
            batch,
            save_dir=f'analysis_results/sample_{i+1}'
        )
        
        print(f"\nResults for sample {i+1}:")
        print(f"Predicted Score: {results['predicted_score']:.3f}")
        print(f"True Score: {results['true_score']:.3f}")
        print("\nTop 3 most important criteria:")
        
        criteria_scores = [
            (k, v['importance_score']) 
            for k, v in results['criteria_importance'].items()
        ]
        criteria_scores.sort(key=lambda x: x[1], reverse=True)
        
        for criterion, score in criteria_scores[:3]:
            print(f"{criterion}: {score:.3f}")
        
        print("\nAnalysis complete. Results saved in 'analysis_results' directory.")
        
        if i >= 20:
            break

if __name__ == "__main__":
    main()