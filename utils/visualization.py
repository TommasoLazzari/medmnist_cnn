import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_learning_curves(history, model_name, save_dir):
    """
    Plots training & validation Loss and Accuracy over epochs.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training History - {model_name}", fontsize=14, fontweight='bold')
    
    #Loss Plot
    axes[0].plot(epochs, history['loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 's--', label='Val Loss', color='#ff7f0e', linewidth=2)
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    #Accuracy Plot
    axes[1].plot(epochs, history['accuracy'], 'o-', label='Train Acc', color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], 's--', label='Val Acc', color='#d62728', linewidth=2)
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir):
    """
    Generates a professionally stylized confusion matrix plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f"Confusion Matrix\n{model_name}",
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > thresh else "black"
            text_str = f"{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)"
            ax.text(j, i, text_str,
                    ha="center", va="center",
                    color=text_color, fontsize=9)
            
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")


def plot_roc_curves(y_true, y_pred_probs, class_names, model_name, save_dir):
    """
    Plots multi-class ROC curves for all classes and computes AUC scores.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_classes = len(class_names)
    
    y_true_onehot = np.eye(num_classes)[y_true]
    
    plt.figure(figsize=(10, 8))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
        
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_pred_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=3,
             label=f'Micro-average (AUC = {auc_micro:.4f})')
             
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves plot to {save_path}")
