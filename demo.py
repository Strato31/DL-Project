"""
Comparaison de 3 méthodes pour l'apprentissage multi-tâches
Utilise le dataset emotion trouvable sur HuggingFace, qui contient des textes labellisés selon 6 émotions. 
Les 6 émotions du dataset : 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise

Avec ce dataset, on se propose de réaliser 2 tâches de classification d'émotions :
- Tâche 1 : Emotion binaire (négatif [0,3,4] versus positif [1,2,5])
- Tâche 2 : Émotions négatives détail (sadness=0, anger=1, fear=2)

On compare 3 méthodes d'apprentissage multi-tâches :
1. Modèles Indépendants : Un modèle par tâche (témoin)
2. Fine-tuning : Entraîner sur Tâche 1 puis fine-tuner sur Tâche 2
3. PNN : Progressive Neural Network avec connexions latérales

Enfin, en ouverture, on teste l'impact du pruning sur le modèle PNN entraîné.
"""

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from pathlib import Path
from PNN import ProgressiveNeuralNetwork
import matplotlib.pyplot as plt

# Configuration
NUM_SAMPLES = 1000      # Échantillons par tâche
FEATURE_DIM = 128       # Dimension des features extraites des données
HIDDEN_DIM = 64         # Dimension de la couche cachée
EPOCHS = 100            # Époques d'entraînement
BATCH_SIZE = 32         # Taille du batch

# Configuration des tâches à comparer (mettre dans l'ordre souhaité ici)
TASK_ORDER = ['binaire', 'negative', 'positive', 'full']
NUM_TASKS = 4  # entre 2 et 4 : par exemple 3 pour comparer 3 tâches successives
# Choix des pourcentages de pruning à tester
prune_percents = [0, 1, 5, 10, 20]

print("=" * 80)
print("COMPARAISON DE MÉTHODES - DATASET emotion")
print("=" * 80)
print(f"Configuration : {NUM_SAMPLES} échantillons/tâche")
print(f"Features : {FEATURE_DIM}D, Hidden : {HIDDEN_DIM}, Epochs : {EPOCHS}")
print("=" * 80)

# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================
print("\n[PRÉPARATION] Chargement du dataset texte...")

#si données pas déjà dans le dossier data, les télécharger
saved_path = Path("data") / "emotion_text_dataset" / "saved"
if not saved_path.exists():
    print("   --> Téléchargement du dataset emotion depuis HuggingFace...")
    dataset = load_dataset("emotion")
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(saved_path))
else:
    print("   --> Chargement du dataset emotion depuis le disque...")
    dataset = load_from_disk(str(saved_path))

# Extraction des données textuelles et tokenization
train_data, test_data = dataset['train'], dataset['test']
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Mapping des labels d'émotions
# 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def extract_features(samples, max_samples):
    """Extrait les features des textes"""
    features = []
    labels = []
    for i, sample in enumerate(samples):
        if i >= max_samples: break
        tokens = tokenizer(sample['text'], padding='max_length', truncation=True, max_length=FEATURE_DIM, return_tensors='pt')
        feat = tokens['input_ids'].squeeze().float().numpy() / 30000.0
        features.append(feat)
        labels.append(sample['label'])
    return np.array(features), np.array(labels)

def create_task_data(original_features, original_labels, task_type, num_samples):
    """
    Crée les données pour une tâche spécifique
    Paramètres:
    -----------
    - original_features (numpy.ndarray): Features extraites originales.
    - original_labels (numpy.ndarray): Labels originaux.
    - task_type:
        'binaire' : binaire négatif vs positif
        'negative' : classification des émotions négatives
        'positive' : classification des émotions positives
        'full' : classification complète 6 classes
    - num_samples (int): Nombre d'échantillons à utiliser.

    Retourne:
    -----------
    - task_features (numpy.ndarray): Features pour la tâche.
    - task_labels (numpy.ndarray): Labels pour la tâche.
    """
    if task_type == 'binaire':
        # 0=négatif (sadness, anger, fear), 1=positif (joy, love, surprise)
        negative_emotions = [0, 3, 4]  # sadness, anger, fear
        positive_emotions = [1, 2, 5]  # joy, love, surprise
        
        new_labels = []
        for label in original_labels:
            if label in negative_emotions:
                new_labels.append(0)
            else:
                new_labels.append(1)
        new_labels = np.array(new_labels)
        
    elif task_type == 'negative':
        # Garder seulement les émotions négatives
        # sadness=0, anger=1, fear=2
        mask = np.isin(original_labels, [0, 3, 4])
        features = original_features[mask]
        labels = original_labels[mask]
        
        # Remapper : sadness(0):0, anger(3):1, fear(4):2
        label_map = {0: 0, 3: 1, 4: 2}
        new_labels = np.array([label_map[l] for l in labels])
        
        # Limiter au nombre d'échantillons
        if len(features) > num_samples:
            indices = np.random.choice(len(features), num_samples, replace=False)
            return features[indices], new_labels[indices]
        return features, new_labels
        
    elif task_type == 'positive':
        # Garder seulement les émotions positives
        # joy=0, love=1, surprise=2
        mask = np.isin(original_labels, [1, 2, 5])
        features = original_features[mask]
        labels = original_labels[mask]
        
        # Remapper : joy(1):0, love(2):1, surprise(5):2
        label_map = {1: 0, 2: 1, 5: 2}
        new_labels = np.array([label_map[l] for l in labels])
        
        if len(features) > num_samples:
            indices = np.random.choice(len(features), num_samples, replace=False)
            return features[indices], new_labels[indices]
        return features, new_labels
        
    elif task_type == 'full':
        new_labels = original_labels
    
    else:
        raise ValueError(f"Task type inconnu : {task_type}")
    
    if len(original_features) > num_samples:
        indices = np.random.choice(len(original_features), num_samples, replace=False)
        return original_features[indices], new_labels[indices]
    
    return original_features, new_labels

# Extraire les features pour train et test
print("   --> Extraction des features...")
train_features, train_labels = extract_features(train_data, 5000)
test_features, test_labels = extract_features(test_data, 2000)
print(f"   OK - Train: {len(train_features)}, Test: {len(test_features)} échantillons extraits")



task_meta = {
    'binaire': ("Sentiment (Négatif vs Positif)", 2),
    'negative': ("Émotions Négatives (sadness, anger, fear)", 3),
    'positive': ("Émotions Positives (joy, love, surprise)", 3),
    'full': ("Toutes Émotions (6 classes)", 6),
}

# Créer dynamiquement les tâches en utilisant train/val/test
print("\n[TÂCHES DÉFINIES]")
tasks = []
for idx, task_type in enumerate(TASK_ORDER[:NUM_TASKS]):
    name, num_classes = task_meta[task_type]
    
    # Créer les données pour chaque split
    train_feats, train_labs = create_task_data(train_features, train_labels, task_type, NUM_SAMPLES)
    test_feats, test_labs = create_task_data(test_features, test_labels, task_type, NUM_SAMPLES // 2)
    
    tasks.append({
        'name': name,
        'type': task_type,
        'features': train_feats,      # Pour entraînement
        'labels': train_labs,
        'test_features': test_feats,  # Pour test
        'test_labels': test_labs,
        'num_classes': num_classes,
    })
    print(f"   Tâche {i+1} : {name}")
    print(f"           --> Train: {len(train_feats)}, Test: {len(test_feats)}")
    print(f"           --> {num_classes} classes")

# ============================================================================
# MÉTHODE 1 : MODÈLES INDÉPENDANTS (BASELINE)
# ============================================================================
print("\n" + "=" * 80)
print("MÉTHODE 1 : MODÈLES INDÉPENDANTS (Baseline)")
print("=" * 80)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(features, labels, hidden_dim, output_dim, epochs, batch_size):
    """Entraîne un modèle de Classifier simple"""
    model = SimpleClassifier(features.shape[1], hidden_dim, output_dim)
    
    dataset = TensorDataset(torch.FloatTensor(features), torch.LongTensor(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_data, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_model(model, features, labels):
    """Évalue un modèle"""
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(features)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted.numpy() == labels).sum()
        accuracy = 100 * correct / len(labels)
    return accuracy

def clone_state_dict(state_dict):
    """Clone proprement un state_dict pour éviter le partage de 
    tenseurs lors de la phase de fine-tuning
    """
    return {k: v.clone() for k, v in state_dict.items()}

accs_independent = [] # stocke les accuracies des modèles indépendants
models_independent = [] # stocke les modèles indépendants

for i, task in enumerate(tasks):
    print(f"   --> Entraînement Tâche {i+1} ({task['name']})...")
    model = train_model(task['features'], task['labels'], HIDDEN_DIM, task['num_classes'], EPOCHS, BATCH_SIZE)
    acc = evaluate_model(model, task['test_features'], task['test_labels'])
    models_independent.append(model)
    accs_independent.append(acc)
    print(f"       OK - Précision (Test) : {acc:.2f}%")

print(f"\n   RÉSULTATS :")
for i, task in enumerate(tasks):
    print(f"   - Tâche {i+1} ({task['name']}) : {accs_independent[i]:.2f}%")

# ============================================================================
# MÉTHODE 2 : FINE-TUNING
# ============================================================================
print("\n" + "=" * 80)
print("MÉTHODE 2 : FINE-TUNING (Sequential Transfer Learning)")
print("=" * 80)

print("   --> Entraînement initial sur Tâche 1 ...")
model_finetuned = train_model(tasks[0]['features'], tasks[0]['labels'], HIDDEN_DIM,
                              tasks[0]['num_classes'], EPOCHS, BATCH_SIZE)

# Accuracies courantes (après apprentissage de chaque tâche)
accs_finetune_current = [evaluate_model(model_finetuned, tasks[0]['test_features'], tasks[0]['test_labels'])]
saved_fc2_states = [clone_state_dict(model_finetuned.fc2.state_dict())]
print(f"   OK - Tâche 1 (Test) : {accs_finetune_current[0]:.2f}%")

# Fine-tuning séquentiel pour les autres tâches
for i in range(1, len(tasks)):
    task = tasks[i]
    print(f"   --> Fine-tuning sur Tâche {i+1} ({task['name']})...")

    # Adapter la couche de sortie au nouveau nombre de classes
    model_finetuned.fc2 = nn.Linear(HIDDEN_DIM, task['num_classes'])

    dataset = TensorDataset(torch.FloatTensor(task['features']), torch.LongTensor(task['labels']))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_finetuned.parameters(), lr=0.001)

    model_finetuned.train()
    for epoch in range(EPOCHS):
        for batch_data, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model_finetuned(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    acc_current = evaluate_model(model_finetuned, task['test_features'], task['test_labels'])
    accs_finetune_current.append(acc_current)
    saved_fc2_states.append(clone_state_dict(model_finetuned.fc2.state_dict()))
    print(f"       OK - Précision Tâche {i+1} (Test) : {acc_current:.2f}%")

# Évaluer l'oubli pour toutes les tâches en restaurant leurs couches
accs_finetune_final = []
for i, task in enumerate(tasks):
    model_finetuned.fc2 = nn.Linear(HIDDEN_DIM, task['num_classes'])
    model_finetuned.fc2.load_state_dict(saved_fc2_states[i])
    acc = evaluate_model(model_finetuned, task['test_features'], task['test_labels'])
    accs_finetune_final.append(acc)

print(f"\n   RÉSULTATS :")
for i, task in enumerate(tasks):
    forget = accs_finetune_current[i] - accs_finetune_final[i]
    print(f"   - Tâche {i+1} ({task['name']}) : {accs_finetune_final[i]:.2f}%   Oubli: {forget:.2f}%")

# ============================================================================
# MÉTHODE 3 : PROGRESSIVE NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 80)
print("MÉTHODE 3 : PROGRESSIVE NEURAL NETWORK (PNN)")
print("=" * 80)

pnn = ProgressiveNeuralNetwork(input_size=FEATURE_DIM, hidden_size=HIDDEN_DIM)
accs_pnn_before = []

for i, task in enumerate(tasks):
    print(f"   --> Entraînement Tâche {i+1} ({task['name']})...")
    pnn.train_new_task(task['features'], task['labels'], hidden_dim=HIDDEN_DIM,
                       epochs=EPOCHS, batch_size=BATCH_SIZE)
    acc = pnn.accuracy(task['test_features'], task['test_labels'], task_index=i)
    accs_pnn_before.append(acc)
    print(f"       OK - Précision Tâche {i+1} (Test) : {acc:.2f}%")

# Évaluer toutes les tâches après entraînement complet
accs_pnn_after = []
for i, task in enumerate(tasks):
    acc = pnn.accuracy(task['test_features'], task['test_labels'], task_index=i)
    accs_pnn_after.append(acc)

print(f"\n   RÉSULTATS :")
for i, task in enumerate(tasks):
    forget = accs_pnn_before[i] - accs_pnn_after[i]
    print(f"   - Tâche {i+1} ({task['name']}) : {accs_pnn_after[i]:.2f}%   Oubli: {forget:.2f}%")

# ============================================================================
# COMPARAISON FINALE
# ============================================================================
print("\n" + "=" * 80)
print("TABLEAU COMPARATIF FINAL")
print("=" * 80)

print("\n┌─────────────────────────┬───────────────┬───────────────┬──────────────┐")
print("│ Méthode                 │ Acc. Moyenne  │ Acc. Dernière │ Oubli moyen  │")
print("├─────────────────────────┼───────────────┼───────────────┼──────────────┤")

# Gabarit de ligne pour un alignement uniforme
row_fmt = "│ {method:<23} │ {acc_mean:>13} │ {acc_last:>13} │ {forget:>12} │"

# Indépendants
acc_mean_indep = float(np.mean(accs_independent)) if len(accs_independent) else 0
row_indep = row_fmt.format(
    method="1. Indépendants",
    acc_mean=f"{acc_mean_indep:6.2f}%",
    acc_last=f"{accs_independent[-1]:6.2f}%" if accs_independent else "N/A",
    forget="N/A",
)
print(row_indep)

# Fine-tuning
forget_ft = [accs_finetune_current[i] - accs_finetune_final[i] for i in range(len(tasks))]
acc_mean_ft = float(np.mean(accs_finetune_final)) if len(accs_finetune_final) else 0
acc_last_ft = accs_finetune_final[-1] if accs_finetune_final else 0
forget_mean_ft = float(np.mean(forget_ft)) if len(forget_ft) else 0
row_ft = row_fmt.format(
    method="2. Fine-tuning",
    acc_mean=f"{acc_mean_ft:6.2f}%",
    acc_last=f"{acc_last_ft:6.2f}%",
    forget=f"{forget_mean_ft:6.2f}%",
)
print(row_ft)

# PNN
forget_pnn = [accs_pnn_before[i] - accs_pnn_after[i] for i in range(len(tasks))]
acc_mean_pnn = float(np.mean(accs_pnn_after)) if len(accs_pnn_after) else 0
acc_last_pnn = accs_pnn_after[-1] if accs_pnn_after else 0
forget_mean_pnn = float(np.mean(forget_pnn)) if len(forget_pnn) else 0
row_pnn = row_fmt.format(
    method="3. PNN (Progressif)",
    acc_mean=f"{acc_mean_pnn:6.2f}%",
    acc_last=f"{acc_last_pnn:6.2f}%",
    forget=f"{forget_mean_pnn:6.2f}%",
)
print(row_pnn)

print("└─────────────────────────┴───────────────┴───────────────┴──────────────┘")

# ==========================================================================
# VISUALISATION DES ACCURACIES PAR TÂCHE ET PAR MÉTHODE
# ==========================================================================
methods = ["Indépendants", "Fine-tuning", "PNN"]
task_labels = [f"T{i+1}" for i in range(len(tasks))]
acc_matrix = np.array([
    accs_independent,
    accs_finetune_final,
    accs_pnn_after,
])

# Création du graphique en barres groupées
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(task_labels))
width = 0.25

for idx, method in enumerate(methods):
    ax.bar(x + (idx - 1) * width, acc_matrix[idx], width, label=method)

ax.set_ylabel("Précision (%)")
ax.set_title("Précision par tâche et par méthode")
ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("comparison_accuracies.png", dpi=150)
print("\nGraphique des accuracies enregistré dans comparison_accuracies.png")

# Graphique de l'oubli (Fine-tuning vs PNN)
fig2, ax2 = plt.subplots(figsize=(10, 6))
forget_methods = ["Fine-tuning", "PNN"]
forget_matrix = np.array([
    forget_ft,
    forget_pnn,
])

for idx, method in enumerate(forget_methods):
    ax2.bar(x + (idx - 0.5) * width, forget_matrix[idx], width, label=method)

ax2.set_ylabel("Oubli (%)")
ax2.set_title("Oubli par tâche et par méthode")
ax2.set_xticks(x)
ax2.set_xticklabels(task_labels)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.legend()

plt.tight_layout()
plt.savefig("comparison_forgetting.png", dpi=150)
print("Graphique de l'oubli enregistré dans comparison_forgetting.png")


#####################################
# Tests : Pruning PNN               #
#####################################

print("\n" + "=" * 80)
print("TEST DE PRUNING SUR LE PNN ENTRAÎNÉ")
print("=" * 80)

total_params = sum(p.numel() for p in pnn.parameters())
print(f"   --> Nombre total de paramètres : {total_params}")

if total_params == 0:
    print("   Attention : Aucun paramètre détecté. Vérifiez que le modèle est bien entraîné.")
else:
    # Sauvegarder l'état du modèle avant pruning
    import copy
    original_state = copy.deepcopy(pnn.state_dict())
    
    pruning_results = []
    
    for perc in prune_percents:
        # Restaurer l'état original pour chaque test de pruning
        pnn.load_state_dict(copy.deepcopy(original_state))
        
        if perc > 0:
            # Afficher un diagnostic avant pruning
            weights_before = [p.clone() for p in pnn.parameters()]
            pnn.prune(perc)
            # Vérifier que les poids ont changé
            weights_after = list(pnn.parameters())
            changed = 0
            for w1, w2 in zip(weights_before, weights_after):
                if w1.shape == w2.shape:
                    changed += (w1 != w2).sum().item()
        
        # Compter les poids à zéro (neurones prunés)
        zero_params = sum((p == 0).sum().item() for p in pnn.parameters())
        non_zero_params = total_params - zero_params
        sparsity = 100 * zero_params / total_params if total_params > 0 else 0
        
        # Évaluer les performances après pruning sur TRAIN et TEST
        try:
            # Évaluation sur train
            accs_train = []
            for i, task in enumerate(tasks):
                acc = pnn.accuracy(task['features'], task['labels'], task_index=i)
                accs_train.append(acc)
            avg_acc_train = sum(accs_train) / len(accs_train)
            
            # Évaluation sur test
            accs_test = []
            for i, task in enumerate(tasks):
                acc = pnn.accuracy(task['test_features'], task['test_labels'], task_index=i)
                accs_test.append(acc)
            avg_acc_test = sum(accs_test) / len(accs_test)
            
            pruning_results.append({
                'percentage': int(perc * 100),
                'sparsity': sparsity,
                'zero_params': zero_params,
                'active_params': non_zero_params,
                'train_accuracies': accs_train,
                'test_accuracies': accs_test,
                'avg_train': avg_acc_train,
                'avg_test': avg_acc_test
            })
        except Exception as e:
            print(f"   Erreur lors de l'évaluation après pruning {int(perc*100)}% : {e}")
    
    # Affichage sous forme de tableau
    print("\n" + "=" * 80)
    print("TABLEAU DES RÉSULTATS DE PRUNING")
    print("=" * 80)
    print(f"┌───────────┬─────────────┬─────────────┬─────────────┬──────────┐")
    print(f"│ Sparsité  │ Poids zéro  │ Poids actifs│Acc. Train %│Acc. Test%│")
    print(f"├───────────┼─────────────┼─────────────┼─────────────┼──────────┤")
    for result in pruning_results:
        print(f"│  {result['sparsity']:>6.2f}% │ {result['zero_params']:>11} │ {result['active_params']:>12} │   {result['avg_train']:>6.2f}%   │ {result['avg_test']:>6.2f}% │")
    print(f"└───────────┴─────────────┴─────────────┴─────────────┴──────────┘")
    
    # Graphique des résultats
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    pruning_levels = [r['percentage'] for r in pruning_results]
    train_accs = [r['avg_train'] for r in pruning_results]
    test_accs = [r['avg_test'] for r in pruning_results]
    
    # Tracer train et test
    ax.plot(pruning_levels, train_accs, 'o-', linewidth=2.5, markersize=10, label='Train', color='tab:green')
    ax.plot(pruning_levels, test_accs, '^-', linewidth=2.5, markersize=10, label='Test', color='tab:red')
    
    ax.set_xlabel('Sparsité (%)', fontsize=13)
    ax.set_ylabel('Précision moyenne (%)', fontsize=13)
    ax.set_title('Impact du pruning sur la précision (Train/Test)', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pruning_results.png', dpi=150, bbox_inches='tight')
    print(f"\nGraphique des résultats de pruning enregistré dans pruning_results.png")

print("\n" + "=" * 80)
print("OK - Test de pruning terminé!")
print("=" * 80)
