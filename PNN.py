import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class ProgressiveNeuralNetwork(nn.Module):
    """
    Une classe représentant un réseau de neurones progressif (PNN) pour l'apprentissage multi-tâches.
    Chaque tâche est apprise par une colonne neuronale distincte, avec des connexions latérales entre les 
    colonnes pour permettre le transfert de connaissances et éviter les oublis catastrophiques.
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Initialiser une nouvelle instance de ProgressiveNeuralNetwork.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.columns = nn.ModuleList()  # Use ModuleList to register modules
        self.hidden_outputs = []  # Liste contenant les sorties des couches cachées pour les connexions latérales
        self.criterion = nn.CrossEntropyLoss()
        torch.manual_seed(0)  # pour la reproductibilité

    def prune(self, percentage):
        """
        Élaguer (Prune) le réseau en supprimant réellement les poids de plus petite magnitude.
        De cette manière, on crée des nouvelles matrices de poids plus petites.
        
        Args:
            percentage: Pourcentage de poids à élaguer (0-100)
        """
        import torch.nn.utils.prune as prune
        
        for column in self.columns:
            for module in column.modules():
                if isinstance(module, nn.Linear):
                    # On commence par appliquer le masque de pruning
                    prune.l1_unstructured(module, name='weight', amount=percentage / 100)
                    
                    # Ensuite, on supprime définitivement les poids (convertit le masque en zéros réels dans le paramètre)
                    prune.remove(module, 'weight')
                    
                    # Maintenant, filtrer les lignes/colonnes nulles en créant une nouvelle matrice de poids plus petite
                    weight = module.weight.data
                    bias = module.bias.data if module.bias is not None else None
                    
                    # Trouver quelles lignes (neurones de sortie) ont des valeurs non nulles
                    active_rows = (weight.abs().sum(dim=1) > 0)
                    
                    if active_rows.sum() > 0 and active_rows.sum() < weight.shape[0]:
                        # Créer une nouvelle couche linéaire de dimension réduite
                        new_out_features = active_rows.sum().item()
                        new_linear = nn.Linear(module.in_features, new_out_features)
                        
                        # Copier les lignes actives dans la nouvelle couche
                        new_linear.weight.data = weight[active_rows, :]
                        if bias is not None:
                            new_linear.bias.data = bias[active_rows]
                        
                        # Remplacer l'ancien module sur place
                        # On doit parcourir les attributs du parent pour faire le remplacement
                        for parent_name, parent in column.named_children():
                            if parent is module:
                                setattr(column, parent_name, new_linear)

    class Column(nn.Module):
        """
        Une réseau de neurones colonne composée d'une seule couche cachée et d'une couche de sortie.
        """

        def __init__(self, input_dim, hidden_dim, output_dim):
            """
            Initialisation

            Paramètres
            ------------
            - input_dim (int): dimension de la couche d'entrée.
            - hidden_dim (int): dimension de la couche cachée.
            - output_dim (int): dimension de la couche de sortie.
            """
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, lateral_input=None):
            """
            Feed forward à travers la colonne.

            Paramètres
            ------------
            - x (torch.Tensor): features d'entrée.
            - lateral_input (torch.Tensor, optionnel): connexions latérales depuis les tâches/colonnes précédentes.

            Retourne
            -----------
            - torch.Tensor: sortie de la colonne.
            """

            if lateral_input is not None:
                x = torch.cat((x, lateral_input), dim=1)  # concaténer l'entrée latérale
            x = torch.relu(self.hidden(x))  # couche cachée avec fonction d'activation ReLU
            x = self.output(x)  # couche de sortie

            return x

    def train_new_task(self, new_data, new_labels, hidden_dim=10, epochs=200, batch_size=32):
        """
        Entraîner une nouvelle colonne pour une nouvelle tâche.

        Paramètres:
        - new_data (numpy.ndarray): Données d'entraînement pour la nouvelle tâche.
        - new_labels (numpy.ndarray): Labels d'entraînement pour la nouvelle tâche.
        - hidden_dim (int, optionnel): Dimension de la couche cachée. Par défaut 10.
        - epochs (int, optionnel): Nombre d'époques d'entraînement. Par défaut 200.
        - batch_size (int, optionnel): Taille du batch pour l'entraînement. Par défaut 32.
        """
        # Préparation des données
        new_data_tensor = torch.FloatTensor(new_data)
        new_labels_tensor = torch.LongTensor(new_labels)
        new_task_dataset = TensorDataset(new_data_tensor, new_labels_tensor)
        new_task_loader = DataLoader(new_task_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Initialiser une nouvelle colonne
        input_dim = new_data.shape[1]
        output_dim = len(set(new_labels))
        lateral_dims = [h[0] for h in self.hidden_outputs]
        total_input_dim = input_dim + sum(lateral_dims)
        new_column = self.Column(input_dim=total_input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        optimizer = optim.Adam(new_column.parameters())

        # Boucle d'entraînement
        for epoch in range(epochs):
            for batch_data, batch_labels in new_task_loader:
                optimizer.zero_grad()

                # Calculer les input latéraux depuis les tâches/colonnes précédentes
                lateral_input = self.calculate_lateral_input(batch_data)

                # Concaténer les input latéraux si disponibles
                if lateral_input is not None:
                    input_to_column = torch.cat((batch_data, lateral_input), dim=1)
                else:
                    input_to_column = batch_data

                # Passage avant et calcul de la perte
                output = new_column(input_to_column)
                loss = self.criterion(output, batch_labels)
                loss.backward()
                optimizer.step()

        # Sauvegarder la colonne neuronale nouvellement entraînée et sa fonction de sortie de couche cachée
        current_hidden = new_column.hidden
        self.hidden_outputs.append(
            (hidden_dim, lambda x: torch.relu(current_hidden(x))))
        self.columns.append(new_column)

    def calculate_lateral_input(self, batch_data, latest_task_index=None):
        """
        Calculer les input latéraux basées sur les sorties cachées des colonnes précédemment entraînées..

        Paramètres:
        - batch_data (torch.Tensor): Les données d'entrée pour le batch actuel.
        - latest_task_index (int, optionnel): Spécifie jusqu'à quelle tâche les sorties cachées doivent
                                               être considérées pour générer les connexions latérales.
                                               Si None, utilise toutes les tâches.

        Retourne:
        - torch.Tensor ou None: Les input latéraux concaténées pour le batch actuel si elles existent,
                                sinon retourne None.
        """
        hidden_outputs = []
        hidden_output = None

        if latest_task_index is None:
            latest_task_index = len(self.hidden_outputs)

        for i, (hidden_dim, hidden_function) in enumerate(self.hidden_outputs[:latest_task_index]):
            with torch.no_grad():  # Pas besoin de calculer les gradients
                # Si c'est la première sortie cachée, la calculer uniquement depuis batch_data
                if hidden_output is None:
                    hidden_output = hidden_function(batch_data)
                # Sinon, concaténer la dernière sortie cachée à batch_data et calculer la suivante
                else:
                    hidden_output = torch.cat((batch_data, torch.cat(hidden_outputs, dim=1)), dim=1)
                    hidden_output = hidden_function(hidden_output)

                hidden_outputs.append(hidden_output)

        if hidden_output is not None:
            lateral_input = torch.cat(hidden_outputs, dim=1)
            return lateral_input
        else:
            return None # Pas d'input latéral disponible pour la première colonne

    def predict(self, data, task_index):
        """
        Prédire les labels des données fournies pour la tâche à l'index donné.

        Paramètres:
        - data (numpy.ndarray): Les données pour lesquelles faire les prédictions.
        - task_index (int): L'index de la tâche pour laquelle faire les prédictions.

        Retourne:
        - numpy.ndarray: Les labels prédits.
        """
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)

            # Si on prédit un seul échantillon, ajouter une dimension pour correspondre à la forme attendue
            if len(data_tensor.shape) == 1:
                data_tensor = data_tensor.unsqueeze(0)

            column = self.columns[task_index]

            lateral_input = self.calculate_lateral_input(data_tensor, task_index)

            # Passage avant à travers la colonne neuronale
            output = column(data_tensor, lateral_input=lateral_input)

            # Si un seul échantillon, retourner directement l'index de la valeur max
            if len(output.shape) == 1:
                predicted = output.argmax()
            # Pour plusieurs échantillons, retourner les indices des valeurs max le long de la dimension 1
            else:
                _, predicted = torch.max(output, 1)

            return predicted.cpu().numpy()

    def accuracy(self, data, labels, task_index):
        """
        Calculer la précision du modèle pour la tâche à l'index donné.

        Paramètres:
        - data (numpy.ndarray): Les données de test.
        - labels (numpy.ndarray): Les vrais labels pour les données de test.
        - task_index (int): L'index de la tâche pour laquelle calculer la précision.

        Retourne:
        - float: La précision du modèle pour la tâche donnée, en pourcentage.
        """
        data = torch.FloatTensor(data)
        labels = torch.LongTensor(labels)

        # Faire les prédictions
        predicted = self.predict(data, task_index)

        # Calculer la précision
        correct = (predicted == labels.cpu().numpy()).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        return accuracy

