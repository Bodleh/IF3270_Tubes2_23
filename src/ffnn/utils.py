import matplotlib.pyplot as plt
import pickle
import numpy as np

from pyvis.network import Network

def visualize_network(model, max_nodes_per_layer=None, limit_connections=False):
    net = Network(height="1000px", width="100%", directed=True, notebook=False)
    layer_nodes = []
    layer_indices = []
    total_layers = len(model.layers) + 1
    layer_spacing = 300
    start_x = -(total_layers-1) * layer_spacing / 2
    input_dim = model.layers[0].input_dim
    input_nodes = []
    input_indices = []
    if max_nodes_per_layer is not None and input_dim > max_nodes_per_layer:
        selected_indices = np.linspace(0, input_dim-1, max_nodes_per_layer, dtype=int)
    else:
        selected_indices = np.arange(input_dim)
    for i, idx in enumerate(selected_indices):
        node_id = f"input_{i}"
        input_nodes.append(node_id)
        input_indices.append(idx)
        y_pos = (i - len(selected_indices)/2) * 60
        net.add_node(node_id, label=f"In {idx}", color="#97c2fc",
                    x=start_x, y=y_pos, fixed=True)
    layer_nodes.append(input_nodes)
    layer_indices.append(input_indices)
    for layer_idx, layer in enumerate(model.layers):
        layer_name = "output" if layer_idx == len(model.layers) - 1 else f"hidden_{layer_idx+1}"
        output_dim = layer.output_dim
        curr_layer_nodes = []
        curr_layer_indices = []
        if max_nodes_per_layer is not None and output_dim > max_nodes_per_layer:
            selected_indices = np.linspace(0, output_dim-1, max_nodes_per_layer, dtype=int)
        else:
            selected_indices = np.arange(output_dim)
        color = "#ff9999" if layer_name == "output" else "#ffb347"
        x_pos = start_x + (layer_idx + 1) * layer_spacing
        for i, idx in enumerate(selected_indices):
            node_id = f"{layer_name}_{i}"
            curr_layer_nodes.append(node_id)
            curr_layer_indices.append(idx)
            label = f"Out {idx}" if layer_name == "output" else f"H{layer_idx+1}_{idx}"
            y_pos = (i - len(selected_indices)/2) * 60
            net.add_node(node_id, label=label, color=color,
                        x=x_pos, y=y_pos, fixed=True)
        layer_nodes.append(curr_layer_nodes)
        layer_indices.append(curr_layer_indices)
        prev_nodes = layer_nodes[layer_idx]
        prev_indices = layer_indices[layer_idx]
        max_connections = None
        if limit_connections and max_nodes_per_layer and max_nodes_per_layer > 10:
            max_connections = 5
        for i, target_node_id in enumerate(curr_layer_nodes):
            target_idx = curr_layer_indices[i]
            if max_connections and len(prev_nodes) > max_connections:
                weights = [abs(layer.W[prev_indices[j], target_idx]) for j in range(len(prev_nodes))]
                source_indices = np.argsort(weights)[-max_connections:]
            else:
                source_indices = range(len(prev_nodes))
            for j in source_indices:
                source_node_id = prev_nodes[j]
                source_idx = prev_indices[j]
                weight = layer.W[source_idx, target_idx]
                gradient = layer.dW[source_idx, target_idx]
                edge_color = "blue" if weight >= 0 else "red"
                width = min(max(abs(weight) * 3, 0.5), 10)
                net.add_edge(source_node_id, target_node_id, 
                            color=edge_color, 
                            width=width, 
                            title=f"Weight: {weight:.4f}\nGradient: {gradient:.4f}")
    net.toggle_physics(False)
    html_file = "ffnn_visualization.html"
    net.save_graph(html_file)
    print(f"Network visualization saved to {html_file}")

def plot_weights_distribution(model, layers_indices: list[int]):
    n_layers = len(layers_indices)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    for i, idx in enumerate(layers_indices):
        if idx < len(model.layers):
            weights = model.layers[idx].W.flatten()
            
            axs[i].hist(weights, bins=30, color=f'C{i}', edgecolor='black', linewidth=0.5)
            axs[i].set_title(f"Layer {idx} Weights")
            axs[i].grid(True, linestyle='--', alpha=0.3)
    
    for i in range(len(layers_indices), len(axs)):
        axs[i].set_visible(False)
    
    fig.text(0.5, 0.02, 'Weight Value', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=12)
    
    fig.suptitle("Weight Distributions", fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.show()

def plot_gradients_distribution(model, layers_indices: list[int]):
    n_layers = len(layers_indices)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    for i, idx in enumerate(layers_indices):
        if idx < len(model.layers):
            gradients = model.layers[idx].dW.flatten()
            
            axs[i].hist(gradients, bins=30, color=f'C{i}', edgecolor='black', linewidth=0.5)
            axs[i].set_title(f"Layer {idx} Gradients")
            axs[i].grid(True, linestyle='--', alpha=0.3)
    
    for i in range(len(layers_indices), len(axs)):
        axs[i].set_visible(False)
    
    fig.text(0.5, 0.02, 'Gradient Value', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=12)
    
    fig.suptitle("Gradient Distributions", fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.show()

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model