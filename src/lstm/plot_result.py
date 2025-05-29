import json
import matplotlib.pyplot as plt
import os

def plot_loss_curves(experiment_results, title):
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    for result in experiment_results:
        history = result['history']
        name = result['name'].replace('_', ' ').title()
        f1_score = result['f1_score']
        plt.plot(history['loss'], label=f'{name} Train Loss')
        plt.plot(history['val_loss'], '--', label=f'{name} Val Loss (F1: {f1_score:.3f})')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Sparse Categorical Crossentropy Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{title.replace(' ', '_').lower()}.png")
    plt.show()

def print_f1_summary(experiment_results, title):
    print(f"\n{title} F1 Score Summary")
    sorted_results = sorted(experiment_results, key=lambda x: x['f1_score'], reverse=True)
    for result in sorted_results:
        name = result['name'].replace('_', ' ').title()
        print(f"{name}: {result['f1_score']:.4f}")


if __name__ == '__main__':
    try:
        with open('results/all_experiment_results.json', 'r') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print("Error: 'results/all_experiment_results.json' not found.")

    plot_loss_curves(all_results['num_layers'], 'Impact of Number of LSTM Layers')
    print_f1_summary(all_results['num_layers'], 'Number of LSTM Layers')

    plot_loss_curves(all_results['cell_count'], 'Impact of LSTM Cell Count')
    print_f1_summary(all_results['cell_count'], 'LSTM Cell Count')
    
    plot_loss_curves(all_results['direction'], 'Impact of LSTM Layer Directionality')
    print_f1_summary(all_results['direction'], 'LSTM Layer Directionality')
