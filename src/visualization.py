import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(data, column):
    """Visualize distribution of a specific column."""
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=column)
    plt.title(f'Distribution of {column}')
    plt.show()
