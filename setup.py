from setuptools import setup

setup(name='D3MUnsupervised',
    version='1.0.0',
    description='Three wrappers for unsupervised clustering algorithms, one wrapper for t-SNE',
    packages=['D3MUnsupervised'],
    install_requires=["typing",
                      "numpy==1.17.3",
                      'scikit-learn == 0.21.3',
                      "Sloth @ git+https://github.com/NewKnowledge/sloth@4f8a1bd547a216672be1aa499928c83d3c46d10f#egg=Sloth-2.0.8",
                      ],
    entry_points = {
        'd3m.primitives': [
            'clustering.k_means.Sloth = D3MUnsupervised:Storc',
            'clustering.hdbscan.Hdbscan = D3MUnsupervised:Hdbscan',
            'dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne = D3MUnsupervised:Tsne',
            'clustering.spectral_graph_clustering.SpectralClustering = D3MUnsupervised:SpectralClustering',
        ],
    },
)

