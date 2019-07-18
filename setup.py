from setuptools import setup

setup(name='D3MUnsupervised',
    version='1.0.0',
    description='Two wrappers for unsupervised clustering algorithms',
    packages=['D3MUnsupervised'],
    install_requires=["typing",
                      "numpy == 1.15.4",
                      'scikit-learn == 0.20.3',
                      'Keras == 2.2.4',
                      "Sloth @ git+https://github.com/NewKnowledge/sloth@c331cec7f9c90642c8726f8cf673c2034493d08b#egg=Sloth-2.0.7",
                      ],
    entry_points = {
        'd3m.primitives': [
            'clustering.k_means.Sloth = D3MUnsupervised:Storc',
            'clustering.hdbscan.Hdbscan = D3MUnsupervised:Hdbscan',
        ],
    },
)
