from setuptools import setup, find_packages

# Read the long description from the README file
with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name="WikiClick",  # Name of the package
    version="1.0.0",    # Version of the package
    author="Your Name", # Author's name
    author_email="your_email@example.com",  # Author's email address
    description="Embedding-Augmented Retrieval system for Wikipedia pages",  # Short description
    long_description=long_description,  # Detailed description from README
    long_description_content_type="text/markdown",  # Type of the long description (Markdown format)
    url="https://github.com/yourusername/WikiClick",  # Link to the project's GitHub repository
    packages=find_packages(where='semanticsearch'),  # Finds all packages under the `semanticsearch` folder
    classifiers=[
        "Programming Language :: Python :: 3",   # Python version compatibility
        "License :: OSI Approved :: MIT License", # License type
        "Operating System :: OS Independent",    # Compatible with all OSes
    ],
    python_requires='>=3.6',  # Minimum required Python version
    install_requires=[  # List of required dependencies
        'numpy>=1.21.0',           # Example: numpy (replace with actual dependencies)
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'faiss-cpu>=1.7.2',         # FAISS for k-NN search
        'sentence-transformers>=2.2.0',  # For embedding models
        'requests>=2.25.1',          # For making API calls (e.g., OpenAI API)
        'pyyaml>=5.4.0',             # For reading YAML configuration
    ],
    extras_require={  # Optional dependencies (e.g., for development)
        'dev': [
            'pytest>=6.2.4',         # Testing framework
            'black>=21.9b0',         # Code formatting
            'tox>=3.24.0',           # Automation tool for testing
        ],
    },
    entry_points={  # Define entry points for command-line scripts if needed
        'console_scripts': [
            'semanticsearch-preprocess=semanticsearch.scripts.preprocess:main',  # Example: preprocess script entry point
            'semanticsearch-inference=semanticsearch.scripts.inference:main',    # Example: inference script entry point
        ],
    },
)
