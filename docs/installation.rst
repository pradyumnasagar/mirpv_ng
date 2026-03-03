Installation
============

Requirements
------------

- Python 3.11+
- ViennaRNA 2.5+
- Bowtie 1.x (for sRNA-seq mode)
- samtools, bedtools (for sRNA-seq mode)

Conda Installation (Recommended)
--------------------------------

The easiest way to install miRPV-NG and all dependencies is using conda/mamba::

    # Clone the repository
    git clone https://github.com/yourorg/mirpv_ng_v3.git
    cd mirpv_ng_v3

    # Create environment
    conda env create -f env.yml
    
    # Activate
    conda activate mirpv-ng

Verifying Installation
----------------------

Check that ViennaRNA is available::

    RNAfold --version
    # Should show: RNAfold 2.x.x

Test the CLI::

    python -m mirpv_ng.cli --help

Run the test suite::

    pytest tests/ -v

Docker (Coming Soon)
--------------------

A Docker image with all dependencies will be available::

    docker pull mirpv/mirpv-ng:latest
