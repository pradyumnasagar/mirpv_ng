Contributing
============

We welcome contributions to miRPV-NG!

Development Setup
-----------------

1. Fork and clone the repository
2. Create the conda environment::

    conda env create -f env.yml
    conda activate mirpv-ng

3. Install development dependencies::

    pip install pytest sphinx sphinx-rtd-theme

Running Tests
-------------

Run the full test suite::

    pytest tests/ -v

Run with coverage::

    pytest tests/ --cov=mirpv_ng --cov-report=html

Building Documentation
----------------------

Build locally::

    cd docs
    make html

View at ``docs/_build/html/index.html``

Code Style
----------

- Follow PEP 8
- Use Google-style docstrings
- Add type hints to function signatures
- Write tests for new features

Pull Request Process
--------------------

1. Create a feature branch
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass
5. Submit PR with clear description
