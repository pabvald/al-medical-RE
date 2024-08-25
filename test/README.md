# Test 

This folder holds tests for the code in the `src/features/` directory, which implements the features used for Random Forest and BiLSTM methods. It also includes tests for the  code in the `src/models/` directory, which contains the data models used for preprocessing and feature computation. The tests were written using [PyTest](https://docs.pytest.org/en/7.2.x/).


## Execute tests 

To run all the tests, execute the following command from the **parent** directory: 

```
python -m pytest test/
```

To run a specific test or subgroup of tests, specify the path. For example, to run only the tests of the features code, execute: 

```
python -m pytest test/features/
```

And to run only the tests of the entity model: 

```
python -m pytest test/models/test_entity.py
```
