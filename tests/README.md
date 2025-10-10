# Signal Test Suite

Comprehensive test suite for the Signal training API.

## Overview

This test suite provides thorough coverage of Signal's core functionality including:

- **Authentication** - API key management and authorization
- **Model Registry** - Model configuration and queries
- **Run Registry** - Training run tracking and metadata
- **API Endpoints** - All REST API routes
- **Client SDK** - Python client library
- **Integration** - End-to-end workflows

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Fixtures and test configuration
├── test_auth.py             # Authentication tests
├── test_models.py           # Model registry tests
├── test_registry.py         # Run registry tests
├── test_schemas.py          # Schema validation tests
├── test_api.py              # API endpoint tests
├── test_client.py           # Client SDK tests
└── test_integration.py      # End-to-end workflow tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

This includes pytest, pytest-cov, and other testing tools.

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_auth.py -v
pytest tests/test_api.py -v
pytest tests/test_integration.py -v
```

### Run Specific Test

```bash
pytest tests/test_auth.py::TestAPIKeyGeneration::test_generate_key_format -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=api --cov=client tests/

# Generate HTML coverage report
pytest --cov=api --cov=client --cov-report=html tests/

# View HTML report
open htmlcov/index.html
```

### Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -n auto
```

## Test Categories

### Unit Tests

Test individual components in isolation:

- `test_auth.py` - API key generation, validation, revocation
- `test_models.py` - Model configuration loading and queries
- `test_registry.py` - Run CRUD operations and metrics
- `test_schemas.py` - Pydantic schema validation

### Integration Tests

Test API endpoints with mocked Modal functions:

- `test_api.py` - All REST endpoints, authentication, authorization
- `test_client.py` - Python client SDK methods

### End-to-End Tests

Test complete workflows:

- `test_integration.py` - Training workflows, gradient accumulation, multi-user scenarios

## Fixtures

Common test fixtures are defined in `conftest.py`:

### File Fixtures
- `temp_api_keys_file` - Temporary API keys storage
- `temp_registry_file` - Temporary run registry storage
- `temp_models_config` - Temporary models configuration

### Manager Fixtures
- `api_key_manager` - APIKeyManager instance
- `run_registry` - RunRegistry instance
- `model_registry` - ModelRegistry instance

### Mock Fixtures
- `mock_modal_create_run` - Mock Modal create_run function
- `mock_modal_forward_backward` - Mock forward-backward function
- `mock_modal_optim_step` - Mock optimizer step function
- `mock_modal_sample` - Mock sampling function
- `mock_modal_save_state` - Mock save state function

### Data Fixtures
- `test_user_id` - Test user identifier
- `test_api_key` - Generated test API key
- `sample_batch` - Sample training data
- `sample_messages_batch` - Sample chat messages data
- `sample_prompts` - Sample generation prompts
- `sample_run_config` - Sample run configuration

### Client Fixture
- `test_client` - FastAPI TestClient with mocked dependencies

## Test Coverage Goals

- **API Endpoints**: 100%
- **Authentication**: 100%
- **Registries**: 100%
- **Client SDK**: 100%
- **Overall**: >90%

## Mocking Strategy

Since Modal functions require GPU infrastructure, all Modal calls are mocked in tests:

- **Unit tests**: Don't call Modal at all
- **API tests**: Mock Modal function calls with `unittest.mock`
- **Integration tests**: Mock entire Modal workflow
- **No GPU required**: All tests run quickly on any machine

## Key Test Scenarios

### Success Scenarios

✅ Complete training loop (forward-backward → optim → sample → save)  
✅ Gradient accumulation across multiple batches  
✅ Multi-step training with metrics tracking  
✅ Both adapter and merged model saving  
✅ Multiple runs per user with isolation  
✅ Chat template format (messages field)  
✅ Custom LoRA configurations  
✅ Various sampling parameters  

### Error Scenarios

🔴 Run not found (404)  
🔴 Unauthorized access (403)  
🔴 Invalid API key (401)  
🔴 Unsupported model (400)  
🔴 Missing gradients  
🔴 Invalid batch data  
🔴 Malformed requests  

## Example Test Run

```bash
$ pytest tests/ -v

tests/test_auth.py::TestAPIKeyGeneration::test_generate_key_format PASSED
tests/test_auth.py::TestAPIKeyValidation::test_validate_correct_key PASSED
tests/test_models.py::TestModelLoading::test_load_models_from_config PASSED
tests/test_registry.py::TestRunCreation::test_create_run_basic PASSED
tests/test_schemas.py::TestRunConfigSchema::test_run_config_defaults PASSED
tests/test_api.py::TestPublicEndpoints::test_root_endpoint PASSED
tests/test_api.py::TestCreateRun::test_create_run_success PASSED
tests/test_client.py::TestSignalClient::test_list_models PASSED
tests/test_integration.py::TestBasicTrainingWorkflow::test_complete_training_flow PASSED

========================== 100+ tests passed in 2.5s ==========================
```

## Continuous Integration

These tests are designed to run in CI/CD environments:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=api --cov=client
```

## Writing New Tests

When adding new features, follow these patterns:

### 1. Add fixtures to `conftest.py` if needed

```python
@pytest.fixture
def my_test_data():
    return {"key": "value"}
```

### 2. Write unit tests first

```python
def test_my_function(my_test_data):
    result = my_function(my_test_data)
    assert result == expected_value
```

### 3. Add API endpoint tests

```python
def test_my_endpoint(test_client, test_api_key):
    response = test_client.post(
        "/my-endpoint",
        headers={"Authorization": f"Bearer {test_api_key}"},
        json={"data": "test"},
    )
    assert response.status_code == 200
```

### 4. Add integration tests for workflows

```python
def test_my_workflow(test_client, test_api_key):
    # Create, train, and verify
    run = create_run()
    train(run)
    assert_results(run)
```

## Troubleshooting

### Tests fail with "No module named 'api'"

Make sure you're running pytest from the project root:

```bash
cd /path/to/signal
pytest tests/
```

### Tests fail with import errors

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Tests are slow

The test suite should run in <10 seconds. If slower:
- Check if you're accidentally calling real Modal functions
- Verify mocks are properly configured
- Use `pytest -v` to see which tests are slow

## Contributing

When contributing tests:

1. ✅ Test both success and error cases
2. ✅ Use descriptive test names
3. ✅ Mock external dependencies (Modal, HuggingFace Hub)
4. ✅ Keep tests fast (<100ms per test)
5. ✅ Add docstrings to test classes/functions
6. ✅ Verify tests pass before submitting PR

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)

