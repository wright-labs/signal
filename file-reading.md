# **Complete Reading Order for Signal Codebase**

This is a structured reading order designed to build your understanding progressively, from high-level concepts down to implementation details.

---

## **Phase 1: High-Level Overview -- DONE**

*Understand what the project does and its architecture*

1. **`README.md`** - Main overview, features, and architecture. Read this to understand the product and its four primitives (forward_backward, optim_step, sample, save_state).

2. **`QUICKSTART.md`** - Setup instructions and architecture diagram. Shows how all components (API, Modal, Supabase) fit together.

3. **`docs/QUICKREF.md`** - Quick reference for API endpoints and common commands. Great for understanding the user-facing API surface.

4. **`docs/DEPLOYMENT.md`** - Deployment guide explaining how to set up the system in production.

5. **`docs/MIGRATION_SUMMARY.md`** - Historical context about architectural decisions (if you want to understand why things are the way they are).

---

## **Phase 2: Configuration -- DONE**

*Understand what models are supported and how they're configured*

6. **`config/models.yaml`** - List of supported models with GPU configurations. Critical for understanding model-to-GPU mapping.

7. **`requirements.txt`** - Full dependencies (though most ML deps are in Modal images).

8. **`requirements-api.txt`** - API server-specific dependencies. Note that ML libs are NOT here since they run on Modal.

---

## **Phase 3: Core API Layer -- DONE**

*The FastAPI server that orchestrates everything*

9. **`main.py`** - Main entry point (imports from api module).

10. **`api/__init__.py`** - API module initialization.

11. **`api/schemas.py`** - Pydantic models for all API requests/responses. Essential for understanding data structures throughout the system.

12. **`api/models.py`** - ModelRegistry class that loads and validates supported models from config.

13. **`api/auth.py`** - Authentication manager supporting both API keys (sk-xxx) and JWT tokens. Includes hybrid auth for web + programmatic access.

14. **`api/supabase_client.py`** - Supabase connection management and user context handling.

15. **`api/registry.py`** - RunRegistry class for managing training runs in Supabase database. CRUD operations for runs, metrics, and artifacts.

16. **`api/frontier_client.py`** - Client for communicating with Frontier Backend for billing/credits (optional, can self-host without it).

17. **`api/logging_config.py`** - Security logging configuration for audit trails.

18. **`main.py`** (in api folder, the actual FastAPI app) - **THE CORE API SERVER**. Defines all endpoints: `/runs`, `/forward_backward`, `/optim_step`, `/sample`, `/save_state`. This orchestrates calls to Modal functions. Read carefully!

19. **`api/openai_compat.py`** - OpenAI-compatible API endpoints for integration with tools like Verifiers. Allows using fine-tuned models via OpenAI SDK format.

---

## **Phase 4: Modal Runtime - GPU Training Infrastructure**

*The actual training code that runs on Modal's GPUs*

20. **`modal_runtime/__init__.py`** - Module initialization.

21. **`modal_runtime/app.py`** - **CRITICAL**: Modal app definition, Docker images, volumes, and secrets. Defines TRAINING_IMAGE and INFERENCE_IMAGE with all ML dependencies. This is where Modal is configured.

22. **`modal_runtime/gpu_utils.py`** - GPU configuration parsing utilities (e.g., "a100-80gb:4" -> 4x A100).

23. **`modal_runtime/model_loader.py`** - Model loading utilities using Transformers + PEFT. Handles quantization (4-bit/8-bit), LoRA application, and model preparation.

24. **`modal_runtime/trainer_utils.py`** - Training utilities: optimizer setup, tokenization, forward/backward passes, gradient handling, checkpoint saving/loading.

25. **`modal_runtime/loss_functions.py`** - Loss functions for different training objectives (causal_lm, DPO, reward modeling, PPO).

26. **`modal_runtime/s3_client.py`** - S3 client for artifact storage with tenant isolation. Handles uploading checkpoints to S3 with signed URLs.

27. **`modal_runtime/manifest.py`** - Manifest generation for saved checkpoints (includes metadata, file sizes, checksums).

28. **`modal_runtime/primitives.py`** - **THE HEART OF THE SYSTEM**: Modal functions for the four primitives (create_run, forward_backward, optim_step, sample, save_state). This is where training actually happens on GPUs. Read this thoroughly!

29. **`modal_runtime/cleanup.py`** - Background job for cleaning up old run data from Modal volumes to save storage costs.

30. **`modal_runtime/axolotl_config_generator.py`** - Legacy Axolotl config generator (may be deprecated).

31. **`modal_runtime/IMPLEMENTATION_SUMMARY.md`** - Summary of Modal runtime implementation details.

32. **`modal_runtime/primitives.py.bak`** - Backup of primitives (skip unless debugging).

---

## **Phase 5: Client SDK**

*Python SDK for interacting with the API*

33. **`client/frontier_signal/__init__.py`** - SDK module exports. Shows all public API.

34. **`client/frontier_signal/_version.py`** - Version string.

35. **`client/frontier_signal/schemas.py`** - Client-side schema definitions (mirrors API schemas).

36. **`client/frontier_signal/exceptions.py`** - Exception hierarchy for error handling.

37. **`client/frontier_signal/client.py`** - **Main synchronous client** (SignalClient, SignalRun). This is what users import to interact with the API.

38. **`client/frontier_signal/training_client.py`** - Specialized training client with convenience methods.

39. **`client/frontier_signal/inference_client.py`** - Specialized inference client.

40. **`client/frontier_signal/async_client.py`** - Async version of main client (AsyncSignalClient).

41. **`client/frontier_signal/async_training_client.py`** - Async training client.

42. **`client/frontier_signal/async_inference_client.py`** - Async inference client.

43. **`client/README.md`** - Client SDK documentation.

44. **`client/HYBRID_CLIENT_GUIDE.md`** - Guide for using both training and inference clients.

45. **`client/PUBLISHING.md`** - Instructions for publishing SDK to PyPI.

46. **`client/pyproject.toml`** - Client package configuration.

---

## **Phase 6: Examples**

*Learn by example*

47. **`examples/sft_example.py`** - **START HERE FOR EXAMPLES**: Complete supervised fine-tuning example showing the full training loop with all four primitives.

48. **`client/examples/basic_sync.py`** - Basic synchronous client example.

49. **`client/examples/basic_async.py`** - Basic async client example.

50. **`client/examples/advanced_training.py`** - Advanced training patterns (gradient accumulation, learning rate schedules).

51. **`client/examples/advanced_inference.py`** - Advanced inference patterns.

---

## **Phase 7: Testing**

*Understand how components are tested*

52. **`tests/conftest.py`** - Pytest configuration and fixtures.

53. **`tests/README.md`** - Testing documentation.

54. **`tests/test_schemas.py`** - Schema validation tests.

55. **`tests/test_models.py`** - Model registry tests.

56. **`tests/test_auth.py`** - Authentication tests.

57. **`tests/test_registry.py`** - Run registry tests.

58. **`tests/test_api.py`** - API endpoint unit tests.

59. **`tests/test_client.py`** - Client SDK tests.

60. **`tests/test_security.py`** - Security middleware tests.

61. **`tests/test_integration.py`** - Integration tests.

62. **`tests/test_api_e2e.py`** - End-to-end API tests.

63. **`tests/test_e2e_simple.py`** - Simple E2E tests.

64. **`client/tests/test_training_client.py`** - Training client tests.

65. **`client/tests/test_inference_client.py`** - Inference client tests.

66. **`client/tests/test_hybrid_api.py`** - Hybrid API tests.

67. **`test_sdk.py`** - SDK quick test.

68. **`client/test_sdk_quick.py`** - Quick SDK test.

---

## **Phase 8: Scripts & Utilities**

*Tools for managing the system*

69. **`scripts/__init__.py`** - Scripts module initialization.

70. **`scripts/manage_keys.py`** - CLI for managing API keys (generate, list, revoke).

71. **`scripts/create_profile.py`** - Script to create user profiles in Supabase.

72. **`scripts/check_supabase.py`** - Verify Supabase connection and schema.

73. **`scripts/create_test_setup.py`** - Set up test data.

74. **`scripts/test_s3_integration.py`** - Test S3 storage integration.

75. **`scripts/finetuneUnsloth.py`** - Legacy Unsloth fine-tuning script (may be deprecated).

76. **`scripts/finetuneAxolotl.py`** - Legacy Axolotl fine-tuning script (may be deprecated).

77. **`scripts/quickstart.sh`** - Quick setup script.

78. **`scripts/migrations/add_s3_storage.sql`** - SQL migration for S3 storage support.

---

## **Phase 9: Deployment & Infrastructure**

*Configuration for hosting*

79. **`Procfile`** - Process file for Heroku/Railway deployment.

80. **`Railway.toml`** - Railway.app configuration.

81. **`nixpacks.toml`** - Nixpacks build configuration.

82. **`start.sh`** - Startup script for production.

83. **`dev_server.sh`** - Development server script.

84. **`docs/README.md`** - Documentation index.

---

## **Phase 10: Housekeeping Files**

*Supporting files*

85. **`LICENSE`** - MIT license.

86. **`client/LICENSE`** - Client SDK license.

87. **`client/MANIFEST.in`** - Package manifest for distribution.

88. **`.gitignore`** (if exists) - Git ignore rules.

89. **`api.log`** - API server logs (generated at runtime).

90. **`logs/security.log`** - Security audit logs (generated at runtime).

---

## **Key Files to Focus On**

If you only have time to read 10 files, read these:

1. **`README.md`** - What is this?
2. **`main.py` (api folder)** - The API server orchestrating everything
3. **`modal_runtime/primitives.py`** - The actual training code on GPUs
4. **`modal_runtime/app.py`** - Modal configuration
5. **`api/schemas.py`** - Data structures
6. **`client/frontier_signal/client.py`** - How users interact with the system
7. **`examples/sft_example.py`** - How it all works together
8. **`api/registry.py`** - Database operations
9. **`modal_runtime/model_loader.py`** - Model loading with LoRA
10. **`config/models.yaml`** - Supported models

---

## **Architecture Summary**

After reading everything, you'll understand:

1. **API Layer** (`api/`): FastAPI server that authenticates users, validates requests, and calls Modal functions
2. **Modal Runtime** (`modal_runtime/`): GPU training code that runs on Modal's serverless infrastructure
3. **Client SDK** (`client/frontier_signal/`): Python library users import to interact with the API
4. **Database** (Supabase): Stores runs, metrics, API keys, user profiles
5. **Storage** (Modal Volumes + S3): Stores model checkpoints and artifacts
6. **Configuration** (`config/models.yaml`): Defines which models are supported and their GPU requirements

The flow is: **User → Client SDK → API Server → Modal Functions (GPU) → Supabase DB + S3 Storage**