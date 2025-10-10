# Supabase Migration Summary

This document summarizes the migration from JSON file-based storage to Supabase.

## What Changed

### Authentication System

**Before:**

- JSON file storage (`/data/api_keys.json`)
- Simple API key validation
- No user management
- No JWT support

**After:**

- Supabase PostgreSQL database
- Hybrid authentication (API keys + JWT tokens)
- Full user management with Google OAuth
- Row Level Security policies
- API key management endpoints for web app

### Data Storage

**Before:**

- JSON file storage (`/data/runs_registry.json`, `/data/api_keys.json`)
- File-based locking
- No relational queries
- Manual JSON parsing

**After:**

- Supabase PostgreSQL database
- ACID transactions
- Efficient queries with indexes
- Automatic timestamp management
- Metrics in separate table with foreign keys

### Files Modified

1. **New Files:**
   - `api/supabase_client.py` - Supabase client wrapper
   - `docs/SUPABASE_SETUP.md` - Complete setup guide
   - `docs/MIGRATION_SUMMARY.md` - This file

2. **Replaced:**
   - `api/auth.py` - Now uses HybridAuthManager with Supabase
   - `api/registry.py` - Now uses Supabase PostgreSQL
   - `scripts/manage_keys.py` - Updated for Supabase
   - `tests/conftest.py` - Mock Supabase for testing

3. **Updated:**
   - `api/main.py` - Hybrid auth + API key management endpoints
   - `api/schemas.py` - Added API key schemas
   - `requirements.txt` - Added Supabase dependencies
   - `.env.example` - Added Supabase configuration
   - `README.md` - Updated setup instructions

## Database Schema

### Tables Created

1. **profiles** - User profiles linked to Supabase Auth
   - id (UUID, foreign key to auth.users)
   - email
   - full_name
   - created_at, updated_at

2. **api_keys** - API keys for programmatic access
   - id (UUID, primary key)
   - user_id (UUID, foreign key to profiles)
   - key_hash (SHA-256 hash)
   - description
   - created_at, last_used
   - is_active

3. **runs** - Training run metadata
   - id (TEXT, primary key like "run_abc123")
   - user_id (UUID, foreign key to profiles)
   - base_model
   - status
   - current_step
   - config (JSONB)
   - created_at, updated_at

4. **run_metrics** - Training metrics over time
   - id (UUID, primary key)
   - run_id (TEXT, foreign key to runs)
   - step
   - loss, grad_norm, learning_rate
   - metrics (JSONB)
   - timestamp

### Security

- Row Level Security (RLS) enabled on all tables
- Users can only access their own data
- Service role key bypasses RLS (backend only)
- API keys hashed with SHA-256 before storage
- JWT tokens validated by Supabase Auth

## New API Endpoints

### API Key Management (for web app)

```http
POST /api-keys
Authorization: Bearer <JWT_TOKEN>
{
  "description": "My API key"
}

GET /api-keys
Authorization: Bearer <JWT_TOKEN>

DELETE /api-keys/{key_id}
Authorization: Bearer <JWT_TOKEN>

GET /profile
Authorization: Bearer <JWT_TOKEN>
```

## Authentication Flow

### For Web App Users:

1. User signs in with Google OAuth via Supabase Auth
2. Frontend gets JWT token from Supabase
3. User calls `/api-keys` with JWT to create API key
4. API key displayed once (never again)
5. User can list/revoke keys from web app

### For API Users:

1. User has API key (sk-xxx format)
2. Include in Authorization header: `Bearer sk-xxx...`
3. Backend validates against hashed keys in database
4. Updates last_used timestamp

## Environment Variables

Required in `.env`:

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-key

# Modal (unchanged)
MODAL_TOKEN_ID=your-modal-id
MODAL_TOKEN_SECRET=your-modal-secret

# HuggingFace (unchanged)
HF_TOKEN=your-hf-token
```

## Backwards Compatibility

- API key format unchanged (`sk-` prefix)
- All existing API endpoints work the same
- Client code doesn't need changes
- Only backend storage mechanism changed

## Testing

- Mocked Supabase client in tests
- All existing tests should pass
- Added fixtures for async operations
- Test environment variables set automatically

## Deployment Checklist

- [ ] Create Supabase project
- [ ] Run SQL migrations
- [ ] Configure Google OAuth
- [ ] Set environment variables
- [ ] Deploy Modal functions
- [ ] Start FastAPI server
- [ ] Create test user in Supabase
- [ ] Generate test API key
- [ ] Test authentication flows

## Rollback Plan

If issues arise:
1. Old code is preserved in git history
2. Can revert to commit before migration
3. No data loss (starting fresh with Supabase)
4. Modal functions unchanged (no rollback needed)

## Benefits of Migration

1. **Scalability**: PostgreSQL handles concurrent requests better than JSON files
2. **Security**: RLS policies, JWT tokens, OAuth integration
3. **Web Integration**: Easy to build web dashboard with Supabase
4. **Audit Trail**: Automatic timestamps, last_used tracking
5. **Relational Data**: Proper foreign keys, cascading deletes
6. **Backups**: Automatic backups via Supabase
7. **Admin UI**: Supabase Dashboard for manual operations

## Next Steps

1. Set up monitoring in Supabase Dashboard
2. Configure email templates for auth
3. Add rate limiting middleware
4. Implement usage tracking
5. Set up alerts for errors
6. Create admin dashboard in web app

## Support

For issues or questions:

- See [docs/SUPABASE_SETUP.md](SUPABASE_SETUP.md) for setup help
- Check Supabase Dashboard logs
- Review RLS policies if auth issues
- Test with Supabase SQL Editor
