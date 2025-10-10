#!/usr/bin/env python3
"""Create a complete test setup with user, profile, and API key for RLS testing."""

import sys
import os
import secrets
import bcrypt
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.supabase_client import get_supabase_admin

# Load environment variables
load_dotenv()


def create_complete_test_setup():
    """Create test user, profile, and API key all at once."""
    supabase = get_supabase_admin()
    
    # Test user details
    test_user_id = "f88c5635-521c-4482-ab30-5657e48f28b2"
    test_email = "rls-test@example.com"
    test_full_name = "RLS Test User"
    
    # Generate API key
    api_key = f"sk-{secrets.token_urlsafe(32)}"
    key_hash = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    key_prefix = api_key[:11]
    
    try:
        print("\n🔧 Creating test setup for RLS testing...")
        print(f"User ID: {test_user_id}")
        print(f"Email: {test_email}")
        
        # Step 1: Create auth user using raw SQL (bypasses FK checks)
        print("\n1. Creating auth user...")
        supabase.rpc('exec_sql', {
            'sql': f"""
            INSERT INTO auth.users (
                id, instance_id, email, encrypted_password,
                email_confirmed_at, created_at, updated_at, aud, role
            )
            VALUES (
                '{test_user_id}'::uuid,
                '00000000-0000-0000-0000-000000000000'::uuid,
                '{test_email}',
                crypt('testpassword123', gen_salt('bf')),
                NOW(), NOW(), NOW(), 'authenticated', 'authenticated'
            )
            ON CONFLICT (id) DO NOTHING;
            """
        }).execute()
        
        # Step 2: Create profile
        print("2. Creating profile...")
        supabase.table("profiles").insert({
            "id": test_user_id,
            "email": test_email,
            "full_name": test_full_name
        }).execute()
        
        # Step 3: Create API key
        print("3. Creating API key...")
        supabase.table("api_keys").insert({
            "user_id": test_user_id,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "description": "RLS Test Key",
            "is_active": True
        }).execute()
        
        print("\n✅ Test setup complete!")
        print(f"\n{'='*70}")
        print(f"Test User ID: {test_user_id}")
        print(f"Test Email: {test_email}")
        print(f"API Key: {api_key}")
        print(f"{'='*70}")
        print(f"\n⚠️  Save this API key - it won't be shown again!")
        print(f"\nTest the RLS implementation with:")
        print(f"  curl -H 'Authorization: Bearer {api_key}' \\")
        print(f"       http://localhost:8000/runs")
        
        return api_key
        
    except Exception as e:
        # Check if it's because items already exist
        if "already exists" in str(e) or "duplicate" in str(e).lower():
            print("\n⚠️  Test user already exists. Trying to create just the API key...")
            try:
                # Just create a new API key
                api_key = f"sk-{secrets.token_urlsafe(32)}"
                key_hash = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                key_prefix = api_key[:11]
                
                supabase.table("api_keys").insert({
                    "user_id": test_user_id,
                    "key_hash": key_hash,
                    "key_prefix": key_prefix,
                    "description": "RLS Test Key (additional)",
                    "is_active": True
                }).execute()
                
                print(f"\n✅ New API key created!")
                print(f"API Key: {api_key}")
                return api_key
            except Exception as e2:
                print(f"\n✗ Error: {e2}")
                sys.exit(1)
        else:
            print(f"\n✗ Error creating test setup: {e}")
            print("\nTry running this SQL in Supabase SQL Editor:")
            print(f"""
INSERT INTO public.profiles (id, email, full_name)
VALUES (
    '{test_user_id}'::uuid,
    '{test_email}',
    '{test_full_name}'
)
ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email;
            """)
            sys.exit(1)


if __name__ == "__main__":
    create_complete_test_setup()

