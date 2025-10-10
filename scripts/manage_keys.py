#!/usr/bin/env python3
"""CLI tool for managing API keys via Supabase.

This is a helper script for self-hosters to generate and manage API keys.
For production use, you should build your own auth service (like Frontier Backend).
"""

import argparse
import asyncio
import sys
import os
import secrets
import bcrypt
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.supabase_client import get_supabase_admin


def _hash_key(api_key: str) -> str:
    """Hash an API key using bcrypt."""
    return bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


async def create_key(args):
    """Create a new API key (for self-hosters without user management)."""
    supabase = get_supabase_admin()
    
    try:
        # Generate secure random key
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        key_hash = _hash_key(api_key)
        
        # Extract prefix for indexed lookup
        key_prefix = api_key[:11] if len(api_key) >= 11 else api_key
        
        # Store hashed key with prefix in database
        supabase.table("api_keys").insert({
            "user_id": args.user_id,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "description": args.description or "",
            "is_active": True
        }).execute()
        
        print(f"\n✓ API Key created successfully!")
        print(f"\nUser ID: {args.user_id}")
        print(f"API Key: {api_key}")
        print(f"Description: {args.description or '(none)'}")
        print(f"\n⚠️  Save this key securely - it won't be shown again!")
        print(f"\nYou can now use this key with:")
        print(f"  export SIGNAL_API_KEY='{api_key}'")
        print(f"  or in your code:")
        print(f"  SignalClient(api_key='{api_key[:20]}...')")
    except Exception as e:
        print(f"\n✗ Error creating API key: {e}")
        sys.exit(1)


async def list_keys_for_user(args):
    """List API keys for a user."""
    supabase = get_supabase_admin()
    
    try:
        result = supabase.table("api_keys").select(
            "id, description, created_at, last_used, is_active"
        ).eq("user_id", args.user_id).order("created_at", desc=True).execute()
        
        keys = result.data or []
        
        if not keys:
            print(f"\nNo API keys found for user: {args.user_id}")
            return
        
        print(f"\nAPI Keys for user: {args.user_id}")
        print("=" * 80)
        
        for i, key in enumerate(keys, 1):
            status = "✓ Active" if key.get("is_active") else "✗ Revoked"
            print(f"\n{i}. {status}")
            print(f"   ID: {key['id']}")
            print(f"   Description: {key.get('description') or '(no description)'}")
            print(f"   Created: {key['created_at']}")
            print(f"   Last used: {key.get('last_used') or 'Never'}")
    except Exception as e:
        print(f"\n✗ Error listing API keys: {e}")
        sys.exit(1)


async def revoke_key_by_id(args):
    """Revoke an API key by its ID."""
    supabase = get_supabase_admin()
    
    try:
        # Update is_active to False (soft delete)
        result = supabase.table("api_keys").update({
            "is_active": False
        }).eq("id", args.key_id).eq("user_id", args.user_id).execute()
        
        if result.data and len(result.data) > 0:
            print(f"✓ API key revoked successfully")
        else:
            print(f"✗ API key not found or already revoked")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error revoking API key: {e}")
        sys.exit(1)


def main():
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    # For key management, we need service role temporarily (admin operation)
    if not os.getenv("SUPABASE_URL"):
        print("\n✗ Error: SUPABASE_URL must be set")
        print("Please set this in your .env file or environment variables")
        sys.exit(1)
    
    # Check for either service role or anon key
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY") and not os.getenv("SUPABASE_ANON_KEY"):
        print("\n✗ Error: Either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY must be set")
        print("Note: Service role key required for creating API keys (admin operation)")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Manage API keys for Signal (self-hosting helper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new API key (self-hosting)
  python manage_keys.py create --user-id YOUR_UUID --description "My dev key"
  
  # List all API keys for a user
  python manage_keys.py list YOUR_UUID
  
  # Revoke an API key
  python manage_keys.py revoke KEY_UUID --user-id YOUR_UUID

Note: 
  - For production, build a proper auth service (see Frontier Backend example)
  - User ID can be any UUID - it's just for organizing keys
  - Generate a UUID with: python -c "import uuid; print(uuid.uuid4())"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create key command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new API key"
    )
    create_parser.add_argument(
        "--user-id",
        dest="user_id",
        required=True,
        help="User UUID (can be any UUID for self-hosting)"
    )
    create_parser.add_argument(
        "--description",
        help="Description for the API key",
        default=""
    )
    
    # List keys command
    list_parser = subparsers.add_parser(
        "list",
        help="List API keys for a user"
    )
    list_parser.add_argument(
        "user_id",
        help="User UUID"
    )
    
    # Revoke key command
    revoke_parser = subparsers.add_parser(
        "revoke",
        help="Revoke an API key"
    )
    revoke_parser.add_argument(
        "key_id",
        help="API key UUID (from database, not the sk- key itself)"
    )
    revoke_parser.add_argument(
        "--user-id",
        dest="user_id",
        required=True,
        help="User UUID (for authorization)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate async function
    if args.command == "create":
        asyncio.run(create_key(args))
    elif args.command == "list":
        asyncio.run(list_keys_for_user(args))
    elif args.command == "revoke":
        asyncio.run(revoke_key_by_id(args))


if __name__ == "__main__":
    main()
