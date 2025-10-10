#!/usr/bin/env python3
"""Create a profile for an existing Supabase Auth user."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.supabase_client import get_supabase_admin

# Load environment variables
load_dotenv()


def create_profile(user_id: str, email: str, full_name: str = None):
    """Create a profile for a user that exists in auth.users."""
    supabase = get_supabase_admin()
    
    try:
        # Check if profile already exists
        existing = supabase.table("profiles").select("*").eq("id", user_id).execute()
        
        if existing.data:
            print(f"✓ Profile already exists for user {user_id}")
            return True
        
        # Create the profile
        result = supabase.table("profiles").insert({
            "id": user_id,
            "email": email,
            "full_name": full_name
        }).execute()
        
        if result.data:
            print(f"\n✓ Profile created successfully!")
            print(f"User ID: {user_id}")
            print(f"Email: {email}")
            return True
        else:
            print(f"✗ Failed to create profile")
            return False
            
    except Exception as e:
        print(f"✗ Error creating profile: {e}")
        return False


def get_auth_user_info(user_id: str):
    """Get user info from auth.users table."""
    supabase = get_supabase_admin()
    
    try:
        # Query the auth.users table through admin API
        # Note: This requires service role key
        result = supabase.auth.admin.get_user_by_id(user_id)
        
        if result and result.user:
            return {
                'id': result.user.id,
                'email': result.user.email,
                'full_name': result.user.user_metadata.get('full_name')
            }
    except Exception as e:
        print(f"Could not fetch user from auth: {e}")
    
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_profile.py <USER_UUID> [email] [full_name]")
        print("\nExample:")
        print("  python scripts/create_profile.py 06fa8c78-86f1-4924-8b3c-c4cec42f47a2 user@example.com")
        sys.exit(1)
    
    user_id = sys.argv[1]
    
    # Try to get user info from auth first
    print(f"Looking up user {user_id}...")
    user_info = get_auth_user_info(user_id)
    
    if user_info:
        print(f"Found user in auth: {user_info['email']}")
        email = user_info['email']
        full_name = user_info.get('full_name')
    else:
        # Fall back to command line arguments
        if len(sys.argv) < 3:
            print("Could not fetch user from auth. Please provide email:")
            print("  python scripts/create_profile.py <USER_UUID> <email> [full_name]")
            sys.exit(1)
        
        email = sys.argv[2]
        full_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Create the profile
    success = create_profile(user_id, email, full_name)
    
    if success:
        print("\n✓ You can now generate an API key:")
        print(f"  python3 scripts/manage_keys.py generate {user_id} --description 'My API key'")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

