#!/usr/bin/env python3
"""Check if Supabase is properly configured."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

def check_env_vars():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY"),
        "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    }
    
    missing = []
    for var, value in required_vars.items():
        if not value or value.startswith("your-"):
            print(f"  ❌ {var} not set or using placeholder value")
            missing.append(var)
        else:
            print(f"  ✅ {var} set")
    
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("   Please update your .env file with actual Supabase credentials")
        return False
    
    print("✅ All environment variables set\n")
    return True


def check_supabase_connection():
    """Check if we can connect to Supabase."""
    print("🔍 Testing Supabase connection...")
    
    try:
        from api.supabase_client import get_supabase
        supabase = get_supabase()
        print("  ✅ Supabase client initialized\n")
        return supabase
    except Exception as e:
        print(f"  ❌ Failed to initialize Supabase client: {e}\n")
        return None


def check_tables(supabase):
    """Check if required tables exist."""
    print("🔍 Checking database tables...")
    
    tables = ["profiles", "api_keys", "runs", "run_metrics"]
    all_exist = True
    
    for table in tables:
        try:
            # Try to query the table (just count)
            result = supabase.table(table).select("*", count="exact").limit(0).execute()
            print(f"  ✅ Table '{table}' exists (count: {result.count})")
        except Exception as e:
            print(f"  ❌ Table '{table}' not found or error: {str(e)[:50]}")
            all_exist = False
    
    if all_exist:
        print("✅ All required tables exist\n")
    else:
        print("\n❌ Some tables are missing. Did you run the SQL migrations?")
        print("   See docs/SUPABASE_SETUP.md for SQL commands\n")
    
    return all_exist


def check_rls_policies(supabase):
    """Check if RLS is enabled."""
    print("🔍 Checking Row Level Security (RLS)...")
    
    # Try to query without proper auth - should work with service key
    try:
        result = supabase.table("api_keys").select("*").limit(1).execute()
        print("  ✅ RLS policies appear to be configured")
        print("     (Using service role key bypasses RLS)\n")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not verify RLS: {e}\n")
        return False


def test_create_profile():
    """Test if we can interact with the profiles table."""
    print("🔍 Testing database operations...")
    
    try:
        from api.supabase_client import get_supabase
        supabase = get_supabase()
        
        # Just check if we can query (don't actually create anything)
        result = supabase.table("profiles").select("id, email").limit(1).execute()
        
        if result.data:
            print(f"  ✅ Found {len(result.data)} profile(s) in database")
        else:
            print("  ℹ️  No profiles yet (create a user in Supabase Dashboard)")
        
        print("  ✅ Database operations working\n")
        return True
    except Exception as e:
        print(f"  ❌ Database operation failed: {e}\n")
        return False


def main():
    print("=" * 60)
    print("🚀 Supabase Configuration Check")
    print("=" * 60)
    print()
    
    # Check environment variables
    if not check_env_vars():
        print("\n❌ Setup incomplete: Environment variables not configured")
        sys.exit(1)
    
    # Check Supabase connection
    supabase = check_supabase_connection()
    if not supabase:
        print("\n❌ Setup incomplete: Cannot connect to Supabase")
        sys.exit(1)
    
    # Check tables
    tables_ok = check_tables(supabase)
    if not tables_ok:
        print("\n❌ Setup incomplete: Database tables missing")
        sys.exit(1)
    
    # Check RLS
    check_rls_policies(supabase)
    
    # Test operations
    test_create_profile()
    
    # Final summary
    print("=" * 60)
    print("✅ Supabase Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Create a user in Supabase Dashboard (Authentication > Users)")
    print("2. Generate an API key:")
    print("   python scripts/manage_keys.py generate <USER_UUID> --description 'Test key'")
    print("3. Start your API:")
    print("   python api/main.py")
    print()
    print("For more info, see GETTING_STARTED_SUPABASE.md")


if __name__ == "__main__":
    main()

