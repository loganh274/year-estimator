import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import boto3
import botocore.exceptions

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import get_aws_credentials

class TestS3Auth(unittest.TestCase):
    
    @patch('boto3.Session')
    def test_gui_auth_success(self, mock_session_cls):
        """Test successful auth when passed via arguments (GUI mode)"""
        print("\nTesting GUI Auth (Success)...")
        
        # Mock Session and Client
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session
        
        # Run with arguments (simulating GUI) - include whitespace to test stripping
        session, bucket = get_aws_credentials(
            access_key=" test_access_key ", 
            secret_key="test_secret_key\n", 
            bucket_name=" test_bucket "
        )
        
        # Verify
        self.assertEqual(bucket, "test_bucket")
        mock_session_cls.assert_called_with(
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            region_name="us-east-1"
        )
        mock_client.list_objects_v2.assert_called_with(Bucket="test_bucket", MaxKeys=1)
        print("✅ GUI Success Test Passed")

    @patch('builtins.input')
    @patch('boto3.Session')
    def test_interactive_auth_success(self, mock_session_cls, mock_input):
        """Test successful interactive authentication (CLI mode)"""
        print("\nTesting Interactive Auth (Success)...")
        
        # Mock Environment (ensure no env vars)
        with patch.dict(os.environ, {}, clear=True):
            # Mock User Input
            mock_input.side_effect = [
                "test_access_key", 
                "test_secret_key", 
                "test_bucket", 
                "us-west-2"
            ]
            
            # Mock Session and Client
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_session_cls.return_value = mock_session
            
            # Run without arguments
            session, bucket = get_aws_credentials()
            
            # Verify
            self.assertEqual(bucket, "test_bucket")
            mock_session_cls.assert_called_with(
                aws_access_key_id="test_access_key",
                aws_secret_access_key="test_secret_key",
                region_name="us-west-2"
            )
            print("✅ Interactive Success Test Passed")

if __name__ == '__main__':
    unittest.main()
