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
    
    @patch('builtins.input')
    @patch('boto3.Session')
    def test_interactive_auth_success(self, mock_session_cls, mock_input):
        """Test successful interactive authentication"""
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
            
            # Run
            session, bucket = get_aws_credentials()
            
            # Verify
            self.assertEqual(bucket, "test_bucket")
            mock_session_cls.assert_called_with(
                aws_access_key_id="test_access_key",
                aws_secret_access_key="test_secret_key",
                region_name="us-west-2"
            )
            # Verify list_objects_v2 called (validation)
            mock_client.list_objects_v2.assert_called_with(Bucket="test_bucket", MaxKeys=1)
            print("✅ Success Test Passed")

    @patch('builtins.input')
    @patch('boto3.Session')
    def test_auth_failure_403(self, mock_session_cls, mock_input):
        """Test authentication failure (403 Forbidden)"""
        print("\nTesting Interactive Auth (403 Failure)...")
        
        with patch.dict(os.environ, {}, clear=True):
            mock_input.side_effect = ["key", "secret", "bucket", "region"]
            
            mock_session = MagicMock()
            mock_client = MagicMock()
            
            # Simulate 403 Error
            error_response = {'Error': {'Code': '403', 'Message': 'Forbidden'}}
            mock_client.list_objects_v2.side_effect = botocore.exceptions.ClientError(error_response, 'ListObjectsV2')
            
            mock_session.client.return_value = mock_client
            mock_session_cls.return_value = mock_session
            
            # Run and expect error
            with self.assertRaises(botocore.exceptions.ClientError):
                get_aws_credentials()
            print("✅ 403 Failure Test Passed")

if __name__ == '__main__':
    unittest.main()
