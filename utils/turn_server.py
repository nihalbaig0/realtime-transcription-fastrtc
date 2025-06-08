import os
from typing import Literal, Dict, Any
from fastrtc import (
    get_twilio_turn_credentials,
    get_cloudflare_turn_credentials_async
)


async def get_rtc_credentials(
    provider: Literal["twilio", "cloudflare", "hf-cloudflare"] = "hf-cloudflare",
    **kwargs
) -> Dict[str, Any]:
    """
    Get RTC configuration for different TURN server providers.
    
    Args:
        provider: The TURN server provider to use ('twilio', 'cloudflare', or 'hf-cloudflare')
        **kwargs: Additional arguments passed to the specific provider's function
    
    Returns:
        Dictionary containing the RTC configuration
    """
    try:
        if provider == "twilio":
            # Twilio TURN Server (sync call wrapped)
            instructions = """
            1. Create a free Twilio account at: https://login.twilio.com/u/signup
            2. Get your Account SID and Auth Token from the Twilio Console
            3. Set environment variables: TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
            """
            account_sid = kwargs.pop("account_sid", os.environ.get("TWILIO_ACCOUNT_SID"))
            auth_token = kwargs.pop("auth_token", os.environ.get("TWILIO_AUTH_TOKEN"))
            if not account_sid or not auth_token:
                raise ValueError(f"Twilio credentials not found. {instructions}")
            return get_twilio_turn_credentials(account_sid=account_sid, auth_token=auth_token)
            
        elif provider == "cloudflare":
            # Cloudflare TURN Server with Cloudflare credentials
            # TODO: Improve Cloudflare instructions
            instructions = """
            1. Create a Cloudflare account at: https://dash.cloudflare.com/
            2. Go to Realtime (Calls) -> TURN Server -> Get Started
            3. Get Turn Token ID and API Token
            4. Set environment variables: TURN_KEY_ID and TURN_KEY_API_TOKEN
            """
            key_id = kwargs.pop("key_id", os.environ.get("TURN_KEY_ID"))
            api_token = kwargs.pop("api_token", os.environ.get("TURN_KEY_API_TOKEN"))
            ttl = kwargs.pop("ttl", 600)  # Default 10 minutes for client-side
            if not key_id or not api_token:
                raise ValueError(f"Cloudflare credentials not found. {instructions}")
            return await get_cloudflare_turn_credentials_async(key_id=key_id, api_token=api_token, ttl=ttl)
            
        elif provider == "hf-cloudflare":
            # Cloudflare with Hugging Face Token (10GB free traffic per month)
            instructions = """
            1. Create a Hugging Face account at huggingface.co
            2. Visit: https://huggingface.co/settings/tokens to create a token
            3. Set HF_TOKEN environment variable or pass token directly
            """
            hf_token = kwargs.pop("hf_token", os.environ.get("HF_TOKEN"))
            ttl = kwargs.pop("ttl", 600)  # Default 10 minutes for client-side
            if not hf_token:
                raise ValueError(f"Hugging Face token not found. {instructions}")
            return await get_cloudflare_turn_credentials_async(hf_token=hf_token, ttl=ttl)
    except Exception as e:
        raise Exception(f"Failed to get RTC credentials ({provider}): {str(e)}")