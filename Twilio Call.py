# Twilio Call and SMS Message Example
# First, install the Twilio Python library: pip install twilio

from twilio.rest import Client

# Your Account SID and Auth Token from twilio.com/console
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)

# Function to make a phone call
def make_call(to_number, from_number, twiml_url):

    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=twiml_url
    )
    print(f"Call SID: {call.sid}")
    return call

# Function to send an SMS message
def send_sms(to_number, from_number, message_body):

    message = client.messages.create(
        to=to_number,
        from_=from_number,
        body=message_body
    )
    print(f"Message SID: {message.sid}")
    return message

# Example usage
if __name__ == "__main__":
    # Set your Twilio phone number and the recipient's number
    twilio_number = '+14129912633'  # Your Twilio phone number
    recipient_number = '+916265586868'  # Recipient's phone number
    
    # URL to a TwiML file that controls the call behavior
    # This example plays a message when the call is answered
    twiml_url = 'https://demo.twilio.com/welcome/voice/'
    
    # Make a call 
    make_call(recipient_number, twilio_number, twiml_url)
    
    # Send an SMS
    send_sms(
        recipient_number,
        twilio_number,
        "Hello! This is a test message from my Twilio application."
    )

# Example TwiML file content (should be hosted at the URL you provide to the call function):
"""
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Enmergency at (Street name).</Say>
    <Play>https://demo.twilio.com/docs/classic.mp3</Play>
</Response>
"""
