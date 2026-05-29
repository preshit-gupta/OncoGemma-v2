import base64
from google.cloud import aiplatform

def test_endpoint():
    print("Initializing AI Platform client...")
    aiplatform.init(project="oncogemma", location="asia-east1")
    
    endpoint_id = "mg-endpoint-72113111-adc0-4ee3-8453-eeb0bfbd3d33"
    print(f"Connecting to endpoint: {endpoint_id}...")
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    # Create a small dummy image for testing
    from PIL import Image
    import io
    img = Image.new("RGB", (224, 224), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    
    query = "Count the number of abnormal mitotic figures in this 40x magnification pathology patch. Respond with a single integer."
    
    instances = [
        {
            "image": {
                "input_bytes": base64_image
            }
        },
        {
            "text": query
        }
    ]
    
    print("Sending predict request...")
    try:
        response = endpoint.predict(instances=instances)
        print("Success! Prediction output:")
        print(response.predictions)
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    test_endpoint()
