import requests
import os
import sys
import json

def run_integration_test():
    url = "http://127.0.0.1:8000/api/analyze"
    test_image_path = r"d:\Projects\oncogemma-app\chest_xray.png"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Skipping image test.")
        # Try to create a dummy small image
        from PIL import Image
        test_image_path = "dummy_test.png"
        img = Image.new("RGB", (512, 512), color="pink")
        img.save(test_image_path)
        print("Created a dummy pink patch image for testing.")
        
    print(f"Sending analysis request with image: {test_image_path}")
    
    files = {
        'image': ('biopsy_patch.png', open(test_image_path, 'rb'), 'image/png')
    }
    
    data = {
        'examType': 'IHC',
        'patientReport': 'Patient is a 54-year-old female with a history of left breast lump. Core biopsy requested for ER/PR/HER2 receptor status.',
        'originalFileName': 'biopsy_patch.png'
    }
    
    try:
        response = requests.post(url, files=files, data=data, timeout=180)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Integration Test SUCCESS!")
            print(f"Analysis ID: {result.get('analysisId')}")
            print(f"Report character length: {len(result.get('report', ''))}")
            print("Vision findings output:")
            print(json.dumps(result.get('json'), indent=2))
            print("Patches count in return payload:")
            print(len(result.get('patches', [])))
            return True
        else:
            print("Integration Test FAILED!")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Test run encountered an exception: {e}")
        return False
    finally:
        # Clean up dummy image if created
        if test_image_path == "dummy_test.png" and os.path.exists(test_image_path):
            os.remove(test_image_path)

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
