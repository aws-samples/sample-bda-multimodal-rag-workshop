import boto3
import json
import uuid
import time
import os
import sagemaker
from datetime import datetime
from IPython.display import Image, clear_output, HTML, display, Markdown
from PIL import Image as PILImage, ImageDraw
import matplotlib.pyplot as plt
import requests
import subprocess
from pathlib import Path

class BDAImageUtils:
    def __init__(self):
        # Initialize AWS session and clients
        self.session = sagemaker.Session()
        self.current_region = boto3.session.Session().region_name
        
        self.sts = boto3.client('sts')
        self.account_id = self.sts.get_caller_identity()['Account']
        
        # Initialize BDA clients
        self.bda_client = boto3.client('bedrock-data-automation')
        self.bda_runtime_client = boto3.client('bedrock-data-automation-runtime')
        self.s3_client = boto3.client('s3')
        
        # Define bucket name using workshop convention
        self.bucket_name = f"bda-workshop-{self.current_region}-{self.account_id}"
        
        # Define S3 locations
        self.data_prefix = "bda-workshop/image"
        self.output_prefix = "bda-workshop/image/output"
        
        # Create bucket if it doesn't exist
        self.create_bucket_if_not_exists()
    
    def download_image(self, url, output_path):
        """Download an image from a URL"""
        import requests
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            raise
    
    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} already exists")
        except:
            print(f"Creating bucket: {self.bucket_name}")
            if self.current_region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.current_region}
                )
            print(f"Bucket {self.bucket_name} created successfully")
    
    def upload_to_s3(self, local_path, s3_key):
        """Upload a local file to S3"""
        self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
        return f's3://{self.bucket_name}/{s3_key}'
    
    def read_json_from_s3(self, s3_uri):
        """Read and parse a JSON file from S3"""
        bucket, key = self.get_bucket_and_key(s3_uri)
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    
    def get_bucket_and_key(self, s3_uri):
        """Extract bucket and key from an S3 URI"""
        path_parts = s3_uri.replace("s3://", "").split("/")
        bucket = path_parts[0]
        key = "/".join(path_parts[1:])
        return bucket, key
    
    def delete_s3_folder(self, prefix):
        """Delete a folder and its contents from S3"""
        objects = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
    
    def wait_for_completion(self, get_status_function, status_kwargs, completion_states, 
                           error_states, status_path_in_response, max_iterations=15, delay=10):
        """Wait for an asynchronous operation to complete"""
        for i in range(max_iterations):
            response = get_status_function(**status_kwargs)
            
            # Extract status from response using the provided path
            status = response
            for key in status_path_in_response.split('.'):
                status = status.get(key, {})
            
            print(f"Current status: {status}")
            
            if status in completion_states:
                print(f"Process completed with status: {status}")
                return response
            elif status in error_states:
                print(f"Process failed with status: {status}")
                return response
            
            print(f"Waiting {delay} seconds...")
            time.sleep(delay)
        
        print(f"Maximum iterations reached. Last status: {status}")
        return response

# Enhanced functions from enhanced_utils.py

def show_business_context(context_type=None, auto_initialize=False):
    """
    Display business context information for a specific context type.
    Uses a simpler approach for better compatibility with Jupyter notebooks.
    
    Args:
        context_type (str): The type of business context to display
        auto_initialize (bool): If True, initialize but don't display (for use in imports)
    """
    contexts = {
        "image_analysis_complete": {
            "title": "Image Analysis with Amazon Bedrock Data Automation",
            "content": """
Amazon Bedrock Data Automation (BDA) enables you to extract structured insights from images by leveraging
generative AI to analyze visual content and transform it into actionable information.

### Business Value

- **Automated Visual Understanding**: Extract text, detect objects, identify logos, and generate descriptions from images at scale
- **Visual Content Integration**: Transform unstructured visual data into structured formats that integrate with business systems
- **Multimodal Intelligence**: Combine visual analysis with other data modalities to create comprehensive understanding
- **Development Acceleration**: Eliminate the need to build and maintain complex computer vision pipelines

### Key Detection Capabilities

- **Text Detection**: Extract printed and handwritten text from images with precise bounding boxes
- **Logo Detection**: Identify corporate logos and brand marks with location information
- **Object Detection**: Recognize common objects, scenery elements, and visual concepts
- **Category Classification**: Automatically categorize images into industry-standard IAB categories

### Industry Applications

- **Retail**: Product recognition for inventory management and customer experiences
- **Marketing**: Measure brand presence through logo detection in digital and print media
- **Content Moderation**: Identify potentially unsafe or inappropriate visual content
- **Document Processing**: Extract text from forms, receipts, and image-based documents
""",
            "sources": [
                {"text": "Amazon Bedrock Data Automation Documentation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda.html"},
                {"text": "Image Processing in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-ouput-image.html"},
                {"text": "AWS Blog: Simplify multimodal generative AI with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/simplify-multimodal-generative-ai-with-amazon-bedrock-data-automation/"},
                {"text": "AWS Blog: Get insights from multimodal content with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/aws/get-insights-from-multimodal-content-with-amazon-bedrock-data-automation-now-generally-available/"}
            ]
        },
        "image_analysis": {
            "title": "Image Analysis with Amazon Bedrock Data Automation",
            "content": """
Amazon Bedrock Data Automation (BDA) enables you to extract structured insights from images by leveraging
generative AI to analyze visual content and transform it into actionable information.

### Business Value

- **Automated Visual Understanding**: Extract text, detect objects, identify logos, and generate descriptions from images at scale
- **Visual Content Integration**: Transform unstructured visual data into structured formats that integrate with business systems
- **Multimodal Intelligence**: Combine visual analysis with other data modalities to create comprehensive understanding
- **Development Acceleration**: Eliminate the need to build and maintain complex computer vision pipelines
""",
            "sources": [
                {"text": "Amazon Bedrock Data Automation Documentation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda.html"},
                {"text": "AWS Blog: Simplify multimodal generative AI with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/simplify-multimodal-generative-ai-with-amazon-bedrock-data-automation/"}
            ]
        },
        "visual_detection": {
            "title": "Visual Detection Capabilities in BDA",
            "content": """
BDA offers powerful visual detection features that transform what your applications can understand from images.

### Key Detection Capabilities

- **Text Detection**: Extract printed and handwritten text from images with precise bounding boxes
- **Logo Detection**: Identify corporate logos and brand marks with location information
- **Object Detection**: Recognize common objects, scenery elements, and visual concepts
- **Category Classification**: Automatically categorize images into industry-standard IAB categories

### Business Applications

- **Retail**: Product recognition for inventory management and customer experiences
- **Marketing**: Measure brand presence through logo detection in digital and print media
- **Content Moderation**: Identify potentially unsafe or inappropriate content across 7 categories: Explicit, Non-Explicit Nudity, Swimwear/Underwear, Violence, Drugs & Tobacco, Alcohol, and Hate Symbols
- **Document Processing**: Extract text from forms, receipts, and image-based documents
""",
            "sources": [
                {"text": "Image Processing in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-ouput-image.html"}
            ]
        },
        "image_classification": {
            "title": "Deriving Business Insights through Image Classification",
            "content": """
BDA's classification capabilities provide actionable insights from visual content that can transform business operations.

### Classification Features

- **IAB Taxonomy**: Classification using the Internet Advertising Bureau's industry-standard categories (supports 24 top-level and 85 second-level categories)
- **Custom Classifications**: Add your own categories through custom blueprints
- **Confidence Scores**: Understand the reliability of classifications for better decision-making
- **Multi-Label Classification**: Images can belong to multiple relevant categories with parent-child relationships

### Business Impact

- **Content Recommendation**: Power recommendation engines with automatic content categorization
- **Advertising Targeting**: Match ads to appropriate visual content based on classifications
- **Digital Asset Management**: Organize and retrieve visual assets using automatic categorization
- **Content Compliance**: Ensure visual content meets industry guidelines and regulations
""",
            "sources": [
                {"text": "Working with Standard Output in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-standard-output.html"},
                {"text": "Image Processing in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-ouput-image.html"},
                {"text": "AWS Blog: Building with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/building-with-amazon-bedrock-data-automation/"}
            ]
        },
        "text_in_image": {
            "title": "Extracting Text from Visual Content",
            "content": """
Text contained within images represents valuable information that traditional systems can't easily access. 
Amazon Bedrock Data Automation helps bridge this gap by extracting text from images.

### Text Extraction Features

- **Text Detection**: Identify and extract text from images with precise bounding boxes
- **Spatial Recognition**: Understand where text appears within the image with polygon mapping
- **Confidence Scoring**: Each detected text element includes a confidence score for validation
- **Hierarchical Structure**: Text is organized by lines and words with parent-child relationships
- **Visual Context Understanding**: Recognize text in relation to other visual elements

### Business Applications

- **Receipt Processing**: Extract purchase information from photographed receipts
- **Sign and Menu Reading**: Capture text from signs, menus, and displays
- **Form Processing**: Extract information from photographed or scanned forms
- **ID Document Reading**: Extract key information from identity documents

For multilingual document processing, the AWS ecosystem offers complementary solutions like Amazon Translate, which can be used together with BDA for comprehensive multilingual workflows.
""",
            "sources": [
                {"text": "Amazon Bedrock Data Automation Documentation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda.html"},
                {"text": "Image Processing in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-ouput-image.html"},
                {"text": "Generate Insights from Content - Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/bedrock/bda/"}
            ]
        },
        "project_architecture": {
            "title": "Understanding BDA Projects for Image Analysis",
            "content": """
Projects are a fundamental architectural component in Amazon Bedrock Data Automation. For image analysis, they provide several key benefits:

1. **Consistent Configuration**: Projects store processing configurations that ensure consistent analysis across multiple images or image batches.

2. **Feature Selection**: Projects let you specify which visual features to extract (text, logos, objects) and which generative outputs to create (summaries, classifications).

3. **Custom Extraction Logic**: Through blueprints, projects can define specialized extraction patterns tailored to your specific visual analysis needs.

4. **Environment Separation**: Different projects can be created for development, testing, and production environments, allowing you to evolve your image analysis capabilities safely.

5. **Access Management**: Projects provide a way to control who can invoke specific types of image analysis, supporting governance requirements.

In production environments, you typically create different projects for different image types or analysis requirements, each with optimized configurations.
""",
            "sources": [
                {"text": "Working with Projects in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-projects.html"}
            ]
        },
        "processing_pipeline": {
            "title": "Behind the Scenes of Image Processing",
            "content": """
When you invoke Amazon Bedrock Data Automation for image analysis, a sophisticated processing pipeline is executed:

### Processing Steps

1. **Image Ingestion**: The system loads and preprocesses the image from S3
2. **Initial Analysis**: Computer vision models identify key visual elements (text, logos, objects)
3. **Spatial Mapping**: Bounding boxes and polygons are calculated for detected elements
4. **Content Moderation**: Optional scanning for inappropriate content across 7 categories
5. **Semantic Analysis**: Generative AI creates descriptions and classifications of image content
6. **IAB Categorization**: Application of standardized advertising taxonomy categories
7. **Result Formation**: All extracted data is assembled into the requested output format

### Performance Considerations

- Processing time depends on image complexity, size, and requested detection types
- High-resolution images with many elements take longer to process
- Typical processing times range from seconds to a minute for complex images
- Response includes important image metadata (dimensions, color depth, encoding)
""",
            "sources": [
                {"text": "Invoking Amazon Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-invoke.html"},
                {"text": "Image Processing in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-ouput-image.html"}
                ]
        },
        "custom_blueprints": {
            "title": "Custom Blueprints: Tailored Visual Understanding",
            "content": """
### What are Blueprints?
Blueprints define structured extraction patterns that allow you to customize how BDA analyzes your specific visual content. They bridge the gap between general-purpose AI capabilities and your unique business requirements.

### Key Advantages of Blueprints

1. **Domain-Specific Extraction**: Target exactly the information that matters to your business case

2. **Consistent Structure**: Ensure all extracted data follows a predictable JSON schema for easier integration

3. **Reduced Post-Processing**: Get data in the format you need without additional transformation steps

4. **Business Logic Integration**: Embed business rules directly into your extraction patterns

5. **Versioned Evolution**: Refine your extraction patterns over time while maintaining compatibility

### Blueprint Components

- **Properties**: Define the fields to extract
- **Inference Types**: Control how fields are populated (extracted vs. inferred)
- **Instructions**: Guide the AI in extracting specific information
- **Definitions**: Create reusable structures for complex extraction patterns

### When to Use Blueprints

Blueprints are particularly valuable when:
- Standard outputs don't capture your specific information needs
- You need consistent structure for downstream processing
- Your domain has specialized terminology or concepts
- You're integrating visual analysis with existing business systems

### Blueprint Capabilities for Images

- **Custom Attributes**: Define specific attributes you want to extract from images
- **Structured Output**: Get results in a consistent JSON schema that matches your business needs
- **Domain-Specific Analysis**: Create specialized extraction for retail, medical, advertising, or other domains
- **Guided LLM Analysis**: Provide instructions that guide the AI in analyzing your specific image types

### Business Applications

- **Product Catalogs**: Extract specific product attributes from product images
- **Marketing Analysis**: Define custom metrics for analyzing advertisement images
- **Quality Control**: Create specialized inspection attributes for manufacturing
- **Scene Understanding**: Define custom scene attributes for security or surveillance applications

### AWS Documentation Resources

- For blueprint schema documentation, see: https://docs.aws.amazon.com/bedrock/latest/userguide/bda-blueprints.html
- For custom extraction patterns, see: https://aws.amazon.com/blogs/machine-learning/custom-extraction-with-amazon-bedrock-data-automation/
- For blueprint best practices, see: https://docs.aws.amazon.com/bedrock/latest/userguide/bda-blueprints-best-practices.html
""",
            "sources": [
                {"text": "Blueprint Management in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-blueprints.html"},
                {"text": "AWS Blog: Custom Extraction with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/custom-extraction-with-amazon-bedrock-data-automation/"}
            ]
        },
        "business_applications": {
            "title": "From Visual Insights to Business Value",
            "content": """
The structured data extracted from images by BDA enables various business applications that transform how organizations work with visual content.

### Practical Applications

- **Product Analysis**: Extract product attributes from catalog images
- **Brand Monitoring**: Track logo appearances across digital and print media
- **Content Organization**: Automatically tag and categorize visual content
- **Text Digitization**: Extract text from images for analysis and search
- **Visual Search**: Enable search functionality based on image content
- **Content Moderation**: Filter inappropriate content across 7 categories (Explicit, Non-Explicit Nudity, Swimwear/Underwear, Violence, Drugs & Tobacco, Alcohol, and Hate Symbols)
- **Ad Targeting**: Match content to appropriate advertising using IAB taxonomy

### Integration Patterns

- **API Integration**: Connect with CRM, DAM, and other business systems
- **Data Pipelines**: Feed extracted information into analytics systems
- **Content Automation**: Trigger workflows based on visual content analysis
- **Multimodal Applications**: Combine with other data sources for richer insights
- **Knowledge Bases**: Feed extracted image insights into Bedrock Knowledge Bases for multimodal RAG
""",
            "sources": [
                {"text": "AWS Blog: Multimodal processing with Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/multimodal-processing-with-bedrock-data-automation/"},
                {"text": "AWS Blog: Visual intelligence applications with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/visual-intelligence-applications-with-amazon-bedrock/"},
                {"text": "AWS Blog: Building a multimodal RAG application with Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/building-a-multimodal-rag-based-application-using-amazon-bedrock-data-automation-and-amazon-bedrock-knowledge-bases/"},
                {"text": "AWS Blog: Get insights from multimodal content with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/aws/get-insights-from-multimodal-content-with-amazon-bedrock-data-automation-now-generally-available/"}
            ]
        }
    }
    
    if auto_initialize:
        return contexts
        
    if context_type not in contexts:
        print(f"Context type '{context_type}' not found.")
        return
    
    context = contexts[context_type]
    
    # Use a simpler approach - Just display the markdown directly
    # This avoids all the HTML formatting issues
    html_header = f"""
    <div style="background-color: #f0f8ff; padding: 15px 15px 5px 15px; border-radius: 10px 10px 0 0;">
        <h3 style="margin-top: 0; cursor: pointer;">{context['title']}</h3>
    </div>
    """
    
    # Display the header
    display(HTML(html_header))
    
    # Display the content as markdown (will render properly)
    display(Markdown(context['content']))
    
    # Build sources HTML and display if available
    if context.get("sources"):
        sources_html = """
        <div style="background-color: #f0f8ff; padding: 5px 15px 15px 15px; border-radius: 0 0 10px 10px; margin-top: -10px;">
            <h4>Sources</h4>
            <ul>
        """
        
        for source in context["sources"]:
            sources_html += f'<li><a href="{source["url"]}" target="_blank">{source["text"]}</a></li>'
            
        sources_html += """
            </ul>
        </div>
        """
        
        display(HTML(sources_html))

def download_image(url, local_path):
    """Download an image with better error handling."""
    try:
        # Only create directories if the local_path has a directory part
        dir_name = os.path.dirname(local_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # Try using curl command which has better SSL handling
        result = subprocess.run(['curl', '-s', '-L', '-o', local_path, url], check=True)
        print(f"Downloaded {url} to {local_path}")
        return local_path
    except subprocess.CalledProcessError:
        # Fall back to requests if curl fails
        try:
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {url} to {local_path}")
            return local_path
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

def ensure_bda_results_dir():
    """Ensure the BDA results directory exists."""
    os.makedirs('../bda-results', exist_ok=True)
    print("Ensured BDA results directory exists.")

def visualize_detections(image_path, standard_output, figsize=(12, 8)):
    """
    Visualize text and logo detections on an image with improved styling
    
    Args:
        image_path (str): Path to the image file
        standard_output (dict): Standard output from BDA
        figsize (tuple): Figure size for the plot
    """
    # Load the original image
    image = PILImage.open(image_path)
    width = standard_output["metadata"]["image_width_pixels"]
    height = standard_output["metadata"]["image_height_pixels"]
    
    # Create a copy for drawing
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    legend_items = []
    
    # Draw bounding boxes for text
    if "text_lines" in standard_output["image"]:
        for txt in standard_output["image"]["text_lines"]:
            for l in txt["locations"]:
                bbox = l["bounding_box"]
                box = (
                    width * bbox["left"],
                    height * bbox["top"],
                    width * (bbox["width"] + bbox["left"]),
                    height * (bbox["height"] + bbox["top"])
                )
                # Use semi-transparent blue with thicker border
                draw.rectangle(box, outline=(0, 0, 255, 230), width=3)
        legend_items.append(("Detected Text", "blue"))
    
    # Draw bounding boxes for logos
    if "logos" in standard_output["image"]:
        for logo in standard_output["image"]["logos"]:
            for l in logo["locations"]:
                bbox = l["bounding_box"]
                box = (
                    width * bbox["left"],
                    height * bbox["top"],
                    width * (bbox["width"] + bbox["left"]),
                    height * (bbox["height"] + bbox["top"])
                )
                # Use semi-transparent red with thicker border
                draw.rectangle(box, outline=(255, 0, 0, 230), width=3)
                
                # Add logo name if available
                if "name" in logo:
                    draw.text(
                        (box[0] + 5, box[1] + 5),
                        logo["name"],
                        fill=(255, 255, 255)
                    )
        legend_items.append(("Detected Logos", "red"))
    
    # Display the image with bounding boxes
    plt.figure(figsize=figsize)
    plt.imshow(image_with_boxes)
    
    # Add a proper legend
    for label, color in legend_items:
        plt.plot([], [], color=color, linewidth=3, label=label)
    
    plt.legend(loc="upper right", frameon=True, fontsize=12)
    plt.title('Visual Content Detection Results', fontsize=14)
    plt.axis("off")
    
    return plt

# Backward compatibility exports
BDAUtils = BDAImageUtils  # Class alias

# Expose enhanced utility functions at module level
__all__ = [
    'BDAImageUtils', 
    'BDAUtils',  # Backward compatibility alias
    'show_business_context',
    'download_image',
    'ensure_bda_results_dir',
    'visualize_detections'
]
