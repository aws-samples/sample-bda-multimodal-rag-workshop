"""
Consolidated utilities for Amazon Bedrock Data Automation (BDA) document workshops.
This file provides functionality for document processing with the "Foundation Building" narrative theme.
"""

import boto3
import json
import time
import os
import requests
import subprocess
from pathlib import Path
import sagemaker
from IPython.display import Image, HTML, display, Markdown

class BDADocumentUtils:
    def __init__(self):
        # Initialize AWS session and clients
        self.session = boto3.Session()
        self.current_region = self.session.region_name
        
        self.sts = boto3.client('sts')
        self.account_id = self.sts.get_caller_identity()['Account']
        
        # Initialize BDA clients
        self.bda_client = boto3.client('bedrock-data-automation')
        self.bda_runtime_client = boto3.client('bedrock-data-automation-runtime')
        self.s3_client = boto3.client('s3')
        
        # Define bucket name using workshop convention
        self.bucket_name = f"bda-workshop-{self.current_region}-{self.account_id}"
        
        # Define S3 locations
        self.input_location = f's3://{self.bucket_name}/bda/input'
        self.output_location = f's3://{self.bucket_name}/bda/output'
        
        # Create bucket if it doesn't exist
        self.create_bucket_if_not_exists()
    
    def get_bucket_and_key(self, s3_uri):
        """Extract bucket and key from an S3 URI"""
        path_parts = s3_uri.replace("s3://", "").split("/")
        bucket = path_parts[0]
        key = "/".join(path_parts[1:])
        return bucket, key
    
    def read_s3_object(self, s3_uri):
        """Read and decode an S3 object"""
        bucket, key = self.get_bucket_and_key(s3_uri)
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    
    def download_and_display_figure(self, s3_uri, width=800):
        """Download a figure from S3 and display it"""
        bucket, key = self.get_bucket_and_key(s3_uri)
        local_path = f"temp_figure_{Path(key).name}"
        self.s3_client.download_file(bucket, key, local_path)
        return Image(filename=local_path, width=width)
    
    def download_document(self, url, output_path):
        """Download a document from a URL with enhanced error handling"""
        return download_file(url, output_path)
    
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
    
    def upload_to_s3(self, local_path, s3_uri):
        """Upload a local file to S3"""
        bucket, key = self.get_bucket_and_key(s3_uri)
        self.s3_client.upload_file(local_path, bucket, key)
        return s3_uri
    
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


def show_progress(current_step, total_steps, step_name):
    """
    Display a progress bar for the workshop.
    
    Args:
        current_step (int): The current step number
        total_steps (int): The total number of steps
        step_name (str): The name of the current step
    """
    percent = (current_step / total_steps) * 100
    progress_html = f"""
    <div style="background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h3>Workshop Progress: Step {current_step} of {total_steps}</h3>
        <div style="background: #ddd; width: 100%; height: 20px; border-radius: 10px;">
            <div style="background: #4CAF50; width: {percent}%; height: 100%; border-radius: 10px;"></div>
        </div>
        <p><strong>Current Step:</strong> {step_name}</p>
    </div>
    """
    display(HTML(progress_html))


def show_business_context(context_type=None, auto_initialize=False):
    """
    Display business context information for a specific context type.
    Uses a simpler approach for better compatibility with Jupyter notebooks.
    
    Args:
        context_type (str): The type of business context to display
        auto_initialize (bool): If True, initialize but don't display (for use in imports)
    """
    contexts = {
        "document_processing": {
            "title": "Document Processing with Amazon Bedrock Data Automation",
            "content": """
Amazon Bedrock Data Automation (BDA) enables you to extract structured information
from unstructured documents by leveraging generative AI to automate transformation into structured formats.

### Business Value

- **Automate Intelligent Document Processing (IDP)**: Process documents at scale without orchestrating complex 
  tasks like classification, extraction, normalization, or validation
- **Streamline Workflows**: Transform unstructured documents into business-specific, structured data that 
  integrates with existing systems
- **Accelerate Development**: Managed experience and customization capabilities help deliver business value faster
- **Reduce Complexity**: Eliminate the need to manage multiple AI models and services through a unified API-driven experience
""",
            "sources": [
                {"text": "Amazon Bedrock Data Automation Documentation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda.html"},
                {"text": "Amazon Bedrock Data Automation Product Page", "url": "https://aws.amazon.com/bedrock/bda/"},
                {"text": "AWS Blog: Simplify multimodal generative AI with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/simplify-multimodal-generative-ai-with-amazon-bedrock-data-automation/"},
                {"text": "AWS Blog: Unleashing the multimodal power of Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/unleashing-the-multimodal-power-of-amazon-bedrock-data-automation-to-transform-unstructured-data-into-actionable-insights/"}
            ]
        },
        "standard_output": {
            "title": "Configuring BDA for Document Processing",
            "content": """
Standard output is the default way of interacting with Amazon Bedrock Data Automation. 
The configuration you choose significantly impacts the quality and usefulness of extracted data.

### Key Configuration Options

- **Granularity Levels**: Control the detail level of extraction
  - **DOCUMENT**: Overall document analysis for summaries and classifications
  - **PAGE**: Page-level extraction for page-specific insights
  - **ELEMENT**: Identification of semantic elements like tables and figures
  - **LINE/WORD**: Detailed text positioning for precise extraction

- **Bounding Box**: When enabled, provides coordinates for document elements, critical for spatial understanding and reconstruction

- **Generative Fields**: AI-generated summaries and descriptions that provide contextual understanding

### Industry Applications

- **Financial Services**: Automated processing of financial statements, invoices, and contracts
- **Healthcare**: Information extraction from clinical notes, medical records, and insurance forms
- **Legal**: Analysis of contracts, legal briefs, and case documentation
""",
            "sources": [
                {"text": "Standard Output in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-standard-output.html"},
                {"text": "AWS Blog: Scalable intelligent document processing using Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/scalable-intelligent-document-processing-using-amazon-bedrock/"},
                {"text": "AWS Blog: Building a multimodal RAG application with Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/building-a-multimodal-rag-based-application-using-amazon-bedrock-data-automation-and-amazon-bedrock-knowledge-bases/"}
            ]
        },
        "processing_pipeline": {
            "title": "Behind the Scenes of Document Processing",
            "content": """
When you invoke Amazon Bedrock Data Automation, the service executes a sophisticated processing pipeline:

### Processing Steps

1. **Document Ingestion**: The system loads and preprocesses the document from S3
2. **Structure Analysis**: AI models identify the document's hierarchical structure and components
3. **Content Extraction**: Text, tables, figures, and other elements are extracted with spatial information
4. **Semantic Enrichment**: Generative AI creates summaries and descriptions of document content
5. **Result Formation**: All extracted data is assembled into the requested output formats

### Processing Configuration

- **Input Configuration**: Specify the S3 location of your document
- **Output Configuration**: Define where processed results should be stored
- **Data Automation Configuration**: Reference your project ARN and stage
- **Data Automation Profile**: Specify the processing profile to use

### Performance Considerations

- Processing time depends on document complexity, size, and requested granularity
- Multi-page documents with many elements take longer to process
- Typical processing times range from seconds to a few minutes for complex documents
""",
            "sources": [
                {"text": "Invoking Amazon Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-invoke.html"},
                {"text": "BDA Runtime API Reference", "url": "https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Runtime_for_Amazon_Bedrock_Data_Automation.html"}
                ]
        },
        "project_architecture": {
            "title": "Understanding BDA Projects",
            "content": """
Projects are a fundamental architectural component in Amazon Bedrock Data Automation. They serve several critical purposes:

1. **Configuration Reusability**: Projects store processing configurations that can be reused across multiple document processing jobs, ensuring consistent results.

2. **Workflow Separation**: Different projects can be created for different document types or processing needs, allowing for specialized processing pipelines.

3. **Versioning and Evolution**: Projects can be updated over time as your needs evolve, while maintaining backward compatibility with existing workflows.

4. **Access Control**: Projects can have specific IAM permissions, allowing you to control which teams or services can access specific document processing configurations.

5. **Cost Management**: By organizing processing jobs under projects, you can track and allocate costs to specific business initiatives or departments.

### Project Stages

Each project can have two stages:
- **LIVE**: Used for production processing of customer requests
- **DEVELOPMENT**: Used for testing and modification before promoting to LIVE

### Project Limits

- Up to 40 document blueprints can be attached to a single project
- Each project can have one standard output configuration per data type
- Project names must be unique within your account and region

In production environments, you would typically create different projects for different document types or processing requirements, each with its own optimized configuration.
""",
            "sources": [
                {"text": "Working with Projects in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-projects.html"},
                {"text": "AWS Blog: Get insights from multimodal content with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/aws/get-insights-from-multimodal-content-with-amazon-bedrock-data-automation-now-generally-available/"}
            ]
        },
        "business_applications": {
            "title": "From Document Insights to Business Value",
            "content": """
The structured data extracted by BDA enables various business applications that transform how organizations work with documents.

### Practical Applications

- **Automated Data Entry**: Extract key data points directly into business systems
- **Contract Analysis**: Identify key terms, obligations, and risks in legal documents
- **Financial Analysis**: Extract and analyze financial metrics from reports and statements
- **Compliance Verification**: Check document content against regulatory requirements
- **Enterprise Search**: Make document content searchable across the organization

### Industry-Specific Use Cases

- **Financial Services**: Automated processing of loan applications, SEC filings, and investment prospectuses
- **Healthcare**: Information extraction from medical records, insurance claims, and clinical trial documentation
- **Legal**: Analysis of contracts, case law, and regulatory documentation
- **Manufacturing**: Processing of quality control reports, supply chain documentation, and safety compliance

### Integration Patterns

- **API Integration**: Connect with CRM, ERP, and other business systems
- **Data Pipelines**: Feed extracted information into data lakes and warehouses
- **Workflow Automation**: Trigger business processes based on document content
- **Knowledge Bases**: Build searchable repositories of document information
""",
            "sources": [
                {"text": "AWS Blog: Automate document processing with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/automate-document-processing-with-amazon-bedrock-prompt-flows/"},
                {"text": "AWS Blog: New Amazon Bedrock capabilities enhance data processing", "url": "https://aws.amazon.com/blogs/aws/new-amazon-bedrock-capabilities-enhance-data-processing-and-retrieval/"},
                {"text": "Guidance for Multimodal Data Processing Using Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/solutions/guidance/multimodal-data-processing-using-amazon-bedrock-data-automation/"}
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


def download_file(url, local_path):
    """Download a file with better error handling."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
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
            print(f"Error downloading file: {e}")
            return None


def ensure_bda_results_dir():
    """Ensure the BDA results directory exists."""
    os.makedirs('../bda-results', exist_ok=True)
    print("Ensured BDA results directory exists.")

# For backward compatibility: Define BDAUtils as an alias to BDADocumentUtils
BDAUtils = BDADocumentUtils
