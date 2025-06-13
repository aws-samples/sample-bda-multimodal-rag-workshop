"""
Consolidated utilities for BDA audio workshops.
This file provides comprehensive functionality for audio processing with Amazon Bedrock Data Automation.
"""

import boto3
import json
import uuid
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import sagemaker
from datetime import datetime
from IPython.display import Audio, clear_output, JSON, HTML, display, Markdown

class BDAAudioUtils:
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
        self.data_prefix = "bda-workshop/audio"
        self.output_prefix = "bda-workshop/audio/output"
        
        # Create bucket if it doesn't exist
        self.create_bucket_if_not_exists()
    
    def download_audio(self, url, output_path):
        """Download an audio file from a URL with enhanced error handling."""
        try:
            # Create directories if needed
            dir_name = os.path.dirname(output_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                
            # Try using curl command which has better SSL handling
            try:
                import subprocess
                result = subprocess.run(['curl', '-s', '-L', '-o', output_path, url], check=True)
                print(f"Downloaded {url} to {output_path}")
                return output_path
            except (subprocess.CalledProcessError, ImportError):
                # Fall back to requests if curl fails
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()  # Raise an exception for HTTP errors
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {url} to {output_path}")
                return output_path
        except Exception as e:
            print(f"Error downloading audio: {e}")
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
    
    def analyze_content_moderation(self, json_data):
        """Analyze content moderation results from BDA output"""
        # Get the content moderation array
        moderation_results = json_data.get('audio', {}).get('content_moderation', [])
        
        high_risk_segments = []
        
        for segment in moderation_results:
            start_time = segment.get('start_timestamp_millis', 0) / 1000  # Convert to seconds
            end_time = segment.get('end_timestamp_millis', 0) / 1000
            
            # Check each moderation category
            for category in segment.get('moderation_categories', []):
                if category['confidence'] > 0.5:
                    high_risk_segments.append({
                        'time_range': f"{start_time:.2f}s - {end_time:.2f}s",
                        'category': category['category'],
                        'confidence': category['confidence']
                    })
        
        if high_risk_segments:
            print("⚠️ High Risk Content Detected:")
            for segment in high_risk_segments:
                print(f"Time Range: {segment['time_range']}")
                print(f"Category: {segment['category']}")
                print(f"Confidence Score: {segment['confidence']:.3f}")
                print("-" * 50)
        else:
            print("✅ No high-risk content detected (all scores below 0.5)")
            
        # Show overall statistics
        print("\nOverall Content Safety Summary:")
        total_segments = len(moderation_results)
        print(f"Total segments analyzed: {total_segments}")
        print(f"Segments with high risk content: {len(high_risk_segments)}")
    
    def visualize_transcript(self, transcript_data, figsize=(14, 6)):
        """
        Visualize speaker segments from transcript data.
        
        Args:
            transcript_data (dict): Transcript data from BDA output
            figsize (tuple): Figure size for the visualization
        """
        if not transcript_data or 'representation' not in transcript_data:
            print("No transcript data available for visualization")
            return
            
        speakers = {}
        segments = []
        
        # Parse the transcript text
        transcript_text = transcript_data.get('representation', {}).get('text', '')
        lines = transcript_text.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # Try to extract speaker and text
            if ':' in line:
                parts = line.split(':', 1)
                speaker = parts[0].strip()
                text = parts[1].strip() if len(parts) > 1 else ""
                
                if speaker not in speakers:
                    speakers[speaker] = len(speakers)
                
                segments.append({
                    'speaker': speaker,
                    'speaker_id': speakers[speaker],
                    'text': text
                })
        
        if not segments:
            print("No speaker segments found in transcript")
            return
            
        # Create the visualization
        plt.figure(figsize=figsize)
        
        # Calculate segment heights
        segment_height = 1
        spacing = 0.2
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(speakers), 10)))
        
        # Draw segments
        y_pos = 0
        for i, segment in enumerate(segments):
            speaker_id = segment['speaker_id']
            plt.barh(y_pos, 1, height=segment_height, color=colors[speaker_id % 10], alpha=0.7)
            
            # Add speaker and text labels
            plt.text(0.02, y_pos, f"{segment['speaker']}: {segment['text'][:50]}{'...' if len(segment['text']) > 50 else ''}", 
                    va='center', fontsize=9, color='black')
            
            y_pos -= (segment_height + spacing)
        
        # Create legend
        legend_handles = [plt.Rectangle((0,0), 1, 1, color=colors[speakers[s] % 10]) for s in speakers]
        plt.legend(legend_handles, list(speakers.keys()), loc='upper right')
        
        # Adjust plot
        plt.ylim(y_pos - spacing, segment_height)
        plt.xlim(0, 1)
        plt.axis('off')
        plt.title('Speaker Distribution in Transcript')
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        # Print statistics
        print(f"Speaker Distribution Statistics:")
        for speaker, count in zip(speakers.keys(), [len([s for s in segments if s['speaker'] == speaker]) for speaker in speakers]):
            print(f"{speaker}: {count} segments")
    
    def generate_moderation_summary(self, json_data):
        """Generate a visual summary of content moderation scores"""
        moderation_results = json_data.get('audio', {}).get('content_moderation', [])
        
        # Initialize category averages
        category_scores = {
            'profanity': [],
            'hate_speech': [],
            'sexual': [],
            'insult': [],
            'violence_or_threat': [],
            'graphic': [],
            'harassment_or_abuse': []
        }
        
        # Collect all scores for each category
        for segment in moderation_results:
            for category in segment.get('moderation_categories', []):
                category_scores[category['category']].append(category['confidence'])
        
        # Calculate averages and prepare data for plotting
        categories = []
        averages = []
        max_scores = []
        
        for category, scores in category_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                categories.append(category.replace('_', ' ').title())
                averages.append(avg_score)
                max_scores.append(max_score)
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        
        # Create bars
        x = range(len(categories))
        bars = plt.bar(x, averages)
        
        # Customize the plot
        plt.title('Average Content Moderation Scores by Category')
        plt.xlabel('Categories')
        plt.ylabel('Average Confidence Score')
        
        # Rotate x-axis labels for better readability
        plt.xticks(x, categories, rotation=45, ha='right')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Add a horizontal line at 0.5 to show the threshold
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Risk Threshold (0.5)')
        plt.legend()
        
        # Set y-axis limits to include some padding
        plt.ylim(0, max(max(averages) * 1.2, 0.6))
        
        # Show the plot
        plt.show()

def show_business_context(context_type=None, auto_initialize=False):
    """
    Display business context information for a specific context type.
    Uses a simpler approach for better compatibility with Jupyter notebooks.
    
    Args:
        context_type (str): The type of business context to display
        auto_initialize (bool): If True, initialize but don't display (for use in imports)
    """
    contexts = {
        "audio_complete": {
            "title": "The Voice of Information: Business Value of Audio Analysis",
            "content": """
Audio data represents one of the most natural and information-rich forms of human communication. 
Amazon Bedrock Data Automation (BDA) enables organizations to extract structured insights from this previously untapped source.

### Core Business Capabilities

- **Unlock Spoken Knowledge**: Convert hours of audio content into structured, searchable information
- **Speaker Intelligence**: Identify and differentiate speakers for nuanced conversation analysis
- **Content Understanding**: Extract topics, sentiments, and key points from audio discussions
- **Compliance & Safety**: Identify potentially sensitive or inappropriate content in audio recordings
- **Accessibility**: Make audio content accessible through accurate transcription and summarization

### Industry Applications

- **Call Centers**
  - Customer sentiment analysis from support calls
  - Compliance verification for regulated conversations
  - Agent performance evaluation and coaching opportunities
  - Identification of common customer issues for process improvement

- **Media & Entertainment**
  - Automated podcast transcription and summarization
  - Content moderation for user-generated audio
  - Content indexing and search for audio libraries
  - Metadata generation for improved content discovery

- **Healthcare**
  - Medical dictation transcription with specialty term recognition
  - Patient interaction analysis for improved care
  - Documentation assistance for clinical encounters
  - Compliance monitoring for patient communications

- **Financial Services**
  - Earnings call analysis for investment insights
  - Advisory call compliance monitoring
  - Client interaction analysis for relationship management
  - Fraud detection in phone banking interactions

### ROI Opportunities

- Reduce manual transcription costs (typically $1-2 per audio minute)
- Increase process efficiency by 60-80% for audio content management
- Unlock previously inaccessible insights from archived audio content
- Enable new use cases through audio content searchability

### Audio Processing Revolution

Before generative AI, audio processing required complex pipelines with separate models for transcription, 
speaker diarization, topic detection, and summarization. BDA unifies these capabilities into a single, 
powerful API that delivers comprehensive audio insights.
""",
            "sources": [
                {"text": "Amazon Bedrock Data Automation Documentation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda.html"},
                {"text": "AWS Blog: Audio Analytics with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"},
                {"text": "AWS Blog: Contact Center Intelligence with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"},
                {"text": "Case Study: Media Organization Improves Content Discovery", "url": "https://aws.amazon.com/solutions/case-studies/?awsf.customer-references-filter-category=*all&awsm.page-customer-references=1"}
            ]
        },
        "voice_foundation": {
            "title": "The Voice of Information",
            "content": """
Audio data represents one of the most natural and information-rich forms of human communication. 
Amazon Bedrock Data Automation (BDA) enables organizations to extract structured insights from this previously untapped source.

### Business Value

- **Unlock Spoken Knowledge**: Convert hours of audio content into structured, searchable information
- **Speaker Intelligence**: Identify and differentiate speakers for nuanced conversation analysis
- **Content Understanding**: Extract topics, sentiments, and key points from audio discussions
- **Compliance & Safety**: Identify potentially sensitive or inappropriate content in audio recordings
- **Accessibility**: Make audio content accessible through accurate transcription and summarization

### Audio Processing Revolution

Before generative AI, audio processing required complex pipelines with separate models for transcription, 
speaker diarization, topic detection, and summarization. BDA unifies these capabilities into a single, 
powerful API that delivers comprehensive audio insights.
""",
            "sources": [
                {"text": "Amazon Bedrock Data Automation Documentation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda.html"},
                {"text": "AWS Blog: Audio Analytics with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"}
            ]
        },
        "audio_analysis_business_value": {
            "title": "Business Value of Audio Analysis",
            "content": """
Audio analysis with BDA creates significant business value across multiple industries by extracting actionable 
insights from spoken content.

### Industry Applications

- **Call Centers**
  - Customer sentiment analysis from support calls
  - Compliance verification for regulated conversations
  - Agent performance evaluation and coaching opportunities
  - Identification of common customer issues for process improvement

- **Media & Entertainment**
  - Automated podcast transcription and summarization
  - Content moderation for user-generated audio
  - Content indexing and search for audio libraries
  - Metadata generation for improved content discovery

- **Healthcare**
  - Medical dictation transcription with specialty term recognition
  - Patient interaction analysis for improved care
  - Documentation assistance for clinical encounters
  - Compliance monitoring for patient communications

- **Financial Services**
  - Earnings call analysis for investment insights
  - Advisory call compliance monitoring
  - Client interaction analysis for relationship management
  - Fraud detection in phone banking interactions

### ROI Opportunities

- Reduce manual transcription costs (typically $1-2 per audio minute)
- Increase process efficiency by 60-80% for audio content management
- Unlock previously inaccessible insights from archived audio content
- Enable new use cases through audio content searchability
""",
            "sources": [
                {"text": "AWS Blog: Contact Center Intelligence with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"},
                {"text": "Case Study: Media Organization Improves Content Discovery", "url": "https://aws.amazon.com/solutions/case-studies/?awsf.customer-references-filter-category=*all&awsm.page-customer-references=1"}
            ]
        },
        "processing_pipeline": {
            "title": "Behind the Scenes of Audio Processing",
            "content": """
When you invoke Amazon Bedrock Data Automation for audio, the service executes a sophisticated processing pipeline:

### Audio Processing Steps

1. **Audio Ingestion**: The system loads and preprocesses the audio file from S3
2. **Speech Recognition**: Advanced AI models convert speech to text with high accuracy
3. **Speaker Diarization**: The system identifies and labels different speakers
4. **Content Analysis**: AI models detect topics, sentiment, and potentially sensitive content
5. **Summarization**: Generative AI creates summaries of the overall audio and key topics
6. **Result Formation**: All extracted data is assembled into the requested output formats

### Performance Considerations

- Processing time depends on audio length, quality, and number of speakers
- Multi-speaker audio requires more processing for accurate speaker separation
- Ambient noise and audio quality impact transcription accuracy
- Typical processing times range from seconds to a few minutes for longer recordings
""",
            "sources": [
                {"text": "Invoking Amazon Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-invoke.html"}            
            ]
        },
        "speaker_identification": {
            "title": "Speaker Identification in Business",
            "content": """
Speaker diarization (identifying who said what) is a powerful capability that transforms how organizations 
understand conversations and interactions.

### Key Applications

- **Meeting Intelligence**: Accurately attribute comments and action items to specific participants
- **Customer Service Quality**: Analyze agent vs. customer speaking patterns and interactions
- **Multi-Party Negotiations**: Track positions and commitments from different stakeholders
- **Interview Analysis**: Separate interviewer and candidate responses for systematic evaluation
- **Panel Discussions**: Identify and track different expert opinions on discussed topics

### Business Impact

- **Communication Analysis**: Identify dominant speakers, interruption patterns, and conversation flow
- **Personalization**: Build speaker profiles based on individual speaking patterns and topics
- **Knowledge Management**: Attribute insights and expertise to specific individuals
- **Compliance**: Verify that required statements were made by authorized individuals
- **Training**: Create better conversational AI by understanding human dialogue patterns

### Technical Evolution

Modern speaker diarization systems can identify speakers even when they briefly overlap, 
distinguish between similar voices, and maintain speaker identity throughout long recordings.
This capability was previously error-prone and limited, but generative AI has dramatically
improved accuracy and robustness.
""",
            "sources": [
                {"text": "AWS Blog: Speaker Diarization with Amazon Transcribe", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-transcribe/"},
                {"text": "Research: Advances in Speaker Diarization", "url": "https://arxiv.org/abs/2303.06876"}
            ]
        },
        "audio_content_moderation": {
            "title": "Audio Content Moderation",
            "content": """
Content moderation for audio is a critical capability for organizations that process user-generated
or public audio content. BDA provides sophisticated detection of potentially sensitive or inappropriate content.

### Business Applications

- **Brand Safety**: Ensure audio content associated with your brand meets appropriate standards
- **User-Generated Content**: Screen uploaded audio for policy violations
- **Compliance**: Identify regulatory violations in recorded communications
- **Risk Management**: Detect potential legal or reputational risks in audio content
- **Safety**: Create safer spaces in audio-based communities and platforms

### Moderation Categories

BDA can detect various categories of potentially inappropriate content:

- **Profanity**: Detection of explicit language
- **Hate Speech**: Identification of discriminatory or hateful content
- **Violence**: References to violent actions or threats
- **Sexual Content**: Detection of explicit sexual references
- **Harassment**: Identification of bullying or abusive language
- **Graphic Content**: References to disturbing or graphic descriptions

### Implementation Approaches

Organizations typically implement audio moderation using one of these patterns:

1. **Pre-publication Review**: Screen content before making it publicly available
2. **Post-publication Monitoring**: Continuously analyze published content
3. **Flagging System**: Automatically flag potential issues for human review
4. **Risk Scoring**: Calculate content risk scores to prioritize review
""",
            "sources": [
                {"text": "Content Moderation Best Practices", "url": "https://aws.amazon.com/solutions/implementations/content-moderation/"},
                {"text": "AWS Blog: Content moderation using artificial intelligence", "url": "https://aws.amazon.com/blogs/machine-learning/content-moderation-using-artificial-intelligence/"}
            ]
        },
        "audio_summarization": {
            "title": "Audio Summarization Applications",
            "content": """
Automatic summarization of audio content creates tremendous business value by making spoken 
information more accessible and actionable.

### Business Use Cases

- **Meeting Summaries**: Extract key discussion points, decisions, and action items
- **Call Center Insights**: Summarize customer interactions for quick review and analysis
- **Media Content**: Create concise summaries of podcasts, interviews, and presentations
- **Educational Content**: Generate lecture summaries for improved learning and review
- **Legal Proceedings**: Summarize testimonies, hearings, and depositions

### Types of Summarization

BDA provides multiple summarization approaches:

1. **Overall Audio Summary**: A comprehensive summary of the entire audio content
2. **Topic Summaries**: Individual summaries for each detected topic or segment
3. **Extractive Summaries**: Key verbatim quotes and statements from the audio
4. **Abstractive Summaries**: AI-generated text that captures the essence of the content

### Integration Patterns

Organizations leverage audio summarization through:

- **Knowledge Management**: Indexing audio content summaries in knowledge bases
- **Documentation**: Automating meeting notes and call records
- **Search Enhancement**: Making audio content discoverable through summary text
- **Workflow Integration**: Triggering actions based on summarized content
- **Dashboard Reporting**: Including audio insights in business intelligence
""",
            "sources": [
                {"text": "AWS Blog: Summarization with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"},
                {"text": "Research: Advances in Abstractive Summarization", "url": "https://arxiv.org/abs/2210.11248"}
            ]
        },
        "project_architecture": {
            "title": "Understanding BDA Projects for Audio",
            "content": """
Projects are a fundamental architectural component in Amazon Bedrock Data Automation. For audio processing,
they serve several critical purposes:

1. **Audio Configuration Profiles**: Projects store processing configurations optimized for different audio types
   (podcasts, calls, meetings, presentations) ensuring consistent results.

2. **Processing Pipeline Separation**: Different projects can be created for different audio types or processing needs,
   such as transcription-focused vs. summarization-focused.

3. **Versioning and Evolution**: As audio processing needs evolve, projects can be updated while maintaining 
   backward compatibility with existing workflows.

4. **Access Control**: Projects can have specific IAM permissions, allowing you to control which teams or services 
   can access specific audio processing configurations.

5. **Cost Management**: By organizing processing jobs under projects, you can track and allocate costs to 
   specific business initiatives or departments.

In production environments, you would typically create different projects for different audio types or processing 
requirements, each with its own optimized configuration.
""",
            "sources": [
                {"text": "Working with Projects in Bedrock Data Automation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda-projects.html"},
                {"text": "AWS Blog: Enterprise Audio Processing with Amazon Bedrock Data Automation", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"}
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

def ensure_bda_results_dir():
    """Ensure the BDA results directory exists."""
    os.makedirs('../bda-results', exist_ok=True)
    print("Ensured BDA results directory exists.")

# Add backward compatibility alias
BDAUtils = BDAAudioUtils
