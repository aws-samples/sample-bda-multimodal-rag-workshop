"""
Consolidated utilities for BDA video workshops.
This file provides comprehensive functionality for video processing with Amazon Bedrock Data Automation.
"""

import boto3
import json
import uuid
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import Video, clear_output, HTML, display, Markdown
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import sagemaker

class BDAVideoUtils:
    def __init__(self):
        # Initialize AWS session and clients
        self.session = sagemaker.Session()
        self.default_bucket = self.session.default_bucket()
        self.current_region = boto3.session.Session().region_name
        
        self.sts = boto3.client('sts')
        self.account_id = self.sts.get_caller_identity()['Account']
        
        # Initialize BDA clients
        self.bda_client = boto3.client('bedrock-data-automation')
        self.bda_runtime_client = boto3.client('bedrock-data-automation-runtime')
        self.s3_client = boto3.client('s3')
        
        # Define S3 locations
        self.data_prefix = "bda-workshop/video"
        self.output_prefix = "bda-workshop/video/output"
    
    def download_video(self, url, output_path):
        """Download a video from a URL with enhanced error handling"""
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
            print(f"Error downloading video: {e}")
            raise
    
    def upload_to_s3(self, local_path, s3_key):
        """Upload a local file to S3"""
        self.s3_client.upload_file(local_path, self.default_bucket, s3_key)
        return f's3://{self.default_bucket}/{s3_key}'
    
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
        objects = self.s3_client.list_objects_v2(Bucket=self.default_bucket, Prefix=prefix)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                self.s3_client.delete_object(Bucket=self.default_bucket, Key=obj['Key'])
    
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
    
    def generate_shot_images(self, video_path, result_data, image_width=120):
        """Generate images for each shot in the video"""
        images = []
        
        # Load the video
        clip = VideoFileClip(video_path)
        
        # Extract shots
        shots = result_data.get("shots", [])
        
        # Generate an image for each shot
        for shot in shots:
            start_time = shot["start_timestamp_millis"] / 1000  # Convert to seconds
            frame = clip.get_frame(start_time)
            images.append((start_time, frame))
        
        clip.close()
        return images
    
    def plot_shots(self, images):
        """Plot shot images in a grid with enhanced visualization"""
        if not images:
            print("No shots to display")
            return
        
        # Calculate grid dimensions
        n_images = len(images)
        cols = 5
        rows = (n_images + cols - 1) // cols
        
        # Create figure with improved styling
        plt.figure(figsize=(15, rows * 3))
        
        # Plot each image with enhanced annotations
        for i, (time, img) in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"Shot {i+1}: {time:.2f}s", fontsize=10, fontweight='bold')
            
            # Add subtle border to highlight the shot
            plt.gca().spines['bottom'].set_color('#1f77b4')
            plt.gca().spines['top'].set_color('#1f77b4')
            plt.gca().spines['right'].set_color('#1f77b4')
            plt.gca().spines['left'].set_color('#1f77b4')
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['top'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            
            plt.axis('off')
        
        plt.suptitle(f"Video Shot Analysis: {n_images} Distinct Visual Segments Detected", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Add room for the title
        plt.show()
        
        print(f"\n🎬 Technical Win: BDA identified {n_images} distinct visual segments in your video!")
        print("This capability enables precise navigation, content indexing, and context-aware understanding of your video.")
    
    def plot_content_moderation(self, video_path, result_data, chapter_index):
        """Plot frames with content moderation labels with enhanced visualization"""
        if chapter_index >= len(result_data["chapters"]):
            print(f"Chapter index {chapter_index} out of range")
            return
        
        chapter = result_data["chapters"][chapter_index]
        clip = VideoFileClip(video_path)
        
        moderation_frames = []
        
        for frame in chapter.get("frames", []):
            if "content_moderation" in frame:
                # Get frame time
                frame_time = frame["timestamp_millis"] / 1000
                
                # Get frame image
                frame_img = clip.get_frame(frame_time)
                
                # Get moderation categories
                categories = []
                confidences = []
                
                for mod in frame["content_moderation"]:
                    categories.append(mod["category"])
                    confidences.append(float(mod["confidence"]))
                
                moderation_frames.append((frame_time, frame_img, categories, confidences))
        
        clip.close()
        
        if not moderation_frames:
            print("No content moderation data found in this chapter")
            return
        
        # Create a more sophisticated visualization
        for frame_time, frame_img, categories, confidences in moderation_frames:
            plt.figure(figsize=(15, 6))
            
            # Left: Plot the video frame
            plt.subplot(1, 2, 1)
            plt.imshow(frame_img)
            plt.title(f"Frame at {frame_time:.2f}s", fontweight='bold')
            plt.axis('off')
            
            # Right: Plot content moderation data with improved styling
            ax = plt.subplot(1, 2, 2)
            bars = plt.barh(categories, confidences, color='#FF9999')
            
            # Add threshold line
            plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
            plt.text(0.51, -0.5, 'Risk Threshold (0.5)', color='red', fontsize=9)
            
            # Highlight bars exceeding threshold
            for i, confidence in enumerate(confidences):
                if confidence > 0.5:
                    bars[i].set_color('#FF3333')
                    plt.text(confidence + 0.02, i, f"{confidence:.2f}", va='center', fontweight='bold')
                else:
                    plt.text(confidence + 0.02, i, f"{confidence:.2f}", va='center')
            
            plt.title("Content Moderation Analysis", fontweight='bold')
            plt.xlabel("Confidence Score")
            plt.xlim(0, 1.1)
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        print(f"\n🛡️ Technical Win: BDA automatically detected potentially sensitive content!")
        print("This powerful moderation capability helps ensure content safety and compliance.")
    
    def visualize_chapters(self, result_data, figsize=(14, 6)):
        """Visualize chapters timeline with summaries"""
        if "chapters" not in result_data or not result_data["chapters"]:
            print("No chapter data available for visualization")
            return
            
        chapters = result_data["chapters"]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Calculate chapter durations and positions
        total_duration = 0
        for chapter in chapters:
            start = chapter.get("start_timestamp_millis", 0) / 1000  # in seconds
            end = chapter.get("end_timestamp_millis", 0) / 1000
            if end > total_duration:
                total_duration = end
                
        # Create color palette for chapters
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(chapters)))
        
        # Plot chapters as segments on a timeline
        y_pos = 0
        for i, chapter in enumerate(chapters):
            start = chapter.get("start_timestamp_millis", 0) / 1000  # in seconds
            end = chapter.get("end_timestamp_millis", 0) / 1000
            duration = end - start
            width = duration / total_duration
            
            # Plot chapter bar
            plt.barh(y_pos, width, left=start/total_duration, height=0.6, color=colors[i], alpha=0.7)
            
            # Add chapter label
            plt.text(start/total_duration + width/2, y_pos, f"Ch.{i+1}", 
                    ha='center', va='center', color='white', fontweight='bold')
            
            # Plot start/end times
            plt.text(start/total_duration, y_pos+0.5, f"{start:.1f}s", ha='center', va='bottom', fontsize=8)
            plt.text(min((start/total_duration + width), 0.98), y_pos+0.5, f"{end:.1f}s", ha='center', va='bottom', fontsize=8)
        
        # Configure plot
        plt.yticks([])
        plt.xticks([])
        plt.xlim(0, 1)
        plt.ylim(-0.5, 1)
        plt.title("Video Chapter Structure", fontweight='bold')
        
        # Add chapter summaries table below
        plt.figtext(0.5, -0.05, "Chapter summaries:", ha='center', fontsize=12, fontweight='bold')
        for i, chapter in enumerate(chapters):
            summary = chapter.get("summary", "No summary available")
            if len(summary) > 100:
                summary = summary[:100] + "..."
            plt.figtext(0.5, -0.15 - (i*0.07), f"Chapter {i+1}: {summary}", ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Adjust figure size to accommodate summaries
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.1 + (len(chapters) * 0.07))
        
        plt.show()
        
        print(f"\n📊 Technical Win: BDA automatically divided the video into {len(chapters)} meaningful chapters!")
        print("This capability enables semantic understanding, improved navigation, and content structuring.")
    
    def visualize_iab_categories(self, result_data):
        """Visualize IAB categories detected in the video"""
        if "chapters" not in result_data or not result_data["chapters"]:
            print("No chapter data available for IAB visualization")
            return
            
        # Collect all IAB categories
        category_counts = {}
        
        for chapter in result_data["chapters"]:
            if "iab_categories" in chapter:
                for iab in chapter["iab_categories"]:
                    category = iab["category"]
                    confidence = float(iab.get("confidence", 0))
                    
                    if confidence > 0.5:  # Only count high-confidence categories
                        if category in category_counts:
                            category_counts[category] += 1
                        else:
                            category_counts[category] = 1
        
        if not category_counts:
            print("No IAB categories detected with confidence > 0.5")
            return
            
        # Sort categories by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        categories = [x[0] for x in sorted_categories]
        counts = [x[1] for x in sorted_categories]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bars for better readability of category names
        colors = plt.cm.tab20c(np.linspace(0, 1, len(categories)))
        bars = plt.barh(categories, counts, color=colors)
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f"{int(width)}", ha='left', va='center')
        
        plt.title("Internet Advertising Bureau (IAB) Categories Detected", fontweight='bold')
        plt.xlabel("Number of Occurrences")
        plt.tight_layout()
        plt.show()
        
        print("\n🏷️ Technical Win: BDA automatically classified your video content into IAB categories!")
        print("This enables improved content discovery, ad targeting, and content recommendations.")

def show_business_context(context_type=None, auto_initialize=False):
    """
    Display business context information for a specific context type.
    Uses a simpler approach for better compatibility with Jupyter notebooks.
    
    Args:
        context_type (str): The type of business context to display
        auto_initialize (bool): If True, initialize but don't display (for use in imports)
    """
    contexts = {
        "video_complete": {
            "title": "Dynamic Content Understanding: Business Value of Video Analysis",
            "content": """
Video is the most information-dense form of content, combining visual elements, motion, audio, and text in a temporal flow. 
Amazon Bedrock Data Automation (BDA) unlocks the hidden value in video content by extracting structured insights from these 
complex, multimodal assets.

### Core Business Capabilities

- **Visual Understanding**: Extract scenes, objects, people, text, and logos from video frames
- **Temporal Analysis**: Understand how content changes over time with chapter detection
- **Content Classification**: Automatically categorize video content using IAB taxonomies
- **Multimodal Integration**: Combine visual analysis with transcript and audio processing
- **Content Moderation**: Identify potentially sensitive or inappropriate content across multiple dimensions
- **Summarization**: Generate concise overviews of video content for improved discoverability

### Industry Applications

- **Media & Entertainment**
  - Automated content cataloging and metadata generation
  - Content moderation for user-generated videos
  - Personalized content recommendations based on visual elements
  - Enhanced search capabilities through extracted visual concepts

- **Advertising & Marketing**
  - Brand and logo detection in sponsored content
  - Competitive analysis of video marketing campaigns
  - Audience engagement analysis for video content
  - Contextual ad placement based on video topics and scenes

- **Retail & E-commerce**
  - Product detection and feature extraction from video reviews
  - Visual merchandising analysis from store videos
  - Creation of shoppable video experiences
  - Trend analysis from fashion and lifestyle videos

- **Security & Compliance**
  - Content policy enforcement for video platforms
  - Identification of unauthorized brand usage
  - Detection of potentially harmful or unsafe content
  - Monitoring of video content for regulatory compliance

### ROI Opportunities

- Reduce manual review costs by 60-80% through automated video analysis
- Increase video asset value by 40-50% through enhanced discoverability
- Accelerate time-to-market for video content by automating metadata creation
- Enable new revenue streams through improved content understanding

### Video Analysis Evolution

Before generative AI, video analysis required complex pipelines with separate models for object detection, 
scene analysis, text recognition, and content classification. These systems were costly, slow, and often 
inaccurate. BDA unifies these capabilities into a single, powerful API that delivers comprehensive video insights 
with unprecedented speed and accuracy.
""",
            "sources": [
                {"text": "Amazon Bedrock Data Automation Documentation", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bda.html"},
                {"text": "AWS Blog: Video Analytics with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"},
                {"text": "AWS Blog: Content Moderation with Generative AI", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"},
                {"text": "Case Study: Media Organization Improves Video Discoverability", "url": "https://aws.amazon.com/solutions/case-studies/?awsf.customer-references-filter-category=*all&awsm.page-customer-references=1"}
            ]
        },
        "chapter_detection": {
            "title": "Chapter Detection: The Backbone of Video Understanding",
            "content": """
Chapter detection is a foundational capability that transforms how organizations work with video content.
By automatically segmenting videos into meaningful chapters, BDA enables deeper understanding and more
efficient video content management.

### Business Applications

- **Improved Navigation**: Enable precise in-video navigation through semantic chapters
- **Content Indexing**: Create more granular, topic-based video indexes
- **Targeted Analysis**: Apply specialized analysis to specific video segments
- **Partial Content Reuse**: Identify self-contained segments for content repurposing
- **Enhanced Search**: Map user queries to specific video segments rather than entire videos

### How Chapter Detection Works

BDA uses multiple signals to identify meaningful chapter boundaries:

1. **Visual Scene Changes**: Major shifts in visual content
2. **Topic Transitions**: Changes in the subject matter being discussed
3. **Speaker Changes**: New speakers or conversation participants
4. **Narrative Structures**: Introduction, main content, and conclusion sections

### Implementation Strategies

Organizations leverage chapter detection in several ways:

1. **Video Libraries**: Enable chapter-based navigation in video platforms
2. **Content Management**: Create chapter-level metadata in DAM systems
3. **Educational Content**: Divide learning videos into concept-based segments
4. **Video Analytics**: Measure engagement at the chapter level
5. **Content Summarization**: Generate summaries for individual chapters
""",
            "sources": [
                {"text": "AWS Blog: Video Indexing with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"},
                {"text": "Research: Multimodal Video Segmentation", "url": "https://arxiv.org/abs/2201.04850"}
            ]
        },
        "scene_detection": {
            "title": "Scene Detection and Shot Analysis",
            "content": """
Scene and shot detection form the foundation of visual understanding in video processing.
This capability breaks down videos into their fundamental visual building blocks.

### Key Concepts

- **Shot**: A continuous segment of video from a single camera perspective
- **Scene**: A collection of shots that form a coherent narrative unit
- **Transition**: The boundary between shots (cut, fade, dissolve, wipe, etc.)

### Business Applications

- **Content Indexing**: Create visual indexes of video content
- **Thumbnail Generation**: Automatically select representative frames for each scene
- **Content Summarization**: Create visual summaries using key frames
- **Editing Analysis**: Study pacing and visual structure of professional videos
- **Content Comparisons**: Compare visual structures across multiple videos

### Technical Evolution

Modern shot detection systems can identify:
- Hard cuts (immediate transitions)
- Soft transitions (fades, dissolves, wipes)
- Camera movements (pans, tilts, zooms)
- Visual consistency within shots

This analysis forms the basis for higher-level understanding of visual narratives.
""",
            "sources": [
                {"text": "AWS Blog: Visual Content Analysis", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-rekognition/"},
                {"text": "Research: Deep Learning for Shot Boundary Detection", "url": "https://arxiv.org/abs/1705.08214"}
            ]
        },
        "content_moderation": {
            "title": "Video Content Moderation",
            "content": """
Content moderation for video is essential for organizations that process user-generated or 
public video content. BDA provides sophisticated detection of potentially sensitive or inappropriate 
content across both visual and audio components.

### Business Applications

- **Platform Safety**: Ensure video platforms provide safe experiences for all users
- **Brand Protection**: Prevent association with inappropriate or harmful content
- **Legal Compliance**: Meet regulatory requirements for content moderation
- **User Trust**: Build confidence through consistent content standards
- **Advertiser Protection**: Ensure ad placement only on appropriate content

### Moderation Dimensions

BDA can detect various categories of potentially inappropriate content:

- **Visual Moderation**: Detection of sensitive images, gestures, objects
- **Audio Moderation**: Detection of inappropriate speech, sounds, music
- **Text Moderation**: Detection of on-screen text containing sensitive content
- **Contextual Moderation**: Understanding content in context rather than isolation

### Implementation Approaches

Organizations typically implement video moderation using one of these patterns:

1. **Pre-publication Review**: Screen content before making it publicly available
2. **Post-publication Monitoring**: Continuously analyze published content
3. **Hybrid Approach**: Automated moderation with human review for edge cases
4. **Progressive Moderation**: Start with strict automated rules and adjust based on performance
""",
            "sources": [
                {"text": "Content Moderation Best Practices", "url": "https://aws.amazon.com/solutions/implementations/content-moderation/"},
                {"text": "AWS Blog: Content moderation using artificial intelligence", "url": "https://aws.amazon.com/blogs/machine-learning/content-moderation-using-artificial-intelligence/"}
            ]
        },
        "video_text_detection": {
            "title": "Text Detection in Videos",
            "content": """
Text detection in videos unlocks valuable information that would otherwise remain inaccessible in the visual stream.
BDA can identify, extract, and make searchable text that appears in frames throughout a video.

### Business Applications

- **Subtitle Verification**: Ensure on-screen text matches spoken content
- **Brand Compliance**: Detect unauthorized text or disclaimers
- **Information Extraction**: Capture statistics, names, dates displayed on screen
- **Document Digitization**: Extract text from documents shown in videos
- **UI/UX Analysis**: Analyze text elements in screen recordings or demos

### Types of Video Text

BDA can detect various types of text in videos:

- **Overlay Text**: Captions, titles, and graphics added in post-production
- **Scene Text**: Text naturally appearing within the filmed environment
- **Document Text**: Text appearing in documents shown in videos
- **Interface Text**: Text in UI elements in screen recordings

### Technical Considerations

- Text detection works across multiple languages
- Motion and partial occlusion handling improves detection in dynamic scenes
- Text tracking across frames ensures consistent extraction
- Confidence scores help prioritize high-quality text extractions
""",
            "sources": [
                {"text": "AWS Blog: Video Text Detection with Amazon Rekognition", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-rekognition/"},
                {"text": "Research: Text Detection and Recognition in Videos", "url": "https://arxiv.org/abs/1904.12581"}
            ]
        },
        "logo_detection": {
            "title": "Logo Detection in Videos",
            "content": """
Logo detection provides powerful brand intelligence by identifying commercial marks and symbols
throughout video content. This capability enables brand monitoring, competitor analysis, and
sponsorship valuation.

### Business Applications

- **Brand Exposure Measurement**: Quantify logo appearances in broadcasts
- **Sponsorship Valuation**: Measure actual airtime of sponsored logos
- **Competitive Intelligence**: Track competitor brand appearances
- **Brand Safety**: Ensure brands aren't associated with inappropriate content
- **Content Monetization**: Identify opportunities for brand placement

### Logo Detection Metrics

BDA provides rich logo detection data:

- **Logo Identification**: Which logos appear in the video
- **Temporal Analysis**: When and for how long logos appear
- **Spatial Analysis**: Where on screen the logos appear
- **Prominence Analysis**: How prominent or visible the logos are
- **Context Analysis**: The surrounding content where logos appear

### Implementation Approaches

Organizations leverage logo detection through:

1. **Brand Monitoring**: Track your brand's presence across media
2. **Competitive Analysis**: Compare your brand exposure to competitors
3. **Content Valuation**: Determine value of media based on brand exposure
4. **Media Planning**: Measure ROI of sponsorship and placement investments
""",
            "sources": [
                {"text": "AWS Blog: Logo Detection with Amazon Rekognition", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-rekognition/"},
                {"text": "Research: Deep Logo Detection in Videos", "url": "https://arxiv.org/abs/1812.00800"}
            ]
        },
        "iab_categorization": {
            "title": "IAB Categorization for Video Content",
            "content": """
IAB categorization provides standardized content classification following the Internet Advertising Bureau (IAB)
taxonomy. This allows organizations to consistently categorize content for advertising, recommendation, 
and discovery purposes.

### Business Applications

- **Ad Targeting**: Align advertisements with relevant content categories
- **Content Discovery**: Improve recommendation systems with standardized categories
- **Content Organization**: Create consistent taxonomies across video libraries
- **Audience Analysis**: Understand viewer interests through content preferences
- **Brand Safety**: Identify content categories to include or exclude for ad placement

### IAB Taxonomy

The IAB Content Taxonomy includes categories such as:

- **Arts & Entertainment**: Movies, Television, Music, Celebrities
- **Business**: Finance, Marketing, Management, Entrepreneurship
- **Computers & Electronics**: Software, Hardware, Mobile, Gaming
- **Health & Fitness**: Exercise, Nutrition, Medical, Wellness
- **Home & Garden**: Interior Design, DIY, Gardening, Real Estate
- **News & Politics**: Current Events, Government, Economy
- **Sports**: Team Sports, Individual Sports, Outdoor Activities
- **Travel**: Destinations, Hotels, Transportation, Tourism

### Implementation Approaches

Organizations apply IAB categorization through:

1. **Content Tagging**: Automatically tag videos with standardized categories
2. **Ad Inventory**: Create categorized ad inventory for programmatic buying
3. **Content Filtering**: Enable category-based filtering for viewers
4. **Analytics**: Track performance and engagement by content category
""",
            "sources": [
                {"text": "IAB Content Taxonomy", "url": "https://www.iab.com/guidelines/content-taxonomy/"},
                {"text": "AWS Blog: Content Classification with Amazon Bedrock", "url": "https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-bedrock/"}
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
BDAUtils = BDAVideoUtils
